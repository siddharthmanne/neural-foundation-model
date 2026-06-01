"""Modal finetune entrypoint for BrainTokenizer on THINGS-MEG (Phase 3b)."""

from __future__ import annotations

from pathlib import Path
import json
import sys

try:
    import neural_tokenizers  # noqa: F401
except ImportError:
    try:
        _REPO_ROOT = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(_REPO_ROOT))
    except IndexError:
        pass

import modal  # noqa: E402

app = modal.App("neural-fm")
project_volume = modal.Volume.from_name("project")
PROJECT_MOUNT = "/project"
BRAINOMNI_REPO = "/root/external/BrainOmni"
CKPT_ROOT = f"{PROJECT_MOUNT}/checkpoints/meg/brainomni"

brainomni_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "mne",
        "scipy",
        "torch",
        "numpy",
        "einops",
        "vector-quantize-pytorch",
        "einx",
        "huggingface_hub",
    )
    .add_local_dir(
        "external/BrainOmni",
        remote_path=BRAINOMNI_REPO,
        ignore=["ckpt_collection", ".cache", "**/__pycache__"],
    )
    .add_local_python_source("neural_tokenizers")
)


def _cap_split_indices(
    indices_per_subject: dict[str, np.ndarray],
    max_trials: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Subsample split indices *before* loading .fif data into RAM."""
    import numpy as np

    if max_trials <= 0:
        return indices_per_subject
    pairs: list[tuple[str, int]] = []
    for subj, idx in indices_per_subject.items():
        for t in idx:
            pairs.append((subj, int(t)))
    if not pairs:
        return indices_per_subject
    rng = np.random.default_rng(seed)
    n = min(max_trials, len(pairs))
    chosen = rng.choice(len(pairs), size=n, replace=False)
    per: dict[str, list[int]] = {}
    for k in chosen:
        subj, t = pairs[int(k)]
        per.setdefault(subj, []).append(t)
    return {k: np.asarray(v, dtype=np.int64) for k, v in per.items()}


@app.function(
    image=brainomni_image,
    volumes={PROJECT_MOUNT: project_volume},
    gpu="L40S",
    cpu=8.0,
    # 64 GB: accommodates the cross-subject averaging path which holds the
    # raw single-trial tensor (~27 GB) AND the f64 sums tensor (~13 GB)
    # concurrently, then frees both after the averaged tensor (~7 GB)
    # is built. 32 GB was enough for non-averaged training but not this.
    memory=64 * 1024,
    timeout=6 * 60 * 60,
)
def finetune_remote(
    lr: float = 1e-5,
    codebook_lr: float = 3e-5,
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 0,
    stage: str = "3b",
    max_train_trials: int = 0,
    max_val_trials: int = 0,
    finetune_mode: str = "adapt",
    grad_clip: float = 1.0,
    codebook_size: int = 0,
    patience: int = 2,
    min_epochs: int = 3,
    averaging: str = "none",
) -> dict:
    """Finetune BrainTokenizer on THINGS-MEG train split."""
    import os
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, BRAINOMNI_REPO)

    from dataclasses import replace

    from neural_tokenizers.meg import BRAINOMNI_DEFAULT, LEARNABLE_SPLIT_DEFAULTS, SplitDefaults
    from neural_tokenizers.meg.brainomni.checkpoint import resolve_ckpt_dir
    from neural_tokenizers.meg.brainomni.config import run_slug
    from neural_tokenizers.meg.brainomni.load import load_braintokenizer
    from neural_tokenizers.meg.brainomni.preprocess import preprocess_for_braintokenizer
    from neural_tokenizers.meg.brainomni.sensor_metadata import load_things_meg_sensor_metadata
    from neural_tokenizers.meg.brainomni.trainer import (
        FinetuneConfig,
        apply_finetune_mode,
        build_optimizer_groups,
        compute_braintokenizer_loss,
    )
    from neural_tokenizers.meg.data import (
        average_trials_by_image,
        list_subjects,
        load_trials_pooled,
    )
    from neural_tokenizers.meg.splits import split_by_image

    ft_cfg = FinetuneConfig(
        mode=finetune_mode,  # type: ignore[arg-type]
        lr=lr,
        codebook_lr=codebook_lr,
        batch_size=batch_size,
        epochs=epochs,
        grad_clip=grad_clip,
        patience=patience,
        min_epochs=min_epochs,
    )
    tok_cfg = (
        replace(BRAINOMNI_DEFAULT, codebook_size=codebook_size)
        if codebook_size > 0
        else BRAINOMNI_DEFAULT
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_cfg = SplitDefaults(
        train_frac=LEARNABLE_SPLIT_DEFAULTS.train_frac,
        val_frac=LEARNABLE_SPLIT_DEFAULTS.val_frac,
        test_frac=LEARNABLE_SPLIT_DEFAULTS.test_frac,
        seed=seed,
    )

    subjects = list_subjects()
    train_per_subj: dict[str, np.ndarray] = {}
    val_per_subj: dict[str, np.ndarray] = {}
    for s in subjects:
        sp = split_by_image(s.image_ids, split_cfg)
        train_per_subj[s.subject] = sp.train
        val_per_subj[s.subject] = sp.val

    train_per_subj = _cap_split_indices(train_per_subj, max_train_trials, seed)
    val_per_subj = _cap_split_indices(val_per_subj, max_val_trials, seed + 1)

    # Single-trial vs cross-subject-averaged data loading. The split is the
    # same image_id partition either way — averaging collapses all trials
    # of a given image_id (across subjects and reps) into one signal, so
    # the cross-image split structure is preserved.
    if averaging == "none":
        X_train, _, _ = load_trials_pooled(subjects, train_per_subj)
        X_val, _, _ = load_trials_pooled(subjects, val_per_subj)
    elif averaging == "cross_subject":
        X_train_raw, train_iids, _ = load_trials_pooled(subjects, train_per_subj)
        X_val_raw, val_iids, _ = load_trials_pooled(subjects, val_per_subj)
        X_train, _ = average_trials_by_image(X_train_raw, train_iids)
        X_val, _ = average_trials_by_image(X_val_raw, val_iids)
        del X_train_raw, X_val_raw
        print(
            f"[finetune] cross-subject averaging: "
            f"{len(train_iids)} train trials → {X_train.shape[0]} averaged | "
            f"{len(val_iids)} val trials → {X_val.shape[0]} averaged"
        )
    else:
        raise ValueError(
            f"averaging must be 'none' or 'cross_subject'; got {averaging!r}"
        )
    print(f"[finetune] train={tuple(X_train.shape)} val={tuple(X_val.shape)}")

    ckpt_dir = resolve_ckpt_dir(None, BRAINOMNI_REPO)
    cb_override = codebook_size if codebook_size > 0 else None
    model = load_braintokenizer(
        ckpt_dir,
        BRAINOMNI_REPO,
        device=device,
        eval_mode=False,
        codebook_size=cb_override,
    )
    if cb_override:
        print(f"[finetune] codebook_size={cb_override} (quantizer reinit if != pretrained)")
    param_counts = apply_finetune_mode(model, ft_cfg.mode)
    print(
        f"[finetune] mode={ft_cfg.mode} "
        f"trainable={param_counts['trainable']:,} / {param_counts['total']:,} params"
    )

    sensor_meta = load_things_meg_sensor_metadata()
    pos_1, st_1 = sensor_meta.batch(1, device)
    pos_1 = pos_1.squeeze(0)
    st_1 = st_1.squeeze(0)

    optimizer = torch.optim.AdamW(build_optimizer_groups(model, ft_cfg))

    def _step(x_batch: torch.Tensor, train: bool) -> float:
        x_batch = x_batch.to(device)
        x_pad, _, mask = preprocess_for_braintokenizer(x_batch, tok_cfg)
        b = x_pad.shape[0]
        pos = pos_1.unsqueeze(0).expand(b, -1, -1)
        st = st_1.unsqueeze(0).expand(b, -1)
        if train:
            model.train()
            loss, _ = compute_braintokenizer_loss(model, x_pad, pos, st, mask, tok_cfg)
            optimizer.zero_grad()
            loss.backward()
            if ft_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    ft_cfg.grad_clip,
                )
            optimizer.step()
            return loss.item()
        model.eval()
        with torch.no_grad():
            loss, _ = compute_braintokenizer_loss(model, x_pad, pos, st, mask, tok_cfg)
        return loss.item()

    slug = run_slug(cfg=tok_cfg, stage=stage)
    out_dir = Path(CKPT_ROOT) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(ckpt_dir, "model_cfg.json")) as f:
        saved_model_cfg = json.load(f)
    saved_model_cfg["codebook_size"] = tok_cfg.codebook_size
    (out_dir / "model_cfg.json").write_text(json.dumps(saved_model_cfg, indent=2))

    best_val = float("inf")
    epochs_since_best = 0
    history: list[dict[str, float]] = []
    early_stopped_at: int | None = None
    for epoch in range(epochs):
        perm = torch.randperm(X_train.shape[0], generator=torch.Generator().manual_seed(seed + epoch))
        train_sum = 0.0
        n_train = 0
        for start in range(0, X_train.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            train_sum += _step(X_train[idx], train=True)
            n_train += 1

        val_sum = 0.0
        n_val = 0
        for start in range(0, X_val.shape[0], batch_size):
            val_sum += _step(X_val[start : start + batch_size], train=False)
            n_val += 1

        train_avg = train_sum / max(n_train, 1)
        val_avg = val_sum / max(n_val, 1)
        history.append({"epoch": float(epoch), "train_loss": train_avg, "val_loss": val_avg})
        print(f"[finetune] epoch {epoch+1}/{epochs} train={train_avg:.4f} val={val_avg:.4f}")
        if val_avg < best_val:
            best_val = val_avg
            epochs_since_best = 0
            torch.save(model.state_dict(), out_dir / "BrainTokenizer.pt")
            print(f"[finetune] saved best checkpoint val={best_val:.4f}")
        else:
            epochs_since_best += 1
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        project_volume.commit()

        # Early stopping: break out of the schedule once val has stopped
        # improving. Best checkpoint is already on disk (saved each time
        # val_avg < best_val), so the on-disk weights at end == best-val.
        # patience<=0 disables. min_epochs absorbs cold-start volatility.
        if patience > 0 and (epoch + 1) >= min_epochs and epochs_since_best >= patience:
            best_epoch = epoch - epochs_since_best + 1  # 0-indexed epoch of best_val
            print(
                f"[finetune] EARLY STOP — val hasn't improved in {patience} "
                f"consecutive epochs (best={best_val:.4f} at epoch {best_epoch + 1})"
            )
            early_stopped_at = epoch + 1
            break

    result = {
        "slug": slug,
        "ckpt_dir": str(out_dir),
        "brainomni_repo": BRAINOMNI_REPO,
        "stage": stage,
        "finetune_mode": finetune_mode,
        "lr": lr,
        "codebook_lr": codebook_lr,
        "codebook_size": tok_cfg.codebook_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_train_trials": max_train_trials,
        "max_val_trials": max_val_trials,
        "seed": seed,
        "best_val_loss": best_val,
        "param_counts": param_counts,
        "history": history,
        "early_stopped_at": early_stopped_at,
        "patience": patience,
        "min_epochs": min_epochs,
        "averaging": averaging,
        "config": {k: getattr(tok_cfg, k) for k in tok_cfg.__dataclass_fields__},
    }
    (out_dir / "config.json").write_text(json.dumps(result, indent=2))
    project_volume.commit()
    return result


@app.local_entrypoint()
def finetune(
    lr: float = 1e-5,
    codebook_lr: float = 3e-5,
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 0,
    stage: str = "3b",
    max_train_trials: int = 0,
    max_val_trials: int = 0,
    output: str = "",
    lr_sweep: bool = False,
    finetune_mode: str = "adapt",
    grad_clip: float = 1.0,
    codebook_size: int = 0,
    patience: int = 2,
    min_epochs: int = 3,
    averaging: str = "none",
):
    """Launch BrainTokenizer finetune on Modal.

    Default ``finetune_mode=adapt`` freezes SEANet convs, trains sensor/cross-attn/RVQ.
    ``--averaging cross_subject`` collapses all trials of an image_id into one
    averaged signal before training (Experiment 2). Recommend `--patience 2`
    early-stopping for the averaged regime — fewer samples → real overfit risk.
    """
    lrs = [3e-6, 1e-5, 3e-5] if lr_sweep else [lr]
    best: dict | None = None
    for trial_lr in lrs:
        print(f"[finetune] mode={finetune_mode} lr={trial_lr} avg={averaging}")
        result = finetune_remote.remote(
            lr=trial_lr,
            codebook_lr=codebook_lr,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            stage=stage,
            max_train_trials=max_train_trials,
            max_val_trials=max_val_trials,
            finetune_mode=finetune_mode,
            grad_clip=grad_clip,
            codebook_size=codebook_size,
            patience=patience,
            min_epochs=min_epochs,
            averaging=averaging,
        )
        if best is None or result["best_val_loss"] < best["best_val_loss"]:
            best = result
    assert best is not None
    slug = best["slug"]
    local_dir = Path(output or f"neural_tokenizers/meg/brainomni/runs/{slug}")
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "config.json").write_text(json.dumps(best, indent=2))
    print(f"[finetune] mode={best['finetune_mode']} lr={best['lr']} val={best['best_val_loss']:.4f}")
    print(f"[finetune] trainable params: {best['param_counts']}")
    print(f"[finetune] remote ckpt: {best['ckpt_dir']}")
    print(f"[finetune] wrote {local_dir / 'config.json'}")


@app.local_entrypoint()
def finetune_detached(
    lr: float = 1e-5,
    codebook_lr: float = 3e-5,
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 0,
    stage: str = "3b",
    max_train_trials: int = 0,
    max_val_trials: int = 0,
    finetune_mode: str = "adapt",
    grad_clip: float = 1.0,
    codebook_size: int = 0,
    patience: int = 2,
    min_epochs: int = 3,
    averaging: str = "none",
):
    """Spawn finetune on Modal — survives laptop sleep/close.

    MUST be launched with ``modal run --detach ...::finetune_detached``.
    Do NOT use ``finetune`` (``.remote()``) for multi-hour jobs; Modal cancels
    those when the local client disconnects even with ``--detach``.
    """
    from neural_tokenizers.meg.brainomni.config import run_slug
    from dataclasses import replace
    from neural_tokenizers.meg import BRAINOMNI_DEFAULT

    tok_cfg = (
        replace(BRAINOMNI_DEFAULT, codebook_size=codebook_size)
        if codebook_size > 0
        else BRAINOMNI_DEFAULT
    )
    slug = run_slug(cfg=tok_cfg, stage=stage)
    remote_ckpt = f"{CKPT_ROOT}/{slug}"

    print(f"[finetune] spawning mode={finetune_mode} lr={lr} epochs={epochs} avg={averaging}")
    fc = finetune_remote.spawn(
        lr=lr,
        codebook_lr=codebook_lr,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        stage=stage,
        max_train_trials=max_train_trials,
        max_val_trials=max_val_trials,
        finetune_mode=finetune_mode,
        grad_clip=grad_clip,
        codebook_size=codebook_size,
        patience=patience,
        min_epochs=min_epochs,
        averaging=averaging,
    )
    print(f"[finetune] spawned function call: {fc.object_id}")
    print(f"[finetune] checkpoint dir: {remote_ckpt}/")
    print(f"[finetune] per-epoch best ckpt + history.json committed to Modal volume")
    print("[finetune] safe to close laptop — monitor at https://modal.com/apps/neural-fm/main")
    print("\nWhen done, copy config locally from Modal volume or re-run eval with:")
    print(f"  neural_tokenizers/meg/brainomni/runs/{slug}/config.json")


@app.local_entrypoint()
def smoke(
    lr: float = 1e-5,
    batch_size: int = 32,
    epochs: int = 3,
    seed: int = 0,
):
    """Quick sanity run: 1k train / 200 val trials, verify loss decreases."""
    result = finetune_remote.remote(
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        stage="3b_smoke",
        max_train_trials=1000,
        max_val_trials=200,
        finetune_mode="adapt",
    )
    local_dir = Path("neural_tokenizers/meg/brainomni/runs/3b_smoke")
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "config.json").write_text(json.dumps(result, indent=2))
    print("\n[smoke] loss history:")
    for row in result["history"]:
        print(f"  epoch {int(row['epoch'])+1}: train={row['train_loss']:.4f} val={row['val_loss']:.4f}")
    print(f"[smoke] wrote {local_dir / 'config.json'}")


def _summarize_run(result: dict) -> dict:
    """Extract sweep-comparable metrics from one finetune result."""
    hist = result["history"]
    first, last = hist[0], hist[-1]
    return {
        "lr": result["lr"],
        "codebook_size": result.get("codebook_size", result.get("config", {}).get("codebook_size")),
        "epochs": result["epochs"],
        "batch_size": result["batch_size"],
        "best_val_loss": result["best_val_loss"],
        "final_train_loss": last["train_loss"],
        "final_val_loss": last["val_loss"],
        "val_drop": first["val_loss"] - last["val_loss"],
        "still_decreasing": last["val_loss"] < hist[-2]["val_loss"] if len(hist) >= 2 else True,
        "history": hist,
    }


@app.local_entrypoint()
def smoke_sweep(
    epochs: int = 5,
    batch_size: int = 32,
    seed: int = 0,
    train_trials: int = 1000,
    val_trials: int = 200,
):
    """LR smoke sweep on a fixed trial subset — pick hyperparams for the full job.

    Runs BrainOmni downstream LR grid {3e-6, 1e-5, 3e-5} with ``adapt`` mode.
    Total wall time ~15–25 min (3 sequential Modal jobs). Writes
    ``runs/3b_smoke_sweep/summary.json`` with the recommended LR.
    """
    lrs = [3e-6, 1e-5, 3e-5]
    runs: list[dict] = []
    for lr in lrs:
        lr_tag = f"lr{lr:.0e}".replace("+", "")
        print(f"\n[smoke_sweep] === lr={lr:.0e} epochs={epochs} batch={batch_size} ===")
        result = finetune_remote.remote(
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            stage=f"3b_smoke_{lr_tag}",
            max_train_trials=train_trials,
            max_val_trials=val_trials,
            finetune_mode="adapt",
        )
        summary = _summarize_run(result)
        runs.append(summary)
        print(f"[smoke_sweep] lr={lr:.0e} best_val={summary['best_val_loss']:.4f} "
              f"val_drop={summary['val_drop']:.4f}")

    # Prefer lowest best_val; tie-break by larger val_drop (still learning).
    runs_sorted = sorted(
        runs,
        key=lambda r: (r["best_val_loss"], -r["val_drop"]),
    )
    best = runs_sorted[0]

    out_dir = Path("neural_tokenizers/meg/brainomni/runs/3b_smoke_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "recommendation": {
            "lr": best["lr"],
            "epochs_full_job": 10,
            "batch_size": batch_size,
            "finetune_mode": "adapt",
            "rationale": "Lowest smoke val loss on 1k/200 trial subset.",
        },
        "runs": runs,
        "ranking": [{"lr": r["lr"], "best_val_loss": r["best_val_loss"]} for r in runs_sorted],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))

    print("\n[smoke_sweep] === ranking (best val loss) ===")
    for i, r in enumerate(runs_sorted, 1):
        print(
            f"  {i}. lr={r['lr']:.0e}  best_val={r['best_val_loss']:.4f}  "
            f"final_val={r['final_val_loss']:.4f}  val_drop={r['val_drop']:.4f}  "
            f"still_decreasing={r['still_decreasing']}"
        )
    print(f"\n[smoke_sweep] recommended for full job: lr={best['lr']:.0e}, "
          f"batch_size={batch_size}, epochs=10, mode=adapt")
    print(f"[smoke_sweep] wrote {out_dir / 'summary.json'}")
    print("\nFull job command:")
    print(
        f"  modal run neural_tokenizers/meg/modal/modal_meg_finetune_brainomni.py::finetune \\\n"
        f"    --finetune-mode adapt --lr {best['lr']} --epochs 10 "
        f"--batch-size {batch_size} --stage 3b"
    )


@app.local_entrypoint()
def codebook_sweep(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 3e-5,
    seed: int = 0,
    train_trials: int = 1000,
    val_trials: int = 200,
):
    """Codebook-size sweep on a fixed trial subset — pick V for the full job.

    Runs V in {256, 512, 1024} with lr=3e-5 (from LR smoke sweep) and adapt mode.
    Non-512 sizes reinitialize RVQ codebooks from pretrained encoder/decoder weights.
    """
    codebook_sizes = [256, 512, 1024]
    runs: list[dict] = []
    for cb in codebook_sizes:
        print(f"\n[codebook_sweep] === V={cb} epochs={epochs} lr={lr:.0e} ===")
        result = finetune_remote.remote(
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            stage="3b_cb_sweep",
            max_train_trials=train_trials,
            max_val_trials=val_trials,
            finetune_mode="adapt",
            codebook_size=cb,
        )
        summary = _summarize_run(result)
        runs.append(summary)
        print(
            f"[codebook_sweep] V={cb} best_val={summary['best_val_loss']:.4f} "
            f"val_drop={summary['val_drop']:.4f}"
        )

    runs_sorted = sorted(
        runs,
        key=lambda r: (r["best_val_loss"], -r["val_drop"]),
    )
    best = runs_sorted[0]

    out_dir = Path("neural_tokenizers/meg/brainomni/runs/3b_codebook_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "recommendation": {
            "codebook_size": best["codebook_size"],
            "lr": lr,
            "epochs_full_job": 10,
            "batch_size": batch_size,
            "finetune_mode": "adapt",
            "rationale": "Lowest smoke val loss on 1k/200 trial subset (10 epochs each).",
        },
        "runs": runs,
        "ranking": [
            {"codebook_size": r["codebook_size"], "best_val_loss": r["best_val_loss"]}
            for r in runs_sorted
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))

    print("\n[codebook_sweep] === ranking (best val loss) ===")
    for i, r in enumerate(runs_sorted, 1):
        print(
            f"  {i}. V={r['codebook_size']}  best_val={r['best_val_loss']:.4f}  "
            f"final_val={r['final_val_loss']:.4f}  val_drop={r['val_drop']:.4f}  "
            f"still_decreasing={r['still_decreasing']}"
        )
    print(
        f"\n[codebook_sweep] recommended for full job: "
        f"codebook_size={best['codebook_size']}, lr={lr:.0e}, "
        f"batch_size={batch_size}, epochs=10, mode=adapt"
    )
    print(f"[codebook_sweep] wrote {out_dir / 'summary.json'}")
    print("\nFull job command:")
    print(
        f"  modal run --detach neural_tokenizers/meg/modal/modal_meg_finetune_brainomni.py::finetune \\\n"
        f"    --finetune-mode adapt --lr {lr} --epochs 10 --batch-size {batch_size} "
        f"--stage 3b --codebook-size {best['codebook_size']}"
    )


@app.function(
    image=brainomni_image,
    volumes={PROJECT_MOUNT: project_volume},
    timeout=10 * 60,
)
def read_config_remote(stage: str = "3b", codebook_size: int = 0) -> dict:
    """Read finetune config.json from the Modal volume."""
    from dataclasses import replace

    from neural_tokenizers.meg import BRAINOMNI_DEFAULT
    from neural_tokenizers.meg.brainomni.config import run_slug

    tok_cfg = (
        replace(BRAINOMNI_DEFAULT, codebook_size=codebook_size)
        if codebook_size > 0
        else BRAINOMNI_DEFAULT
    )
    path = Path(CKPT_ROOT) / run_slug(cfg=tok_cfg, stage=stage) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"No config at {path}")
    return json.loads(path.read_text())


@app.local_entrypoint()
def fetch_config(
    stage: str = "3b",
    codebook_size: int = 0,
    output: str = "",
):
    """Pull config.json from Modal volume to local runs/ directory."""
    result = read_config_remote.remote(stage=stage, codebook_size=codebook_size)
    slug = result["slug"]
    local_dir = Path(output or f"neural_tokenizers/meg/brainomni/runs/{slug}")
    local_dir.mkdir(parents=True, exist_ok=True)
    out_path = local_dir / "config.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[fetch] best_val_loss={result['best_val_loss']:.4f} lr={result['lr']}")
    print(f"[fetch] wrote {out_path}")


@app.local_entrypoint()
def lr_sweep_coarse(
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 0,
    train_trials: int = 1000,
    val_trials: int = 200,
):
    """LR sweep on small subset: {1e-4, 1e-3} × epochs (default 10).

    Writes ``runs/3b_lr_sweep_coarse/summary.json``.
    """
    lrs = [1e-4, 1e-3]
    runs: list[dict] = []
    for lr in lrs:
        lr_tag = f"lr{lr:.0e}".replace("+", "")
        print(f"\n[lr_sweep_coarse] === lr={lr:.0e} epochs={epochs} ===")
        result = finetune_remote.remote(
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            stage=f"3b_lr_{lr_tag}",
            max_train_trials=train_trials,
            max_val_trials=val_trials,
            finetune_mode="adapt",
        )
        summary = _summarize_run(result)
        runs.append(summary)
        print(
            f"[lr_sweep_coarse] lr={lr:.0e} best_val={summary['best_val_loss']:.4f} "
            f"val_drop={summary['val_drop']:.4f}"
        )

    runs_sorted = sorted(runs, key=lambda r: (r["best_val_loss"], -r["val_drop"]))
    best = runs_sorted[0]
    out_dir = Path("neural_tokenizers/meg/brainomni/runs/3b_lr_sweep_coarse")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "recommendation": {"lr": best["lr"], "epochs": epochs, "batch_size": batch_size},
        "runs": runs,
        "ranking": [{"lr": r["lr"], "best_val_loss": r["best_val_loss"]} for r in runs_sorted],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print("\n[lr_sweep_coarse] === ranking ===")
    for i, r in enumerate(runs_sorted, 1):
        print(
            f"  {i}. lr={r['lr']:.0e}  best_val={r['best_val_loss']:.4f}  "
            f"final_val={r['final_val_loss']:.4f}  still_decreasing={r['still_decreasing']}"
        )
    print(f"[lr_sweep_coarse] wrote {out_dir / 'summary.json'}")
