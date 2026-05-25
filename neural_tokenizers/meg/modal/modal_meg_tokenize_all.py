"""Tokenize all THINGS-MEG trials with the BrainOmni 3b finetuned checkpoint.

What this produces on the `project` Modal Volume:
  /project/data/things-meg/tokens/brainomni_3b_V512_rvq4/
    config.json   tokenizer + checkpoint metadata (single source of truth)
    P1.npz        per-subject token cache (see schema below)
    P2.npz
    P3.npz
    P4.npz

Per-subject .npz schema:
  tokens             (N, 16, 8, 4) int16   RVQ codes per trial
  meg_trigger_codes  (N,)          int64   THINGS image_nr per trial
  trial_idx          (N,)          int64   0-indexed occurrence within (subject, trigger)
  subject            ()            unicode "P1".."P4"  (scalar)

Filtering policy:
  Only trials whose trigger code is present in the bridge file
  `/project/data/things-meg/labels/meg_trigger_to_image_id.json` are kept.
  This drops THINGS-MEG `catch` trials (artificial oddball stimuli), which
  account for ~2400 of the 27048 trials per subject.

Idempotency:
  Re-running skips any subject whose <subject>.npz already exists alongside
  a config.json that matches the current checkpoint+codebook configuration.
  Useful for resuming after a crash. Pass --force to re-tokenize anyway.

Serial across subjects (per-user policy in Step 2 plan):
  P1 → P2 → P3 → P4 inside one GPU job, ~30 min total on L40S.
  If you want parallel, just call `tokenize_one_subject_remote.spawn(...)` 4×.

Cost: ~$1 (L40S × ~30 min).

Run from inner repo root:
    modal run neural_tokenizers/meg/modal/modal_meg_tokenize_all.py::tokenize_all
    # Or to force re-tokenize:
    modal run neural_tokenizers/meg/modal/modal_meg_tokenize_all.py::tokenize_all --force
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("neural-fm")
project_volume = modal.Volume.from_name("project")
PROJECT_MOUNT = "/project"
BRAINOMNI_REPO = "/root/external/BrainOmni"

# Layout on the Volume.
# Tokens dir slug == checkpoint slug so the link from tokens → producing
# checkpoint is unambiguous. The `brainomni/` subfolder leaves room for
# future MEG tokenizer outputs (cho2026, mu_transform, …) to be co-located.
CHECKPOINT_SLUG = "V512_rvq4_win512_sf256_3b"
CKPT_DIR = f"/project/checkpoints/meg/brainomni/{CHECKPOINT_SLUG}"
BRIDGE_PATH = "/project/data/things-meg/labels/meg_trigger_to_image_id.json"
TOKENS_DIR = f"/project/data/things-meg/tokens/brainomni/{CHECKPOINT_SLUG}"

tokenize_image = (
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


def _config_payload(codebook_size: int, num_quantizers: int) -> dict:
    """Tokenizer + checkpoint metadata written next to the per-subject .npz."""
    return {
        "version": "1",
        "tokenizer": "brainomni_braintokenizer",
        "stage": "3b_finetuned_adapt",
        "ckpt_dir": CKPT_DIR,
        "codebook_size": codebook_size,
        "num_quantizers": num_quantizers,
        "expected_token_shape": [16, 8, 4],
        "token_dtype": "int16",
        "source_sfreq_hz": 200.0,
        "target_sfreq_hz": 256.0,
        "window_length": 512,
        "n_subjects": 4,
        "bridge_path": BRIDGE_PATH,
        "filtered_trial_types": ["exp", "test"],
        "skipped_trial_types": ["catch"],
    }


@app.function(
    image=tokenize_image,
    volumes={PROJECT_MOUNT: project_volume},
    gpu="L40S",
    cpu=8.0,
    memory=32 * 1024,
    timeout=6 * 60 * 60,
)
def tokenize_all_remote(force: bool = False, batch_size: int = 64) -> dict:
    """Serial-over-subjects BrainOmni 3b tokenization."""
    import os
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, BRAINOMNI_REPO)

    from neural_tokenizers.meg.brainomni.adapter import BrainOmniTokenizer
    from neural_tokenizers.meg.meg_config import BRAINOMNI_DEFAULT
    from neural_tokenizers.meg.data import list_subjects, load_trials

    # ----- preconditions ------------------------------------------------
    if not os.path.exists(BRIDGE_PATH):
        raise FileNotFoundError(
            f"Missing bridge file {BRIDGE_PATH}. "
            f"Run modal_download_meg_image_bridge.py::build first."
        )
    if not os.path.isfile(os.path.join(CKPT_DIR, "BrainTokenizer.pt")):
        raise FileNotFoundError(f"Missing checkpoint at {CKPT_DIR}/BrainTokenizer.pt")

    os.makedirs(TOKENS_DIR, exist_ok=True)

    bridge = json.loads(Path(BRIDGE_PATH).read_text())
    trigger_to_image_id = {
        int(k): v for k, v in bridge["trigger_to_image_id"].items()
    }
    print(f"[tokenize] bridge: {len(trigger_to_image_id)} trigger codes")

    # ----- config + idempotency check ----------------------------------
    cfg = BRAINOMNI_DEFAULT  # codebook_size=512, num_quantizers=4
    payload_cfg = _config_payload(cfg.codebook_size, cfg.num_quantizers)
    config_path = os.path.join(TOKENS_DIR, "config.json")
    if os.path.exists(config_path) and not force:
        existing = json.loads(Path(config_path).read_text())
        # Compare just the load-bearing fields.
        load_bearing = {"tokenizer", "stage", "ckpt_dir", "codebook_size", "num_quantizers", "expected_token_shape"}
        if all(existing.get(k) == payload_cfg.get(k) for k in load_bearing):
            print("[tokenize] config matches existing tokens; will only fill missing subjects.")
        else:
            raise RuntimeError(
                f"Existing config at {config_path} does not match current run; "
                f"refusing to overwrite. Pass --force to override."
            )
    Path(config_path).write_text(json.dumps(payload_cfg, indent=2))

    # ----- load tokenizer ----------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tokenize] loading BrainOmni 3b from {CKPT_DIR} on {device}")
    tokenizer = BrainOmniTokenizer.from_checkpoint(
        ckpt_dir=CKPT_DIR,
        brainomni_repo=BRAINOMNI_REPO,
        cfg=cfg,
        device=device,
    )

    # ----- iterate subjects --------------------------------------------
    subjects = list_subjects()  # uses MEG_DATA.data_dir = /project/data/things-meg/preprocessed
    subject_summaries: dict[str, dict] = {}

    for subject_idx in subjects:
        subj = subject_idx.subject
        out_path = os.path.join(TOKENS_DIR, f"{subj}.npz")
        if os.path.exists(out_path) and not force:
            print(f"[tokenize] {subj}: cache exists, skip")
            arr = np.load(out_path)
            subject_summaries[subj] = {
                "n_trials_kept": int(arr["tokens"].shape[0]),
                "from_cache": True,
            }
            continue

        t0 = time.time()
        # 1. Trigger codes per trial in .fif order (already loaded eagerly).
        triggers = subject_idx.image_ids.astype(np.int64)
        n_total = len(triggers)

        # 2. Filter to trials whose trigger is in the bridge (drop catch).
        in_bridge_mask = np.array(
            [int(t) in trigger_to_image_id for t in triggers], dtype=bool
        )
        kept_trial_indices = np.flatnonzero(in_bridge_mask)
        n_kept = len(kept_trial_indices)
        n_dropped = n_total - n_kept
        print(
            f"[tokenize] {subj}: {n_total} total trials, "
            f"{n_kept} kept ({n_dropped} catch/unknown dropped)"
        )

        # 3. Compute trial_idx (0-indexed occurrence within (subject, trigger))
        # using the kept-trial sequence in .fif order.
        running: dict[int, int] = {}
        trial_idx_array = np.empty(n_kept, dtype=np.int64)
        kept_triggers = np.empty(n_kept, dtype=np.int64)
        for i, fif_pos in enumerate(kept_trial_indices):
            t = int(triggers[fif_pos])
            kept_triggers[i] = t
            trial_idx_array[i] = running.get(t, 0)
            running[t] = running.get(t, 0) + 1

        # 4. Load the kept trials into a CPU tensor.
        X, _img = load_trials(subject_idx, kept_trial_indices)
        # X shape: (n_kept, C=271, T=281), float32 on CPU.
        assert X.shape[0] == n_kept, f"loader returned {X.shape[0]} trials, expected {n_kept}"

        # 5. Batched tokenize → int16 tokens (n_kept, 16, 8, 4).
        all_tokens = np.empty((n_kept, 16, 8, 4), dtype=np.int16)
        n_batches = (n_kept + batch_size - 1) // batch_size
        for b in range(n_batches):
            lo, hi = b * batch_size, min((b + 1) * batch_size, n_kept)
            x_batch = X[lo:hi].to(device)
            tokens = tokenizer.tokenize(x_batch)  # (B, 16, 8, 4) long on device
            if tokens.shape[1:] != (16, 8, 4):
                raise RuntimeError(
                    f"unexpected token shape {tuple(tokens.shape)} for {subj} batch {b}"
                )
            if tokens.max().item() > 32767 or tokens.min().item() < -32768:
                raise RuntimeError(
                    f"tokens out of int16 range for {subj}: "
                    f"min={tokens.min().item()}, max={tokens.max().item()}"
                )
            all_tokens[lo:hi] = tokens.cpu().numpy().astype(np.int16)
            if b == 0 or (b + 1) % 50 == 0 or b == n_batches - 1:
                print(f"[tokenize]   {subj} batch {b+1}/{n_batches}")

        # 6. Save .npz.
        np.savez(
            out_path,
            tokens=all_tokens,
            meg_trigger_codes=kept_triggers,
            trial_idx=trial_idx_array,
            subject=np.array(subj, dtype="<U2"),
        )
        elapsed = time.time() - t0
        print(
            f"[tokenize] {subj}: wrote {out_path} "
            f"({os.path.getsize(out_path) / 1e6:.1f} MB, {elapsed:.0f}s)"
        )
        project_volume.commit()
        subject_summaries[subj] = {
            "n_trials_total_in_fif": int(n_total),
            "n_trials_kept": int(n_kept),
            "n_trials_dropped": int(n_dropped),
            "from_cache": False,
            "elapsed_s": round(elapsed, 1),
        }

    return {"per_subject": subject_summaries, "tokens_dir": TOKENS_DIR}


@app.local_entrypoint()
def tokenize_all(force: bool = False, batch_size: int = 64):
    summary = tokenize_all_remote.remote(force=force, batch_size=batch_size)
    print("\n[tokenize_all] summary:")
    for subj, info in summary["per_subject"].items():
        print(f"  {subj}: {info}")
    print(f"\n[tokenize_all] tokens dir: {summary['tokens_dir']}")
