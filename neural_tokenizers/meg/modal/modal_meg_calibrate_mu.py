"""Phase-1 calibration on Modal: fit μ-transform on the THINGS-MEG train split.

What it does (one short remote function):
  1. Discover P1–P4 epoch files on the `project` Volume.
  2. Build per-subject by-image splits with MU_SPLIT_DEFAULTS — guarantees
     the test trials match those used by Phase 2+ (cross-phase invariant
     enforced by splits.py).
  3. Uniformly subsample N trials from the pooled train split.
  4. Move the subsample to GPU, run `fit_calibration` (one torch.quantile
     reduction per channel), return the resulting calibration dict.

The calibration dict is small (~6 KB JSON). It is returned to the local
entrypoint via `.remote()` and written to disk locally, so it lands directly
in git-trackable form (see meg/CLAUDE.md §10: "JSON goes in git").

We DO NOT touch the test split here — that comes in modal_meg_eval.py.

Run:
    modal run neural_tokenizers/meg/modal/modal_meg_calibrate_mu.py::calibrate \\
        --n-sample 2000 \\
        --output neural_tokenizers/meg/mu_transform/calibration.json
"""

from __future__ import annotations

from pathlib import Path
import sys

# Local-only sys.path tweak: makes `neural_tokenizers.*` importable when the
# script is invoked from anywhere on the laptop. On Modal's remote container,
# `add_local_python_source("neural_tokenizers")` already mounts the package
# (and the script itself lives at /root/script.py, where parents[3] would
# IndexError), so we skip the tweak whenever the import already works OR the
# file isn't deep enough on disk to have a sensible parents[3].
try:
    import neural_tokenizers  # noqa: F401
except ImportError:
    try:
        _REPO_ROOT = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(_REPO_ROOT))
    except IndexError:
        # Remote container — neural_tokenizers should already be on sys.path
        # via add_local_python_source. If it isn't, fail loudly downstream.
        pass

import modal  # noqa: E402

# ---------- Modal setup (kept inline; sharing across scripts via a sibling
# common.py turned out to be fragile — the remote container re-imports the
# entrypoint script during function hydration and `from common import ...`
# isn't reachable as a top-level module there).
app = modal.App("neural-fm")
project_volume = modal.Volume.from_name("project")
PROJECT_MOUNT = "/project"

meg_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("mne", "scipy", "torch", "numpy")
    .add_local_python_source("neural_tokenizers")
)


@app.function(
    image=meg_image,
    volumes={PROJECT_MOUNT: project_volume},
    gpu="A10",
    cpu=4.0,
    memory=16 * 1024,
    timeout=60 * 30,
)
def fit_remote(
    n_sample: int = 2000,
    seed: int = 0,
    mu: float = 255.0,
    vocab_size: int = 256,
    clip_lo_pct: float = 0.5,
    clip_hi_pct: float = 99.5,
    channel_mode: str = "per_channel",
) -> dict:
    """Fit μ-transform calibration on the THINGS-MEG train split.

    Args:
        n_sample: trials pooled from the train split. 2000 × 281 = 562k
            samples per channel — plenty for tight 0.5%/99.5% estimates.
        seed: split seed. Threads into split_by_image so identical seeds
            give identical (train, test) trials across runs.
        mu, vocab_size, clip_lo_pct, clip_hi_pct, channel_mode: μ-transform
            hyperparameters (defaults reproduce the paper baseline).
            Override to sweep V or clip percentiles.

    Returns:
        The calibration dict (MuCalibration.to_json()), small enough to ship
        back through `.remote()`.
    """
    import torch

    from neural_tokenizers.meg import (
        MU_SPLIT_DEFAULTS,
        MuTransformConfig,
        SplitDefaults,
        fit_calibration,
    )
    from neural_tokenizers.meg.data import list_subjects, sample_train_trials
    from neural_tokenizers.meg.splits import split_by_image

    # Compose a per-run config; do NOT mutate the global MU_TRANSFORM_DEFAULT.
    cfg = MuTransformConfig(
        mu=mu,
        vocab_size=vocab_size,
        clip_lo_pct=clip_lo_pct,
        clip_hi_pct=clip_hi_pct,
        channel_mode=channel_mode,
    )
    print(f"[calib] config: {cfg}")

    split_cfg = SplitDefaults(
        train_frac=MU_SPLIT_DEFAULTS.train_frac,
        val_frac=MU_SPLIT_DEFAULTS.val_frac,
        test_frac=MU_SPLIT_DEFAULTS.test_frac,
        seed=seed,
    )

    subjects = list_subjects()
    print(f"[calib] discovered subjects: {[s.subject for s in subjects]}")

    # Per-subject by-image splits → pool train indices for the subsample.
    train_per_subj: dict[str, "list[int]"] = {}
    for s in subjects:
        sp = split_by_image(s.image_ids, split_cfg)
        train_per_subj[s.subject] = sp.train
        print(
            f"[calib] {s.subject}: train={len(sp.train)} val={len(sp.val)} test={len(sp.test)}"
        )

    X_sample = sample_train_trials(
        subjects, train_per_subj, n_sample=n_sample, seed=seed
    )
    print(f"[calib] subsample shape: {tuple(X_sample.shape)} dtype={X_sample.dtype}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_sample = X_sample.to(device)
    print(f"[calib] fitting on device={device} ...")
    calib = fit_calibration(X_sample, cfg)
    print(
        f"[calib] done. scaler stats: "
        f"min={calib.scaler.min().item():.3e} "
        f"max={calib.scaler.max().item():.3e} "
        f"mean={calib.scaler.mean().item():.3e}"
    )
    return calib.to_json()


@app.local_entrypoint()
def calibrate(
    n_sample: int = 2000,
    seed: int = 0,
    mu: float = 255.0,
    vocab_size: int = 256,
    clip_lo_pct: float = 0.5,
    clip_hi_pct: float = 99.5,
    channel_mode: str = "per_channel",
    runs_dir: str = "neural_tokenizers/meg/mu_transform/runs",
):
    """Run calibration remotely; write the returned JSON locally under
    `<runs_dir>/<slug>/calibration.json`.

    Per-config subdir (slug) prevents sweeps from overwriting each other.
    The slug encodes (V, μ, clip percentiles, channel mode, seed) — anything
    that changes the calibration math gets its own directory.

    Invoke:
        modal run neural_tokenizers/meg/modal/modal_meg_calibrate_mu.py::calibrate \\
            --vocab-size 256 --mu 255 --seed 0
    """
    import json

    from neural_tokenizers.meg.mu_transform import run_slug

    payload = fit_remote.remote(
        n_sample=n_sample, seed=seed, mu=mu, vocab_size=vocab_size,
        clip_lo_pct=clip_lo_pct, clip_hi_pct=clip_hi_pct, channel_mode=channel_mode,
    )
    slug = run_slug(mu, vocab_size, clip_lo_pct, clip_hi_pct, channel_mode, seed)
    out_dir = Path(runs_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration.json"
    out_path.write_text(json.dumps(payload, indent=2))

    # Drop a tiny `config.json` next to it so the directory is self-describing.
    (out_dir / "config.json").write_text(json.dumps({
        "slug": slug, "mu": mu, "vocab_size": vocab_size,
        "clip_lo_pct": clip_lo_pct, "clip_hi_pct": clip_hi_pct,
        "channel_mode": channel_mode, "seed": seed, "n_sample": n_sample,
    }, indent=2))

    print(f"[calib] slug: {slug}")
    print(f"[calib] wrote {out_path}")
    print(f"[calib] preview (first 3 channels):")
    for k in ("clip_lo", "clip_hi", "scaler"):
        print(f"  {k}[:3] = {payload[k][:3]}")
