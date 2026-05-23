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
# IndexError), so we skip the tweak whenever the import already works.
try:
    import neural_tokenizers  # noqa: F401
except ImportError:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))

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
def fit_remote(n_sample: int = 2000, seed: int = 0) -> dict:
    """Fit μ-transform calibration on the THINGS-MEG train split.

    Args:
        n_sample: number of trials to pool from the train split. 2000 trials
            × 281 samples = 562k samples per channel — plenty for the
            0.5%/99.5% percentile estimate to be tight.
        seed: rng seed (also threads into split_by_image so identical seeds
              produce identical (train, test) sets across runs).

    Returns:
        The calibration dict (MuCalibration.to_json()). Small enough to ship
        back through `.remote()`.
    """
    import torch

    from neural_tokenizers.meg import (
        MU_SPLIT_DEFAULTS,
        MU_TRANSFORM_DEFAULT,
        SplitDefaults,
        fit_calibration,
    )
    from neural_tokenizers.meg.data import list_subjects, sample_train_trials
    from neural_tokenizers.meg.splits import split_by_image

    # Allow caller-provided seed to override the dataclass default.
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
    calib = fit_calibration(X_sample, MU_TRANSFORM_DEFAULT)
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
    output: str = "neural_tokenizers/meg/mu_transform/calibration.json",
):
    """Run calibration remotely, write the returned JSON to `output` locally.

    Default output path lands the calibration in the git-tracked location
    declared by meg/CLAUDE.md §7.

    Invoke:
        modal run neural_tokenizers/meg/modal/modal_meg_calibrate_mu.py::calibrate
    """
    import json

    payload = fit_remote.remote(n_sample=n_sample, seed=seed)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[calib] wrote calibration to {out_path}")
    print(f"[calib] preview (first 3 channels):")
    for k in ("clip_lo", "clip_hi", "scaler"):
        print(f"  {k}[:3] = {payload[k][:3]}")
