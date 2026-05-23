"""Generic Modal evaluator for any MEG tokenizer.

This script knows only three things:
  1. How to load THINGS-MEG trials from the `project` Volume (delegates to
     `meg/data.py`).
  2. How to pick the test split (delegates to `meg/splits.py`).
  3. How to invoke the §5 harness with MEG-appropriate EvalConfig defaults
     (delegates to `meg/meg_config.py::EvalDefaults`).

It does NOT know how to construct a particular tokenizer — that's a small
per-phase factory function. Adding Phase 2 will mean adding one factory
entry, not a new script.

Usage:
    modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \\
        --tokenizer mu_transform \\
        --calibration neural_tokenizers/meg/mu_transform/calibration.json \\
        --n-test 3000
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
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

# Modal scaffold — inlined rather than shared via a sibling common.py
# (the remote container re-imports this script during function hydration,
# and a sibling top-level module isn't reachable there).
app = modal.App("neural-fm")
project_volume = modal.Volume.from_name("project")
PROJECT_MOUNT = "/project"

meg_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("mne", "scipy", "torch", "numpy")
    .add_local_python_source("neural_tokenizers")
)


# ---------- per-tokenizer factory dispatch -------------------------------
#
# Adding a new tokenizer phase = adding ONE function below. The Modal entry
# point picks the right factory based on --tokenizer <name>.

def _build_mu_transform(calibration_payload: dict[str, Any]):
    """Construct a Phase-1 MuTransformTokenizer from a calibration JSON dict."""
    from neural_tokenizers.meg import MuCalibration, MuTransformTokenizer

    calib = MuCalibration.from_json(calibration_payload)
    return MuTransformTokenizer(calib)


_TOKENIZER_FACTORIES = {
    "mu_transform": _build_mu_transform,
    # "cho2026":    _build_cho2026,   # add when Phase 2 lands
}


# ---------- the remote eval ----------------------------------------------

@app.function(
    image=meg_image,
    volumes={PROJECT_MOUNT: project_volume},
    gpu="A10",
    cpu=4.0,
    memory=32 * 1024,
    timeout=60 * 60,
)
def evaluate_remote(
    tokenizer_name: str,
    tokenizer_payload: dict[str, Any],
    n_test: int = 3000,
    seed: int = 0,
    use_mu_split: bool = True,
) -> dict[str, Any]:
    """Run the §5 four-axis harness against the named tokenizer.

    Args:
        tokenizer_name: which factory to call (e.g. "mu_transform").
        tokenizer_payload: tokenizer-specific config (e.g. calibration dict
            for mu_transform; later, a checkpoint path or weights for cho2026).
        n_test: cap on the number of test trials materialized into memory
            (per-trial = 271*281*4 B ≈ 305 KB → 3000 trials ≈ 0.9 GB f32).
            Set 0 / negative to use the full test split.
        seed: split seed. Must match the seed used at calibration time.
        use_mu_split: True for Phase-1 (no val), False for learnable phases.

    Returns:
        A flat dict of {axis_name: {metric: value}} from the §5 harness.
        Small enough to print or post-process locally.
    """
    import numpy as np
    import torch

    from neural_tokenizers.evaluation import EvalConfig, evaluate
    from neural_tokenizers.meg import (
        EVAL_DEFAULTS,
        LEARNABLE_SPLIT_DEFAULTS,
        MU_SPLIT_DEFAULTS,
        SplitDefaults,
    )
    from neural_tokenizers.meg.data import list_subjects, load_trials_pooled
    from neural_tokenizers.meg.splits import split_by_image

    if tokenizer_name not in _TOKENIZER_FACTORIES:
        raise ValueError(
            f"unknown tokenizer {tokenizer_name!r}; "
            f"available: {sorted(_TOKENIZER_FACTORIES)}"
        )

    base_split = MU_SPLIT_DEFAULTS if use_mu_split else LEARNABLE_SPLIT_DEFAULTS
    split_cfg = SplitDefaults(
        train_frac=base_split.train_frac,
        val_frac=base_split.val_frac,
        test_frac=base_split.test_frac,
        seed=seed,
    )

    subjects = list_subjects()
    print(f"[eval] discovered subjects: {[s.subject for s in subjects]}")

    # Per-subject test indices.
    test_per_subj: dict[str, np.ndarray] = {}
    total_test = 0
    for s in subjects:
        sp = split_by_image(s.image_ids, split_cfg)
        test_per_subj[s.subject] = sp.test
        total_test += len(sp.test)
    print(f"[eval] total test trials available: {total_test}")

    # Cap at n_test if requested.
    if n_test > 0 and total_test > n_test:
        per_subj_cap = n_test // len(subjects)
        rng = np.random.default_rng(seed)
        for k, idx in test_per_subj.items():
            if len(idx) > per_subj_cap:
                pick = rng.choice(len(idx), size=per_subj_cap, replace=False)
                test_per_subj[k] = np.sort(idx[pick])
        total_test = sum(len(v) for v in test_per_subj.values())
        print(f"[eval] capped to {total_test} test trials")

    X, image_ids, _ = load_trials_pooled(subjects, test_per_subj)
    print(f"[eval] materialized X={tuple(X.shape)} dtype={X.dtype}")

    # Build the tokenizer inside the container.
    factory = _TOKENIZER_FACTORIES[tokenizer_name]
    tok = factory(tokenizer_payload)

    # Labels for the §5.3 probe: re-encode image_ids to a contiguous 0..K-1
    # range. The full THINGS image-id → concept_id mapping is not yet wired
    # (see meg/CLAUDE.md §8), so this is a placeholder dense relabeling.
    # Probe numbers should be read relative to the in-trial random baseline
    # the harness also computes, NOT as absolute concept accuracy yet.
    _, dense_labels = np.unique(image_ids, return_inverse=True)
    labels = torch.from_numpy(dense_labels.astype("int64"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    print(f"[eval] running harness on device={device} ...")
    cfg = EvalConfig(
        sample_rate_hz=EVAL_DEFAULTS.sample_rate_hz,
        bands=dict(EVAL_DEFAULTS.bands),
        device=device,
        batch_size=64,
        seed=seed,
        psd_nperseg=EVAL_DEFAULTS.psd_nperseg,
        probe_epochs=EVAL_DEFAULTS.probe_epochs,
        probe_top_k=EVAL_DEFAULTS.probe_top_k,
        probe_test_frac=EVAL_DEFAULTS.probe_test_frac,
    )
    report = evaluate(tok, X, labels, cfg)
    print(report)

    return {
        m.name: dict(m.values)
        for m in (report.reconstruction, report.codebook, report.sequence, report.probe)
        if m is not None
    }


@app.local_entrypoint()
def run(
    tokenizer: str = "mu_transform",
    calibration: str = "neural_tokenizers/meg/mu_transform/calibration.json",
    n_test: int = 3000,
    seed: int = 0,
    output: str = "",
):
    """Run the §5 harness remotely and print / save the report.

    Invoke:
        modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \\
            --tokenizer mu_transform \\
            --calibration neural_tokenizers/meg/mu_transform/calibration.json
    """
    import json

    payload = json.loads(Path(calibration).read_text())
    use_mu_split = tokenizer == "mu_transform"
    report = evaluate_remote.remote(
        tokenizer_name=tokenizer,
        tokenizer_payload=payload,
        n_test=n_test,
        seed=seed,
        use_mu_split=use_mu_split,
    )
    print("\n[eval] report (local view):")
    for axis, values in report.items():
        print(f"  [{axis}]")
        for k, v in values.items():
            print(f"    {k:<32s} {v:.4f}")

    if output:
        Path(output).write_text(json.dumps(report, indent=2))
        print(f"[eval] wrote report to {output}")
