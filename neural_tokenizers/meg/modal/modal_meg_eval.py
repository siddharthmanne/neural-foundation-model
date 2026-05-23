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

# Modal scaffold — inlined rather than shared via a sibling common.py
# (the remote container re-imports this script during function hydration,
# and a sibling top-level module isn't reachable there).
app = modal.App("neural-fm")
project_volume = modal.Volume.from_name("project")
PROJECT_MOUNT = "/project"

meg_image = (
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
        remote_path="/root/external/BrainOmni",
        # Submodule source only; weights fetched at runtime if absent.
        ignore=["ckpt_collection", ".cache", "**/__pycache__"],
    )
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


def _build_brainomni(payload: dict[str, Any]):
    """Construct Phase-3 BrainOmniTokenizer from a config.json dict."""
    import os
    import sys

    repo = payload.get("brainomni_repo", "/root/external/BrainOmni")
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from neural_tokenizers.meg.brainomni.adapter import build_tokenizer_from_payload
    from neural_tokenizers.meg.brainomni.checkpoint import resolve_ckpt_dir

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    ckpt_dir = resolve_ckpt_dir(payload.get("ckpt_dir"), repo)
    # Finetuned runs store on the project volume.
    if not os.path.isdir(ckpt_dir) and payload.get("ckpt_dir", "").startswith("/project"):
        ckpt_dir = payload["ckpt_dir"]
    payload = {**payload, "device": device, "ckpt_dir": ckpt_dir, "brainomni_repo": repo}
    return build_tokenizer_from_payload(payload)


_TOKENIZER_FACTORIES = {
    "mu_transform": _build_mu_transform,
    "brainomni": _build_brainomni,
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
        BRAINOMNI_EVAL_DEFAULTS,
        EVAL_DEFAULTS,
        LEARNABLE_SPLIT_DEFAULTS,
        MU_SPLIT_DEFAULTS,
        SplitDefaults,
    )
    from neural_tokenizers.meg.data import (
        ConceptMapping,
        list_subjects,
        load_trials_pooled,
    )
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

    # Labels for the §5.3 probe: THINGS concept IDs (1..1854) → dense
    # [0..K-1] for cross-entropy. Two-step densification:
    #   (a) image_id → THINGS concept_id (via ConceptMapping; full 1854 space)
    #   (b) collapse to ONLY the concepts that appear in our 3000-trial eval
    #       set. CRITICAL: skipping (b) gives the probe ~800 "phantom" output
    #       columns (concepts in THINGS but absent here), which weight decay
    #       pulls toward zero and distorts argmax. With (b), n_classes equals
    #       the actual unique concept count in the eval set.
    concept_map = ConceptMapping.load()
    full_labels, valid = concept_map.encode(image_ids)
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        print(f"[eval] dropping {n_dropped} trials with unmapped image_ids")
        keep_idx = torch.from_numpy(np.nonzero(valid)[0].astype("int64"))
        X = X[keep_idx]
        full_labels = full_labels[valid]
    # Step (b): re-densify to [0, K_observed).
    _, dense_labels = np.unique(full_labels, return_inverse=True)
    labels = torch.from_numpy(dense_labels.astype("int64"))
    print(
        f"[eval] probe labels: {len(np.unique(dense_labels))} concepts in eval set "
        f"({X.shape[0]} trials, out of {concept_map.n_concepts} THINGS total)"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    print(f"[eval] running harness on device={device} ...")
    eval_defaults = BRAINOMNI_EVAL_DEFAULTS if tokenizer_name == "brainomni" else EVAL_DEFAULTS
    cfg = EvalConfig(
        sample_rate_hz=eval_defaults.sample_rate_hz,
        bands=dict(eval_defaults.bands),
        device=device,
        batch_size=64,
        seed=seed,
        psd_nperseg=eval_defaults.psd_nperseg,
        probe_epochs=eval_defaults.probe_epochs,
        probe_top_k=eval_defaults.probe_top_k,
        probe_test_frac=eval_defaults.probe_test_frac,
    )
    report = evaluate(tok, X, labels, cfg)
    print(report)

    return {
        m.name: dict(m.values)
        for m in (report.reconstruction, report.codebook, report.sequence, report.probe)
        if m is not None
    }


def _eval_slug(n_test: int, seed: int) -> str:
    """Filename suffix that distinguishes eval invocations against one calibration.

    Two eval runs with the same (n_test, seed) are deterministic and identical,
    so overwriting is fine. Different (n_test, seed) get different files so
    they don't clobber each other.
    """
    n_part = "full" if n_test <= 0 else f"n{n_test}"
    return f"ntest={n_part}_s{seed}"


@app.local_entrypoint()
def run(
    tokenizer: str = "mu_transform",
    calibration: str = "",
    n_test: int = 3000,
    seed: int = 0,
    output: str = "",
):
    """Run the §5 harness remotely; report lands next to the calibration in
    `<calibration_dir>/evals/eval_<eval_slug>.json`.

    Args:
        tokenizer: which factory to dispatch.
        calibration: path to the calibration.json. If empty, picks the most
            recent run under
            `neural_tokenizers/meg/mu_transform/runs/*/calibration.json`.
        n_test: cap on test-trial count (0 = use full test split).
        seed: split seed (must match calibration's seed).
        output: optional explicit report path. If empty, writes
            `<calibration_dir>/evals/eval_<eval_slug>.json` — same calibration
            can host multiple eval reports under different eval configs without
            overwriting.

    Invoke:
        modal run neural_tokenizers/meg/modal/modal_meg_eval.py::run \\
            --calibration neural_tokenizers/meg/mu_transform/runs/<slug>/calibration.json
    """
    import json

    calib_path = Path(_resolve_config(tokenizer, calibration))
    payload = json.loads(calib_path.read_text())
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
        out_path = Path(output)
    else:
        evals_dir = calib_path.parent / "evals"
        evals_dir.mkdir(parents=True, exist_ok=True)
        out_path = evals_dir / f"eval_{_eval_slug(n_test, seed)}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[eval] wrote report to {out_path}")


def _resolve_config(tokenizer: str, config_path: str) -> str:
    """Resolve --calibration/config path for each tokenizer phase."""
    if config_path:
        return config_path
    if tokenizer == "brainomni":
        runs_dir = Path("neural_tokenizers/meg/brainomni/runs")
        candidates = sorted(
            runs_dir.glob("*/config.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            # Fall back to bundled pretrained checkpoint config.
            default = Path("neural_tokenizers/meg/brainomni/runs/V512_rvq4_win512_sf256_3a/config.json")
            if default.exists():
                return str(default)
            raise FileNotFoundError(
                f"No config.json under {runs_dir}. "
                "Pass --calibration <path> or create the 3a config."
            )
        print(f"[eval] using newest brainomni config: {candidates[0]}")
        return str(candidates[0])
    return _resolve_calibration(config_path)


def _resolve_calibration(calibration: str) -> str:
    """Resolve --calibration: explicit path, or the newest under runs_dir."""
    if calibration:
        return calibration
    runs_dir = Path("neural_tokenizers/meg/mu_transform/runs")
    candidates = sorted(runs_dir.glob("*/calibration.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No calibration.json found under {runs_dir}. "
            f"Pass --calibration <path> or run modal_meg_calibrate_mu.py first."
        )
    print(f"[eval] using newest calibration: {candidates[0]}")
    return str(candidates[0])
