"""CPU-only Modal evaluator for the LaBraM EEG tokenizer.

Mirrors neural_tokenizers/meg/modal/modal_meg_eval.py but adapted for EEG:

  - Tokens are pre-cached in npz files on the project volume (no re-tokenizing
    at eval time, no GPU needed).
  - Only the 8192×64 codebook embedding table is loaded from the checkpoint
    (~2 MB vs 94 MB for the full LaBraM model).
  - Reconstruction axis is disabled: npz files contain tokens only, not raw
    waveforms.
  - "raw" probe feature = token IDs cast to float (sparse categorical, 17
    positions × int ∈ [0,8192)). Labeled clearly in the output JSON.

Usage:
    # Linear probe, category27 (default):
    modal run neural_tokenizers/eeg/modal/modal_eeg_eval.py::run

    # MLP on animacy:
    modal run neural_tokenizers/eeg/modal/modal_eeg_eval.py::run \\
        --probe-classifier mlp --probe-label-space animacy

    # CNN on subject ID (pipeline sanity):
    modal run neural_tokenizers/eeg/modal/modal_eeg_eval.py::run \\
        --probe-classifier cnn --probe-label-space subject

    # Include EEG1 subjects as well (60 subjects total, much more data):
    modal run neural_tokenizers/eeg/modal/modal_eeg_eval.py::run \\
        --sources eeg1,eeg2 --n-test 0

Results: <repo>/neural_tokenizers/eeg/evals/eval_<slug>.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

# Local sys.path tweak for laptop invocation. Skipped in Modal container where
# add_local_python_source already mounts the package.
try:
    import neural_tokenizers  # noqa: F401
except ImportError:
    try:
        _REPO_ROOT = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(_REPO_ROOT))
    except IndexError:
        pass

import modal  # noqa: E402

# Modal scaffold inlined — remote containers can't import a sibling _app.py
# (the function re-imports this script during hydration; only installed packages
# and add_local_python_source mounts are reachable).
app = modal.App("neural-fm")
_project_volume = modal.Volume.from_name("project", create_if_missing=False)
PROJECT_MOUNT = "/project"

# Default checkpoint path on the project volume — mirrored from EEGDataSpec so
# the local entrypoint doesn't have to import neural_tokenizers.eeg (which
# pulls in torch via __init__.py, which may not be installed locally).
_DEFAULT_CHECKPOINT = (
    "/project/checkpoints/eeg/labram/"
    "V8192_d64_ch17_sr200_train-eeg1+2_e5/checkpoint.pt"
)

# CUDA image: probe training (MLP/CNN) benefits from GPU.
eeg_eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .pip_install(
        "torch",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_python_source("neural_tokenizers")
)


# ---------- per-tokenizer factory --------------------------------------------

def _build_labram(payload: dict[str, Any]):
    from neural_tokenizers.eeg.labram_adapter import LaBraMEvalAdapter
    return LaBraMEvalAdapter.from_config(payload)


_TOKENIZER_FACTORIES = {
    "labram": _build_labram,
}


# ---------- remote function --------------------------------------------------

@app.function(
    image=eeg_eval_image,
    volumes={PROJECT_MOUNT: _project_volume},
    gpu="A100",
    cpu=8.0,
    memory=32 * 1024,
    timeout=60 * 60 * 2,
)
def evaluate_remote(
    tokenizer_name: str,
    tokenizer_payload: dict[str, Any],
    n_test: int = 20000,
    seed: int = 0,
    sources: list[str] | None = None,
    probe_classifier: str = "linear",
    probe_label_space: str = "category27",
    probe_class_weighted: bool = True,
) -> dict[str, Any]:
    """Run the §5 four-axis harness against the named EEG tokenizer.

    Reconstruction is disabled (npz cache has tokens only). The four axes
    that run: codebook, sequence, probe (linear/mlp/cnn), retrieval.

    Args:
        tokenizer_name: factory key (currently only "labram").
        tokenizer_payload: {"ckpt_path": ..., "device": "cpu"}.
        n_test: cap on total trials to load (0 = use all).
        seed: RNG seed for trial subsampling and probe CV.
        sources: EEG sources to include ("eeg1", "eeg2", or both).
            Defaults to ["eeg2"] (10 subjects, ~66k trials).
        probe_classifier: "linear" | "mlp" | "cnn".
        probe_label_space: "category27" | "animacy" | "subject".

    Returns:
        Flat dict {axis_name: {metric: value}} — small enough to print.
    """
    import numpy as np
    import torch

    from neural_tokenizers.evaluation import EvalConfig, evaluate
    from neural_tokenizers.eeg.data import list_subjects, load_tokens_pooled, image_ids_to_int
    from neural_tokenizers.eeg.eeg_config import EVAL_DEFAULTS
    from neural_tokenizers.eeg.labels import (
        AnimacyMapping,
        ConceptMapping,
        SuperordinateMapping,
    )

    if tokenizer_name not in _TOKENIZER_FACTORIES:
        raise ValueError(
            f"unknown tokenizer {tokenizer_name!r}; "
            f"available: {sorted(_TOKENIZER_FACTORIES)}"
        )

    _sources = tuple(sources or ["eeg2"])
    subjects = list_subjects(sources=_sources)
    print(f"[eval] found {len(subjects)} (source, subject) npz files")
    for s in subjects:
        print(f"  {s.source}/{s.subject}  {s.n_trials} trials")

    tokens, image_ids_str, subject_ids = load_tokens_pooled(subjects)
    total = tokens.shape[0]
    print(f"[eval] loaded {total} trials, shape {tuple(tokens.shape)}")

    if n_test > 0 and total > n_test:
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(total, size=n_test, replace=False))
        tokens = tokens[pick]
        image_ids_str = image_ids_str[pick]
        subject_ids = subject_ids[pick]
        print(f"[eval] subsampled to {n_test} trials")

    # Signal passed to evaluate(): (B, 17, 1) float.
    # tokenize() in LaBraMEvalAdapter squeezes the dummy time axis → (B, 17).
    # The 'raw' probe feature becomes token-IDs-as-floats — not raw EEG, but
    # an informative baseline: 17 sparse categoricals, each ∈ [0, 8192).
    signal = tokens.float().unsqueeze(-1)   # (B, 17, 1)

    # ---- Label encoding ----
    if probe_label_space == "subject":
        labels = torch.from_numpy(subject_ids.astype("int64"))
        n_unique = int(np.unique(subject_ids).size)
        print(
            f"[eval] subject labels: {n_unique}/{len(subjects)} subjects "
            f"in {len(labels)} trials"
        )
    else:
        concept_map = ConceptMapping.load()
        super_map = SuperordinateMapping.load()
        if probe_label_space == "category27":
            label_encoder = super_map
        elif probe_label_space == "animacy":
            label_encoder = AnimacyMapping.from_super_map(super_map)
        else:
            raise ValueError(
                f"probe_label_space must be 'category27', 'animacy', or 'subject'; "
                f"got {probe_label_space!r}"
            )
        image_ids_int = image_ids_to_int(image_ids_str)
        encoded_labels, valid = label_encoder.encode_image_ids(image_ids_int, concept_map)
        n_dropped = int((~valid).sum())
        if n_dropped > 0:
            print(f"[eval] dropping {n_dropped} trials — no {probe_label_space} label")
            keep = np.nonzero(valid)[0].astype("int64")
            signal = signal[torch.from_numpy(keep)]
            subject_ids = subject_ids[keep]
            encoded_labels = encoded_labels[valid]
        labels = torch.from_numpy(encoded_labels.astype("int64"))
        n_classes_used = int(np.unique(encoded_labels).size)
        n_classes_total = label_encoder.n_categories
        print(
            f"[eval] {probe_label_space} labels: "
            f"{n_classes_used}/{n_classes_total} classes in {len(labels)} trials"
        )

    tok = _TOKENIZER_FACTORIES[tokenizer_name](tokenizer_payload)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    signal = signal.to(device)
    print(f"[eval] harness: device={device}  signal={tuple(signal.shape)}  "
          f"classifier={probe_classifier!r}  labels={probe_label_space!r}")

    cfg = EvalConfig(
        sample_rate_hz=EVAL_DEFAULTS.sample_rate_hz,
        bands=dict(EVAL_DEFAULTS.bands),
        device=device,
        batch_size=512,
        seed=seed,
        psd_nperseg=EVAL_DEFAULTS.psd_nperseg,
        probe_epochs=EVAL_DEFAULTS.probe_epochs,
        probe_top_k=EVAL_DEFAULTS.probe_top_k,
        probe_test_frac=EVAL_DEFAULTS.probe_test_frac,
        probe_n_folds=EVAL_DEFAULTS.probe_n_folds,
        probe_rvq_layers=EVAL_DEFAULTS.probe_rvq_layers,
        probe_class_weighted=probe_class_weighted,
        probe_classifier=probe_classifier,
        probe_mlp_hidden=EVAL_DEFAULTS.probe_mlp_hidden,
        probe_mlp_dropout=EVAL_DEFAULTS.probe_mlp_dropout,
        probe_cnn_hidden=EVAL_DEFAULTS.probe_cnn_hidden,
        run_reconstruction=False,   # no raw waveform in the npz cache
        run_codebook=True,
        run_sequence=True,
        run_probe=True,
        run_retrieval=True,
    )

    report = evaluate(tok, signal, labels, cfg)
    print(report)

    return {
        m.name: dict(m.values)
        for m in (
            report.reconstruction,
            report.codebook,
            report.sequence,
            report.probe,
            report.retrieval,
        )
        if m is not None
    }


# ---------- result slug ------------------------------------------------------

def _eval_slug(
    n_test: int,
    seed: int,
    probe_classifier: str = "linear",
    probe_label_space: str = "category27",
    sources: list[str] | None = None,
    probe_class_weighted: bool = True,
) -> str:
    n_part = "full" if n_test <= 0 else f"n{n_test}"
    clf_part = "" if probe_classifier == "linear" else f"_{probe_classifier}"
    label_part = "" if probe_label_space == "category27" else f"_{probe_label_space}"
    src_part = (
        "" if (sources is None or sorted(sources) == ["eeg2"])
        else "_" + "+".join(sorted(sources))
    )
    wt_part = "" if probe_class_weighted else "_unweighted"
    return f"ntest={n_part}_s{seed}{clf_part}{label_part}{src_part}{wt_part}"


# ---------- local entrypoint -------------------------------------------------

@app.local_entrypoint()
def run(
    tokenizer: str = "labram",
    ckpt: str = "",
    n_test: int = 20000,
    seed: int = 0,
    sources: str = "eeg2",
    output: str = "",
    probe_classifier: str = "linear",
    probe_label_space: str = "category27",
    class_weighted: bool = True,
):
    """Run the §5 harness remotely on the EEG token cache.

    Args:
        tokenizer: factory key ("labram" — the only option for now).
        ckpt: path to the LaBraM checkpoint ON THE PROJECT VOLUME.
            Defaults to the V8192 production checkpoint.
        n_test: trial cap (0 = use all available trials).
        seed: RNG seed for subsampling and probe CV.
        sources: comma-separated EEG source(s) to include.
            "eeg2" = 10 subjects. "eeg1,eeg2" = 60 subjects.
        output: explicit local output path for the JSON report.
            Defaults to neural_tokenizers/eeg/evals/eval_<slug>.json.
        probe_classifier: "linear" | "mlp" | "cnn".
        probe_label_space: "category27" | "animacy" | "subject".

    Examples:
        modal run neural_tokenizers/eeg/modal/modal_eeg_eval.py::run
        modal run neural_tokenizers/eeg/modal/modal_eeg_eval.py::run \\
            --probe-classifier mlp --probe-label-space animacy --n-test 0
        modal run neural_tokenizers/eeg/modal/modal_eeg_eval.py::run \\
            --sources eeg1,eeg2 --n-test 0
    """
    import json

    ckpt_path = ckpt or _DEFAULT_CHECKPOINT
    sources_list = [s.strip() for s in sources.split(",") if s.strip()]
    payload = {"ckpt_path": ckpt_path, "device": "cpu"}

    print(f"[run] tokenizer={tokenizer!r}  ckpt={ckpt_path}")
    print(f"[run] sources={sources_list}  n_test={n_test}  seed={seed}")
    print(f"[run] classifier={probe_classifier!r}  label_space={probe_label_space!r}")

    report = evaluate_remote.remote(
        tokenizer_name=tokenizer,
        tokenizer_payload=payload,
        n_test=n_test,
        seed=seed,
        sources=sources_list,
        probe_classifier=probe_classifier,
        probe_label_space=probe_label_space,
        probe_class_weighted=class_weighted,
    )

    print("\n[eval] report:")
    for axis, values in report.items():
        print(f"  [{axis}]")
        for k, v in values.items():
            print(f"    {k:<32s} {v:.4f}")

    if output:
        out_path = Path(output)
    else:
        slug = _eval_slug(n_test, seed, probe_classifier, probe_label_space, sources_list, class_weighted)
        # Write locally, not to the volume (ckpt_path is a volume path).
        evals_dir = Path("neural_tokenizers/eeg/evals")
        evals_dir.mkdir(parents=True, exist_ok=True)
        out_path = evals_dir / f"eval_{slug}.json"

    # Add a legend so readers of the JSON know what 'raw' means here.
    # In the EEG eval, signal = token_ids.float(), so 'raw' probe features
    # are 17 sparse categoricals (token IDs as floats), NOT raw EEG waveforms.
    # This differs from MEG evals where 'raw' = actual sensor recordings.
    report["_feature_legend"] = {
        "tokens": "mean-pooled 64-D codebook embeddings (one per channel-second)",
        "raw": "token IDs cast to float32 — 17 sparse categoricals in [0, 8192), NOT raw EEG",
        "random": "same featurization as tokens but with uniformly random IDs (lower bound)",
    }

    out_path.write_text(json.dumps(report, indent=2))
    print(f"[eval] wrote report to {out_path}")
