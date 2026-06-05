"""
Experiment 1.5: probe on cross-subject/rep averaged EEG2 waveform tokens.

Diagnostic for whether single-trial noise is the bottleneck in the §5.3 probe.
Mirrors MEG's cross-subject averaging run (modal_meg_eval.py averaging="cross_subject").

Pipeline:
 1. Load preprocessed_eeg_training.npy for all 10 EEG2 subjects.
 2. Apply the image-level 80/20 val split (seed 20200220, same as LaBraM
    finetuning). Keep only the ~3,308 val-assigned image conditions.
    The 200 EEG2 test images (80 reps each) are excluded — using training-pool
    images keeps the averaging depth consistent at 4 reps × 10 subjects = 40
    trials per image across all M averaged waveforms.
 3. Per subject: calibrate to µV, upsample 100→200 Hz, bandpass 0.1–75 Hz
    + 50 Hz notch (production preprocessing, matches LaBraM finetuning).
 4. Average across all 10 subjects × 4 reps per image →
    (M, 17, 200) averaged waveforms.
 5. Tokenize averaged waveforms via the full LaBraM model. Input is already
    at 200 Hz — tile directly to 8 patches, no second upsample.
 6. Run §5 harness (codebook + sequence + probe + retrieval) using
    LaBraMEvalAdapter (embedding-table only). Reconstruction disabled.

Compare vs eval_ntest=full_s0.json (single-trial tokens, same label space).
If averaged probe >> single-trial probe, single-trial noise was the bottleneck.
If similar, the tokenizer has discarded the task-relevant signal.

Output: neural_tokenizers/eeg/evals/eval_ntest=full_s0_avgcross_subject.json

Run:
    modal run neural_tokenizers/eeg/modal/eval_labram_avg_probe.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import urllib.request
import warnings
from pathlib import Path

import modal
import numpy as np


# ---- Modal scaffold (inlined — no _app.py dependency) ----------------------

app = modal.App("neural-fm")
_project_volume = modal.Volume.from_name("project", create_if_missing=False)
PROJECT_MOUNT = "/project"

# ---- Constants --------------------------------------------------------------

SLUG             = "V8192_d64_ch17_sr200_train-eeg1+2_e5"
CHECKPOINT_PATH  = f"/project/checkpoints/eeg/labram/{SLUG}/checkpoint.pt"
CATALOG_PATH     = "/project/data/things_catalog.json"
EEG2_RAW_ROOT    = "/project/data/raw/things-eeg2"
EEG2_IMAGE_META  = "/project/data/things-eeg/labels/eeg2_image_metadata.npy"
EEG1_STD_PATH    = "/project/data/things-eeg/labels/eeg1_per_channel_std.json"

EEG2_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]
THINGS_EEG2_CH_NAMES = [
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "PZ",
    "PO3", "PO4", "PO7", "PO8", "POZ",
    "O1", "O2", "OZ",
]

N_TRAIN_IMAGES = 16540
N_TEST_IMAGES  = 200
SPLIT_SEED     = 20200220
VAL_FRAC       = 0.2

# ---- Container image --------------------------------------------------------

_this_dir = os.path.dirname(os.path.abspath(__file__))

avg_probe_image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel")
    .pip_install([
        "einops",
        "timm==0.6.12",
        "tensorboardX",
        "h5py",
        "mne",
        "scipy",
        "numpy<2",
        "pyhealth",
    ])
    .add_local_python_source("neural_tokenizers")
)


# ---- Pure-Python helpers (run locally and remotely) -------------------------

def _basename(s: str) -> str:
    for sep in ("\\", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s


def _download_labram(dest: str = "/tmp/LaBraM") -> str:
    if not os.path.isdir(dest):
        print("Downloading LaBraM source...")
        tmp_archive = "/tmp/labram.tar.gz"
        urllib.request.urlretrieve(
            "https://github.com/935963004/LaBraM/archive/refs/heads/main.tar.gz",
            tmp_archive,
        )
        with tarfile.open(tmp_archive, "r:gz") as tf:
            tf.extractall("/tmp/labram_extract")
        shutil.move("/tmp/labram_extract/LaBraM-main", dest)
        os.remove(tmp_archive)
        print("LaBraM downloaded.")
    return dest


def _val_condition_mask() -> np.ndarray:
    """(N_TRAIN_IMAGES,) bool: True for val-assigned training image conditions.

    Replicates the split used during LaBraM finetuning: permute all 16,740
    image conditions (train + EEG2 test), assign the top 20% to val. Keep
    only the entries that index into the 16,540 training-pool images.
    """
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = rng.permutation(N_TRAIN_IMAGES + N_TEST_IMAGES)
    n_train_split = int((N_TRAIN_IMAGES + N_TEST_IMAGES) * (1 - VAL_FRAC))
    val_global = set(int(x) for x in shuffled[n_train_split:])
    return np.array([i in val_global for i in range(N_TRAIN_IMAGES)], dtype=bool)


def _preprocess_subject(
    subject: str,
    eeg1_ch_std: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess EEG2 training trials for one subject.

    Filters to val-assigned conditions immediately after loading to keep peak
    memory low (~230 MB per subject instead of 4.5 GB).

    Returns:
        trials:   (n_val_trials, 17, 200) float32 — calibrated µV, 200 Hz
        cond_idx: (n_val_trials,) int64 — condition index ∈ [0, N_TRAIN_IMAGES)
    """
    import scipy.signal
    import mne

    train_path = os.path.join(EEG2_RAW_ROOT, subject, "preprocessed_eeg_training.npy")
    test_path  = os.path.join(EEG2_RAW_ROOT, subject, "preprocessed_eeg_test.npy")
    data      = np.load(train_path, allow_pickle=True).item()["preprocessed_eeg_data"]
    data_test = np.load(test_path,  allow_pickle=True).item()["preprocessed_eeg_data"]
    # train: (16540, 4, 17, 100), test: (200, 80, 17, 100), MVNN-whitened, 100 Hz

    n_images, n_reps = data.shape[0], data.shape[1]

    # Per-subject calibration: compute std over train + test combined, exactly
    # as modal_eeg_produce_tokens.py::_extract_eeg2_trials() does.
    # Using training data only gives a different scale because the 200 test
    # images have 80 reps each and dominate the pooled variance.
    all_flat = np.concatenate([
        data.reshape(-1, 17, 100),
        data_test.reshape(-1, 17, 100),
    ], axis=0).astype(np.float32)
    eeg2_ch_std = all_flat.std(axis=(0, 2)).clip(min=1e-6)       # (17,)
    scale = (eeg1_ch_std / eeg2_ch_std).astype(np.float32)       # (17,)

    # Filter to val conditions before upsample/filter to save memory
    val_cond_indices = np.where(val_mask)[0]                      # (3308,)
    val_data = data[val_cond_indices]                             # (3308, 4, 17, 100)
    del data, data_test, all_flat                                 # free ~5 GB

    # Flatten reps: (3308, 4, 17, 100) → (13232, 17, 100)
    n_val = len(val_cond_indices)
    flat = val_data.reshape(n_val * n_reps, 17, 100).astype(np.float32)
    cond_idx = np.repeat(val_cond_indices, n_reps).astype(np.int64)
    del val_data

    # Upsample 100 → 200 Hz
    up = scipy.signal.resample_poly(flat, up=2, down=1, axis=-1)  # (13232, 17, 200)
    del flat

    # Bandpass 0.1–75 Hz + 50 Hz notch at 200 Hz (matches LaBraM training conversion)
    n_tr, n_ch, n_t = up.shape
    arr = up.reshape(n_tr * n_ch, n_t).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr = mne.filter.filter_data(
            arr, sfreq=200.0, l_freq=0.1, h_freq=75.0,
            method="fir", fir_window="hamming", verbose=False,
        )
        arr = mne.filter.notch_filter(arr, Fs=200.0, freqs=50.0, verbose=False)
    up = arr.reshape(n_tr, n_ch, n_t).astype(np.float32)

    # Apply µV calibration scale
    up *= scale[np.newaxis, :, np.newaxis]

    print(f"  [{subject}] {n_val * n_reps} val trials  "
          f"scale mean={scale.mean():.3f} std={scale.std():.3f}")
    return up, cond_idx


def _average_by_condition(
    all_trials: np.ndarray,
    all_cond_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average (N, 17, 200) trials by condition index.

    Returns:
        X_avg:       (M, 17, 200) float32
        sorted_conds: (M,) int64 — sorted unique condition indices
    """
    unique_conds = np.unique(all_cond_idx)
    M = len(unique_conds)
    cond_to_dense = {int(c): i for i, c in enumerate(unique_conds)}
    dense_idx = np.array([cond_to_dense[int(c)] for c in all_cond_idx], dtype=np.int64)

    sums   = np.zeros((M, 17, 200), dtype=np.float64)
    counts = np.zeros(M, dtype=np.int64)
    np.add.at(sums,   dense_idx, all_trials.astype(np.float64))
    np.add.at(counts, dense_idx, 1)

    X_avg = (sums / counts[:, np.newaxis, np.newaxis]).astype(np.float32)
    return X_avg, unique_conds


def _tokenize_200hz(
    model,
    input_chans: list[int],
    X: np.ndarray,
    device,
    batch_size: int = 128,
) -> np.ndarray:
    """Tokenize (N, 17, 200) 200 Hz waveforms using the full LaBraM model.

    Input is already at 200 Hz — no upsample. Tiles each trial to 8 identical
    patches (input_size=1600 = 8×200-sample patches) and takes patch-0 code
    per channel, matching LaBraMTokenizer.tokenize() for 200 Hz input.

    Returns (N, 17) int16 token array.
    """
    import torch

    N = len(X)
    out = np.empty((N, 17), dtype=np.int16)
    model.eval()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = torch.from_numpy(X[start:end]).to(device)           # (B, 17, 200)
        x_nats = batch.unsqueeze(2).expand(-1, -1, 8, -1).contiguous()  # (B, 17, 8, 200)
        with torch.no_grad():
            tokens_all = model.get_codebook_indices(x_nats, input_chans=input_chans)
            tokens = tokens_all[:, ::8]                             # (B, 17)
        out[start:end] = tokens.cpu().to(torch.int16).numpy()

        if (start // batch_size) % 10 == 0:
            print(f"  tokenized {end}/{N}")

    return out


# ---- Remote function --------------------------------------------------------

@app.function(
    image=avg_probe_image,
    volumes={PROJECT_MOUNT: _project_volume},
    gpu="A10G",
    cpu=8,
    memory=32 * 1024,
    timeout=60 * 60 * 3,
)
def run_avg_probe(
    checkpoint_path: str = CHECKPOINT_PATH,
    probe_classifier: str = "linear",
    seed: int = 0,
) -> dict:
    import torch

    # ---- LaBraM on sys.path ----
    labram_root = _download_labram()
    if labram_root not in sys.path:
        sys.path.insert(0, labram_root)

    from neural_tokenizers.eeg.labram_tokenizer import _load_vqnsp, _build_input_chans
    from neural_tokenizers.eeg.labram_adapter import LaBraMEvalAdapter
    from neural_tokenizers.eeg.labels import ConceptMapping, SuperordinateMapping
    from neural_tokenizers.evaluation import EvalConfig, evaluate
    from neural_tokenizers.eeg.eeg_config import EVAL_DEFAULTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[avg_probe] device={device}")

    # ---- Load full LaBraM model for tokenization ----
    print(f"[avg_probe] loading LaBraM model from {checkpoint_path}")
    model = _load_vqnsp(checkpoint_path, torch.device(device))
    input_chans = _build_input_chans(THINGS_EEG2_CH_NAMES)

    # ---- EEG1 per-channel std for calibration ----
    with open(EEG1_STD_PATH) as f:
        eeg1_ch_std = np.array(json.load(f)["per_channel_std"], dtype=np.float32)  # (17,)

    # ---- Val split mask ----
    val_mask = _val_condition_mask()
    n_val_conds = int(val_mask.sum())
    print(f"[avg_probe] val training conditions: {n_val_conds} / {N_TRAIN_IMAGES}")

    # ---- Load + preprocess all subjects, serial to keep peak mem bounded ----
    print("[avg_probe] loading and preprocessing subjects...")
    all_trials_parts: list[np.ndarray] = []
    all_cond_parts:   list[np.ndarray] = []

    for subject in EEG2_SUBJECTS:
        trials, cond_idx = _preprocess_subject(subject, eeg1_ch_std, val_mask)
        all_trials_parts.append(trials)
        all_cond_parts.append(cond_idx)

    all_trials   = np.concatenate(all_trials_parts, axis=0)   # (13232*10, 17, 200)
    all_cond_idx = np.concatenate(all_cond_parts,   axis=0)
    del all_trials_parts, all_cond_parts
    print(f"[avg_probe] pooled: {len(all_trials)} val trials across 10 subjects")

    # Sanity: each val condition should have exactly n_subjects * n_reps = 40 trials
    _, counts_per_cond = np.unique(all_cond_idx, return_counts=True)
    print(f"[avg_probe] trials/condition: min={counts_per_cond.min()} "
          f"max={counts_per_cond.max()} median={int(np.median(counts_per_cond))} "
          f"(expected 40 = 10 subjects × 4 reps)")

    # ---- Average by condition ----
    print("[avg_probe] averaging across subjects and reps...")
    X_avg, sorted_conds = _average_by_condition(all_trials, all_cond_idx)
    del all_trials
    M = len(X_avg)
    print(f"[avg_probe] averaged waveforms: {M} × {X_avg.shape[1:]}  "
          f"dtype={X_avg.dtype}")

    # ---- Tokenize averaged waveforms (200 Hz, no second upsample) ----
    print(f"[avg_probe] tokenizing {M} averaged waveforms...")
    tokens = _tokenize_200hz(model, input_chans, X_avg, device)  # (M, 17) int16
    del model, X_avg
    print(f"[avg_probe] tokens: {tokens.shape}  dtype={tokens.dtype}")

    # ---- Map condition indices → 9-digit THINGS catalog image_ids ----
    md = np.load(EEG2_IMAGE_META, allow_pickle=True).item()
    train_filenames = [_basename(s) for s in md["train_img_files"]]

    with open(CATALOG_PATH) as f:
        cat = json.load(f)
    filename_to_id = {v: k for k, v in cat["image_id_to_filename"].items()}

    image_ids = np.array(
        [filename_to_id[train_filenames[int(c)]] for c in sorted_conds],
        dtype="<U9",
    )

    # ---- Encode superordinate-27 labels ----
    image_ids_int = np.array([int(x) for x in image_ids], dtype=np.int64)
    concept_map = ConceptMapping.load()
    super_map   = SuperordinateMapping.load()
    encoded_labels, valid = super_map.encode_image_ids(image_ids_int, concept_map)

    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"[avg_probe] dropping {n_dropped} waveforms — no category27 label")
        tokens         = tokens[valid]
        encoded_labels = encoded_labels[valid]

    labels     = torch.from_numpy(encoded_labels.astype("int64"))
    n_classes  = int(np.unique(encoded_labels).size)
    n_samples  = len(labels)
    print(f"[avg_probe] {n_samples} samples  {n_classes}/27 categories  "
          f"~{n_samples // n_classes} samples/class")

    # ---- Run harness ----
    # Pass pre-tokenized data as (B, 17, 1) float — same convention as
    # modal_eeg_eval.py. LaBraMEvalAdapter.tokenize() squeezes the dummy axis.
    signal  = torch.from_numpy(tokens.astype(np.int64)).float().unsqueeze(-1)  # (M, 17, 1)
    adapter = LaBraMEvalAdapter(ckpt_path=checkpoint_path, device="cpu")

    cfg = EvalConfig(
        sample_rate_hz=EVAL_DEFAULTS.sample_rate_hz,
        bands=dict(EVAL_DEFAULTS.bands),
        device="cpu",
        batch_size=512,
        seed=seed,
        psd_nperseg=EVAL_DEFAULTS.psd_nperseg,
        probe_epochs=EVAL_DEFAULTS.probe_epochs,
        probe_top_k=EVAL_DEFAULTS.probe_top_k,
        probe_test_frac=EVAL_DEFAULTS.probe_test_frac,
        probe_n_folds=EVAL_DEFAULTS.probe_n_folds,
        probe_rvq_layers=EVAL_DEFAULTS.probe_rvq_layers,
        probe_class_weighted=True,
        probe_classifier=probe_classifier,
        probe_mlp_hidden=EVAL_DEFAULTS.probe_mlp_hidden,
        probe_mlp_dropout=EVAL_DEFAULTS.probe_mlp_dropout,
        probe_cnn_hidden=EVAL_DEFAULTS.probe_cnn_hidden,
        run_reconstruction=False,
        run_codebook=True,
        run_sequence=True,
        run_probe=True,
        run_retrieval=True,
    )

    print(f"[avg_probe] running harness: {n_samples} averaged-waveform tokens, "
          f"classifier={probe_classifier!r}...")
    report = evaluate(adapter, signal, labels, cfg)
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


# ---- Local entrypoint -------------------------------------------------------

@app.local_entrypoint()
def run(
    checkpoint_path: str = CHECKPOINT_PATH,
    probe_classifier: str = "linear",
    seed: int = 0,
    output: str = "",
):
    """Run Exp 1.5 averaged-probe eval remotely.

    Args:
        checkpoint_path: LaBraM checkpoint on the project volume.
        probe_classifier: "linear" | "mlp" | "cnn".
        seed: RNG seed for probe CV.
        output: explicit local output path. Defaults to
            neural_tokenizers/eeg/evals/eval_ntest=full_s<seed>_avgcross_subject.json
    """
    print(f"[run] ckpt={checkpoint_path}")
    print(f"[run] probe_classifier={probe_classifier!r}  seed={seed}")

    report = run_avg_probe.remote(
        checkpoint_path=checkpoint_path,
        probe_classifier=probe_classifier,
        seed=seed,
    )

    print("\n[eval] report:")
    for axis, values in report.items():
        print(f"  [{axis}]")
        for k, v in values.items():
            print(f"    {k:<32s} {v:.4f}")

    if output:
        out_path = Path(output)
    else:
        evals_dir = Path("neural_tokenizers/eeg/evals")
        evals_dir.mkdir(parents=True, exist_ok=True)
        clf_part = "" if probe_classifier == "linear" else f"_{probe_classifier}"
        out_path = evals_dir / f"eval_ntest=full_s{seed}{clf_part}_avgcross_subject.json"

    report["_feature_legend"] = {
        "tokens": (
            "mean-pooled 64-D codebook embeddings (LaBraMEvalAdapter); "
            "input was averaged across 40 trials (4 reps × 10 EEG2 subjects)"
        ),
        "raw": "averaged-waveform token IDs cast to float32 — 17 sparse categoricals",
        "random": "uniformly random token IDs (lower bound)",
        "_averaging": "4 reps × 10 subjects = 40 trials per image condition",
        "_n_train_images_val": "~3308 val-assigned training-pool images",
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[eval] wrote report to {out_path}")
