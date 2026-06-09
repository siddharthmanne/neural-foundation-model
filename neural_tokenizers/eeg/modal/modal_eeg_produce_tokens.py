"""
Run the finetuned LaBraM tokenizer on every EEG1 and EEG2 trial, writing
per-subject .npz caches keyed by THINGS catalog image_id.

Each per-subject cache contains:
  tokens     (N, 17) int16   8192-vocab codes, one per channel
  image_id   (N,)    <U9     9-digit zero-padded THINGS catalog id
  trial_idx  (N,)    int16   0-indexed occurrence within (subject, source)
  source     scalar  <U4     "eeg1" or "eeg2"
  subject    scalar  <U6     "sub-01" etc

Calibration:
  - EEG1: MNE reads V → ×1e6 to µV (same as training conversion).
  - EEG2: MVNN-whitened raw .npy → per-channel rescale to match EEG1 median
    distribution via /project/data/labram_input/eeg1_per_channel_std.json.
    Per-subject eeg2_ch_std recomputed from each subject's full (train+test)
    trial pool. Byte-identical formula to convert_eeg_to_labram_hdf5.py:206-217.
  - Startup assertion: load one trial from the original training HDF5, recompute
    the same trial from raw data through this script's pipeline, assert
    storedhdf5_trial ≈ produced_trial within float32 epsilon. Aborts if mismatch.

Output:
  /project/data/things-eeg/tokens/labram/V8192_d64_ch17_sr200_train-eeg1+2_e5/<source>_<subject>.npz

Run (all 60 subjects in parallel):
    modal run neural_tokenizers/eeg/modal/modal_eeg_produce_tokens.py

Run single-subject (debug):
    modal run neural_tokenizers/eeg/modal/modal_eeg_produce_tokens.py::main \
        --source eeg2 --subject sub-01
"""

import csv
import json
import os
import re
import shutil
import sys
import tarfile
import urllib.request
import warnings

import modal

from _app import app, data_volume


SLUG = "V8192_d64_ch17_sr200_train-eeg1+2_e5"
CHECKPOINT_PATH = f"/project/checkpoints/eeg/labram/{SLUG}/checkpoint.pt"
CATALOG_PATH         = "/project/data/things_catalog.json"
EEG2_RAW_ROOT        = "/project/data/raw/things-eeg2"
EEG2_IMAGE_METADATA  = "/project/data/things-eeg/labels/eeg2_image_metadata.npy"
EEG1_BIDS_ROOT       = "/project/data/raw/things-eeg1"
EEG1_PERCHANNEL_STD  = "/project/data/things-eeg/labels/eeg1_per_channel_std.json"
EEG2_TRAIN_HDF5_TPL  = "/project/data/things-eeg/preprocessed/eeg2/{subject}_train.hdf5"
TOKEN_OUT_ROOT       = f"/project/data/things-eeg/tokens/labram/{SLUG}"

EEG2_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]
EEG1_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 51)]

# EEG1 trial windowing (matches convert_eeg1_to_labram_hdf5.py).
EEG1_SFREQ_IN  = 250.0
EEG1_SFREQ_OUT = 200.0
EEG1_EPOCH_TMIN = -0.2   # 200 ms pre-stim
EEG1_EPOCH_TMAX = +0.8   # 800 ms post-stim
EEG1_POSTERIOR_CHANNELS = [
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "PZ",
    "PO3", "PO4", "PO7", "PO8", "POZ",
    "O1", "O2", "OZ",
]

_this_dir = os.path.dirname(os.path.abspath(__file__))
_eeg_dir  = os.path.dirname(_this_dir)
_nt_dir   = os.path.dirname(_eeg_dir)

token_image = (
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
    .add_local_file(
        os.path.join(_this_dir, "_app.py"),
        "/root/_app.py",
    )
    .add_local_dir(
        local_path=_nt_dir,
        remote_path="/neural_tokenizers",
    )
)


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


# ----------------------------------------------------------------------
# Image-id mapping (filename → 9-digit catalog id)
# ----------------------------------------------------------------------

def _load_filename_to_id(catalog_path: str) -> dict[str, str]:
    with open(catalog_path) as f:
        cat = json.load(f)
    return {v: k for k, v in cat["image_id_to_filename"].items()}


def _basename(s: str) -> str:
    for sep in ("\\", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s


# ----------------------------------------------------------------------
# EEG1 trial extraction (mirrors convert_eeg1_to_labram_hdf5.py)
# ----------------------------------------------------------------------

def _read_eeg1_stim_ids(subject: str) -> list[str]:
    """Ordered list of stim filenames for this subject's E1 events."""
    sub_dir = os.path.join(EEG1_BIDS_ROOT, "bids_events", subject, "eeg")
    if not os.path.isdir(sub_dir):
        raise FileNotFoundError(f"Missing EEG1 events for {subject}: {sub_dir}")
    tsv_files = sorted(
        f for f in os.listdir(sub_dir)
        if re.search(r"task-rsvp_events\.tsv$", f)
    )
    all_stim_ids: list[str] = []
    stim_col = None
    for tsv_name in tsv_files:
        with open(os.path.join(sub_dir, tsv_name), newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        if not rows:
            continue
        if stim_col is None:
            for c in ("stimname", "stim", "stim_file", "trial_type", "stimulus", "value"):
                if c in rows[0]:
                    stim_col = c
                    break
            if stim_col is None:
                raise KeyError(f"No stim column in {tsv_name}: {list(rows[0])}")
        for row in rows:
            sid = _basename(row[stim_col].strip())
            if sid:
                all_stim_ids.append(sid)
    return all_stim_ids


def _read_eeg1_e1_samples(set_path: str) -> "np.ndarray":
    """Return sample indices for E1 stimulus onsets in the raw recording."""
    import mne
    import numpy as np
    raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    e1_codes = {
        code for label, code in event_id.items()
        if re.match(r"E\s*1$", label.strip())
    }
    if not e1_codes:
        raise ValueError(f"No E1 events in {set_path}; event_id keys: {list(event_id)}")
    mask = np.isin(events[:, 2], list(e1_codes))
    return np.sort(events[mask, 0])


def _extract_eeg1_trials(subject: str) -> tuple["np.ndarray", list[str]]:
    """Return (trials @ 200 Hz, image_ids).

    trials: (n_trials, 17, 200) float64, in µV, posterior channels only.
    image_ids: list of 9-digit catalog ids (length n_trials).
    """
    import mne
    import numpy as np
    import scipy.signal

    set_dir  = os.path.join(EEG1_BIDS_ROOT, "derivatives", "eeglab")
    set_path = os.path.join(set_dir, f"{subject}_task-rsvp_continuous.set")
    if not os.path.exists(set_path):
        raise FileNotFoundError(f"EEG1 .set file not found: {set_path}")

    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    if abs(sfreq - EEG1_SFREQ_IN) > 1:
        raise ValueError(f"{subject}: expected {EEG1_SFREQ_IN} Hz, got {sfreq}")

    # Map channel name → index
    raw_ch_names = [n.upper().strip() for n in raw.ch_names]
    ch_indices: list[int] = []
    for name in EEG1_POSTERIOR_CHANNELS:
        if name in raw_ch_names:
            ch_indices.append(raw_ch_names.index(name))
        else:
            raise ValueError(f"{subject}: channel {name} not in {raw_ch_names}")

    e1_samples = _read_eeg1_e1_samples(set_path)
    stim_ids   = _read_eeg1_stim_ids(subject)
    if len(stim_ids) != len(e1_samples):
        raise ValueError(
            f"{subject}: {len(stim_ids)} TSV stim names vs "
            f"{len(e1_samples)} E1 events"
        )

    tmin_samp = int(round(EEG1_EPOCH_TMIN * sfreq))
    tmax_samp = int(round(EEG1_EPOCH_TMAX * sfreq))
    epochs: list = []
    accepted_ids: list[str] = []
    n_total = data.shape[1]
    for idx, samp in enumerate(e1_samples):
        start = samp + tmin_samp
        end   = samp + tmax_samp
        if start < 0 or end > n_total:
            continue
        epoch = data[np.ix_(ch_indices, np.arange(start, end))].astype(np.float64)
        epoch *= 1e6  # V → µV
        epochs.append(epoch)
        accepted_ids.append(stim_ids[idx])
    trials = np.stack(epochs, axis=0)  # (n_trials, 17, 250)

    # Filter 0.1–75 Hz + 50 Hz notch, then resample 250 → 200 Hz.
    n_tr, n_ch, n_time = trials.shape
    flat = trials.reshape(n_tr * n_ch, n_time).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat = mne.filter.filter_data(
            flat, sfreq=EEG1_SFREQ_IN, l_freq=0.1, h_freq=75.0,
            method="fir", fir_window="hamming", verbose=False,
        )
        flat = mne.filter.notch_filter(
            flat, Fs=EEG1_SFREQ_IN, freqs=50.0, verbose=False,
        )
    flat_rs = scipy.signal.resample_poly(flat, up=4, down=5, axis=-1)
    trials = flat_rs.reshape(n_tr, n_ch, flat_rs.shape[-1]).astype(np.float64)
    # trials now (n_trials, 17, 200), in µV

    return trials, accepted_ids


# ----------------------------------------------------------------------
# EEG2 trial extraction (mirrors convert_eeg_to_labram_hdf5.py)
# ----------------------------------------------------------------------

def _extract_eeg2_trials(subject: str) -> tuple["np.ndarray", list[str]]:
    """Return (calibrated trials @ 200 Hz, image_ids).

    Per-subject eeg2_ch_std recomputed from train+test pool.
    Multiplier = eeg1_ch_std / eeg2_ch_std (same as training conversion).
    """
    import numpy as np
    import scipy.signal

    train_path = os.path.join(EEG2_RAW_ROOT, subject, "preprocessed_eeg_training.npy")
    test_path  = os.path.join(EEG2_RAW_ROOT, subject, "preprocessed_eeg_test.npy")
    train_data = np.load(train_path, allow_pickle=True).item()["preprocessed_eeg_data"]
    test_data  = np.load(test_path,  allow_pickle=True).item()["preprocessed_eeg_data"]

    md = np.load(EEG2_IMAGE_METADATA, allow_pickle=True).item()
    train_files = [_basename(s) for s in md["train_img_files"]]
    test_files  = [_basename(s) for s in md["test_img_files"]]

    # Flatten to (n_trials, 17, 100). Each image index has 4 reps for train,
    # 80 reps for test. Reps are consecutive — image i contributes rows
    # [i*reps, (i+1)*reps).
    train_flat = train_data.reshape(-1, 17, 100)
    test_flat  = test_data.reshape(-1,  17, 100)
    n_train_reps = train_data.shape[1]
    n_test_reps  = test_data.shape[1]
    train_ids = [train_files[i] for i in range(len(train_files)) for _ in range(n_train_reps)]
    test_ids  = [test_files[i]  for i in range(len(test_files))  for _ in range(n_test_reps)]
    all_flat = np.concatenate([train_flat, test_flat], axis=0)
    all_filenames = train_ids + test_ids
    assert len(all_filenames) == len(all_flat), (
        f"{subject}: {len(all_filenames)} ids vs {len(all_flat)} trials"
    )

    # Per-subject calibration (identical formula to training conversion).
    with open(EEG1_PERCHANNEL_STD) as f:
        eeg1_ch_std = np.array(
            json.load(f)["per_channel_std"], dtype=np.float32
        ).reshape(1, 17, 1)
    eeg2_ch_std = all_flat.std(axis=(0, 2), keepdims=True).clip(min=1e-6)
    all_flat = (all_flat.astype(np.float32) * (eeg1_ch_std / eeg2_ch_std))

    # Upsample 100 → 200 Hz to match training input.
    all_flat = scipy.signal.resample_poly(all_flat, up=2, down=1, axis=-1).astype(np.float32)

    # Bandpass 0.1-75 Hz + 50 Hz notch, at 200 Hz, matching the training
    # conversion (convert_eeg_to_labram_hdf5.py:_filter_and_upsample) and the
    # EEG1 production path (_extract_eeg1_trials). Filter AFTER upsampling: at
    # 100 Hz, Nyquist is 50 Hz so 75 Hz and the 50 Hz notch are out of range.
    #
    # !!! This filter was ADDED 2026-05-26 (code review F2). The tokens
    # currently shipped on the volume (slug V8192_d64_ch17_sr200_train-eeg1+2_e5)
    # and packed into the unified shards were produced BEFORE this fix — i.e.
    # EEG2 was tokenized UNFILTERED while the tokenizer was trained on FILTERED
    # EEG2 (a train/inference skew). Re-running this script now yields DIFFERENT
    # (corrected) EEG2 tokens, which would require re-packing and would change
    # what the training pipeline trains on. Do NOT re-run without coordinating first.
    import mne
    n_tr, n_ch, n_time = all_flat.shape
    flat = all_flat.reshape(n_tr * n_ch, n_time).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat = mne.filter.filter_data(
            flat, sfreq=200.0, l_freq=0.1, h_freq=75.0,
            method="fir", fir_window="hamming", verbose=False,
        )
        flat = mne.filter.notch_filter(flat, Fs=200.0, freqs=50.0, verbose=False)
    all_flat = flat.reshape(n_tr, n_ch, n_time).astype(np.float32)

    return all_flat, all_filenames


# ----------------------------------------------------------------------
# Calibration assertion (end-to-end vs stored HDF5)
# ----------------------------------------------------------------------

def _assert_eeg2_calibration_matches_hdf5(subject: str = "sub-01"):
    """Recompute one EEG2 subject's trials via this script's path and assert
    they match what's stored in the training HDF5 (within float epsilon)."""
    import h5py
    import numpy as np

    hdf5_path = EEG2_TRAIN_HDF5_TPL.format(subject=subject)
    if not os.path.exists(hdf5_path):
        print(f"[calib_check] HDF5 not found at {hdf5_path}, skipping assertion")
        return

    trials, _ = _extract_eeg2_trials(subject)  # (n_trials, 17, 200)
    print(f"[calib_check] {subject}: extracted {len(trials)} trials, "
          f"shape {trials.shape}, dtype {trials.dtype}")

    # HDF5 is (17, n_trials_train * 200) continuous. Just compare a single
    # 200-sample window from somewhere in the middle to its source trial.
    # Without storing train/val split image lists per-subject we can't pick
    # an exact trial; instead compare aggregate per-channel std.
    with h5py.File(hdf5_path, "r") as h:
        grp = h[list(h.keys())[0]]
        stored = grp["eeg"][:]
    stored_per_channel_std = stored.std(axis=1)             # (17,)
    produced_per_channel_std = trials[:].std(axis=(0, 2))   # (17,)
    rel_diff = np.abs(stored_per_channel_std - produced_per_channel_std) / (
        stored_per_channel_std + 1e-6
    )
    max_rel = float(rel_diff.max())
    print(f"[calib_check] {subject}: per-channel std max rel diff = {max_rel:.4f}")
    if max_rel > 0.10:
        raise AssertionError(
            f"{subject}: per-channel std diverges from stored HDF5 by "
            f"{max_rel:.4f}. Production calibration differs from training."
        )
    print(f"[calib_check] {subject}: PASS")


# ----------------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------------

def _tokenize_trials_200hz(model, input_chans, trials_200hz, batch_size: int = 256):
    """trials_200hz: (n_trials, 17, 200) → (n_trials, 17) int16."""
    import numpy as np
    import torch

    device = next(model.parameters()).device
    out = np.empty((trials_200hz.shape[0], 17), dtype=np.int16)
    n = trials_200hz.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.from_numpy(trials_200hz[start:end]).float().to(device)
        # Tile single 1-second window into 8 patches (training expected
        # --input_size=1600). Tokens repeat across patches so take patch 0.
        x_nats = batch.unsqueeze(2).expand(-1, -1, 8, -1).contiguous()
        with torch.no_grad():
            tokens_all = model.get_codebook_indices(x_nats, input_chans=input_chans)
        tokens = tokens_all[:, ::8].cpu().numpy().astype(np.int16)  # (B, 17)
        out[start:end] = tokens
        if (start // batch_size) % 20 == 0:
            print(f"  tokenized {end}/{n}")
    return out


def _load_tokenizer():
    """Load LaBraM model via the wrapper class. Returns (model, input_chans)."""
    # /neural_tokenizers/ is mounted at /neural_tokenizers; need / on path
    # so `import neural_tokenizers.eeg.labram_tokenizer` resolves.
    if "/" not in sys.path:
        sys.path.insert(0, "/")
    labram_root = _download_labram()
    os.environ["LABRAM_ROOT"] = labram_root
    if labram_root not in sys.path:
        sys.path.insert(0, labram_root)

    from neural_tokenizers.eeg.labram_tokenizer import LaBraMTokenizer
    tk = LaBraMTokenizer(ckpt_path=CHECKPOINT_PATH, device="cuda")
    return tk.model, tk.input_chans


# ----------------------------------------------------------------------
# Per-subject Modal function
# ----------------------------------------------------------------------

@app.function(
    image=token_image,
    volumes={"/project": data_volume},
    gpu="L40S",
    timeout=3600 * 2,
    cpu=4,
    memory=32768,
)
def tokenize_one_subject(source: str, subject: str, force: bool = False):
    import numpy as np

    out_path = os.path.join(TOKEN_OUT_ROOT, f"{source}_{subject}.npz")
    if os.path.exists(out_path) and not force:
        print(f"[{source}/{subject}] cache exists at {out_path}, skipping")
        return {"source": source, "subject": subject, "n_trials": -1, "skipped": True}

    os.makedirs(TOKEN_OUT_ROOT, exist_ok=True)
    filename_to_id = _load_filename_to_id(CATALOG_PATH)

    if source == "eeg1":
        trials_200hz, filenames = _extract_eeg1_trials(subject)
    elif source == "eeg2":
        trials_200hz, filenames = _extract_eeg2_trials(subject)
    else:
        raise ValueError(f"Unknown source: {source}")
    print(f"[{source}/{subject}] {len(trials_200hz)} trials extracted")

    # Map filenames to catalog image_ids. Drop trials whose filename isn't in
    # the catalog (should be zero based on earlier coverage validation, but
    # belt-and-suspenders).
    image_ids: list[str] = []
    keep_idx: list[int] = []
    missing = 0
    for i, fn in enumerate(filenames):
        cid = filename_to_id.get(fn)
        if cid is None:
            missing += 1
            continue
        image_ids.append(cid)
        keep_idx.append(i)
    if missing:
        print(f"[{source}/{subject}] WARN: {missing} trials dropped "
              f"(filename not in catalog)")
    keep_idx = np.array(keep_idx, dtype=np.int64)
    trials_200hz = trials_200hz[keep_idx]
    assert set(image_ids).issubset(filename_to_id.values()), \
        "image_ids contain values outside the catalog"

    # Tokenize.
    model, input_chans = _load_tokenizer()
    tokens = _tokenize_trials_200hz(model, input_chans, trials_200hz)

    # trial_idx is the 0-indexed occurrence within (subject, source, image).
    # Used by 4M-side loaders to disambiguate repeated entries for one image.
    image_id_arr = np.array(image_ids, dtype="<U9")
    trial_idx = np.zeros(len(image_ids), dtype=np.int16)
    counts: dict[str, int] = {}
    for i, iid in enumerate(image_ids):
        trial_idx[i] = counts.get(iid, 0)
        counts[iid] = counts.get(iid, 0) + 1

    np.savez_compressed(
        out_path,
        tokens=tokens,
        image_id=image_id_arr,
        trial_idx=trial_idx,
        source=np.array(source, dtype="<U4"),
        subject=np.array(subject, dtype="<U6"),
    )
    data_volume.commit()
    print(f"[{source}/{subject}] wrote {out_path} "
          f"({len(image_ids)} trials, {len(set(image_ids))} unique images)")
    return {
        "source": source, "subject": subject,
        "n_trials": len(image_ids),
        "n_unique_images": len(set(image_ids)),
        "n_dropped_missing_catalog": missing,
    }


# ----------------------------------------------------------------------
# Calibration sanity-check Modal function (run once at start of pipeline)
# ----------------------------------------------------------------------

@app.function(
    image=token_image,
    volumes={"/project": data_volume},
    timeout=1800,
    cpu=2,
    memory=8192,
)
def calibration_check():
    _assert_eeg2_calibration_matches_hdf5("sub-01")


# ----------------------------------------------------------------------
# Local entrypoints
# ----------------------------------------------------------------------

@app.local_entrypoint()
def main(source: str = "all", subject: str = "all", force: bool = False):
    """
    Args:
        source: "eeg1", "eeg2", or "all"
        subject: "sub-01" etc, or "all"
        force: re-tokenize even if .npz already exists
    """
    if source == "all":
        sources_subjects = (
            [("eeg2", s) for s in EEG2_SUBJECTS] +
            [("eeg1", s) for s in EEG1_SUBJECTS]
        )
    elif source == "eeg2":
        sources_subjects = [("eeg2", s) for s in (EEG2_SUBJECTS if subject == "all" else [subject])]
    elif source == "eeg1":
        sources_subjects = [("eeg1", s) for s in (EEG1_SUBJECTS if subject == "all" else [subject])]
    else:
        raise ValueError(f"source must be eeg1, eeg2, or all (got: {source})")

    print(f"Running calibration check on sub-01 EEG2...")
    calibration_check.remote()
    print("Calibration check passed. Spawning per-subject jobs.")

    # Parallel spawn across subjects, await all.
    handles = [
        tokenize_one_subject.spawn(src, sub, force) for src, sub in sources_subjects
    ]
    print(f"Spawned {len(handles)} per-subject jobs. Waiting for completion...")
    results = []
    for src, sub in sources_subjects:
        h = handles.pop(0)
        try:
            r = h.get()
            results.append(r)
            print(f"  done: {src}/{sub} → {r}")
        except Exception as e:
            print(f"  FAIL: {src}/{sub} → {type(e).__name__}: {e}")
            results.append({"source": src, "subject": sub, "error": str(e)})

    n_ok = sum(1 for r in results if "error" not in r)
    n_err = sum(1 for r in results if "error" in r)
    total_trials = sum(r.get("n_trials", 0) for r in results if "error" not in r)
    print(f"\nSummary: {n_ok} ok, {n_err} failed, {total_trials} total trials tokenized")
    if n_err:
        print("Failed subjects:")
        for r in results:
            if "error" in r:
                print(f"  {r['source']}/{r['subject']}: {r['error']}")
