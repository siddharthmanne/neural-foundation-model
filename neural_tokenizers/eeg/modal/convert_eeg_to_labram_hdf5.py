"""
Convert THINGS-EEG2 preprocessed .npy files to LaBraM HDF5 format.

Reads per-subject .npy files from /project/data/raw/things-eeg2/sub-XX/,
applies image-level 80/20 train/val split, filters, upsamples to 200 Hz,
concatenates trials end-to-end, and writes HDF5 files that LaBraM's
SingleShockDataset can consume.

Run with:
    modal run neural_tokenizers/eeg/modal/convert_eeg_to_labram_hdf5.py
"""

import json
import os
import warnings

import modal
from _app import app, data_volume

# No LaBraM needed for conversion; keep the image lean.
convert_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["numpy", "scipy", "mne", "h5py"])
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_app.py"),
        "/root/_app.py",
    )
)

# All 17 posterior+occipital channels in LaBraM's uppercase 10-20 format.
# Verified against gifale95/eeg_encoding_model preprocessing (regex ^O *|^P *).
CHANNEL_NAMES = [
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "PZ",
    "PO3", "PO4", "PO7", "PO8", "POZ",
    "O1", "O2", "OZ",
]

# LaBraM standard_1020 list (utils.py:42-57). Used to validate channel names
# before writing HDF5 — missing names raise ValueError at training time.
LABRAM_STANDARD_1020 = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ", "FC1",
    "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6", "FT9", "FT10",
    "TP9", "TP10", "P1", "P2", "P5", "P6", "AF3", "AF4", "AF7", "AF8",
    "PO3", "PO4", "PO7", "PO8", "POZ", "OZ", "FPZ", "FCZ", "CPZ",
    "TP7", "TP8", "T7", "T8", "P7", "P8", "FT7", "FT8", "FC3", "FC4",
    "CP3", "CP4", "F1", "F2", "F5", "F6", "C1", "C2", "C5", "C6",
]


def _find_npy_file(subject_dir: str, pattern: str) -> str:
    """Return the first file in subject_dir whose name contains pattern."""
    for fname in os.listdir(subject_dir):
        if pattern in fname and fname.endswith(".npy"):
            return os.path.join(subject_dir, fname)
    raise FileNotFoundError(
        f"No file matching '{pattern}' in {subject_dir}. "
        f"Found: {os.listdir(subject_dir)}"
    )


def _filter_and_upsample(trials: "np.ndarray") -> "np.ndarray":
    """
    Upsample to 200 Hz first, then apply MNE bandpass + notch filter.

    Upsample before filter: at 100 Hz the Nyquist is 50 Hz, so h_freq=75 is
    above Nyquist and notch=50 is exactly at Nyquist — both are invalid. After
    upsampling to 200 Hz the Nyquist is 100 Hz and both are well inside range.
    This matches LaBraM's own dataset_maker/shock/utils/eegUtils.py order.

    Input:  (n_trials, 17, 100)  MVNN-whitened, 100 Hz
    Output: (n_trials, 17, 200)  200 Hz, filtered
    """
    import numpy as np
    import scipy.signal
    import mne

    n_trials, n_ch, _ = trials.shape

    # Step 1: upsample 100 -> 200 Hz (exact integer ratio, no edge artefacts).
    upsampled = scipy.signal.resample_poly(
        trials.astype(float), up=2, down=1, axis=-1
    )  # (n_trials, 17, 200)
    _, _, n_times_up = upsampled.shape

    # Step 2: filter at 200 Hz — Nyquist is 100 Hz, so 0.1–75 Hz and 50 Hz
    # notch are valid.
    flat = upsampled.reshape(n_trials * n_ch, n_times_up)
    warnings.filterwarnings("ignore")
    flat = mne.filter.filter_data(
        flat, sfreq=200.0, l_freq=0.1, h_freq=75.0,
        method="fir", fir_window="hamming", verbose=False,
    )
    flat = mne.filter.notch_filter(
        flat, Fs=200.0, freqs=50.0, verbose=False,
    )

    return flat.reshape(n_trials, n_ch, n_times_up).astype(np.float64)


@app.function(
    image=convert_image,
    volumes={"/project": data_volume},
    cpu=4,
    memory=32768,
    timeout=3600 * 3,
)
def convert_things_eeg2(subjects: list[str] | None = None) -> dict:
    """
    Convert THINGS-EEG2 subjects to LaBraM HDF5.

    Returns the manifest dict (also written to disk).
    """
    import numpy as np
    import h5py

    warnings.filterwarnings("ignore")

    if subjects is None:
        subjects = [f"sub-{i:02d}" for i in range(1, 11)]

    raw_root = "/project/data/raw/things-eeg2"
    out_root = "/project/data/things-eeg/preprocessed/eeg2"
    os.makedirs(out_root, exist_ok=True)

    # Image IDs: training images 0..16539, test images 16540..16739.
    n_train_images = 16540
    n_test_images = 200
    all_image_ids = np.arange(n_train_images + n_test_images)

    rng = np.random.default_rng(20200220)
    shuffled = rng.permutation(all_image_ids)
    n_train_split = int(len(all_image_ids) * 0.8)
    train_image_ids = set(shuffled[:n_train_split].tolist())
    # val_image_ids = shuffled[n_train_split:] -- implied by complement

    manifest = {}

    for subject in subjects:
        print(f"[{subject}] Starting conversion...")
        subject_dir = os.path.join(raw_root, subject)

        train_npy_path = _find_npy_file(subject_dir, "training")
        test_npy_path = _find_npy_file(subject_dir, "test")

        train_data = np.load(train_npy_path, allow_pickle=True).item()
        test_data = np.load(test_npy_path, allow_pickle=True).item()

        # Shapes: (16540, 4, 17, 100) and (200, 80, 17, 100)
        eeg_train = train_data["preprocessed_eeg_data"]  # (16540, 4, 17, 100)
        eeg_test = test_data["preprocessed_eeg_data"]    # (200,  80, 17, 100)

        # Read actual channel order from the data dict — Gifford's preprocessing
        # writes a 'ch_names' key. Using this instead of CHANNEL_NAMES ensures
        # chOrder in the HDF5 matches the actual row order of the (17, T) array,
        # so LaBraM's pos_embed slicing assigns the right positions to each channel.
        ch_names_raw = train_data.get("ch_names", None)
        if ch_names_raw is None:
            raise KeyError(
                f"[{subject}] preprocessed .npy dict missing 'ch_names' key. "
                "Cannot determine channel order for HDF5 chOrder attribute. "
                "Keys present: " + str(list(train_data.keys()))
            )
        ch_names_actual = [c.upper() for c in ch_names_raw]

        n_train_img, n_train_rep, n_ch, n_time = eeg_train.shape
        n_test_img,  n_test_rep,  _,    _      = eeg_test.shape

        assert n_train_img == n_train_images, f"Expected {n_train_images} train images, got {n_train_img}"
        assert n_test_img == n_test_images, f"Expected {n_test_images} test images, got {n_test_img}"
        assert n_ch == len(ch_names_actual), f"Expected {len(ch_names_actual)} channels (from ch_names), got {n_ch}"

        # Validate every channel name against LaBraM's standard_1020 list.
        for ch in ch_names_actual:
            if ch not in LABRAM_STANDARD_1020:
                raise ValueError(
                    f"[{subject}] Channel '{ch}' not in LaBraM standard_1020. "
                    "LaBraM will raise ValueError at training time."
                )

        # Flatten to (n_total_trials, 17, 100) with matching image ID per trial.
        # Training block: each image has n_train_rep repetitions.
        train_trials_list = []
        train_image_id_list = []
        for img_idx in range(n_train_img):
            for rep in range(n_train_rep):
                train_trials_list.append(eeg_train[img_idx, rep])
                train_image_id_list.append(img_idx)

        # Test block: image IDs 16540..16739.
        test_trials_list = []
        test_image_id_list = []
        for img_idx in range(n_test_img):
            for rep in range(n_test_rep):
                test_trials_list.append(eeg_test[img_idx, rep])
                test_image_id_list.append(n_train_images + img_idx)

        all_trials = np.array(train_trials_list + test_trials_list, dtype=np.float32)
        all_image_ids_per_trial = train_image_id_list + test_image_id_list

        # Per-channel calibration: rescale EEG2 channel-wise to match EEG1's
        # per-channel std vector (µV). This preserves inter-channel structure
        # (occipital channels legitimately have higher variance than parietal)
        # which the pretrained codebook expects. The EEG1 std vector is computed
        # once by compute_eeg1_channel_stats() and loaded from JSON.
        eeg1_stats_path = "/project/data/things-eeg/labels/eeg1_per_channel_std.json"
        if not os.path.exists(eeg1_stats_path):
            raise FileNotFoundError(
                f"EEG1 per-channel std not found at {eeg1_stats_path}. "
                "Run: modal run neural_tokenizers/eeg/modal/convert_eeg1_to_labram_hdf5.py"
                "::compute_channel_stats_standalone"
            )
        with open(eeg1_stats_path) as _f:
            _eeg1_stats = json.load(_f)
        eeg1_ch_std = np.array(_eeg1_stats["per_channel_std"], dtype=np.float32).reshape(1, 17, 1)
        eeg2_ch_std = all_trials.std(axis=(0, 2), keepdims=True).clip(min=1e-6)  # (1, 17, 1)
        all_trials *= eeg1_ch_std / eeg2_ch_std
        print(f"[{subject}] Channel-wise µV calibration: "
              f"EEG2 pre-std mean={eeg2_ch_std.squeeze().mean():.4f}, "
              f"EEG1 target mean={eeg1_ch_std.squeeze().mean():.3f} µV")

        # Apply the image-level split to individual trials.
        split_mask_train = np.array(
            [iid in train_image_ids for iid in all_image_ids_per_trial]
        )

        trials_train = all_trials[split_mask_train]   # (n_train_trials, 17, 100)
        trials_val   = all_trials[~split_mask_train]  # (n_val_trials,   17, 100)

        print(
            f"[{subject}] Split: {len(trials_train)} train trials, "
            f"{len(trials_val)} val trials"
        )

        # Filter + upsample in batches of 500 to cap per-batch memory.
        batch_size = 500

        def process_in_batches(trials_block):
            processed = []
            for start in range(0, len(trials_block), batch_size):
                batch = trials_block[start : start + batch_size]
                processed.append(_filter_and_upsample(batch))
            return np.concatenate(processed, axis=0)

        print(f"[{subject}] Filtering + upsampling train trials...")
        filtered_train = process_in_batches(trials_train)  # (n_train_trials, 17, 200)
        print(f"[{subject}] Filtering + upsampling val trials...")
        filtered_val   = process_in_batches(trials_val)    # (n_val_trials,   17, 200)

        # Concatenate trials end-to-end per the LaBraM continuous-recording format.
        # Shape: (17, n_trials * 200). stride_size=window_size=200 during training
        # means each window aligns exactly to one original trial.
        def trials_to_continuous(trials_upsampled):
            # (n_trials, 17, 200) -> (17, n_trials * 200)
            return trials_upsampled.transpose(1, 0, 2).reshape(
                n_ch, len(trials_upsampled) * 200
            )

        eeg_train_cont = trials_to_continuous(filtered_train)
        eeg_val_cont   = trials_to_continuous(filtered_val)

        # Write HDF5 files.
        train_hdf5_path = os.path.join(out_root, f"{subject}_train.hdf5")
        val_hdf5_path   = os.path.join(out_root, f"{subject}_val.hdf5")

        for hdf5_path, eeg_cont in [
            (train_hdf5_path, eeg_train_cont),
            (val_hdf5_path,   eeg_val_cont),
        ]:
            with h5py.File(hdf5_path, "w") as f:
                grp = f.create_group(subject)
                ds = grp.create_dataset(
                    "eeg",
                    data=eeg_cont,
                    dtype=np.float64,
                    chunks=(n_ch, 200),
                )
                ds.attrs["lFreq"]   = 0.1
                ds.attrs["hFreq"]   = 75.0
                ds.attrs["rsFreq"]  = 200
                # chOrder: actual row order from the .npy dict, uppercased.
                ds.attrs["chOrder"] = ch_names_actual
            print(f"[{subject}] Wrote {hdf5_path} ({eeg_cont.shape})")

        manifest[subject] = {
            "train":          train_hdf5_path,
            "val":            val_hdf5_path,
            "n_train_trials": int(len(filtered_train)),
            "n_val_trials":   int(len(filtered_val)),
            "ch_names":       ch_names_actual,
        }

    manifest_path = os.path.join(out_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")

    data_volume.commit()
    return manifest


@app.function(
    image=convert_image,
    volumes={"/project": data_volume},
    cpu=2,
    memory=8192,
    timeout=600,
)
def inspect_eeg2_npy() -> None:
    """Print all keys and metadata from sub-01's EEG2 .npy dicts.

    Answers: does the .npy store image filenames, or only positional indices?
    We need image filenames to build a unified EEG1+EEG2 train/val split.
    Also checks /project/data/raw/things-eeg2/sub-01/ for sidecar files.
    """
    import numpy as np

    sub01_dir = "/project/data/raw/things-eeg2/sub-01"

    # Directory listing for sidecar files.
    print("=== Directory listing: /project/data/raw/things-eeg2/sub-01/ ===")
    try:
        entries = sorted(os.listdir(sub01_dir))
        for e in entries:
            import stat
            full = os.path.join(sub01_dir, e)
            sz = os.path.getsize(full)
            print(f"  {e:60s}  {sz:>12,} bytes")
    except FileNotFoundError:
        print("  [not found]")

    print()

    for npy_basename, label in [
        ("preprocessed_eeg_training.npy", "TRAINING  (16,540 images × n_reps × 17ch × 100t)"),
        ("preprocessed_eeg_test.npy",     "TEST      (200 images × n_reps × 17ch × 100t)"),
    ]:
        npy_path = os.path.join(sub01_dir, npy_basename)
        print(f"=== {label} ===")
        print(f"    file: {npy_path}")
        if not os.path.exists(npy_path):
            print("    [FILE NOT FOUND]")
            print()
            continue

        d = np.load(npy_path, allow_pickle=True).item()
        print(f"    top-level keys ({len(d)}):")
        for key, val in sorted(d.items()):
            if isinstance(val, np.ndarray):
                print(f"      '{key}': ndarray  shape={val.shape}  dtype={val.dtype}")
                # If array contains strings/bytes, print first 5.
                if val.dtype.kind in ("U", "S", "O"):
                    flat = val.flat
                    samples = [str(next(flat)) for _ in range(min(5, val.size))]
                    print(f"        first 5 values: {samples}")
            elif isinstance(val, (list, tuple)):
                n = len(val)
                print(f"      '{key}': {type(val).__name__}  len={n}")
                # Print first 5 if they look like strings.
                if n > 0 and isinstance(val[0], (str, bytes)):
                    print(f"        first 5: {[str(v) for v in val[:5]]}")
            else:
                print(f"      '{key}': {type(val).__name__}  value={repr(val)[:120]}")
        print()


@app.local_entrypoint()
def inspect_eeg2_standalone():
    inspect_eeg2_npy.remote()


@app.local_entrypoint()
def convert_standalone():
    result = convert_things_eeg2.remote()
    print("Conversion complete. Manifest:")
    for subject, info in result.items():
        print(
            f"  {subject}: {info['n_train_trials']} train trials, "
            f"{info['n_val_trials']} val trials"
        )
