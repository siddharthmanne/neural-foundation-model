"""
Convert THINGS-EEG1 continuous EEGLAB data to LaBraM HDF5 format.

Inputs (must exist on the volume before running):
  /project/data/raw/things-eeg1/derivatives/eeglab/sub-XX_task-rsvp_continuous.set
  /project/data/raw/things-eeg1/bids_events/sub-XX/eeg/sub-XX_task-rsvp_run-NN_events.tsv

Run download_things_eeg1_events first if the TSV files are missing.

Output:
  /project/data/labram_input/things-eeg1/sub-XX_train.hdf5
  /project/data/labram_input/things-eeg1/sub-XX_val.hdf5
  /project/data/labram_input/things-eeg1/manifest.json

Pipeline per subject:
  1. Parse all events TSV files (sorted by run) → ordered list of concept names
  2. Find all E1 events in the continuous MNE recording (stimulus onsets)
  3. Pair TSV concept names to E1 events by position (same count, time-ordered)
  4. Extract -0.2..+0.8 s epochs (250 samples @ 250 Hz)
  5. Select the 17 THINGS-EEG2-compatible channels from the 64-channel montage
  6. Apply image-level 80/20 train/val split on unique image filenames (seed 20200220)
  7. Filter: 0.1-75 Hz bandpass + 50 Hz notch
  8. Resample: 250 -> 200 Hz (resample_poly up=4, down=5)
  9. Concatenate per split and write LaBraM HDF5

Run with:
    modal run neural_tokenizers/eeg/modal/convert_eeg1_to_labram_hdf5.py
    modal run neural_tokenizers/eeg/modal/convert_eeg1_to_labram_hdf5.py::inspect_events
"""

import json
import os
import warnings

import modal
from _app import app, data_volume

convert_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["numpy", "scipy", "mne", "h5py"])
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_app.py"),
        "/root/_app.py",
    )
)

# The 17 posterior+occipital channels shared with THINGS-EEG2.
# These are a subset of EEG1's 64-channel montage and are verified to be
# present in THINGS-EEG1 (confirmed by MNE channel list inspection).
TARGET_CHANNELS = [
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "PZ",
    "PO3", "PO4", "PO7", "PO8", "POZ",
    "O1", "O2", "OZ",
]

LABRAM_STANDARD_1020 = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ", "FC1",
    "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6", "FT9", "FT10",
    "TP9", "TP10", "P1", "P2", "P5", "P6", "AF3", "AF4", "AF7", "AF8",
    "PO3", "PO4", "PO7", "PO8", "POZ", "OZ", "FPZ", "FCZ", "CPZ",
    "TP7", "TP8", "T7", "T8", "P7", "P8", "FT7", "FT8", "FC3", "FC4",
    "CP3", "CP4", "F1", "F2", "F5", "F6", "C1", "C2", "C5", "C6",
]

# Epoch window matching THINGS-EEG2 (-200..+800 ms = 1 s = 250 samples @ 250 Hz).
EPOCH_TMIN = -0.2
EPOCH_TMAX = 0.8
SFREQ_IN   = 250.0
SFREQ_OUT  = 200.0


def _load_events_tsv(subject: str, bids_root: str) -> list[str]:
    """Return ordered list of stimulus concept names for this subject.

    Reads all sub-XX_task-rsvp_run-NN_events.tsv files, sorted by run number.
    Within each file, rows are in presentation order. Concatenates across runs.

    The events TSV must have a column identifying the stimulus: we try
    'trial_type', 'stim_file', and 'value' (in that order).

    Returns: list of concept strings, one per stimulus trial (length = n_E1_events).
    """
    import csv
    import re

    sub_dir = os.path.join(bids_root, subject, "eeg")
    if not os.path.isdir(sub_dir):
        raise FileNotFoundError(
            f"BIDS events directory not found: {sub_dir}\n"
            "Run: modal run download_things_eeg.py::download_things_eeg1_events"
        )

    # THINGS-EEG1 BIDS has one TSV per subject (no run numbers in filename).
    # Pattern: sub-XX_task-rsvp_events.tsv
    tsv_files = sorted(
        f for f in os.listdir(sub_dir)
        if re.search(r"task-rsvp_events\.tsv$", f)
    )
    if not tsv_files:
        raise FileNotFoundError(
            f"No events TSV files found in {sub_dir}.\n"
            f"Files present: {os.listdir(sub_dir)}"
        )

    all_stim_ids = []
    stim_col = None

    for tsv_name in tsv_files:
        tsv_path = os.path.join(sub_dir, tsv_name)
        with open(tsv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        if not rows:
            continue

        # Detect the stimulus identity column on the first TSV.
        # THINGS-EEG1 uses 'stimname' (bare image filename, e.g. carousel_11s.jpg)
        # and 'stim' (full path). We prefer stimname for image-level identity.
        if stim_col is None:
            for candidate in ("stimname", "stim", "stim_file", "trial_type", "stimulus", "value"):
                if candidate in rows[0]:
                    stim_col = candidate
                    break
            if stim_col is None:
                raise KeyError(
                    f"Cannot find stimulus identity column in {tsv_path}.\n"
                    f"Columns present: {list(rows[0].keys())}\n"
                    "Expected one of: stimname, stim_file, trial_type, stimulus, value."
                )
            print(f"  Using column '{stim_col}' for stimulus identity")

        for row in rows:
            stim_id = row[stim_col].strip()
            # stimname already gives bare filename (e.g. 'carousel_11s.jpg').
            # If a full path was used, take the basename.
            if "\\" in stim_id:
                stim_id = stim_id.rsplit("\\", 1)[-1]
            if "/" in stim_id:
                stim_id = stim_id.rsplit("/", 1)[-1]
            all_stim_ids.append(stim_id)

    return all_stim_ids


def _find_e1_events(raw) -> "np.ndarray":
    """Return E1 stimulus-onset sample positions from a continuous MNE Raw object.

    THINGS-EEG1 uses event type 'E  1' (with two spaces) or 'E1'.
    Returns sorted array of sample indices (int).
    """
    import mne
    import numpy as np

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    # event_id maps annotation label -> integer code; events[:,2] is the code.
    # We want the code(s) whose label starts with 'E' followed by optional spaces and '1'.
    import re
    e1_codes = {
        code for label, code in event_id.items()
        if re.match(r"E\s*1$", label.strip())
    }
    if not e1_codes:
        raise ValueError(
            f"No E1 events found. event_id keys: {list(event_id.keys())}"
        )

    mask = np.isin(events[:, 2], list(e1_codes))
    e1_samples = np.sort(events[mask, 0])
    return e1_samples


def _extract_epochs(
    data: "np.ndarray",
    e1_samples: "np.ndarray",
    ch_indices: list[int],
    sfreq: float,
) -> "np.ndarray":
    """Extract fixed-length epochs around each E1 event.

    data: (n_total_channels, n_times)
    e1_samples: sorted sample indices of stimulus onsets
    ch_indices: indices into data's channel axis to keep
    sfreq: sampling rate (should be SFREQ_IN=250)

    Returns: (n_trials, len(ch_indices), epoch_samples) float64
    """
    import numpy as np

    tmin_samp = int(round(EPOCH_TMIN * sfreq))   # -50 samples @ 250 Hz
    tmax_samp = int(round(EPOCH_TMAX * sfreq))    # +200 samples @ 250 Hz
    epoch_len = tmax_samp - tmin_samp             # 250 samples

    epochs = []
    n_total = data.shape[1]
    skipped = 0

    for samp in e1_samples:
        start = samp + tmin_samp
        end   = samp + tmax_samp
        if start < 0 or end > n_total:
            skipped += 1
            continue
        epoch = data[np.ix_(ch_indices, np.arange(start, end))].astype(np.float64)
        # MNE reads EEGLAB data in V (SI units); LaBraM expects µV.
        # Confirmed V-scale by inspect_eeg1_events: abs-mean ~1.94e-5 V (~19.4 µV).
        epoch *= 1e6
        epochs.append(epoch)

    if skipped:
        print(f"  [warn] Skipped {skipped} epochs near recording boundary")

    return np.stack(epochs, axis=0)  # (n_trials, n_ch, epoch_len)


def _filter_and_resample(trials: "np.ndarray") -> "np.ndarray":
    """Apply 0.1-75 Hz + 50 Hz notch at SFREQ_IN, then resample to SFREQ_OUT.

    Input:  (n_trials, 17, 250)  at 250 Hz
    Output: (n_trials, 17, 200)  at 200 Hz
    """
    import numpy as np
    import scipy.signal
    import mne

    n_trials, n_ch, n_time = trials.shape
    flat = trials.reshape(n_trials * n_ch, n_time).astype(float)

    warnings.filterwarnings("ignore")
    flat = mne.filter.filter_data(
        flat, sfreq=SFREQ_IN, l_freq=0.1, h_freq=75.0,
        method="fir", fir_window="hamming", verbose=False,
    )
    flat = mne.filter.notch_filter(flat, Fs=SFREQ_IN, freqs=50.0, verbose=False)

    # Resample 250 -> 200 Hz: up=4, down=5  (250 * 4/5 = 200, exact integer ratio)
    flat_rs = scipy.signal.resample_poly(flat, up=4, down=5, axis=-1)
    n_time_out = flat_rs.shape[-1]  # should be 200

    return flat_rs.reshape(n_trials, n_ch, n_time_out).astype(np.float64)


@app.function(
    image=convert_image,
    volumes={"/project": data_volume},
    cpu=4,
    memory=32768,
    timeout=3600 * 4,
)
def inspect_eeg1_events(subject: str = "sub-01") -> None:
    """Print events TSV structure and E1 event count for one subject.

    Run this before the full conversion to verify the column structure.
    """
    import mne
    import numpy as np

    bids_root = "/project/data/raw/things-eeg1/bids_events"
    set_dir   = "/project/data/raw/things-eeg1/derivatives/eeglab"
    set_file  = os.path.join(set_dir, f"{subject}_task-rsvp_continuous.set")

    print(f"=== {subject} ===")

    print("\n--- Events TSV ---")
    stim_ids = _load_events_tsv(subject, bids_root)
    print(f"Total stimulus entries: {len(stim_ids)}")
    unique_stims = sorted(set(stim_ids))
    print(f"Unique images: {len(unique_stims)}")
    print(f"First 5: {stim_ids[:5]}")
    print(f"Last  5: {stim_ids[-5:]}")
    print(f"Image filename examples: {unique_stims[:10]}")

    print("\n--- MNE E1 events ---")
    raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
    print(f"Channels: {len(raw.ch_names)}, sfreq: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.1f} s")

    # Scale check: MNE returns EEGLAB data in V (SI units); EEGLAB stored µV.
    # Healthy reading: abs-mean ~ 1e-5 to 1e-4 V (= 10-100 µV).
    # If abs-mean is already ~10-100, it's already in µV — no rescaling needed.
    data_sample = raw.get_data()[:, :5000]
    abs_mean = float(np.abs(data_sample).mean())
    print(f"data abs-mean (first 5000 samples): {abs_mean:.3e}")
    print(f"  std: {data_sample.std():.3e}")
    if abs_mean < 1e-3:
        print(f"  -> V-scale (MNE SI units). ×1e6 needed: {abs_mean*1e6:.1f} µV")
    else:
        print(f"  -> Already µV-scale. No rescaling needed.")

    e1_samples = _find_e1_events(raw)
    print(f"E1 events: {len(e1_samples)}")

    if len(stim_ids) != len(e1_samples):
        print(f"[WARN] TSV count ({len(stim_ids)}) != E1 count ({len(e1_samples)})")
    else:
        print("[OK] TSV count matches E1 event count")

    print("\n--- Channel coverage ---")
    raw_ch_upper = [ch.upper() for ch in raw.ch_names]
    missing = [ch for ch in TARGET_CHANNELS if ch not in raw_ch_upper]
    print(f"Target channels present: {len(TARGET_CHANNELS) - len(missing)}/{len(TARGET_CHANNELS)}")
    if missing:
        print(f"  Missing: {missing}")
    else:
        print("  All 17 target channels present")


@app.function(
    image=convert_image,
    volumes={"/project": data_volume},
    cpu=4,
    memory=32768,
    timeout=3600 * 12,
)
def convert_things_eeg1(subjects: list[str] | None = None) -> dict:
    """Convert THINGS-EEG1 subjects to LaBraM HDF5.

    Returns the manifest dict (also written to disk).
    """
    import mne
    import numpy as np
    import h5py

    warnings.filterwarnings("ignore")

    if subjects is None:
        subjects = [f"sub-{i:02d}" for i in range(1, 51)]

    bids_root = "/project/data/raw/things-eeg1/bids_events"
    set_dir   = "/project/data/raw/things-eeg1/derivatives/eeglab"
    out_root  = "/project/data/labram_input/things-eeg1"
    os.makedirs(out_root, exist_ok=True)

    # Collect image filenames from TSV files for all subjects.
    # Each entry is a bare filename like 'carousel_11s.jpg' (stimname column).
    print("Loading events TSV files...")
    all_unique_stims: set[str] = set()
    per_subject_stims: dict[str, list[str]] = {}

    for subject in subjects:
        try:
            stim_ids = _load_events_tsv(subject, bids_root)
            per_subject_stims[subject] = stim_ids
            all_unique_stims.update(stim_ids)
        except FileNotFoundError as e:
            print(f"[{subject}] Skipping: {e}")

    if not all_unique_stims:
        raise RuntimeError(
            "No events TSV data found for any subject. "
            "Run download_things_eeg1_events first."
        )

    sorted_stims = sorted(all_unique_stims)
    stim_to_id: dict[str, int] = {s: i for i, s in enumerate(sorted_stims)}
    n_stims = len(sorted_stims)
    print(f"  {n_stims} unique images across {len(per_subject_stims)} subjects")

    # Image-level 80/20 split — keyed on image filename (stimname column).
    #
    # Unified split: if /project/data/labram_input/split_manifest.json exists
    # (produced by build_unified_split, which merges EEG1 + EEG2 image sets),
    # load train_images from it and intersect with EEG1's stim set.
    # Otherwise fall back to EEG1-only 80/20 with a warning.
    split_manifest_path = "/project/data/labram_input/split_manifest.json"
    if os.path.exists(split_manifest_path):
        print(f"  Using unified split from {split_manifest_path}")
        with open(split_manifest_path) as f:
            split_manifest = json.load(f)
        train_image_set = set(split_manifest["train_images"])
        train_stim_ids = {stim_to_id[s] for s in sorted_stims if s in train_image_set}
        n_train_unified = len(train_image_set & all_unique_stims)
        print(f"  {n_train_unified}/{n_stims} EEG1 images in unified train split")
    else:
        print(
            "  [warn] No split_manifest.json — using EEG1-only 80/20 split.\n"
            "  EEG1 and EEG2 splits are independent; probe accuracy may be\n"
            "  slightly inflated. Build unified split when EEG2 image list is available."
        )
        rng = np.random.default_rng(20200220)
        shuffled = rng.permutation(n_stims)
        n_train_split = int(n_stims * 0.8)
        train_stim_ids = set(shuffled[:n_train_split].tolist())

    # Save image-ID map for reproducibility.
    stim_map_path = os.path.join(out_root, "stimname_to_image_id.json")
    with open(stim_map_path, "w") as f:
        json.dump(stim_to_id, f, indent=2)
    print(f"  Image ID map saved to {stim_map_path}")

    manifest = {}
    manifest_path = os.path.join(out_root, "manifest.json")

    for subject in subjects:
        if subject not in per_subject_stims:
            print(f"[{subject}] No event TSV data — skipping")
            continue

        print(f"\n[{subject}] Starting conversion...")

        set_file = os.path.join(set_dir, f"{subject}_task-rsvp_continuous.set")
        if not os.path.exists(set_file):
            print(f"[{subject}] .set file not found at {set_file} — skipping")
            continue

        # --- Load stim filename list from TSV ---
        stim_ids_for_subject = per_subject_stims[subject]
        image_ids = [stim_to_id[s] for s in stim_ids_for_subject]

        # --- Load continuous EEG and find E1 events ---
        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
        sfreq = raw.info["sfreq"]
        if abs(sfreq - SFREQ_IN) > 1:
            raise ValueError(
                f"[{subject}] Expected {SFREQ_IN} Hz, got {sfreq} Hz. "
                "Check if resampling was applied in derivatives."
            )

        e1_samples = _find_e1_events(raw)
        n_e1 = len(e1_samples)
        n_tsv = len(stim_ids_for_subject)

        if n_e1 != n_tsv:
            raise ValueError(
                f"[{subject}] E1 event count ({n_e1}) != TSV row count ({n_tsv}). "
                "Cannot do positional matching — check TSV files."
            )
        print(f"  {n_e1} trials, {n_stims} unique images")

        # --- Select channels ---
        raw_ch_upper = [ch.upper() for ch in raw.ch_names]
        ch_indices = []
        for tch in TARGET_CHANNELS:
            if tch not in raw_ch_upper:
                raise ValueError(
                    f"[{subject}] Channel '{tch}' not found in recording. "
                    f"Available channels: {raw_ch_upper}"
                )
            ch_indices.append(raw_ch_upper.index(tch))

        data = raw.get_data()  # (n_all_ch, n_times)

        # --- Extract epochs ---
        epochs = _extract_epochs(data, e1_samples, ch_indices, sfreq)
        # epochs: (n_trials, 17, 250)  — some may be dropped for boundary issues

        # Reconcile dropped trials: if any E1 events near boundary were skipped,
        # the epoch count may be < n_e1. Rebuild image_id list from valid events.
        n_epochs_actual = len(epochs)
        if n_epochs_actual < n_e1:
            print(
                f"  [warn] {n_e1 - n_epochs_actual} trials dropped (boundary). "
                "Using surviving trials only."
            )
            tmin_samp = int(round(EPOCH_TMIN * sfreq))
            tmax_samp = int(round(EPOCH_TMAX * sfreq))
            n_total = data.shape[1]
            kept_mask = np.array([
                (samp + tmin_samp >= 0) and (samp + tmax_samp <= n_total)
                for samp in e1_samples
            ])
            image_ids = [iid for iid, keep in zip(image_ids, kept_mask) if keep]

        # --- Image-level split ---
        split_mask_train = np.array([iid in train_stim_ids for iid in image_ids])

        trials_train = epochs[split_mask_train]
        trials_val   = epochs[~split_mask_train]
        print(
            f"  Split: {len(trials_train)} train trials, {len(trials_val)} val trials"
        )

        # --- Filter + resample in batches ---
        batch_size = 300

        def process_in_batches(block):
            out = []
            for start in range(0, len(block), batch_size):
                out.append(_filter_and_resample(block[start:start + batch_size]))
            return np.concatenate(out, axis=0)

        print("  Filtering + resampling train...")
        filtered_train = process_in_batches(trials_train)
        print("  Filtering + resampling val...")
        filtered_val   = process_in_batches(trials_val)

        n_ch = len(TARGET_CHANNELS)
        assert filtered_train.shape[-1] == 200, f"Expected 200 samples after resample, got {filtered_train.shape[-1]}"

        # --- Concatenate to continuous format ---
        def to_continuous(arr):
            return arr.transpose(1, 0, 2).reshape(n_ch, len(arr) * 200)

        eeg_train_cont = to_continuous(filtered_train)
        eeg_val_cont   = to_continuous(filtered_val)

        # --- Validate channels against LaBraM ---
        for ch in TARGET_CHANNELS:
            if ch not in LABRAM_STANDARD_1020:
                raise ValueError(f"Channel '{ch}' not in LaBraM standard_1020")

        # --- Write HDF5 ---
        train_hdf5 = os.path.join(out_root, f"{subject}_train.hdf5")
        val_hdf5   = os.path.join(out_root, f"{subject}_val.hdf5")

        for hdf5_path, eeg_cont in [(train_hdf5, eeg_train_cont), (val_hdf5, eeg_val_cont)]:
            with h5py.File(hdf5_path, "w") as fh:
                grp = fh.create_group(subject)
                ds = grp.create_dataset(
                    "eeg",
                    data=eeg_cont,
                    dtype=np.float64,
                    chunks=(n_ch, 200),
                )
                ds.attrs["lFreq"]   = 0.1
                ds.attrs["hFreq"]   = 75.0
                ds.attrs["rsFreq"]  = 200
                ds.attrs["chOrder"] = TARGET_CHANNELS
            print(f"  Wrote {hdf5_path} {eeg_cont.shape}")

        manifest[subject] = {
            "train":          train_hdf5,
            "val":            val_hdf5,
            "n_train_trials": int(len(filtered_train)),
            "n_val_trials":   int(len(filtered_val)),
            "ch_names":       TARGET_CHANNELS,
        }

        # Commit after each subject so a timeout doesn't lose completed work.
        manifest_path = os.path.join(out_root, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        data_volume.commit()
        print(f"  [{subject}] committed to volume")

    print(f"\nManifest written to {manifest_path}")
    return manifest


@app.function(
    image=convert_image,
    volumes={"/project": data_volume},
    cpu=2,
    memory=8192,
    timeout=600,
)
def compute_eeg1_channel_stats() -> dict:
    """
    Compute per-channel std (17-vector) across all EEG1 subjects from the
    already-converted HDF5 files. Saves to:
      /project/data/labram_input/eeg1_per_channel_std.json

    Used by EEG2 conversion to rescale MVNN-whitened data to match EEG1's
    actual per-channel µV distribution, preserving inter-channel structure.
    """
    import numpy as np
    import h5py

    manifest_path = "/project/data/labram_input/things-eeg1/manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Accumulate per-subject per-channel stds, then average across subjects.
    # Use train HDF5 only (larger, more representative).
    per_subject_stds = []
    for subject, info in manifest.items():
        with h5py.File(info["train"], "r") as f:
            grp = f[subject]
            data = grp["eeg"][:]          # (17, T)
        ch_std = data.std(axis=-1)        # (17,)
        per_subject_stds.append(ch_std)
        print(f"  {subject}: per-channel std mean={ch_std.mean():.3f} µV, "
              f"min={ch_std.min():.3f}, max={ch_std.max():.3f}")

    # Use median (not mean) to pool across subjects — several EEG1 subjects
    # have artifact-contaminated channels with std >100 µV that would corrupt
    # a mean-based calibration target (mean pooled to 50.8 µV vs ~12 µV for
    # clean subjects). Median is robust to these outliers.
    pooled_std = np.median(np.stack(per_subject_stds), axis=0)  # (17,)
    print(f"\nPooled EEG1 per-channel std — median across subjects (µV):")
    for i, (name, s) in enumerate(zip(manifest[list(manifest.keys())[0]]["ch_names"], pooled_std)):
        print(f"  {name}: {s:.3f}")

    out = {
        "per_channel_std": pooled_std.tolist(),
        "n_subjects": len(per_subject_stds),
        "channel_names": manifest[list(manifest.keys())[0]]["ch_names"],
    }
    out_path = "/project/data/labram_input/eeg1_per_channel_std.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    data_volume.commit()
    print(f"\nSaved to {out_path}")
    return out


@app.local_entrypoint()
def compute_channel_stats_standalone():
    result = compute_eeg1_channel_stats.remote()
    print(f"Done. Pooled std mean: {sum(result['per_channel_std'])/len(result['per_channel_std']):.3f} µV")


@app.local_entrypoint()
def inspect_events():
    """Inspect events TSV structure for sub-01 before running full conversion."""
    inspect_eeg1_events.remote(subject="sub-01")


@app.local_entrypoint()
def convert_standalone():
    result = convert_things_eeg1.remote()
    print("Conversion complete. Manifest:")
    for subject, info in result.items():
        print(
            f"  {subject}: {info['n_train_trials']} train trials, "
            f"{info['n_val_trials']} val trials"
        )
