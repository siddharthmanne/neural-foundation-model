"""
Download THINGS-EEG1 and THINGS-EEG2 preprocessed data into the `project`
Modal volume. Resume-safe: re-running is a no-op if data is already there.

Usage from the modal/ directory:
    modal run download_things_eeg.py::download_things_eeg2
    modal run download_things_eeg.py::download_things_eeg1
    modal run download_things_eeg.py::verify

After both downloads finish:
    /project/data/raw/things-eeg2/sub-01/preprocessed_eeg_training.npy   (and _test.npy)
    /project/data/raw/things-eeg2/sub-02/...
    ...
    /project/data/raw/things-eeg2/sub-10/...
    /project/data/raw/things-eeg1/derivatives/eeglab/sub-01_task-rsvp_continuous.set    (and .fdt)
    ...

Sources:
- THINGS-EEG2 (Gifford et al. 2022): OSF component anp5v (within project 3jk45).
    Downloads per-file via osfclient Python API (not the unreliable ?zip= endpoint).
    Format: numpy .npy dicts, 100 Hz, 17 occipital+posterior channels,
    MVNN-whitened, epoched -200..+800 ms.
- THINGS-EEG1 (Grootswagers et al. 2022): OpenNeuro ds003825 (DOI
    10.18112/openneuro.ds003825). Preprocessed = EEGLAB .set/.fdt in
    derivatives/. We pull derivatives/ only via aws s3 sync (no-sign-request).

LaBraM compatibility notes (handled in a follow-up conversion script,
not at download time):
- THINGS-EEG2 is at 100 Hz; LaBraM expects 200 Hz. Resample at convert step.
- THINGS-EEG2 channels are subset 17 occipital+posterior; LaBraM is
    channel-flexible via the standard_1020 lookup, so 17 channels works.
- THINGS-EEG1 is at 1000 Hz raw; preprocessed derivatives sampling rate
    will be confirmed by `verify()`.
"""

import subprocess
from pathlib import Path

import modal


app = modal.App("neural-fm-eeg-downloads")

# Container image with everything needed for OSF + OpenNeuro pulls + verification.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "curl", "unzip", "git")
    .pip_install(
        "osfclient>=0.0.5",     # OSF Python client for THINGS-EEG2
        "awscli",                # for OpenNeuro S3 sync (THINGS-EEG1)
        "mne>=1.6.0",            # to load EEGLAB .set/.fdt for verify
        "numpy",
        "tqdm",
    )
)

# Reference the `project` volume (you confirmed this is your volume name).
data_volume = modal.Volume.from_name("project", create_if_missing=True)


# ---------------------------------------------------------------------------
# THINGS-EEG2: OSF project 3jk45 (Gifford et al. 2022)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/project": data_volume},
    timeout=60 * 60 * 4,   # 4h — OSF can be slow under load
)
def download_things_eeg2():
    """Pull THINGS-EEG2 preprocessed EEG (10 subjects) to /project/data/raw/things-eeg2/.

    Uses osfclient Python API to enumerate and download each .npy file from
    OSF component anp5v individually. More reliable than the ?zip= endpoint,
    and naturally resume-safe (skips files that already exist).
    """
    from osfclient import OSF

    target_dir = Path("/project/data/raw/things-eeg2")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale zip from any prior failed attempt
    stale_zip = target_dir / "preprocessed.zip"
    stale_zip.unlink(missing_ok=True)

    if _has_complete_eeg2(target_dir):
        print(f"[skip] {target_dir} already has all 10 subjects with train+test files.")
        return

    print("[osf] Connecting to OSF component anp5v (THINGS-EEG2 preprocessed)...")
    osf = OSF()
    project = osf.project("anp5v")

    downloaded = 0
    skipped = 0
    for store in project.storages:
        for osf_file in store.files:
            # osf_file.path looks like /sub-01/preprocessed_eeg_training.npy
            rel = osf_file.path.lstrip("/")
            # Skip 63-channel variants — we only want the standard 17-channel files
            if "63_channels" in rel:
                continue
            local_path = target_dir / rel
            if local_path.exists():
                skipped += 1
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[osf] downloading {rel}")
            with open(local_path, "wb") as f:
                osf_file.write_to(f)
            downloaded += 1

    print(f"[osf] done: {downloaded} downloaded, {skipped} already present")

    # OSF stores each subject as a zip. Unzip and delete.
    for i in range(1, 11):
        sub_zip = target_dir / f"sub-{i:02d}.zip"
        if sub_zip.exists():
            print(f"[unzip] {sub_zip.name}")
            subprocess.run(["unzip", "-n", str(sub_zip), "-d", str(target_dir)], check=True)
            sub_zip.unlink()

    if not _has_complete_eeg2(target_dir):
        print(
            "[WARN] Not all 10 subjects have both train+test .npy files. "
            "Check component anp5v on OSF for the actual file listing."
        )

    data_volume.commit()
    print(f"[done] {target_dir} contents:")
    subprocess.run(["du", "-sh", str(target_dir)], check=False)
    subprocess.run(["ls", "-la", str(target_dir)], check=False)


# ---------------------------------------------------------------------------
# THINGS-EEG1: OpenNeuro ds003825 (Grootswagers et al. 2022)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/project": data_volume},
    timeout=60 * 60 * 6,   # 6h — derivatives is large
)
def download_things_eeg1():
    """Pull THINGS-EEG1 derivatives/ (EEGLAB .set/.fdt for 50 subjects) to /project/data/raw/things-eeg1/.

    Uses `aws s3 sync --no-sign-request` against the OpenNeuro public bucket.
    aws s3 sync is natively idempotent / resume-safe — re-running only fetches
    files that are missing or changed on the remote side.
    """
    target_dir = Path("/project/data/raw/things-eeg1")
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[aws] s3 sync openneuro.org/ds003825/derivatives -> {target_dir}/derivatives")
    result = subprocess.run(
        [
            "aws", "s3", "sync",
            "--no-sign-request",
            "s3://openneuro.org/ds003825/derivatives/",
            str(target_dir / "derivatives"),
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"aws s3 sync failed (rc={result.returncode})")

    # Also grab participants.tsv and dataset_description.json for metadata
    for name in ("participants.tsv", "participants.json", "dataset_description.json", "README"):
        subprocess.run(
            [
                "aws", "s3", "cp",
                "--no-sign-request",
                f"s3://openneuro.org/ds003825/{name}",
                str(target_dir / name),
            ],
            check=False,  # missing files are OK
        )

    data_volume.commit()
    print(f"[done] {target_dir} contents:")
    subprocess.run(["du", "-sh", str(target_dir)], check=False)
    subprocess.run(["ls", "-la", str(target_dir)], check=False)


# ---------------------------------------------------------------------------
# Verify both datasets landed and are loadable
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/project": data_volume},
    timeout=60 * 30,
)
def verify():
    """List both dataset directories and load one subject's array from each
    to confirm files are uncorrupted and the format is what we expect.
    """
    import numpy as np

    print("=" * 70)
    print("THINGS-EEG2 (/project/data/raw/things-eeg2)")
    print("=" * 70)
    subprocess.run(["du", "-sh", "/project/data/raw/things-eeg2"], check=False)
    subprocess.run(["ls", "-la", "/project/data/raw/things-eeg2"], check=False)

    eeg2_train = Path("/project/data/raw/things-eeg2/sub-01/preprocessed_eeg_training.npy")
    eeg2_test = Path("/project/data/raw/things-eeg2/sub-01/preprocessed_eeg_test.npy")
    if eeg2_train.exists():
        d = np.load(eeg2_train, allow_pickle=True).item()
        arr = d["preprocessed_eeg_data"]
        ch_names = d.get("ch_names", [])
        times = d.get("times", [])
        sfreq = (1.0 / (times[1] - times[0])) if len(times) > 1 else None
        print(f"\n[sub-01 train] shape={arr.shape}, dtype={arr.dtype}")
        print(f"  channels (n={len(ch_names)}): {ch_names}")
        print(f"  timepoints (n={len(times)}), window=[{times[0]:.3f}, {times[-1]:.3f}]s")
        if sfreq is not None:
            print(f"  inferred sampling rate: {sfreq:.1f} Hz")
        # Expected shape: (image_conditions ~ 16540, repetitions=4, channels=17, timepoints=100)
    else:
        print(f"[WARN] {eeg2_train} not found")

    if eeg2_test.exists():
        d = np.load(eeg2_test, allow_pickle=True).item()
        arr = d["preprocessed_eeg_data"]
        print(f"[sub-01 test ] shape={arr.shape}, dtype={arr.dtype}")
        # Expected shape: (image_conditions=200, repetitions=80, channels=17, timepoints=100)
    else:
        print(f"[WARN] {eeg2_test} not found")

    print()
    print("=" * 70)
    print("THINGS-EEG1 (/project/data/raw/things-eeg1)")
    print("=" * 70)
    subprocess.run(["du", "-sh", "/project/data/raw/things-eeg1"], check=False)
    subprocess.run(["ls", "-la", "/project/data/raw/things-eeg1"], check=False)

    eeg1_dir = Path("/project/data/raw/things-eeg1/derivatives")
    if eeg1_dir.exists():
        set_files = sorted(eeg1_dir.rglob("*.set"))
        print(f"\n[eeg1] found {len(set_files)} .set files in derivatives/")
        if set_files:
            try:
                import mne
                raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=False, verbose=False)
                print(f"  example file: {set_files[0].relative_to(eeg1_dir)}")
                print(f"  channels: {len(raw.ch_names)}")
                print(f"  sampling rate: {raw.info['sfreq']} Hz")
                print(f"  duration: {raw.times[-1]:.1f}s")
            except Exception as e:
                print(f"  [WARN] mne load failed: {e}")
    else:
        print(f"[WARN] {eeg1_dir} not found")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _has_complete_eeg2(target_dir: Path) -> bool:
    """Return True iff all 10 subjects have train+test .npy files present."""
    for i in range(1, 11):
        sub_dir = target_dir / f"sub-{i:02d}"
        if not (sub_dir / "preprocessed_eeg_training.npy").exists():
            return False
        if not (sub_dir / "preprocessed_eeg_test.npy").exists():
            return False
    return True


# ---------------------------------------------------------------------------
# THINGS-EEG1 BIDS events TSVs: image identity for each trial
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/project": data_volume},
    timeout=60 * 60,   # 1h — TSV files are small, but 50 subjects × N runs each
)
def download_things_eeg1_events():
    """Pull THINGS-EEG1 BIDS events TSV files from OpenNeuro ds003825.

    These TSVs are the only place that maps trial position to stimulus concept
    (image identity). Required for the image-level 80/20 train/val split and
    for shard packing. The main EEG1 download (download_things_eeg1) only pulled
    derivatives/; this fetches the BIDS-level sub-XX/eeg/*.tsv files.

    Saves to: /project/data/raw/things-eeg1/bids_events/sub-XX/eeg/*.tsv
    """
    target_dir = Path("/project/data/raw/things-eeg1/bids_events")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("[aws] Syncing BIDS events TSVs from ds003825 sub-*/eeg/*.tsv ...")
    result = subprocess.run(
        [
            "aws", "s3", "sync",
            "--no-sign-request",
            "s3://openneuro.org/ds003825/",
            str(target_dir),
            "--exclude", "*",
            "--include", "sub-*/eeg/*.tsv",
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"aws s3 sync failed (rc={result.returncode})")

    # Count what we got
    tsv_files = sorted(target_dir.rglob("*.tsv"))
    print(f"[done] {len(tsv_files)} TSV files downloaded to {target_dir}")
    if tsv_files:
        print(f"  first: {tsv_files[0].relative_to(target_dir)}")
        print(f"  last:  {tsv_files[-1].relative_to(target_dir)}")
        # Print column names from first file so we can verify image identity field
        with open(tsv_files[0]) as f:
            header = f.readline().strip()
        print(f"  columns: {header}")
        # Print first data row
        with open(tsv_files[0]) as f:
            f.readline()
            first_row = f.readline().strip()
        print(f"  first row: {first_row}")

    data_volume.commit()


@app.local_entrypoint()
def main():
    print("Available functions:")
    print("  modal run download_things_eeg.py::download_things_eeg2        # OSF 3jk45 -> /project/data/raw/things-eeg2")
    print("  modal run download_things_eeg.py::download_things_eeg1        # OpenNeuro ds003825 derivatives -> /project/data/raw/things-eeg1")
    print("  modal run download_things_eeg.py::download_things_eeg1_events # BIDS events TSVs -> /project/data/raw/things-eeg1/bids_events/")
    print("  modal run download_things_eeg.py::verify                      # list + load-test one subject from each")
    print()
    print("For long downloads use --detach:")
    print("  modal run --detach download_things_eeg.py::download_things_eeg1")
