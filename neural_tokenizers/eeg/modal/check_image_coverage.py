"""
Quick coverage check: which THINGS catalog images are covered by EEG1 / EEG2 /
their union?

EEG1 image IDs come from the bids_events TSVs (stimname column).
EEG2 image IDs come from Gifford's image_metadata.npy (in raw subject dirs or
fetched from OSF if not present).

Run:
    modal run neural_tokenizers/eeg/modal/check_image_coverage.py
"""

import csv
import json
import os
import re

import modal

from _app import app, data_volume


coverage_image = (
    modal.Image.debian_slim()
    .pip_install(["numpy<2"])
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_app.py"),
        "/root/_app.py",
    )
)


def _collect_eeg1_filenames(bids_root: str) -> set[str]:
    """Return set of unique image basenames seen across all EEG1 subjects' TSV events."""
    stims: set[str] = set()
    for sub in sorted(os.listdir(bids_root)):
        if not sub.startswith("sub-"):
            continue
        sub_dir = os.path.join(bids_root, sub, "eeg")
        if not os.path.isdir(sub_dir):
            continue
        tsv_files = sorted(
            f for f in os.listdir(sub_dir)
            if re.search(r"task-rsvp_events\.tsv$", f)
        )
        for tsv_name in tsv_files:
            with open(os.path.join(sub_dir, tsv_name), newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)
            if not rows:
                continue
            stim_col = None
            for candidate in ("stimname", "stim", "stim_file", "trial_type", "stimulus", "value"):
                if candidate in rows[0]:
                    stim_col = candidate
                    break
            if stim_col is None:
                continue
            for row in rows:
                sid = row[stim_col].strip()
                if "\\" in sid:
                    sid = sid.rsplit("\\", 1)[-1]
                elif "/" in sid:
                    sid = sid.rsplit("/", 1)[-1]
                if sid:
                    stims.add(sid)
    return stims


def _collect_eeg2_filenames(eeg2_raw_root: str) -> set[str]:
    """Look for image_metadata.npy in EEG2 raw root or any subject dir."""
    import numpy as np
    candidates = [
        os.path.join(eeg2_raw_root, "image_metadata.npy"),
        os.path.join(eeg2_raw_root, "sub-01", "image_metadata.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True).item()
            train_names = list(data.get("train_img_files", data.get("train_files", [])))
            test_names  = list(data.get("test_img_files",  data.get("test_files",  [])))
            return set(_basename(n) for n in train_names + test_names)
    return set()


def _basename(name: str) -> str:
    for sep in ("\\", "/"):
        if sep in name:
            name = name.rsplit(sep, 1)[-1]
    return name


@app.function(
    image=coverage_image,
    volumes={"/project": data_volume},
    timeout=900,
    cpu=2,
)
def check_coverage():
    cat_path = "/project/data/things_catalog.json"
    with open(cat_path) as f:
        cat = json.load(f)
    id_to_filename = cat["image_id_to_filename"]  # 26,107 entries
    catalog_filenames = set(id_to_filename.values())
    print(f"Catalog: {len(catalog_filenames)} unique image filenames")

    eeg1_root = "/project/data/raw/things-eeg1"
    bids_events_root = os.path.join(eeg1_root, "bids_events")
    derivatives_root = os.path.join(eeg1_root, "derivatives")
    if os.path.isdir(bids_events_root):
        scan_root = bids_events_root
    elif os.path.isdir(derivatives_root):
        scan_root = derivatives_root
    else:
        scan_root = eeg1_root

    print(f"\nScanning EEG1 events under {scan_root}...")
    eeg1_filenames = _collect_eeg1_filenames(scan_root)
    print(f"EEG1: {len(eeg1_filenames)} unique image filenames extracted")
    print(f"  examples: {sorted(eeg1_filenames)[:5]}")

    print("\nScanning EEG2 image_metadata.npy...")
    eeg2_filenames = _collect_eeg2_filenames("/project/data/raw/things-eeg2")
    print(f"EEG2: {len(eeg2_filenames)} unique image filenames extracted "
          f"(0 means image_metadata.npy not on volume)")
    if eeg2_filenames:
        print(f"  examples: {sorted(eeg2_filenames)[:5]}")

    eeg1_in_catalog = eeg1_filenames & catalog_filenames
    eeg2_in_catalog = eeg2_filenames & catalog_filenames
    union_in_catalog = (eeg1_filenames | eeg2_filenames) & catalog_filenames

    print("\n" + "=" * 60)
    print(f"EEG1 ∩ catalog:           {len(eeg1_in_catalog):>6}")
    print(f"EEG2 ∩ catalog:           {len(eeg2_in_catalog):>6}")
    print(f"(EEG1 ∪ EEG2) ∩ catalog:  {len(union_in_catalog):>6}")
    print(f"EEG1 ∩ EEG2:              {len(eeg1_filenames & eeg2_filenames):>6}")
    print(f"Catalog total:            {len(catalog_filenames):>6}")
    print(f"Catalog NOT covered:      {len(catalog_filenames - (eeg1_filenames | eeg2_filenames)):>6}")
    print("=" * 60)

    return {
        "catalog_total": len(catalog_filenames),
        "eeg1_covered": len(eeg1_in_catalog),
        "eeg2_covered": len(eeg2_in_catalog),
        "union_covered": len(union_in_catalog),
        "eeg1_only": len(eeg1_filenames - eeg2_filenames),
        "eeg2_only": len(eeg2_filenames - eeg1_filenames),
        "overlap": len(eeg1_filenames & eeg2_filenames),
    }


@app.local_entrypoint()
def main():
    result = check_coverage.remote()
    print("\nResult:")
    for k, v in result.items():
        print(f"  {k:<24s} {v}")
