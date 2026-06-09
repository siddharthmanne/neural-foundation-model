"""
Download Gifford's image_metadata.npy for THINGS-EEG2 from OSF and re-check
EEG1 ∪ EEG2 coverage of the THINGS catalog.

OSF project 3jk45 (Gifford et al. 2022) contains several components; the image
metadata file lives outside the preprocessed-EEG component (anp5v) that we
already downloaded. This script enumerates project 3jk45's storage looking
for image_metadata.npy, downloads it to the volume, and reports coverage.

Run:
    modal run neural_tokenizers/eeg/modal/download_eeg2_image_metadata.py
"""

import csv
import json
import os
import re

import modal

from _app import app, data_volume


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "osfclient>=0.0.5",
        "numpy<2",
        "requests",
    ])
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_app.py"),
        "/root/_app.py",
    )
)


def _collect_eeg1_filenames(bids_events_root: str) -> set[str]:
    """Same logic as check_image_coverage.py — extract EEG1 stim filenames."""
    stims: set[str] = set()
    for sub in sorted(os.listdir(bids_events_root)):
        if not sub.startswith("sub-"):
            continue
        sub_dir = os.path.join(bids_events_root, sub, "eeg")
        if not os.path.isdir(sub_dir):
            continue
        for tsv_name in sorted(
            f for f in os.listdir(sub_dir)
            if re.search(r"task-rsvp_events\.tsv$", f)
        ):
            with open(os.path.join(sub_dir, tsv_name), newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)
            if not rows:
                continue
            stim_col = None
            for c in ("stimname", "stim", "stim_file", "trial_type", "stimulus", "value"):
                if c in rows[0]:
                    stim_col = c
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


def _find_and_download_image_metadata(dest_dir: str) -> str | None:
    """Enumerate OSF project 3jk45 storage looking for image_metadata.npy.

    Returns the local path it was written to, or None if not found.
    """
    from osfclient import OSF

    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "image_metadata.npy")
    if os.path.exists(dest_path):
        print(f"[osf] image_metadata.npy already at {dest_path}")
        return dest_path

    # Try multiple known OSF components for the THINGS-EEG2 project.
    # The image set / metadata may live in a different component than anp5v
    # (preprocessed EEG). We probe a few candidates.
    osf = OSF()
    candidates = ["3jk45", "y63gw", "crxs4", "anp5v"]
    for guid in candidates:
        try:
            print(f"[osf] probing component {guid}...")
            project = osf.project(guid)
            for store in project.storages:
                for osf_file in store.files:
                    name = os.path.basename(osf_file.path)
                    if name == "image_metadata.npy":
                        print(f"[osf] found {osf_file.path} in {guid}, downloading...")
                        with open(dest_path, "wb") as f:
                            osf_file.write_to(f)
                        print(f"[osf] wrote {dest_path} ({os.path.getsize(dest_path)} bytes)")
                        return dest_path
        except Exception as e:
            print(f"[osf] {guid}: {type(e).__name__}: {e}")

    print("[osf] image_metadata.npy NOT FOUND in candidate components.")
    print("[osf] You may need to download it manually from "
          "https://osf.io/3jk45/ → image set component → image_metadata.npy")
    return None


@app.function(
    image=image,
    volumes={"/project": data_volume},
    timeout=900,
    cpu=2,
)
def download_and_check():
    import numpy as np

    metadata_dir = "/project/data/raw/things-eeg2"
    md_path = _find_and_download_image_metadata(metadata_dir)
    data_volume.commit()

    cat_path = "/project/data/things_catalog.json"
    with open(cat_path) as f:
        cat = json.load(f)
    catalog_filenames = set(cat["image_id_to_filename"].values())
    print(f"\nCatalog: {len(catalog_filenames)} unique image filenames")

    eeg1_filenames = _collect_eeg1_filenames("/project/data/raw/things-eeg1/bids_events")
    print(f"EEG1: {len(eeg1_filenames)} unique image filenames")

    eeg2_filenames: set[str] = set()
    if md_path and os.path.exists(md_path):
        md = np.load(md_path, allow_pickle=True).item()
        print(f"\nimage_metadata.npy keys: {list(md.keys())}")
        train_names = list(md.get("train_img_files",
                                 md.get("train_files",
                                       md.get("train_img_concepts", []))))
        test_names  = list(md.get("test_img_files",
                                 md.get("test_files",
                                       md.get("test_img_concepts",  []))))
        eeg2_filenames = set(_basename(n) for n in train_names + test_names)
        print(f"EEG2: {len(eeg2_filenames)} unique image filenames "
              f"(train={len(train_names)}, test={len(test_names)})")
        print(f"  examples: {sorted(eeg2_filenames)[:5]}")

    eeg1_in_cat = eeg1_filenames & catalog_filenames
    eeg2_in_cat = eeg2_filenames & catalog_filenames
    union_in_cat = (eeg1_filenames | eeg2_filenames) & catalog_filenames

    print("\n" + "=" * 60)
    print(f"EEG1 ∩ catalog:           {len(eeg1_in_cat):>6}")
    print(f"EEG2 ∩ catalog:           {len(eeg2_in_cat):>6}")
    print(f"(EEG1 ∪ EEG2) ∩ catalog:  {len(union_in_cat):>6}")
    print(f"EEG1 ∩ EEG2:              {len(eeg1_filenames & eeg2_filenames):>6}")
    print(f"EEG2 only (not in EEG1):  {len(eeg2_filenames - eeg1_filenames):>6}")
    print(f"Catalog total:            {len(catalog_filenames):>6}")
    print(f"Catalog NOT covered:      {len(catalog_filenames - (eeg1_filenames | eeg2_filenames)):>6}")
    print("=" * 60)

    return {
        "catalog_total": len(catalog_filenames),
        "eeg1_covered": len(eeg1_in_cat),
        "eeg2_covered": len(eeg2_in_cat),
        "union_covered": len(union_in_cat),
        "overlap": len(eeg1_filenames & eeg2_filenames),
        "eeg2_only": len(eeg2_filenames - eeg1_filenames),
        "uncovered": len(catalog_filenames - (eeg1_filenames | eeg2_filenames)),
    }


def _basename(name: str) -> str:
    for sep in ("\\", "/"):
        if sep in name:
            name = name.rsplit(sep, 1)[-1]
    return name


@app.local_entrypoint()
def main():
    result = download_and_check.remote()
    print("\nFinal:")
    for k, v in result.items():
        print(f"  {k:<24s} {v}")
