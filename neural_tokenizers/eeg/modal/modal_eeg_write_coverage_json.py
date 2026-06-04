"""
Write a JSON file listing which THINGS catalog image_ids are covered by EEG.

Maps EEG1 stim filenames (from BIDS event TSVs) and EEG2 stim filenames (from
Gifford's image_metadata.npy) onto things_catalog.json's image_id space, then
writes a single JSON for computing the EEG ∩ MEG image set for
shared cross-modal val splits.

Output path on Volume:
    /project/data/eeg_coverage.json

Schema:
    {
      "schema_version": 1,
      "catalog_n_images": 26107,
      "image_ids_eeg1":   ["000000001", ...]  # sorted, 9-digit zero-padded
      "image_ids_eeg2":   ["000000001", ...]
      "image_ids_union":  ["000000001", ...]
      "image_ids_intersection": ["000000001", ...]
      "image_ids_uncovered":    ["000000123", ...]   # in catalog, no EEG anywhere
      "n_eeg1": 22448,
      "n_eeg2": 16740,
      "n_union": 22470,
      "n_intersection": 16718,
      "n_uncovered": 3637
    }

Run:
    modal run neural_tokenizers/eeg/modal/modal_eeg_write_coverage_json.py
"""

import csv
import json
import os
import re

import modal

from _app import app, data_volume


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["numpy<2"])
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_app.py"),
        "/root/_app.py",
    )
)


def _basename(name: str) -> str:
    for sep in ("\\", "/"):
        if sep in name:
            name = name.rsplit(sep, 1)[-1]
    return name


def _collect_eeg1_filenames(bids_events_root: str) -> set[str]:
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
                sid = _basename(sid)
                if sid:
                    stims.add(sid)
    return stims


def _collect_eeg2_filenames(eeg2_metadata_path: str) -> set[str]:
    import numpy as np
    if not os.path.exists(eeg2_metadata_path):
        return set()
    md = np.load(eeg2_metadata_path, allow_pickle=True).item()
    names = list(md.get("train_img_files", [])) + list(md.get("test_img_files", []))
    return set(_basename(n) for n in names)


@app.function(
    image=image,
    volumes={"/project": data_volume},
    timeout=600,
    cpu=2,
)
def write_coverage_json():
    with open("/project/data/things_catalog.json") as f:
        catalog = json.load(f)
    id_to_filename = catalog["image_id_to_filename"]      # {"000000001": "aardvark_01b.jpg", ...}
    filename_to_id = {v: k for k, v in id_to_filename.items()}

    eeg1_filenames = _collect_eeg1_filenames("/project/data/raw/things-eeg1/bids_events")
    eeg2_filenames = _collect_eeg2_filenames("/project/data/raw/things-eeg2/image_metadata.npy")

    def to_ids(filenames: set[str]) -> set[str]:
        return {filename_to_id[fn] for fn in filenames if fn in filename_to_id}

    ids_eeg1   = to_ids(eeg1_filenames)
    ids_eeg2   = to_ids(eeg2_filenames)
    ids_union  = ids_eeg1 | ids_eeg2
    ids_inter  = ids_eeg1 & ids_eeg2
    ids_uncov  = set(id_to_filename.keys()) - ids_union

    payload = {
        "schema_version": 1,
        "catalog_n_images": len(id_to_filename),
        "image_ids_eeg1":   sorted(ids_eeg1),
        "image_ids_eeg2":   sorted(ids_eeg2),
        "image_ids_union":  sorted(ids_union),
        "image_ids_intersection": sorted(ids_inter),
        "image_ids_uncovered":    sorted(ids_uncov),
        "n_eeg1":  len(ids_eeg1),
        "n_eeg2":  len(ids_eeg2),
        "n_union": len(ids_union),
        "n_intersection": len(ids_inter),
        "n_uncovered":    len(ids_uncov),
    }

    out_path = "/project/data/eeg_coverage.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    data_volume.commit()

    print(f"Wrote {out_path}")
    print(f"  catalog images:           {payload['catalog_n_images']}")
    print(f"  EEG1 covered:             {payload['n_eeg1']}")
    print(f"  EEG2 covered:             {payload['n_eeg2']}")
    print(f"  Union (any EEG):          {payload['n_union']}")
    print(f"  Intersection (both EEG):  {payload['n_intersection']}")
    print(f"  Uncovered (no EEG):       {payload['n_uncovered']}")
    print(f"  Sample image_ids_union[:5]: {payload['image_ids_union'][:5]}")
    return {k: v for k, v in payload.items() if not isinstance(v, list)}


@app.local_entrypoint()
def main():
    result = write_coverage_json.remote()
    print("\nFinal:")
    for k, v in result.items():
        print(f"  {k:<20s} {v}")
