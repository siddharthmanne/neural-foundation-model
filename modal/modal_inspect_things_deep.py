"""Deep sample inspection for THINGS modalities."""

from __future__ import annotations

import io
import os
import re
import tarfile
from collections import Counter, defaultdict

import modal

app = modal.App("inspect-things-deep")
project_volume = modal.Volume.from_name("project")
PROJECT = "/project"
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")


def _load_npy(path: str):
    import numpy as np

    return np.load(path, allow_pickle=False)


@app.function(image=image, volumes={PROJECT: project_volume}, timeout=60 * 15)
def inspect():
    import numpy as np

    # --- EEG filename patterns ---
    eeg_re = re.compile(
        r"^(?P<image_id>\d{9})_(?P<dataset>eeg[12])sub(?P<subj>\d+)_t(?P<trial>\d+)\.eeg\.npy$"
    )
    meg_re = re.compile(
        r"^(?P<image_id>\d{9})_(?P<subj>P\d+)_t(?P<trial>\d+)\.meg\.npy$"
    )

    def inspect_eeg_source(label: str, path: str):
        print(f"\n=== EEG {label}: {path} ===")
        if os.path.isdir(path):
            all_files = os.listdir(path)
            print(f"  n_loose_total={len(all_files)}")
            sample_names = sorted(all_files)[:200]
        elif os.path.isfile(path):
            with tarfile.open(path, "r") as tar:
                sample_names = [m.name for m in tar][:200]
                all_files = None
        else:
            print("  missing")
            return
        shapes = Counter()
        per_image: dict[str, int] = Counter()
        for name in sample_names:
            if not name.endswith(".eeg.npy"):
                continue
            m = eeg_re.match(name)
            if not m:
                print(f"  bad name: {name}")
                continue
            per_image[m.group("image_id")] += 1
            if os.path.isdir(path):
                arr = _load_npy(os.path.join(path, name))
            else:
                with tarfile.open(path, "r") as tar:
                    f = tar.extractfile(name)
                    arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
            shapes[str(arr.shape) + " " + str(arr.dtype)] += 1
        print(f"  shapes: {dict(shapes)}")
        if per_image:
            counts = Counter(per_image.values())
            print(f"  entries per image distribution (top): {counts.most_common(5)}")
            print(f"  sample image_ids: {sorted(per_image.keys())[:3]}")

    inspect_eeg_source(
        "train shard_000",
        f"{PROJECT}/data/train/things/tok_eeg/shard_000.tar",
    )
    inspect_eeg_source(
        "val loose dir",
        f"{PROJECT}/data/val/things/tok_eeg",
    )

    # --- MEG: trials per image in one shard ---
    print("\n=== MEG train shard_000 trial stacking ===")
    meg_path = f"{PROJECT}/data/train/things/tok_meg/shard_000.tar"
    per_image_meg: dict[str, list] = defaultdict(list)
    with tarfile.open(meg_path, "r") as tar:
        for member in tar:
            m = meg_re.match(member.name)
            if not m:
                continue
            per_image_meg[m.group("image_id")].append(member.name)
    trial_counts = Counter(len(v) for v in per_image_meg.values())
    print(f"  unique images in shard_000: {len(per_image_meg)}")
    print(f"  trials-per-image distribution: {dict(trial_counts)}")

    # --- tok_rgb@224: check shape and whether catalog slot ---
    print("\n=== tok_rgb@224 shard-00000 ===")
    rgb_path = f"{PROJECT}/data/train/things/tok_rgb@224/shard-00000.tar"
    with tarfile.open(rgb_path, "r") as tar:
        names = [m.name for m in tar][:3]
    for name in names:
        with tarfile.open(rgb_path, "r") as tar:
            f = tar.extractfile(name)
            arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
        print(f"  {name}: shape={arr.shape} dtype={arr.dtype}")

    # --- things_split vs legacy manifest overlap ---
    import json

    split = json.loads(open(f"{PROJECT}/data/things_split.json").read())
    train_manifest = json.loads(open(f"{PROJECT}/data/train/things_manifest.json").read())
    val_manifest = json.loads(open(f"{PROJECT}/data/val/things_manifest.json").read())
    legacy_train = set()
    legacy_val = set()
    for s in train_manifest["shards"].values():
        legacy_train.update(s["image_ids"])
    for s in val_manifest["shards"].values():
        legacy_val.update(s["image_ids"])
    new_train = set(split["train_image_ids"])
    new_val = set(split["val_image_ids"])
    print("\n=== Split comparison ===")
    print(f"  legacy train={len(legacy_train)} val={len(legacy_val)}")
    print(f"  things_split train={len(new_train)} val={len(new_val)}")
    print(f"  legacy val - new val: {len(legacy_val - new_val)}")
    print(f"  new val - legacy val: {len(new_val - legacy_val)}")
    print(f"  legacy train ∩ new val: {len(legacy_train & new_val)}")

    # crop_settings?
    for p in [
        f"{PROJECT}/data/train/things/crop_settings",
        f"{PROJECT}/data/train/cc12m/crop_settings",
    ]:
        print(f"\n  exists {p}: {os.path.isdir(p)}")


@app.local_entrypoint()
def main():
    inspect.remote()
