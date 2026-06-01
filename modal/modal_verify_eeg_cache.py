"""Verify LaBraM EEG cache has everything needed for 4M repack.

Run: modal run modal_verify_eeg_cache.py::main
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from glob import glob

import modal

app = modal.App("verify-eeg-cache")
vol = modal.Volume.from_name("project")
PROJECT = "/project"
img = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")

EEG_CACHE = (
    f"{PROJECT}/data/things-eeg/tokens/labram/"
    "V8192_d64_ch17_sr200_train-eeg1+2_e5"
)
EEG_COVERAGE = f"{PROJECT}/data/eeg_coverage.json"
THINGS_SPLIT = f"{PROJECT}/data/things_split.json"
CATALOG_PATH = f"{PROJECT}/data/things_catalog.json"


@app.function(image=img, volumes={PROJECT: vol}, timeout=60 * 10, memory=16 * 1024)
def verify() -> dict:
    import numpy as np

    print("=== LaBraM EEG cache verification ===\n")

    cfg = json.loads(open(os.path.join(EEG_CACHE, "config.json")).read())
    print(f"config slug: {cfg['slug']}")
    print(f"expected_token_shape: {cfg['expected_token_shape']}")
    print(f"token_dtype: {cfg['token_dtype']}")

    npz_paths = sorted(glob(os.path.join(EEG_CACHE, "*.npz")))
    print(f"\nnpz files: {len(npz_paths)}")

    # Aggregate all trials from cache
    trials_by_image: dict[str, list[tuple[str, int, np.ndarray]]] = defaultdict(list)
    total_trials = 0
    bad_shapes = 0
    bad_dtypes = 0

    for p in npz_paths:
        z = np.load(p)
        required = {"tokens", "image_id", "trial_idx", "source", "subject"}
        missing = required - set(z.files)
        if missing:
            raise RuntimeError(f"{p} missing keys {missing}")

        tokens = z["tokens"]
        image_ids = z["image_id"]
        trial_idx = z["trial_idx"]
        source = str(z["source"])
        subject = str(z["subject"])

        if tokens.dtype != np.int16:
            bad_dtypes += 1
        n = tokens.shape[0]
        total_trials += n
        for i in range(n):
            tok = tokens[i]
            if tok.shape != (17,):
                bad_shapes += 1
            iid = str(image_ids[i])
            tidx = int(trial_idx[i])
            trials_by_image[iid].append((source, subject, tidx, tok))

        print(f"  {os.path.basename(p)}: {n} trials  source={source}  subject={subject}")

    cache_image_ids = set(trials_by_image.keys())
    print(f"\ntotal trials in cache: {total_trials}")
    print(f"unique image_ids in cache: {len(cache_image_ids)}")
    print(f"bad token shapes: {bad_shapes}, bad dtypes: {bad_dtypes}")

    # Trial count distribution
    trial_counts = Counter(len(v) for v in trials_by_image.values())
    print(f"trials-per-image distribution (top 8): {trial_counts.most_common(8)}")

    # Load coverage + split + catalog
    eeg_cov = json.loads(open(EEG_COVERAGE).read())
    eeg_union = set(str(x) for x in eeg_cov["image_ids_union"])
    eeg1 = set(str(x) for x in eeg_cov["image_ids_eeg1"])
    eeg2 = set(str(x) for x in eeg_cov["image_ids_eeg2"])
    eeg_intersection = set(str(x) for x in eeg_cov.get("image_ids_intersection", eeg1 & eeg2))

    split = json.loads(open(THINGS_SPLIT).read())
    train = set(split["train_image_ids"])
    val = set(split["val_image_ids"])
    catalog = set(json.loads(open(CATALOG_PATH).read())["image_id_to_filename"])

    print("\n=== Coverage comparison ===")
    print(f"eeg_coverage union:        {len(eeg_union)}")
    print(f"eeg_coverage eeg1:         {len(eeg1)}")
    print(f"eeg_coverage eeg2:         {len(eeg2)}")
    print(f"eeg_coverage intersection: {len(eeg_intersection)}")
    print(f"catalog:                   {len(catalog)}")
    print(f"things_split train:        {len(train)}")
    print(f"things_split val:          {len(val)}")

    union_missing_from_cache = sorted(eeg_union - cache_image_ids)
    cache_extra_not_in_union = sorted(cache_image_ids - eeg_union)
    cache_not_in_catalog = sorted(cache_image_ids - catalog)

    print(f"\ncache ⊇ eeg_union?  missing={len(union_missing_from_cache)}")
    if union_missing_from_cache[:5]:
        print(f"  first missing: {union_missing_from_cache[:5]}")
    print(f"cache extras not in union: {len(cache_extra_not_in_union)}")
    if cache_extra_not_in_union[:5]:
        print(f"  first extra: {cache_extra_not_in_union[:5]}")
    print(f"cache ids not in catalog: {len(cache_not_in_catalog)}")

    # For repack we need: every image in catalog gets either real EEG or sentinel.
    # Real EEG needed for all image_ids in eeg_union (per our mask logic).
    # Confirm every eeg_union id has trials in cache.
    can_cover_union = len(union_missing_from_cache) == 0

    # Val images with EEG in split - any val id in eeg_union should be in cache
    val_with_eeg = val & eeg_union
    val_eeg_missing = sorted(val_with_eeg - cache_image_ids)
    print(f"\nval ∩ eeg_union: {len(val_with_eeg)}")
    print(f"val EEG missing from cache: {len(val_eeg_missing)}")

    # Spot-check stacking: compare trial count for a few ids
    sample_ids = sorted(cache_image_ids)[:3] + sorted(val_with_eeg)[:3]
    print("\n=== Spot-check trial counts ===")
    for iid in sample_ids:
        entries = trials_by_image[iid]
        sources = Counter(e[0] for e in entries)
        print(f"  {iid}: {len(entries)} trials  by_source={dict(sources)}")

    # Stacked shape sanity
    sample_iid = next(iter(cache_image_ids))
    stacked = np.stack([t[3] for t in sorted(trials_by_image[sample_iid], key=lambda x: (x[0], x[1], x[2]))], axis=0)
    print(f"\nstacked shape for {sample_iid}: {stacked.shape} dtype={stacked.dtype}")

    ok = (
        bad_shapes == 0
        and bad_dtypes == 0
        and can_cover_union
        and len(cache_not_in_catalog) == 0
        and len(val_eeg_missing) == 0
        and cfg["expected_token_shape"] == [17]
    )

    print("\n=== VERDICT ===")
    if ok:
        print("PASS — LaBraM cache has all data needed for repack.")
    else:
        print("FAIL — see gaps above.")

    return {
        "ok": ok,
        "n_npz": len(npz_paths),
        "n_trials": total_trials,
        "n_cache_images": len(cache_image_ids),
        "n_eeg_union": len(eeg_union),
        "union_missing_from_cache": len(union_missing_from_cache),
        "val_eeg_missing": len(val_eeg_missing),
        "expected_token_shape": cfg["expected_token_shape"],
    }


@app.local_entrypoint()
def main():
    result = verify.remote()
    print("\n", json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)
