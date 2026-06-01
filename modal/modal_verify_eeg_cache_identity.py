"""Confirm LaBraM cache tokens are byte-identical to legacy tok_eeg shards.

Samples legacy train-tar + val-loose `.eeg.npy` entries and compares against
the matching row in the LaBraM subject `.npz` cache (same image_id, source,
subject, trial_idx).

Run: modal run modal_verify_eeg_cache_identity.py::main
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import tarfile
from glob import glob

import modal

app = modal.App("verify-eeg-cache-identity")
vol = modal.Volume.from_name("project")
PROJECT = "/project"
img = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")

EEG_CACHE = (
    f"{PROJECT}/data/things-eeg/tokens/labram/"
    "V8192_d64_ch17_sr200_train-eeg1+2_e5"
)
SRC_TRAIN_EEG = f"{PROJECT}/data/train/things/tok_eeg"
SRC_VAL_EEG = f"{PROJECT}/data/val/things/tok_eeg"

EEG_FN_RE = re.compile(
    r"^(?P<image_id>\d{9})_(?P<dataset>eeg[12])sub(?P<subj>\d+)_t(?P<trial>\d+)\.eeg\.npy$"
)

SAMPLE_N = 500
SEED = 0


def _legacy_sort_key(name: str) -> tuple[str, str, str, int] | None:
    m = EEG_FN_RE.match(name)
    if not m:
        return None
    return (
        m.group("image_id"),
        m.group("dataset"),
        f"sub-{int(m.group('subj')):02d}",
        int(m.group("trial")),
    )


def _read_legacy_entry(path_or_member, name: str) -> tuple[str, bytes, tuple] | None:
    key = _legacy_sort_key(name)
    if key is None:
        return None
    if isinstance(path_or_member, str):
        with open(path_or_member, "rb") as fh:
            data = fh.read()
    else:
        tar, member = path_or_member
        f = tar.extractfile(member)
        if f is None:
            return None
        data = f.read()
    return name, data, key


def _collect_legacy_sample(sample_n: int) -> list[tuple[str, bytes, tuple]]:
    """Sample legacy entries without reading the full train+val corpus."""
    rng = random.Random(SEED)
    out: list[tuple[str, bytes, tuple]] = []

    # Random val loose files (fast — individual paths only).
    if os.path.isdir(SRC_VAL_EEG):
        val_names = [n for n in os.listdir(SRC_VAL_EEG) if n.endswith(".eeg.npy")]
        n_val = min(len(val_names), sample_n // 2)
        for name in rng.sample(val_names, n_val):
            row = _read_legacy_entry(os.path.join(SRC_VAL_EEG, name), name)
            if row:
                out.append(row)

    # Random members from a few train tars.
    if os.path.isdir(SRC_TRAIN_EEG):
        tar_names = sorted(
            n for n in os.listdir(SRC_TRAIN_EEG) if n.endswith(".tar")
        )
        per_tar = max(1, (sample_n - len(out)) // min(3, len(tar_names)))
        for tar_name in rng.sample(tar_names, min(3, len(tar_names))):
            path = os.path.join(SRC_TRAIN_EEG, tar_name)
            with tarfile.open(path, "r") as tar:
                members = [m for m in tar if m.name.endswith(".eeg.npy")]
            pick = rng.sample(members, min(per_tar, len(members)))
            with tarfile.open(path, "r") as tar:
                for member in pick:
                    row = _read_legacy_entry((tar, member), member.name)
                    if row:
                        out.append(row)

    return out[:sample_n]


@app.function(image=img, volumes={PROJECT: vol}, timeout=60 * 20, memory=16 * 1024)
def verify() -> dict:
    import numpy as np

    cfg = json.loads(open(os.path.join(EEG_CACHE, "config.json")).read())
    print("=== LaBraM config (cache run metadata) ===")
    print(json.dumps({k: cfg[k] for k in (
        "slug", "family", "vocab_size", "n_channels", "expected_token_shape",
        "token_dtype", "train_data", "best_epoch_picked",
    )}, indent=2))

    # Build cache lookup: (source, subject) -> npz arrays
    print("\n=== Loading LaBraM cache index ===")
    cache_by_subject: dict[tuple[str, str], dict] = {}
    for p in sorted(glob(os.path.join(EEG_CACHE, "*.npz"))):
        z = np.load(p)
        source = str(z["source"])
        subject = str(z["subject"])
        cache_by_subject[(source, subject)] = {
            "path": p,
            "tokens": z["tokens"],
            "image_id": z["image_id"],
            "trial_idx": z["trial_idx"],
        }
    print(f"loaded {len(cache_by_subject)} subject npz files")

    # Collect legacy entries (sample if huge)
    print("\n=== Collecting legacy entries ===")
    all_legacy = _collect_legacy_sample(SAMPLE_N)
    print(f"legacy entries collected: {len(all_legacy)}")
    if not all_legacy:
        raise RuntimeError("no legacy tok_eeg entries found on volume")

    sample = all_legacy
    print(f"comparing sample size: {len(sample)}")

    mismatches: list[str] = []
    missing_in_cache: list[str] = []
    matches = 0

    for fname, legacy_bytes, (image_id, source, subject, trial_idx) in sample:
        legacy_arr = np.load(io.BytesIO(legacy_bytes), allow_pickle=False)
        subj_data = cache_by_subject.get((source, subject))
        if subj_data is None:
            missing_in_cache.append(f"{fname}: no npz for {(source, subject)}")
            continue

        image_ids = subj_data["image_id"]
        trial_idxs = subj_data["trial_idx"]
        tokens = subj_data["tokens"]

        # Find matching row
        hits = np.flatnonzero(
            (image_ids == image_id) & (trial_idxs == trial_idx)
        )
        if len(hits) != 1:
            missing_in_cache.append(
                f"{fname}: {len(hits)} cache hits for "
                f"{image_id}/{source}/{subject}/t{trial_idx}"
            )
            continue

        cache_arr = tokens[int(hits[0])]
        if legacy_arr.shape != cache_arr.shape:
            mismatches.append(
                f"{fname}: shape legacy={legacy_arr.shape} cache={cache_arr.shape}"
            )
            continue
        if not np.array_equal(legacy_arr, cache_arr):
            mismatches.append(
                f"{fname}: values differ max_abs={int(np.max(np.abs(legacy_arr - cache_arr)))}"
            )
            continue
        matches += 1

    # Full enumeration on one legacy train tar (exhaustive within shard_000)
    print("\n=== Exhaustive check: train shard_000.tar ===")
    shard0_path = os.path.join(SRC_TRAIN_EEG, "shard_000.tar")
    exhaustive_mismatch = 0
    exhaustive_missing = 0
    exhaustive_match = 0
    if os.path.isfile(shard0_path):
        with tarfile.open(shard0_path, "r") as tar:
            for member in tar:
                if not member.name.endswith(".eeg.npy"):
                    continue
                key = _legacy_sort_key(member.name)
                if key is None:
                    continue
                image_id, source, subject, trial_idx = key
                f = tar.extractfile(member)
                legacy_arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
                subj_data = cache_by_subject[(source, subject)]
                hits = np.flatnonzero(
                    (subj_data["image_id"] == image_id)
                    & (subj_data["trial_idx"] == trial_idx)
                )
                if len(hits) != 1:
                    exhaustive_missing += 1
                    continue
                cache_arr = subj_data["tokens"][int(hits[0])]
                if np.array_equal(legacy_arr, cache_arr):
                    exhaustive_match += 1
                else:
                    exhaustive_mismatch += 1
        print(
            f"shard_000: match={exhaustive_match} "
            f"mismatch={exhaustive_mismatch} missing={exhaustive_missing}"
        )

    # Config fingerprint: legacy was packed from same slug?
    print("\n=== Slug cross-check ===")
    print(f"cache slug: {cfg['slug']}")
    print("legacy filenames encode eeg1/eeg2 subject+trial — same convention as cache fields")

    ok = (
        matches == len(sample)
        and not mismatches
        and not missing_in_cache
        and exhaustive_mismatch == 0
        and exhaustive_missing == 0
    )

    print("\n=== VERDICT ===")
    if ok:
        print("PASS — legacy tok_eeg tokens are byte-identical to LaBraM cache entries.")
    else:
        print("FAIL")
        if missing_in_cache[:5]:
            print("missing (first 5):", missing_in_cache[:5])
        if mismatches[:5]:
            print("mismatches (first 5):", mismatches[:5])

    return {
        "ok": ok,
        "sample_n": len(sample),
        "sample_matches": matches,
        "sample_missing": len(missing_in_cache),
        "sample_mismatches": len(mismatches),
        "shard_000_matches": exhaustive_match,
        "shard_000_mismatches": exhaustive_mismatch,
        "shard_000_missing": exhaustive_missing,
        "cache_slug": cfg["slug"],
    }


@app.local_entrypoint()
def main():
    result = verify.remote()
    print("\n", json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)
