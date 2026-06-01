"""Pure logic for packing per-trial MEG tokens into WebDataset-style tar shards.

What this module owns:
  - The MEG sample filename convention (one tar entry per trial).
  - The mapping (image_id → which RGB shard) → which MEG shard it goes in.
  - Writing a tar from a list of (filename, ndarray) entries.

What this module does NOT touch:
  - MNE / .fif loading.
  - BrainOmni model code or checkpoints.
  - Modal (this file imports nothing from the modal SDK).

The Modal entrypoint in modal_meg_pack_shards.py wires this against the
real token cache + manifests on the project Volume.

Filename convention:
  <image_id>_<subject>_t<trial_idx>.meg.npy
  e.g. 000000001_P1_t0.meg.npy

  - image_id:  9-digit zero-padded universal THINGS catalog id
  - subject:   P1..P4
  - trial_idx: 0-indexed occurrence within (subject, image_id).
               Almost always 0 (one trial per subject for `exp` images);
               0..11 for the ~200 `test` repeat images.
  - .meg.npy:  binary numpy array, shape (16, 8, 4), dtype int16
               (BrainOmni 3b RVQ codes; codebook_size=512 fits in 9 bits).

The MEG shard layout MIRRORS the RGB split (uses 4M's `tok_<modality>/`
convention, matching the sibling `tok_eeg/` directory already on the volume):
  /project/data/train/things/tok_meg/shard_NNN.tar   for image_ids in train manifest
  /project/data/val/things/tok_meg/shard_NNN.tar     for image_ids in val manifest
"""

from __future__ import annotations

import io
import os
import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np


# ---------- filename convention ------------------------------------------

# Captures (image_id, subject, trial_idx).
MEG_FILENAME_RE = re.compile(
    r"^(?P<image_id>\d{9})_(?P<subject>P\d+)_t(?P<trial_idx>\d+)\.meg\.npy$"
)

SUBJECT_RE = re.compile(r"^P\d+$")

# Expected per-trial token shape after BrainOmni 3b tokenize().
EXPECTED_TOKEN_SHAPE: tuple[int, int, int] = (16, 8, 4)
EXPECTED_TOKEN_DTYPE = np.int16


def meg_filename(image_id: str, subject: str, trial_idx: int) -> str:
    """Canonical sample filename for one MEG trial inside a shard tar."""
    if not (isinstance(image_id, str) and len(image_id) == 9 and image_id.isdigit()):
        raise ValueError(f"image_id must be 9-digit zero-padded str; got {image_id!r}")
    if not SUBJECT_RE.match(subject):
        raise ValueError(f"subject must match P\\d+; got {subject!r}")
    if not (isinstance(trial_idx, int) and trial_idx >= 0):
        raise ValueError(f"trial_idx must be non-negative int; got {trial_idx!r}")
    return f"{image_id}_{subject}_t{trial_idx}.meg.npy"


def parse_meg_filename(name: str) -> tuple[str, str, int] | None:
    """Inverse of meg_filename. Returns (image_id, subject, trial_idx) or None."""
    m = MEG_FILENAME_RE.match(name)
    if not m:
        return None
    return m.group("image_id"), m.group("subject"), int(m.group("trial_idx"))


# ---------- entry type ----------------------------------------------------

@dataclass(frozen=True)
class MegEntry:
    """One MEG trial to be packed into a shard tar.

    tokens.shape must equal EXPECTED_TOKEN_SHAPE. dtype is coerced to
    EXPECTED_TOKEN_DTYPE on write (int16 — codebook_size=512 fits).
    """

    image_id: str
    subject: str
    trial_idx: int
    tokens: np.ndarray

    def __post_init__(self):
        if self.tokens.shape != EXPECTED_TOKEN_SHAPE:
            raise ValueError(
                f"tokens shape {self.tokens.shape} != expected {EXPECTED_TOKEN_SHAPE} "
                f"for {self.image_id}/{self.subject}/t{self.trial_idx}"
            )

    @property
    def filename(self) -> str:
        return meg_filename(self.image_id, self.subject, self.trial_idx)


# ---------- planning ------------------------------------------------------

def build_image_id_to_shard(split_manifest: dict) -> dict[str, str]:
    """Invert per-split manifest: {image_id: shard_id} for fast lookup.

    Raises if an image_id appears in more than one shard of the same split.
    """
    out: dict[str, str] = {}
    for shard_id, info in split_manifest["shards"].items():
        for image_id in info["image_ids"]:
            if image_id in out:
                raise ValueError(
                    f"image_id {image_id} in multiple shards: "
                    f"{out[image_id]} and {shard_id}"
                )
            out[image_id] = shard_id
    return out


def group_entries_by_shard(
    entries: Iterable[MegEntry],
    image_id_to_shard: dict[str, str],
) -> dict[str, list[MegEntry]]:
    """Group MegEntries by shard_id for the given split.

    Entries whose image_id is NOT in image_id_to_shard are SKIPPED — those
    images belong to the other split (or aren't in the catalog at all).
    Duplicate (image_id, subject, trial_idx) within a shard raises.
    """
    by_shard: dict[str, list[MegEntry]] = defaultdict(list)
    seen_per_shard: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        shard_id = image_id_to_shard.get(e.image_id)
        if shard_id is None:
            continue
        if e.filename in seen_per_shard[shard_id]:
            raise ValueError(
                f"duplicate MEG entry {e.filename} within {shard_id}"
            )
        seen_per_shard[shard_id].add(e.filename)
        by_shard[shard_id].append(e)
    return dict(by_shard)


# ---------- tar writing ---------------------------------------------------

def _serialize_tokens(arr: np.ndarray) -> bytes:
    """Convert (16,8,4) tokens to a .npy byte blob (int16, contiguous)."""
    if arr.dtype != EXPECTED_TOKEN_DTYPE:
        if arr.min() < np.iinfo(EXPECTED_TOKEN_DTYPE).min or arr.max() > np.iinfo(EXPECTED_TOKEN_DTYPE).max:
            raise ValueError(
                f"token values out of int16 range: min={arr.min()}, max={arr.max()}"
            )
        arr = arr.astype(EXPECTED_TOKEN_DTYPE)
    arr = np.ascontiguousarray(arr)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def write_meg_shard_tar(out_path: str, entries: list[MegEntry]) -> None:
    """Write a tar of `<filename>.meg.npy` entries, sorted by filename.

    Writes to `<out_path>.tmp` then atomically renames into place — same
    pattern as `things_manifest.write_shard_from_locations`.
    """
    ordered = sorted(entries, key=lambda e: e.filename)
    tmp_path = out_path + ".tmp"
    with tarfile.open(tmp_path, "w") as tar:
        for e in ordered:
            data = _serialize_tokens(e.tokens)
            info = tarfile.TarInfo(name=e.filename)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    os.replace(tmp_path, out_path)


def read_meg_shard_tar(tar_path: str) -> dict[str, np.ndarray]:
    """Inverse of write_meg_shard_tar — for verification + tests.

    Returns {filename: tokens_ndarray}. Filenames are the same strings
    produced by `meg_filename`.
    """
    out: dict[str, np.ndarray] = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.name.endswith(".meg.npy"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            out[member.name] = np.load(io.BytesIO(f.read()), allow_pickle=False)
    return out
