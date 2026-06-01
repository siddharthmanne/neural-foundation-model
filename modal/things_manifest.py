"""Pure logic for the THINGS image catalog + train/val repack on the project Volume.

Why this module is import-clean (no `modal`, no `torch`):
  The Modal entrypoint in `modal_things_repack.py` runs in a container with
  the project Volume mounted. Everything domain-specific (parsing tarballs,
  building manifests, computing the split, planning the repack) is here
  instead so it can be unit-tested on a laptop against synthetic tar
  fixtures (`test_things_manifest.py`). Modal is only a deployment target,
  not a dependency of the logic.

System-level note on the split policy:
  Splits are at the image level (each THINGS concept appears in both train
  and val, different exemplars held out) because the project target is
  neural-to-image decoding with 4M. Standard brain-decoding evals
  (MindEye, Takagi & Nishimoto, BrainDiffuser, the THINGS-EEG/MEG decoding
  line) hold out images, not concepts. Concept-level holdout is a
  secondary "zero-shot decoding" eval, not the primary one. If we ever
  want that, we can derive it from the same catalog without rerunning the
  repack — see `modal_things_repack.py` docstring.

THINGS shard layout (verified 2026-05-23 against /project/data/train/things/rgb):
  Each tar contains paired entries `<image_id>.jpg` + `<image_id>.txt`.
  - `image_id` is a 9-digit zero-padded alphabetical rank of the THINGS
    filename (e.g. `aardvark_01b.jpg` → `000000001`).
  - `<image_id>.txt` body is the canonical THINGS filename.
  - 26,107 unique images total across 27 shards × 1000 (last partial).
"""

from __future__ import annotations

import io
import os
import random
import re
import tarfile
from dataclasses import dataclass
from typing import Any, Iterable


# ---------- constants -----------------------------------------------------

SHARD_NAME_RE = re.compile(r"^shard_(\d{3})\.tar$")
IMAGE_ID_TXT_RE = re.compile(r"^(\d{9})\.txt$")
IMAGE_ID_JPG_RE = re.compile(r"^(\d{9})\.jpg$")
IMAGES_PER_SHARD = 1000


# ---------- catalog construction ------------------------------------------

def extract_shard_contents(tar_path: str) -> dict[str, str]:
    """Read a single THINGS shard tar; return ``{image_id: filename}``.

    Reads only the `.txt` entries (filename strings); ignores the `.jpg`
    bytes. Cheap enough to call on all 27 shards back-to-back when building
    the catalog.
    """
    out: dict[str, str] = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            m = IMAGE_ID_TXT_RE.match(member.name)
            if not m:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            out[m.group(1)] = f.read().decode("utf-8").strip()
    return out


def build_catalog(image_id_to_filename: dict[str, str]) -> dict:
    """Return the global THINGS catalog JSON payload.

    The catalog is split-invariant — any future modality (MEG, EEG, fMRI)
    consults this single file to resolve image_id ↔ filename, regardless
    of which split each image lives in.
    """
    ordered = dict(sorted(image_id_to_filename.items()))
    return {
        "version": "1",
        "n_images": len(ordered),
        "id_format": (
            "9-digit zero-padded alphabetical rank of the THINGS image filename"
        ),
        "image_id_to_filename": ordered,
    }


# ---------- split logic ----------------------------------------------------

def image_level_split(
    image_ids: Iterable[str],
    val_frac: float = 0.15,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Deterministic image-level shuffle into (train, val).

    Each THINGS concept's ~14 images get distributed across both splits.
    This matches the brain-decoding-eval convention (image-level holdout):
    the model sees the concept at train time and must use the held-out
    trial's neural signal to pick the right exemplar at inference.

    Determinism: sorts ids first so callers can pass any iterable and still
    get the same split for a given seed.
    """
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1); got {val_frac}")
    ids = sorted(image_ids)
    n_val = round(len(ids) * val_frac)
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    val_set = set(shuffled[:n_val])
    train_ids = [i for i in ids if i not in val_set]
    val_ids = [i for i in ids if i in val_set]
    return train_ids, val_ids


ID_FORMAT = (
    "9-digit zero-padded alphabetical rank of the THINGS image filename"
)


def load_coverage_json(payload: dict[str, Any] | list[Any]) -> set[str]:
    """Extract catalog image IDs from a coverage JSON payload.

    Supports:
      - ``image_ids`` (meg_coverage.json)
      - ``image_ids_intersection`` (legacy eeg field — EEG1 ∩ EEG2)
      - ``image_ids_union`` (eeg_coverage.json — EEG1 ∪ EEG2)
      - a bare list at the top level
    """
    if isinstance(payload, list):
        return set(str(x) for x in payload)
    if "image_ids" in payload:
        return set(str(x) for x in payload["image_ids"])
    if "image_ids_union" in payload:
        return set(str(x) for x in payload["image_ids_union"])
    if "image_ids_intersection" in payload:
        return set(str(x) for x in payload["image_ids_intersection"])
    raise ValueError(
        "coverage JSON must contain 'image_ids', 'image_ids_union', or "
        f"'image_ids_intersection' (got keys: {sorted(payload)})"
    )


def parse_eeg_coverage(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse eeg_coverage.json into per-dataset sets and counts."""
    eeg1 = set(str(x) for x in payload["image_ids_eeg1"])
    eeg2 = set(str(x) for x in payload["image_ids_eeg2"])
    intersection = set(str(x) for x in payload.get("image_ids_intersection", eeg1 & eeg2))
    union = set(str(x) for x in payload.get("image_ids_union", eeg1 | eeg2))
    return {
        "n_eeg1": len(eeg1),
        "n_eeg2": len(eeg2),
        "n_eeg_intersection": len(intersection),
        "n_eeg_union": len(union),
        "eeg1_image_ids": eeg1,
        "eeg2_image_ids": eeg2,
        "intersection_image_ids": intersection,
        "union_image_ids": union,
    }


def build_meg_coverage_payload(
    trigger_to_image_id: dict[str, str],
    *,
    source: str = "things-meg/labels/meg_trigger_to_image_id.json",
) -> dict[str, Any]:
    """Build meg_coverage.json from unique bridge ``trigger_to_image_id`` values."""
    image_ids = sorted(set(trigger_to_image_id.values()))
    return {
        "version": "1",
        "modality": "meg",
        "n_image_ids": len(image_ids),
        "id_format": ID_FORMAT,
        "source": source,
        "image_ids": image_ids,
    }


def neural_intersection_split(
    catalog_image_ids: Iterable[str],
    meg_image_ids: Iterable[str],
    eeg_val_pool_ids: Iterable[str],
    val_frac: float = 0.20,
    seed: int = 0,
    *,
    eeg_stats: dict[str, int] | None = None,
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """Split catalog into train/val with val drawn from MEG ∩ EEG val pool.

    Val IDs are a deterministic ``val_frac`` sample of
    ``catalog ∩ meg ∩ eeg_val_pool``. Train gets all remaining catalog IDs.

    Returns:
        (train_ids, val_ids, val_pool_ids, stats)
    """
    catalog = sorted(set(catalog_image_ids))
    meg = set(meg_image_ids)
    eeg_pool = set(eeg_val_pool_ids)
    val_pool = sorted(set(catalog) & meg & eeg_pool)
    if not val_pool:
        raise ValueError("val pool is empty — check meg/eeg coverage inputs")
    val_ids, _ = _val_sample_from_pool(val_pool, val_frac=val_frac, seed=seed)
    val_set = set(val_ids)
    train_ids = [i for i in catalog if i not in val_set]
    stats: dict[str, int] = {
        "n_catalog": len(catalog),
        "n_meg": len(meg),
        "n_intersection": len(val_pool),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
    }
    if eeg_stats:
        stats.update(eeg_stats)
    return train_ids, val_ids, val_pool, stats


def _val_sample_from_pool(
    val_pool_ids: list[str],
    val_frac: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Return (val_ids, unused_train_from_pool) from a shuffled val_frac sample."""
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1); got {val_frac}")
    ids = sorted(val_pool_ids)
    n_val = round(len(ids) * val_frac)
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    val_set = set(shuffled[:n_val])
    val_ids = [i for i in ids if i in val_set]
    train_from_pool = [i for i in ids if i not in val_set]
    return val_ids, train_from_pool


def build_proposed_shard_layout(
    train_image_ids: list[str],
    val_image_ids: list[str],
    images_per_shard: int = IMAGES_PER_SHARD,
) -> dict[str, Any]:
    """Preview shard assignment without writing tars or manifests."""
    train_shards = pack_into_shards(train_image_ids, images_per_shard)
    val_shards = pack_into_shards(val_image_ids, images_per_shard)

    def _shard_map(shards: list[list[str]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for n, ids in enumerate(shards):
            out[f"shard_{n:03d}"] = {"n_images": len(ids), "image_ids": list(ids)}
        return out

    return {
        "images_per_shard": images_per_shard,
        "train": {
            "n_shards": len(train_shards),
            "n_images": len(train_image_ids),
            "shard_to_image_ids": _shard_map(train_shards),
        },
        "val": {
            "n_shards": len(val_shards),
            "n_images": len(val_image_ids),
            "shard_to_image_ids": _shard_map(val_shards),
        },
    }


def build_things_split_payload(
    catalog_image_ids: Iterable[str],
    meg_image_ids: Iterable[str],
    eeg_parsed: dict[str, Any],
    *,
    val_frac: float = 0.20,
    seed: int = 0,
    sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the canonical things_split.json payload."""
    eeg_val_pool = sorted(eeg_parsed["intersection_image_ids"])
    eeg_stats = {
        "n_eeg1": eeg_parsed["n_eeg1"],
        "n_eeg2": eeg_parsed["n_eeg2"],
        "n_eeg_intersection": eeg_parsed["n_eeg_intersection"],
        "n_eeg_union": eeg_parsed["n_eeg_union"],
    }
    train_ids, val_ids, val_pool_ids, stats = neural_intersection_split(
        catalog_image_ids,
        meg_image_ids,
        eeg_val_pool,
        val_frac=val_frac,
        seed=seed,
        eeg_stats=eeg_stats,
    )
    payload: dict[str, Any] = {
        "version": "1",
        "policy": "image_level_val_from_neural_intersection",
        "seed": seed,
        "val_frac": val_frac,
        "intersection_definition": (
            "catalog ∩ meg_coverage ∩ (eeg1 ∩ eeg2)"
        ),
        "eeg": {
            "n_eeg1": eeg_stats["n_eeg1"],
            "n_eeg2": eeg_stats["n_eeg2"],
            "n_intersection": eeg_stats["n_eeg_intersection"],
            "n_union": eeg_stats["n_eeg_union"],
            "note": (
                "EEG1 and EEG2 are separate datasets with different coverage "
                f"(n_eeg1={eeg_stats['n_eeg1']}, n_eeg2={eeg_stats['n_eeg2']}, "
                f"n_union={eeg_stats['n_eeg_union']}). Val is sampled from "
                "catalog ∩ meg ∩ (eeg1 ∩ eeg2). n_union is reference only."
            ),
        },
        "sources": sources
        or {
            "catalog": "things_catalog.json",
            "meg_coverage": "things-meg/labels/meg_coverage.json",
            "eeg_coverage": "eeg_coverage.json",
        },
        "legacy_references": {
            "note": (
                "Existing train/val manifests use the prior 85/15 split; "
                "not overwritten by things_split.json."
            ),
            "train_manifest": "train/things_manifest.json",
            "val_manifest": "val/things_manifest.json",
        },
        **stats,
        "intersection_image_ids": val_pool_ids,
        "train_image_ids": train_ids,
        "val_image_ids": val_ids,
    }
    return payload


def catalog_slot_id(image_id: str, images_per_shard: int = IMAGES_PER_SHARD) -> str:
    """Return ``shard_NNN`` for catalog-slot ownership (Option 1 / 4M layout).

    THINGS image ids are 1-indexed (lowest is ``000000001``), so we subtract
    one before integer-dividing by ``images_per_shard``.
    """
    return f"shard_{(int(image_id) - 1) // images_per_shard:03d}"


def pack_by_catalog_slot(
    image_ids: Iterable[str],
    images_per_shard: int = IMAGES_PER_SHARD,
) -> dict[str, list[str]]:
    """Group image_ids by catalog slot. Returns ``{shard_id: [ids...]}``.

    Ids within each shard are sorted ascending. Slots with no matching ids
    are omitted from the returned dict (empty val shards are valid at read
    time — write an empty tar or skip; 4M accepts zero-sample shards).
    """
    grouped: dict[str, list[str]] = {}
    for image_id in sorted(set(image_ids)):
        slot = catalog_slot_id(image_id, images_per_shard)
        grouped.setdefault(slot, []).append(image_id)
    return grouped


def pack_into_shards(
    image_ids: list[str],
    images_per_shard: int = IMAGES_PER_SHARD,
) -> list[list[str]]:
    """Sort `image_ids` and chunk into fixed-size shards (last may be partial).

    Sorting (rather than keeping random order) preserves the
    `image_id // images_per_shard == shard_index` math within a split,
    which is useful for debugging and for any modality that wants to
    locate a shard from an image_id without consulting the manifest.
    """
    ordered = sorted(image_ids)
    return [
        ordered[i : i + images_per_shard]
        for i in range(0, len(ordered), images_per_shard)
    ]


def build_image_id_to_shard_map(
    image_ids: Iterable[str],
    images_per_shard: int = IMAGES_PER_SHARD,
) -> dict[str, str]:
    """Map each image_id to ``shard_NNN`` after sorted fixed-size packing."""
    shards = pack_into_shards(sorted(set(image_ids)), images_per_shard)
    out: dict[str, str] = {}
    for n, ids in enumerate(shards):
        shard_id = f"shard_{n:03d}"
        for image_id in ids:
            prior = out.get(image_id)
            if prior is not None:
                raise ValueError(
                    f"image_id {image_id} in multiple shards: {prior} and {shard_id}"
                )
            out[image_id] = shard_id
    return out


def build_split_manifest(
    split_name: str,
    shards: list[list[str]],
    shards_subpath: str,
) -> dict:
    """Build the per-split manifest JSON.

    Shape:
      {
        "split": "train" | "val",
        "n_shards": int,
        "n_images": int,
        "images_per_shard": int,
        "shards_subpath": str,                    # e.g. "things/rgb"
        "shards": {"shard_NNN": {"n_images": ..., "image_ids": [...]}}
      }
    """
    shards_payload: dict[str, dict] = {}
    total = 0
    for n, ids in enumerate(shards):
        shard_id = f"shard_{n:03d}"
        shards_payload[shard_id] = {
            "n_images": len(ids),
            "image_ids": list(ids),
        }
        total += len(ids)
    return {
        "split": split_name,
        "n_shards": len(shards),
        "n_images": total,
        "images_per_shard": IMAGES_PER_SHARD,
        "shards_subpath": shards_subpath,
        "shards": shards_payload,
    }


# ---------- repack execution ----------------------------------------------

@dataclass(frozen=True)
class SourceLocation:
    """Where a single image_id currently lives in the source shard layout."""

    src_tar_path: str
    image_id: str


def index_source_shards(shard_paths: list[str]) -> dict[str, SourceLocation]:
    """Walk every source shard once; return ``{image_id: SourceLocation}``.

    Errors loudly if the same image_id appears in more than one source tar —
    that would mean the shards are inconsistent and the repack is unsafe.
    """
    locations: dict[str, SourceLocation] = {}
    for p in shard_paths:
        for image_id in extract_shard_contents(p):
            prior = locations.get(image_id)
            if prior is not None:
                raise ValueError(
                    f"image_id {image_id} found in both "
                    f"{prior.src_tar_path} and {p}"
                )
            locations[image_id] = SourceLocation(p, image_id)
    return locations


def write_shard_from_locations(
    out_path: str,
    image_ids_in_order: list[str],
    source_index: dict[str, SourceLocation],
) -> None:
    """Build a new tar at `out_path` from existing source shards.

    Copies `.jpg` and `.txt` bytes verbatim (no JPEG re-encoding). Groups
    reads by source tar so each source is opened at most once per call.
    Writes to `<out_path>.tmp` then atomically renames into place.
    """
    by_src: dict[str, list[str]] = {}
    for image_id in image_ids_in_order:
        src = source_index[image_id].src_tar_path
        by_src.setdefault(src, []).append(image_id)

    extracted: dict[str, tuple[bytes, bytes]] = {}  # id -> (jpg, txt)
    for src_path, ids in by_src.items():
        needed = set(ids)
        with tarfile.open(src_path, "r") as src:
            for member in src:
                if member.name.endswith(".jpg"):
                    iid = member.name[:-4]
                    if iid not in needed:
                        continue
                    f = src.extractfile(member)
                    if f is None:
                        continue
                    cur = extracted.get(iid, (b"", b""))
                    extracted[iid] = (f.read(), cur[1])
                elif member.name.endswith(".txt"):
                    iid = member.name[:-4]
                    if iid not in needed:
                        continue
                    f = src.extractfile(member)
                    if f is None:
                        continue
                    cur = extracted.get(iid, (b"", b""))
                    extracted[iid] = (cur[0], f.read())

    missing = [i for i in image_ids_in_order if i not in extracted]
    if missing:
        raise RuntimeError(
            f"write_shard_from_locations: {len(missing)} ids missing from "
            f"sources (first 5: {missing[:5]})"
        )

    tmp_path = out_path + ".tmp"
    with tarfile.open(tmp_path, "w") as out:
        for image_id in image_ids_in_order:
            jpg_bytes, txt_bytes = extracted[image_id]
            if not jpg_bytes or not txt_bytes:
                raise RuntimeError(
                    f"missing payload for {image_id}: "
                    f"jpg_bytes={len(jpg_bytes)}, txt_bytes={len(txt_bytes)}"
                )
            for name, data in (
                (f"{image_id}.jpg", jpg_bytes),
                (f"{image_id}.txt", txt_bytes),
            ):
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                out.addfile(info, io.BytesIO(data))
    os.replace(tmp_path, out_path)
