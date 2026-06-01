"""Repack THINGS modalities into 4M Option 1 catalog-slot layout.

Target layout (per ``4m_training/README.md``)::

  /project/data/train/things/{tok_rgb,tok_depth,tok_meg,tok_eeg,meg_mask,eeg_mask}/shard_NNN.tar
  /project/data/val/things/{...}/shard_NNN.tar

Each tar entry is ``<image_id>.npy`` keyed for 4M ``multi_tarfile_samples`` zip.
Shard index NNN = ``(int(image_id) - 1) // 1000`` (catalog slot, 000..026).

Train/val membership comes from ``/project/data/things_split.json``.
Does **not** touch ``cc12m`` paths.

Phases:
    modal run modal_things_4m_repack.py::plan     # JSON plan only
    modal run modal_things_4m_repack.py::repack   # stage + swap + verify + cleanup legacy

Legacy sources (read-only during repack):
    tok_rgb@224 / tok_depth@224   — full-catalog ``shard-00000.tar`` layout
    things-meg token cache        — MEG stacked trials
    things-eeg LaBraM token cache — EEG stacked trials (60 subject npz files)
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import modal

import meg_token_shard
import things_manifest

app = modal.App("things-4m-repack")
project_volume = modal.Volume.from_name("project")
PROJECT = "/project"

repack_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .add_local_python_source("things_manifest", "meg_token_shard")
)

THINGS_SPLIT = f"{PROJECT}/data/things_split.json"
CATALOG_PATH = f"{PROJECT}/data/things_catalog.json"
MEG_COVERAGE = f"{PROJECT}/data/things-meg/labels/meg_coverage.json"
EEG_COVERAGE = f"{PROJECT}/data/eeg_coverage.json"
MEG_CACHE = f"{PROJECT}/data/things-meg/tokens/brainomni/V512_rvq4_win512_sf256_3b"
MEG_BRIDGE = f"{PROJECT}/data/things-meg/labels/meg_trigger_to_image_id.json"
EEG_CACHE = (
    f"{PROJECT}/data/things-eeg/tokens/labram/"
    "V8192_d64_ch17_sr200_train-eeg1+2_e5"
)

STAGING = f"{PROJECT}/data/staging/things_4m_repack"
PLAN_PATH = f"{STAGING}/repack_plan.json"

# Legacy read paths
SRC_RGB_TOK = f"{PROJECT}/data/train/things/tok_rgb@224"
SRC_DEPTH_TOK = f"{PROJECT}/data/train/things/tok_depth@224"
SRC_TRAIN_MEG = f"{PROJECT}/data/train/things/tok_meg"
SRC_VAL_MEG = f"{PROJECT}/data/val/things/tok_meg"
SRC_TRAIN_EEG = f"{PROJECT}/data/train/things/tok_eeg"
SRC_VAL_EEG = f"{PROJECT}/data/val/things/tok_eeg"

MODALITIES = (
    "tok_rgb",
    "tok_depth",
    "tok_meg",
    "tok_eeg",
    "meg_mask",
    "eeg_mask",
)

LEGACY_DIRS_TO_REMOVE = (
    SRC_RGB_TOK,
    SRC_DEPTH_TOK,
)

MEG_SENTINEL_SHAPE = (1, 16, 8, 4)
EEG_SENTINEL_SHAPE = (1, 17)


def _npy_bytes(arr) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


def _write_image_shard(out_path: str, entries: list[tuple[str, bytes]]) -> None:
    """Write ``{image_id}.npy`` members sorted by id."""
    tmp = out_path + ".tmp"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with tarfile.open(tmp, "w") as tar:
        for image_id, data in sorted(entries, key=lambda x: x[0]):
            info = tarfile.TarInfo(name=f"{image_id}.npy")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    os.replace(tmp, out_path)


def _catalog_src_tar_path(src_dir: str, slot_id: str) -> str:
    """Map ``shard_NNN`` -> legacy ``shard-NNNNN.tar`` in tok_rgb@224 layout."""
    n = int(slot_id.split("_")[1])
    return os.path.join(src_dir, f"shard-{n:05d}.tar")


def _index_catalog_token_tar(tar_path: str) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    if not os.path.isfile(tar_path):
        return out
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.name.endswith(".npy"):
                continue
            image_id = member.name[:-4]
            f = tar.extractfile(member)
            if f is None:
                continue
            out[image_id] = f.read()
    return out


def _squeeze_rgb_depth(raw: bytes):
    import numpy as np

    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    if arr.shape == (1, 196):
        arr = arr.reshape(196)
    elif arr.shape != (196,):
        raise ValueError(f"unexpected rgb/depth token shape {arr.shape}")
    return arr.astype(np.int16, copy=False)


def _load_eeg_from_labram_cache() -> dict:
    """Load LaBraM cache incrementally; return ``{image_id: (n_trials, 17) int16}``.

    Processes one subject ``.npz`` at a time (60 files, not 2M loose paths).
    Copies rows immediately so no npz stays pinned via millions of views.
    """
    import numpy as np
    from glob import glob

    out: dict[str, np.ndarray] = {}
    npz_paths = sorted(glob(os.path.join(EEG_CACHE, "*.npz")))
    if not npz_paths:
        raise FileNotFoundError(f"No LaBraM npz files in {EEG_CACHE}")

    for fi, p in enumerate(npz_paths):
        z = np.load(p)
        source = str(z["source"])
        subject = str(z["subject"])
        tokens = z["tokens"]
        image_ids = z["image_id"]
        trial_idx = z["trial_idx"]
        n = tokens.shape[0]

        local: dict[str, list[tuple[tuple[str, str, int], np.ndarray]]] = defaultdict(list)
        for i in range(n):
            row = tokens[i]
            if row.shape != (17,):
                raise ValueError(f"{p} row {i}: bad shape {row.shape}")
            sort_key = (source, subject, int(trial_idx[i]))
            local[str(image_ids[i])].append(
                (sort_key, np.asarray(row, dtype=np.int16))
            )
        del z, tokens, image_ids, trial_idx

        for image_id, chunks in local.items():
            ordered = sorted(chunks, key=lambda x: x[0])
            block = np.stack([c[1] for c in ordered], axis=0)
            if image_id in out:
                out[image_id] = np.concatenate([out[image_id], block], axis=0)
            else:
                out[image_id] = block
        del local

        if (fi + 1) % 10 == 0 or fi + 1 == len(npz_paths):
            print(f"    ... EEG cache {fi + 1}/{len(npz_paths)} npz files")

    return out


def _load_split() -> tuple[set[str], set[str], list[str]]:
    split = json.loads(open(THINGS_SPLIT).read())
    train = set(split["train_image_ids"])
    val = set(split["val_image_ids"])
    catalog = sorted(split["train_image_ids"] + split["val_image_ids"])
    if train & val:
        raise ValueError("things_split train/val overlap")
    return train, val, catalog


def _load_masks(catalog: Iterable[str]) -> tuple[set[str], set[str]]:
    meg = things_manifest.load_coverage_json(json.loads(open(MEG_COVERAGE).read()))
    eeg_cov = json.loads(open(EEG_COVERAGE).read())
    eeg = things_manifest.load_coverage_json(eeg_cov)  # union by default field
    return meg, eeg


@dataclass
class RepackPlan:
    train_ids: list[str]
    val_ids: list[str]
    meg_ids: set[str]
    eeg_ids: set[str]
    slots: list[str]


def _build_plan() -> RepackPlan:
    train, val, catalog = _load_split()
    meg_ids, eeg_ids = _load_masks(catalog)
    slots = [f"shard_{i:03d}" for i in range(27)]
    return RepackPlan(sorted(train), sorted(val), meg_ids, eeg_ids, slots)


def _dst(split: str, modality: str) -> str:
    return f"{PROJECT}/data/{split}/things/{modality}"


def _stage_dst(split: str, modality: str) -> str:
    return f"{STAGING}/{split}/{modality}"


@app.function(
    image=repack_image,
    volumes={PROJECT: project_volume},
    cpu=4.0,
    memory=16 * 1024,
    timeout=60 * 30,
)
def plan_remote() -> dict:
    plan = _build_plan()
    train_pack = things_manifest.pack_by_catalog_slot(plan.train_ids)
    val_pack = things_manifest.pack_by_catalog_slot(plan.val_ids)

    payload = {
        "version": "1",
        "split_source": THINGS_SPLIT,
        "n_train": len(plan.train_ids),
        "n_val": len(plan.val_ids),
        "n_meg_coverage": len(plan.meg_ids),
        "n_eeg_coverage": len(plan.eeg_ids),
        "modalities": list(MODALITIES),
        "train_shards": {k: len(v) for k, v in sorted(train_pack.items())},
        "val_shards": {k: len(v) for k, v in sorted(val_pack.items())},
        "notes": {
            "rgb_depth_source": "train/things/tok_rgb@224 + tok_depth@224 (full catalog)",
            "meg_source": f"token cache {MEG_CACHE} (stacked trials)",
            "eeg_source": f"LaBraM cache {EEG_CACHE} (stacked trials)",
            "legacy_rgb_jpg": "train/things/rgb left unchanged (not in 4M data_path)",
        },
    }
    os.makedirs(STAGING, exist_ok=True)
    open(PLAN_PATH, "w").write(json.dumps(payload, indent=2))
    project_volume.commit()
    print(json.dumps(payload, indent=2))
    return payload


@app.function(
    image=repack_image,
    volumes={PROJECT: project_volume},
    cpu=8.0,
    memory=16 * 1024,
    timeout=60 * 180,
)
def repack_remote() -> dict:
    import numpy as np

    if not os.path.exists(PLAN_PATH):
        raise FileNotFoundError(f"Run plan first; missing {PLAN_PATH}")

    plan = _build_plan()
    train_pack = things_manifest.pack_by_catalog_slot(plan.train_ids)
    val_pack = things_manifest.pack_by_catalog_slot(plan.val_ids)

    # --- EEG from LaBraM cache (60 subject npz files) ---
    print("[repack] loading LaBraM EEG cache...")
    eeg_by_image = _load_eeg_from_labram_cache()
    print(f"[repack] EEG cache covers {len(eeg_by_image)} image_ids")

    # --- MEG from token cache ---
    print("[repack] loading MEG token cache...")
    bridge = {int(k): v for k, v in json.loads(open(MEG_BRIDGE).read())["trigger_to_image_id"].items()}
    meg_by_image: dict[str, list[meg_token_shard.MegEntry]] = defaultdict(list)
    for subj in ("P1", "P2", "P3", "P4"):
        npz = np.load(os.path.join(MEG_CACHE, f"{subj}.npz"))
        tokens = npz["tokens"]
        triggers = npz["meg_trigger_codes"]
        trial_idx = npz["trial_idx"]
        for i in range(tokens.shape[0]):
            trig = int(triggers[i])
            image_id = bridge.get(trig)
            if image_id is None:
                continue
            meg_by_image[image_id].append(
                meg_token_shard.MegEntry(
                    image_id=image_id,
                    subject=subj,
                    trial_idx=int(trial_idx[i]),
                    tokens=tokens[i],
                )
            )
    print(f"[repack] MEG cache covers {len(meg_by_image)} image_ids")

    shutil.rmtree(STAGING, ignore_errors=True)

    for split_name, id_pack in (("train", train_pack), ("val", val_pack)):
        for slot_id, image_ids in sorted(id_pack.items()):
            # RGB + depth from catalog token tars
            src_rgb_tar = _catalog_src_tar_path(SRC_RGB_TOK, slot_id)
            src_depth_tar = _catalog_src_tar_path(SRC_DEPTH_TOK, slot_id)
            rgb_raw = _index_catalog_token_tar(src_rgb_tar)
            depth_raw = _index_catalog_token_tar(src_depth_tar)

            rgb_entries: list[tuple[str, bytes]] = []
            depth_entries: list[tuple[str, bytes]] = []
            meg_entries: list[tuple[str, bytes]] = []
            eeg_entries: list[tuple[str, bytes]] = []
            meg_mask_entries: list[tuple[str, bytes]] = []
            eeg_mask_entries: list[tuple[str, bytes]] = []

            for image_id in image_ids:
                if image_id not in rgb_raw:
                    raise RuntimeError(f"{split_name}/{slot_id}: missing tok_rgb {image_id}")
                if image_id not in depth_raw:
                    raise RuntimeError(f"{split_name}/{slot_id}: missing tok_depth {image_id}")

                rgb_entries.append((image_id, _npy_bytes(_squeeze_rgb_depth(rgb_raw[image_id]))))
                depth_entries.append((image_id, _npy_bytes(_squeeze_rgb_depth(depth_raw[image_id]))))

                if image_id in meg_by_image:
                    trials = sorted(
                        meg_by_image[image_id],
                        key=lambda e: (e.subject, e.trial_idx),
                    )
                    stacked = np.stack(
                        [t.tokens.astype(np.int16, copy=False) for t in trials], axis=0
                    )
                    meg_entries.append((image_id, _npy_bytes(stacked)))
                    meg_mask_entries.append((image_id, _npy_bytes(np.array([1], dtype=np.uint8))))
                else:
                    sentinel = np.full(MEG_SENTINEL_SHAPE, -1, dtype=np.int16)
                    meg_entries.append((image_id, _npy_bytes(sentinel)))
                    meg_mask_entries.append((image_id, _npy_bytes(np.array([0], dtype=np.uint8))))

                if image_id in eeg_by_image:
                    eeg_entries.append((image_id, _npy_bytes(eeg_by_image[image_id])))
                    eeg_mask_entries.append((image_id, _npy_bytes(np.array([1], dtype=np.uint8))))
                else:
                    sentinel = np.full(EEG_SENTINEL_SHAPE, -1, dtype=np.int16)
                    eeg_entries.append((image_id, _npy_bytes(sentinel)))
                    eeg_mask_entries.append((image_id, _npy_bytes(np.array([0], dtype=np.uint8))))

            out_map = {
                "tok_rgb": rgb_entries,
                "tok_depth": depth_entries,
                "tok_meg": meg_entries,
                "tok_eeg": eeg_entries,
                "meg_mask": meg_mask_entries,
                "eeg_mask": eeg_mask_entries,
            }
            for mod, entries in out_map.items():
                out_path = os.path.join(_stage_dst(split_name, mod), f"{slot_id}.tar")
                _write_image_shard(out_path, entries)
            print(
                f"[repack] {split_name}/{slot_id}: {len(image_ids)} images "
                f"(meg_real={sum(1 for i in image_ids if i in meg_by_image)}, "
                f"eeg_real={sum(1 for i in image_ids if i in eeg_by_image)})"
            )

    project_volume.commit()

    # Swap staged -> production (only things/ 4M modalities; never cc12m).
    for split_name in ("train", "val"):
        for mod in MODALITIES:
            staged_dir = _stage_dst(split_name, mod)
            final_dir = _dst(split_name, mod)
            os.makedirs(final_dir, exist_ok=True)
            expected = {f"{slot}.tar" for slot in plan.slots}
            for fname in os.listdir(final_dir):
                if fname.endswith(".tar") and fname not in expected:
                    os.remove(os.path.join(final_dir, fname))
            for fname in sorted(os.listdir(staged_dir)):
                os.replace(
                    os.path.join(staged_dir, fname),
                    os.path.join(final_dir, fname),
                )

    shutil.rmtree(STAGING, ignore_errors=True)
    project_volume.commit()

    _verify_on_volume()
    removed = _cleanup_legacy_on_volume()
    project_volume.commit()
    return {
        "status": "ok",
        "n_train": len(plan.train_ids),
        "n_val": len(plan.val_ids),
        "legacy_removed": removed,
    }


def _cleanup_legacy_on_volume() -> dict:
    """Drop superseded @224 dirs and leftover legacy EEG loose/tar files."""
    removed: dict[str, int | str] = {}
    for path in LEGACY_DIRS_TO_REMOVE:
        if os.path.isdir(path):
            n = sum(len(files) for _, _, files in os.walk(path))
            shutil.rmtree(path)
            removed[path] = n
            print(f"[cleanup] removed dir {path} ({n} files)")
    # Legacy EEG: remove loose .eeg.npy and old sequential tars if any remain
    # alongside the new catalog-slot tars (do NOT rmtree tok_eeg — that's the destination).
    for split in ("train", "val"):
        eeg_dir = _dst(split, "tok_eeg")
        if not os.path.isdir(eeg_dir):
            continue
        n_loose = n_stale = 0
        for fname in os.listdir(eeg_dir):
            fpath = os.path.join(eeg_dir, fname)
            if fname.endswith(".eeg.npy"):
                os.remove(fpath)
                n_loose += 1
            elif fname.endswith(".tar") and fname.startswith("shard_"):
                try:
                    idx = int(fname[6:9])
                except ValueError:
                    idx = 999
                if idx > 26:
                    os.remove(fpath)
                    n_stale += 1
        if n_loose:
            removed[f"{eeg_dir}/*.eeg.npy"] = n_loose
            print(f"[cleanup] removed {n_loose} loose EEG files from {eeg_dir}")
        if n_stale:
            removed[f"{eeg_dir}/stale_tars"] = n_stale
    return removed


def _verify_on_volume() -> dict:
    train, val, _catalog = _load_split()
    train_pack = things_manifest.pack_by_catalog_slot(train)
    val_pack = things_manifest.pack_by_catalog_slot(val)

    for split_name, id_pack, expected_ids in (
        ("train", train_pack, train),
        ("val", val_pack, val),
    ):
        seen: set[str] = set()
        for slot_id, image_ids in sorted(id_pack.items()):
            url_dir = _dst(split_name, "tok_rgb")
            tar_path = os.path.join(url_dir, f"{slot_id}.tar")
            if not os.path.isfile(tar_path):
                raise RuntimeError(f"missing {tar_path}")
            with tarfile.open(tar_path, "r") as tar:
                keys = sorted(m.name[:-4] for m in tar if m.name.endswith(".npy"))
            if keys != sorted(image_ids):
                raise RuntimeError(f"{split_name}/{slot_id} tok_rgb keys mismatch")
            seen.update(keys)
        if seen != expected_ids:
            raise RuntimeError(f"{split_name} tok_rgb ids != things_split")

    # Spot-check zip alignment on shard_000
    for split_name in ("train", "val"):
        mods = ["tok_rgb", "tok_depth", "tok_meg", "tok_eeg", "meg_mask", "eeg_mask"]
        keys_per_mod = {}
        for mod in mods:
            p = os.path.join(_dst(split_name, mod), "shard_000.tar")
            if not os.path.isfile(p):
                continue
            with tarfile.open(p, "r") as tar:
                keys_per_mod[mod] = [m.name for m in tar if m.name.endswith(".npy")]
        if keys_per_mod and len(set(map(tuple, keys_per_mod.values()))) != 1:
            raise RuntimeError(f"{split_name}/shard_000 key mismatch across modalities")
        print(f"[verify] {split_name}/shard_000 aligned: {len(next(iter(keys_per_mod.values())))} keys")

    print("[verify] all checks passed")
    return {"ok": True}


@app.function(
    image=repack_image,
    volumes={PROJECT: project_volume},
    cpu=2.0,
    memory=8 * 1024,
    timeout=60 * 20,
)
def verify_remote() -> dict:
    return _verify_on_volume()


@app.local_entrypoint()
def plan():
    print(json.dumps(plan_remote.remote(), indent=2))


@app.local_entrypoint()
def repack():
    print(json.dumps(repack_remote.remote(), indent=2))


@app.local_entrypoint()
def verify():
    print(json.dumps(verify_remote.remote(), indent=2))
