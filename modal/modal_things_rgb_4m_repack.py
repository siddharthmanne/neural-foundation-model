"""Repack raw THINGS RGB JPEG shards into 4M catalog-slot layout.

Target layout (aligned with ``tok_rgb`` / ``things_split.json``)::

  /project/data/train/things/rgb/shard_NNN.tar
  /project/data/val/things/rgb/shard_NNN.tar

Each tar entry is paired ``<image_id>.jpg`` + ``<image_id>.txt``.
Shard index NNN = ``(int(image_id) - 1) // 1000`` (catalog slot, 000..026).

Sources: existing train/val ``things/rgb`` tars (legacy 85/15 sequential layout).
Does **not** touch ``cc12m`` or tokenized modalities.

Run from modal/::
    modal run modal_things_rgb_4m_repack.py::plan
    modal run --detach modal_things_rgb_4m_repack.py::repack
    modal run modal_things_rgb_4m_repack.py::verify
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile

import modal

import things_manifest

app = modal.App("things-rgb-4m-repack")
project_volume = modal.Volume.from_name("project")
PROJECT = "/project"

repack_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    "things_manifest"
)

THINGS_SPLIT = f"{PROJECT}/data/things_split.json"
STAGING = f"{PROJECT}/data/staging/things_rgb_4m_repack"
PLAN_PATH = f"{STAGING}/repack_plan.json"
SLOTS = [f"shard_{i:03d}" for i in range(27)]


def _rgb_dir(split: str) -> str:
    return f"{PROJECT}/data/{split}/things/rgb"


def _stage_dir(split: str) -> str:
    return f"{STAGING}/{split}"


def _list_rgb_shards(split: str) -> list[str]:
    d = _rgb_dir(split)
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if things_manifest.SHARD_NAME_RE.match(f)
    )


def _load_split() -> tuple[set[str], set[str]]:
    split = json.loads(open(THINGS_SPLIT).read())
    train = set(split["train_image_ids"])
    val = set(split["val_image_ids"])
    if train & val:
        raise ValueError("things_split train/val overlap")
    return train, val


@app.function(
    image=repack_image,
    volumes={PROJECT: project_volume},
    cpu=4.0,
    memory=8 * 1024,
    timeout=60 * 15,
)
def plan_remote() -> dict:
    train, val = _load_split()
    train_pack = things_manifest.pack_by_catalog_slot(sorted(train))
    val_pack = things_manifest.pack_by_catalog_slot(sorted(val))

    src_train = _list_rgb_shards("train")
    src_val = _list_rgb_shards("val")
    payload = {
        "version": "1",
        "split_source": THINGS_SPLIT,
        "n_train": len(train),
        "n_val": len(val),
        "src_train_shards": len(src_train),
        "src_val_shards": len(src_val),
        "train_shards": {k: len(v) for k, v in sorted(train_pack.items())},
        "val_shards": {k: len(v) for k, v in sorted(val_pack.items())},
        "target_slots": SLOTS,
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
    if not os.path.isfile(PLAN_PATH):
        raise FileNotFoundError(f"Run plan first; missing {PLAN_PATH}")

    train, val = _load_split()
    train_pack = things_manifest.pack_by_catalog_slot(sorted(train))
    val_pack = things_manifest.pack_by_catalog_slot(sorted(val))

    src_paths = _list_rgb_shards("train") + _list_rgb_shards("val")
    if not src_paths:
        raise FileNotFoundError("No rgb shard_*.tar found under train/val things/rgb")

    print(f"[repack] indexing {len(src_paths)} source rgb shards...")
    src_index = things_manifest.index_source_shards(src_paths)
    print(f"[repack] indexed {len(src_index)} image_ids")

    expected_catalog = train | val
    missing = sorted(expected_catalog - set(src_index))
    if missing:
        raise RuntimeError(
            f"{len(missing)} split image_ids missing from rgb sources "
            f"(first 5: {missing[:5]})"
        )

    shutil.rmtree(STAGING, ignore_errors=True)

    for split_name, id_pack in (("train", train_pack), ("val", val_pack)):
        os.makedirs(_stage_dir(split_name), exist_ok=True)
        for slot_id, image_ids in sorted(id_pack.items()):
            out_path = os.path.join(_stage_dir(split_name), f"{slot_id}.tar")
            things_manifest.write_shard_from_locations(
                out_path, image_ids, src_index
            )
            print(f"[repack] {split_name}/{slot_id}: {len(image_ids)} images")
        project_volume.commit()

    expected_tars = {f"{s}.tar" for s in SLOTS}
    for split_name in ("train", "val"):
        final_dir = _rgb_dir(split_name)
        os.makedirs(final_dir, exist_ok=True)
        for fname in os.listdir(final_dir):
            if fname.endswith(".tar") and fname not in expected_tars:
                os.remove(os.path.join(final_dir, fname))
                print(f"[repack] removed stale {final_dir}/{fname}")
        staged_dir = _stage_dir(split_name)
        for fname in sorted(os.listdir(staged_dir)):
            os.replace(
                os.path.join(staged_dir, fname),
                os.path.join(final_dir, fname),
            )

    shutil.rmtree(STAGING, ignore_errors=True)
    project_volume.commit()

    _verify_on_volume()
    return {
        "status": "ok",
        "n_train": len(train),
        "n_val": len(val),
        "n_source_shards": len(src_paths),
    }


def _verify_on_volume() -> dict:
    train, val = _load_split()
    train_pack = things_manifest.pack_by_catalog_slot(sorted(train))
    val_pack = things_manifest.pack_by_catalog_slot(sorted(val))

    for split_name, id_pack, expected_ids in (
        ("train", train_pack, train),
        ("val", val_pack, val),
    ):
        seen: set[str] = set()
        for slot_id, image_ids in sorted(id_pack.items()):
            tar_path = os.path.join(_rgb_dir(split_name), f"{slot_id}.tar")
            if not os.path.isfile(tar_path):
                raise RuntimeError(f"missing {tar_path}")
            contents = things_manifest.extract_shard_contents(tar_path)
            if sorted(contents.keys()) != sorted(image_ids):
                raise RuntimeError(f"{split_name}/{slot_id} rgb keys mismatch")
            seen.update(contents.keys())
        if seen != expected_ids:
            raise RuntimeError(f"{split_name} rgb ids != things_split")

        # Cross-check keys match tok_rgb for shard_000
        tok_path = os.path.join(
            f"{PROJECT}/data/{split_name}/things/tok_rgb", "shard_000.tar"
        )
        if os.path.isfile(tok_path):
            with tarfile.open(tok_path, "r") as tar:
                tok_keys = sorted(m.name[:-4] for m in tar if m.name.endswith(".npy"))
            rgb_path = os.path.join(_rgb_dir(split_name), "shard_000.tar")
            rgb_keys = sorted(
                things_manifest.extract_shard_contents(rgb_path).keys()
            )
            if tok_keys != rgb_keys:
                raise RuntimeError(f"{split_name}/shard_000 rgb vs tok_rgb key mismatch")
            print(f"[verify] {split_name}/shard_000 aligned with tok_rgb: {len(tok_keys)} keys")

    print("[verify] all checks passed")
    return {"ok": True}


@app.function(
    image=repack_image,
    volumes={PROJECT: project_volume},
    cpu=2.0,
    memory=4 * 1024,
    timeout=60 * 15,
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
