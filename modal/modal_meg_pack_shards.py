"""Pack BrainOmni 3b MEG tokens into per-split shard tars aligned with things_split.json.

What it produces:
  /project/data/train/things/tok_meg/shard_NNN.tar
  /project/data/val/things/tok_meg/shard_NNN.tar
  /project/data/train/things_meg_manifest.json
  /project/data/val/things_meg_manifest.json

Train/val membership comes from /project/data/things_split.json (not legacy RGB
manifests). Val must contain trials for exactly the 3,344 val_image_ids.

Run from modal/:
    modal run modal_meg_pack_shards.py::plan
    modal run modal_meg_pack_shards.py::pack
    modal run modal_meg_pack_shards.py::verify
"""

from __future__ import annotations

import json
import os
import shutil

import modal

import meg_token_shard
import things_manifest
from modal_app import app, image


project_volume = modal.Volume.from_name("project")
PROJECT_MOUNT = "/project"

CHECKPOINT_SLUG = "V512_rvq4_win512_sf256_3b"
TOKENS_DIR = f"/project/data/things-meg/tokens/brainomni/{CHECKPOINT_SLUG}"
BRIDGE_PATH = "/project/data/things-meg/labels/meg_trigger_to_image_id.json"
THINGS_SPLIT_PATH = "/project/data/things_split.json"

DST_TRAIN_MEG = "/project/data/train/things/tok_meg"
DST_VAL_MEG = "/project/data/val/things/tok_meg"
STAGING_ROOT = "/project/data/staging/tok_meg"
STAGING_TRAIN = f"{STAGING_ROOT}/train"
STAGING_VAL = f"{STAGING_ROOT}/val"
PLAN_PATH = f"{STAGING_ROOT}/pack_plan.json"

TRAIN_MEG_MANIFEST = "/project/data/train/things_meg_manifest.json"
VAL_MEG_MANIFEST = "/project/data/val/things_meg_manifest.json"

SUBJECTS = ("P1", "P2", "P3", "P4")
EXPECTED_VAL_IMAGES = 3344

pack_image = (
    image
    .pip_install("numpy")
    .add_local_python_source("meg_token_shard", "things_manifest", "modal_app")
)


def _load_things_split() -> tuple[dict, dict[str, str], dict[str, str], set[str], set[str]]:
    if not os.path.exists(THINGS_SPLIT_PATH):
        raise FileNotFoundError(f"Missing {THINGS_SPLIT_PATH}")
    split = json.loads(open(THINGS_SPLIT_PATH).read())
    train_ids = split["train_image_ids"]
    val_ids = split["val_image_ids"]
    if len(val_ids) != EXPECTED_VAL_IMAGES:
        raise ValueError(
            f"things_split.json has {len(val_ids)} val_image_ids; "
            f"expected {EXPECTED_VAL_IMAGES}"
        )
    train_set, val_set = set(train_ids), set(val_ids)
    if train_set & val_set:
        raise ValueError("train_image_ids and val_image_ids overlap")
    train_inv = things_manifest.build_image_id_to_shard_map(train_ids)
    val_inv = things_manifest.build_image_id_to_shard_map(val_ids)
    return split, train_inv, val_inv, train_set, val_set


def _load_bridge() -> dict[int, str]:
    raw = json.loads(open(BRIDGE_PATH).read())
    return {int(k): v for k, v in raw["trigger_to_image_id"].items()}


def _iter_subject_entries(tokens_dir: str, subject: str, trigger_to_image_id: dict[int, str]):
    import numpy as np

    path = os.path.join(tokens_dir, f"{subject}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing token cache {path}. Run modal_meg_tokenize_all.py::tokenize_all first."
        )
    arr = np.load(path)
    tokens = arr["tokens"]
    triggers = arr["meg_trigger_codes"]
    trial_idx = arr["trial_idx"]
    cache_subject = str(arr["subject"])
    if cache_subject != subject:
        raise RuntimeError(
            f"{path} has subject={cache_subject!r}, expected {subject!r}"
        )

    skipped_unknown = 0
    for i in range(tokens.shape[0]):
        trig = int(triggers[i])
        image_id = trigger_to_image_id.get(trig)
        if image_id is None:
            skipped_unknown += 1
            continue
        yield meg_token_shard.MegEntry(
            image_id=image_id,
            subject=subject,
            trial_idx=int(trial_idx[i]),
            tokens=tokens[i],
        )
    if skipped_unknown:
        print(f"[pack]   {subject}: skipped {skipped_unknown} unresolved triggers")


def _summarize_shard_plan(grouped: dict[str, list]) -> dict:
    return {
        sid: {"n_entries": len(entries)} for sid, entries in sorted(grouped.items())
    }


def _load_all_entries(bridge: dict[int, str]) -> tuple[list, dict[str, int]]:
    all_entries: list = []
    per_subj_stats: dict[str, int] = {}
    for subj in SUBJECTS:
        prev = len(all_entries)
        for e in _iter_subject_entries(TOKENS_DIR, subj, bridge):
            all_entries.append(e)
        per_subj_stats[subj] = len(all_entries) - prev
        print(f"[plan] {subj}: {per_subj_stats[subj]} entries")
    return all_entries, per_subj_stats


def _unique_image_ids(entries: list) -> set[str]:
    return {e.image_id for e in entries}


@app.function(
    image=pack_image,
    volumes={PROJECT_MOUNT: project_volume},
    cpu=4.0,
    memory=16 * 1024,
    timeout=60 * 30,
)
def plan_remote() -> dict:
    """Compute shard assignment from things_split.json and write manifests."""
    for p in (BRIDGE_PATH, THINGS_SPLIT_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    cfg_path = os.path.join(TOKENS_DIR, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing tokenizer cache config at {cfg_path}")

    cfg = json.loads(open(cfg_path).read())
    split, train_inv, val_inv, train_set, val_set = _load_things_split()
    bridge = _load_bridge()
    print(
        f"[plan] things_split v{split.get('version')}: "
        f"train_imgs={len(train_set)} val_imgs={len(val_set)}"
    )

    all_entries, per_subj_stats = _load_all_entries(bridge)
    train_grouped = meg_token_shard.group_entries_by_shard(all_entries, train_inv)
    val_grouped = meg_token_shard.group_entries_by_shard(all_entries, val_inv)

    train_image_ids = _unique_image_ids(
        e for entries in train_grouped.values() for e in entries
    )
    val_image_ids = _unique_image_ids(
        e for entries in val_grouped.values() for e in entries
    )
    n_train_entries = sum(len(v) for v in train_grouped.values())
    n_val_entries = sum(len(v) for v in val_grouped.values())
    n_orphan = len(all_entries) - n_train_entries - n_val_entries

    if n_orphan != 0:
        raise RuntimeError(
            f"{n_orphan} MEG entries didn't land in either train or val."
        )
    if len(val_image_ids) != EXPECTED_VAL_IMAGES:
        raise RuntimeError(
            f"val shard unique image_ids={len(val_image_ids)}, "
            f"expected {EXPECTED_VAL_IMAGES}"
        )
    if val_image_ids != val_set:
        missing = val_set - val_image_ids
        extra = val_image_ids - val_set
        raise RuntimeError(
            f"val MEG image_ids mismatch things_split: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    print(
        f"[plan] entries: total={len(all_entries)} "
        f"train={n_train_entries} ({len(train_image_ids)} imgs) "
        f"val={n_val_entries} ({len(val_image_ids)} imgs)"
    )

    train_meg_manifest = {
        "split": "train",
        "split_source": THINGS_SPLIT_PATH,
        "n_shards": len(train_grouped),
        "n_entries": n_train_entries,
        "n_images": len(train_image_ids),
        "shards_subpath": "things/tok_meg",
        "tokenizer_cfg": cfg,
        "shards": _summarize_shard_plan(train_grouped),
    }
    val_meg_manifest = {
        "split": "val",
        "split_source": THINGS_SPLIT_PATH,
        "n_shards": len(val_grouped),
        "n_entries": n_val_entries,
        "n_images": EXPECTED_VAL_IMAGES,
        "shards_subpath": "things/tok_meg",
        "tokenizer_cfg": cfg,
        "shards": _summarize_shard_plan(val_grouped),
    }

    for p in (TRAIN_MEG_MANIFEST, VAL_MEG_MANIFEST, PLAN_PATH):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    open(TRAIN_MEG_MANIFEST, "w").write(json.dumps(train_meg_manifest, indent=2))
    open(VAL_MEG_MANIFEST, "w").write(json.dumps(val_meg_manifest, indent=2))

    plan_payload = {
        "version": "2",
        "split_source": THINGS_SPLIT_PATH,
        "tokenizer_cfg": cfg,
        "subjects": list(SUBJECTS),
        "n_total_entries": len(all_entries),
        "n_train_entries": n_train_entries,
        "n_val_entries": n_val_entries,
        "n_train_images": len(train_image_ids),
        "n_val_images": len(val_image_ids),
        "per_subject_entries": per_subj_stats,
        "train_shard_paths": [
            os.path.join(DST_TRAIN_MEG, f"{sid}.tar") for sid in sorted(train_grouped)
        ],
        "val_shard_paths": [
            os.path.join(DST_VAL_MEG, f"{sid}.tar") for sid in sorted(val_grouped)
        ],
    }
    open(PLAN_PATH, "w").write(json.dumps(plan_payload, indent=2))
    project_volume.commit()

    return {
        "n_total_entries": len(all_entries),
        "n_train_entries": n_train_entries,
        "n_val_entries": n_val_entries,
        "n_train_images": len(train_image_ids),
        "n_val_images": len(val_image_ids),
        "train_n_shards": len(train_grouped),
        "val_n_shards": len(val_grouped),
    }


@app.function(
    image=pack_image,
    volumes={PROJECT_MOUNT: project_volume},
    cpu=4.0,
    memory=16 * 1024,
    timeout=60 * 90,
)
def pack_remote() -> dict:
    """Build MEG shard tars from things_split.json and swap into place."""
    if not os.path.exists(PLAN_PATH):
        raise FileNotFoundError(f"Run plan first; {PLAN_PATH} missing")
    plan_payload = json.loads(open(PLAN_PATH).read())
    _, train_inv, val_inv, _, _ = _load_things_split()
    bridge = _load_bridge()

    all_entries, _ = _load_all_entries(bridge)
    train_grouped = meg_token_shard.group_entries_by_shard(all_entries, train_inv)
    val_grouped = meg_token_shard.group_entries_by_shard(all_entries, val_inv)

    os.makedirs(STAGING_TRAIN, exist_ok=True)
    os.makedirs(STAGING_VAL, exist_ok=True)

    for split_name, grouped, staging in (
        ("train", train_grouped, STAGING_TRAIN),
        ("val", val_grouped, STAGING_VAL),
    ):
        for shard_id, entries in sorted(grouped.items()):
            out_path = os.path.join(staging, f"{shard_id}.tar")
            meg_token_shard.write_meg_shard_tar(out_path, entries)
            print(
                f"[pack] {split_name}/{shard_id}: "
                f"{len(entries)} entries, {os.path.getsize(out_path) / 1e6:.2f} MB"
            )
    project_volume.commit()

    expected_train = len(plan_payload["train_shard_paths"])
    expected_val = len(plan_payload["val_shard_paths"])
    n_train_staged = len([f for f in os.listdir(STAGING_TRAIN) if f.endswith(".tar")])
    n_val_staged = len([f for f in os.listdir(STAGING_VAL) if f.endswith(".tar")])
    if n_train_staged != expected_train or n_val_staged != expected_val:
        raise RuntimeError(
            f"staged train={n_train_staged}/{expected_train} "
            f"val={n_val_staged}/{expected_val}"
        )

    os.makedirs(DST_TRAIN_MEG, exist_ok=True)
    os.makedirs(DST_VAL_MEG, exist_ok=True)

    # Remove stale shard tars not in the new layout.
    for dst_dir, grouped in ((DST_TRAIN_MEG, train_grouped), (DST_VAL_MEG, val_grouped)):
        expected = {f"{sid}.tar" for sid in grouped}
        for fname in os.listdir(dst_dir):
            if fname.endswith(".tar") and fname not in expected:
                os.remove(os.path.join(dst_dir, fname))
                print(f"[pack] removed stale {dst_dir}/{fname}")

    for f in sorted(os.listdir(STAGING_TRAIN)):
        os.replace(os.path.join(STAGING_TRAIN, f), os.path.join(DST_TRAIN_MEG, f))
    for f in sorted(os.listdir(STAGING_VAL)):
        os.replace(os.path.join(STAGING_VAL, f), os.path.join(DST_VAL_MEG, f))
    shutil.rmtree(STAGING_ROOT, ignore_errors=True)
    project_volume.commit()

    return {"train_n_shards": n_train_staged, "val_n_shards": n_val_staged}


@app.function(
    image=pack_image,
    volumes={PROJECT_MOUNT: project_volume},
    cpu=2.0,
    memory=4 * 1024,
    timeout=60 * 15,
)
def verify_remote() -> dict:
    """Verify MEG shards match things_split.json."""
    import numpy as np

    split, _, _, train_set, val_set = _load_things_split()
    train_meg_manifest = json.loads(open(TRAIN_MEG_MANIFEST).read())
    val_meg_manifest = json.loads(open(VAL_MEG_MANIFEST).read())

    all_val_image_ids: set[str] = set()
    all_train_image_ids: set[str] = set()

    for split_name, meg_manifest, dst_dir, expected_ids in (
        ("train", train_meg_manifest, DST_TRAIN_MEG, train_set),
        ("val", val_meg_manifest, DST_VAL_MEG, val_set),
    ):
        on_disk = sorted(f for f in os.listdir(dst_dir) if f.endswith(".tar"))
        expected = sorted(f"{s}.tar" for s in meg_manifest["shards"])
        if on_disk != expected:
            raise RuntimeError(f"{split_name} shard list mismatch")

        split_image_ids: set[str] = set()
        for shard_id in meg_manifest["shards"]:
            tar_path = os.path.join(dst_dir, f"{shard_id}.tar")
            entries = meg_token_shard.read_meg_shard_tar(tar_path)
            n_loaded = len(entries)
            expected_count = meg_manifest["shards"][shard_id]["n_entries"]
            if n_loaded != expected_count:
                raise RuntimeError(
                    f"{split_name}/{shard_id}: expected {expected_count} entries, "
                    f"got {n_loaded}"
                )
            for fn, arr in entries.items():
                parsed = meg_token_shard.parse_meg_filename(fn)
                if parsed is None:
                    raise RuntimeError(f"bad filename {fn!r}")
                image_id = parsed[0]
                if image_id not in expected_ids:
                    raise RuntimeError(
                        f"{split_name}/{shard_id}: {image_id} not in things_split "
                        f"{split_name} set"
                    )
                split_image_ids.add(image_id)
                if arr.shape != (16, 8, 4) or arr.dtype != np.int16:
                    raise RuntimeError(f"{fn}: bad array {arr.shape} {arr.dtype}")

        if split_name == "train":
            all_train_image_ids = split_image_ids
        else:
            all_val_image_ids = split_image_ids

        print(
            f"[verify] {split_name}: {len(on_disk)} shards, "
            f"{meg_manifest['n_entries']} entries, "
            f"{len(split_image_ids)} unique image_ids"
        )

    if len(all_val_image_ids) != EXPECTED_VAL_IMAGES:
        raise RuntimeError(
            f"val unique image_ids={len(all_val_image_ids)}, "
            f"expected {EXPECTED_VAL_IMAGES}"
        )
    if all_val_image_ids != val_set:
        raise RuntimeError("val MEG image_ids != things_split val_image_ids")
    if all_train_image_ids & all_val_image_ids:
        raise RuntimeError("train and val MEG image_ids overlap")

    print("[verify] all checks passed")
    return {
        "split_source": THINGS_SPLIT_PATH,
        "train_n_shards": train_meg_manifest["n_shards"],
        "val_n_shards": val_meg_manifest["n_shards"],
        "train_n_entries": train_meg_manifest["n_entries"],
        "val_n_entries": val_meg_manifest["n_entries"],
        "train_n_images": len(all_train_image_ids),
        "val_n_images": len(all_val_image_ids),
    }


@app.local_entrypoint()
def plan():
    summary = plan_remote.remote()
    print("\n[plan] summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


@app.local_entrypoint()
def pack():
    summary = pack_remote.remote()
    print("\n[pack] summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


@app.local_entrypoint()
def verify():
    summary = verify_remote.remote()
    print("\n[verify] summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
