"""One-shot maintenance job: repack THINGS RGB shards into image-level train/val.

What it produces on the `project` Modal Volume:
  /project/data/train/things/rgb/shard_NNN.tar   23 shards (~85% of images)
  /project/data/val/things/rgb/shard_NNN.tar      4 shards (~15% of images)
  /project/data/things_catalog.json               image_id ↔ filename for all 26107 images
  /project/data/train/things_manifest.json        per-shard image_ids for train
  /project/data/val/things_manifest.json          per-shard image_ids for val

Why three phases instead of one:
  The Volume is shared with the team. The `plan` phase writes only JSONs
  (small, reversible) and a `repack_plan.json` describing the exact moves.
  Eyeball that, then run `repack`. If `repack` crashes mid-way, staged
  shards remain in /project/data/staging/ — re-running `repack` resumes
  (skips shards already in staging); only the swap-into-place step is
  destructive.

Split policy (system-level decision recorded here so it's discoverable):
  Image-level random split, seed=0, val_frac=0.15. Each THINGS concept's
  ~14 images get distributed across both splits, matching the brain-decoding
  eval convention (MindEye, Takagi & Nishimoto, THINGS-EEG/MEG decoding line).
  Concept-level holdout is a separate "zero-shot" eval and is not produced
  here — see things_manifest.image_level_split if you want to derive it
  later from the same catalog.

Cost: ~$1 / job (CPU container, ~5 GB read + ~5 GB write per `repack`).

Run from the modal/ subdirectory (so pip_install_from_requirements resolves):
    cd modal
    modal run modal_things_repack.py::plan
    # eyeball:
    #   /project/data/things_catalog.json
    #   /project/data/train/things_manifest.json
    #   /project/data/val/things_manifest.json
    #   /project/data/staging/repack_plan.json
    modal run modal_things_repack.py::repack
    modal run modal_things_repack.py::verify
"""

from __future__ import annotations

import json
import os
import shutil

import modal

import things_manifest
from modal_app import app, image


project_volume = modal.Volume.from_name("project")

repack_image = image.add_local_python_source("things_manifest", "modal_app")

PROJECT_MOUNT = "/project"

# Paths on the volume — single source of truth.
SRC_RGB_DIR = "/project/data/train/things/rgb"      # current 27 shards
DST_TRAIN_RGB = "/project/data/train/things/rgb"    # train destination (same dir)
DST_VAL_RGB = "/project/data/val/things/rgb"        # val destination (new)
STAGING_ROOT = "/project/data/staging"
STAGING_TRAIN = f"{STAGING_ROOT}/things/rgb/train"
STAGING_VAL = f"{STAGING_ROOT}/things/rgb/val"

CATALOG_PATH = "/project/data/things_catalog.json"
TRAIN_MANIFEST = "/project/data/train/things_manifest.json"
VAL_MANIFEST = "/project/data/val/things_manifest.json"
PLAN_PATH = f"{STAGING_ROOT}/repack_plan.json"

VAL_FRAC = 0.15
SEED = 0


def _list_source_shards(dir_: str) -> list[str]:
    if not os.path.isdir(dir_):
        return []
    return sorted(
        os.path.join(dir_, f)
        for f in os.listdir(dir_)
        if things_manifest.SHARD_NAME_RE.match(f)
    )


@app.function(
    image=repack_image,
    volumes={PROJECT_MOUNT: project_volume},
    cpu=4.0,
    memory=8 * 1024,
    timeout=60 * 30,
)
def plan_remote() -> dict:
    """Phase 1: compute split + write JSONs + plan. Tar files untouched."""
    src_paths = _list_source_shards(SRC_RGB_DIR)
    if not src_paths:
        raise FileNotFoundError(f"No shard_*.tar found in {SRC_RGB_DIR}")
    print(f"[plan] {len(src_paths)} source shards in {SRC_RGB_DIR}")

    id_to_fn: dict[str, str] = {}
    for p in src_paths:
        contents = things_manifest.extract_shard_contents(p)
        print(f"  {os.path.basename(p)}: {len(contents)} images")
        for image_id, fn in contents.items():
            prior = id_to_fn.get(image_id)
            if prior is not None and prior != fn:
                raise ValueError(
                    f"image_id {image_id} disagrees across shards: "
                    f"{prior!r} vs {fn!r}"
                )
            id_to_fn[image_id] = fn
    print(f"[plan] total unique images: {len(id_to_fn)}")

    catalog = things_manifest.build_catalog(id_to_fn)
    train_ids, val_ids = things_manifest.image_level_split(
        id_to_fn.keys(), val_frac=VAL_FRAC, seed=SEED
    )
    train_shards = things_manifest.pack_into_shards(train_ids)
    val_shards = things_manifest.pack_into_shards(val_ids)
    train_manifest = things_manifest.build_split_manifest(
        "train", train_shards, "things/rgb"
    )
    val_manifest = things_manifest.build_split_manifest(
        "val", val_shards, "things/rgb"
    )
    total = train_manifest["n_images"] + val_manifest["n_images"]
    if total != catalog["n_images"]:
        raise RuntimeError(
            f"split mismatch: train+val={total} != catalog={catalog['n_images']}"
        )

    for p in (CATALOG_PATH, TRAIN_MANIFEST, VAL_MANIFEST, PLAN_PATH):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(CATALOG_PATH, "w") as fh:
        json.dump(catalog, fh, indent=2)
    with open(TRAIN_MANIFEST, "w") as fh:
        json.dump(train_manifest, fh, indent=2)
    with open(VAL_MANIFEST, "w") as fh:
        json.dump(val_manifest, fh, indent=2)

    plan_payload = {
        "version": "1",
        "val_frac": VAL_FRAC,
        "seed": SEED,
        "src_paths": src_paths,
        "src_n_shards": len(src_paths),
        "n_images": catalog["n_images"],
        "train_shard_paths": [
            os.path.join(DST_TRAIN_RGB, f"shard_{n:03d}.tar")
            for n in range(len(train_shards))
        ],
        "val_shard_paths": [
            os.path.join(DST_VAL_RGB, f"shard_{n:03d}.tar")
            for n in range(len(val_shards))
        ],
    }
    with open(PLAN_PATH, "w") as fh:
        json.dump(plan_payload, fh, indent=2)

    print(f"[plan] wrote {CATALOG_PATH}")
    print(
        f"[plan] wrote {TRAIN_MANIFEST}  "
        f"({train_manifest['n_shards']} shards, {train_manifest['n_images']} imgs)"
    )
    print(
        f"[plan] wrote {VAL_MANIFEST}  "
        f"({val_manifest['n_shards']} shards, {val_manifest['n_images']} imgs)"
    )
    print(f"[plan] wrote {PLAN_PATH}")
    project_volume.commit()

    return {
        "src_n_shards": len(src_paths),
        "n_images": catalog["n_images"],
        "train_n_shards": train_manifest["n_shards"],
        "train_n_images": train_manifest["n_images"],
        "val_n_shards": val_manifest["n_shards"],
        "val_n_images": val_manifest["n_images"],
    }


@app.function(
    image=repack_image,
    volumes={PROJECT_MOUNT: project_volume},
    cpu=4.0,
    memory=16 * 1024,
    timeout=60 * 90,
)
def repack_remote() -> dict:
    """Phase 2: build new shards in staging, validate, atomic swap into place."""
    if not os.path.exists(PLAN_PATH):
        raise FileNotFoundError(f"Run plan first; {PLAN_PATH} missing")
    with open(PLAN_PATH) as fh:
        plan_payload = json.load(fh)
    with open(TRAIN_MANIFEST) as fh:
        train_manifest = json.load(fh)
    with open(VAL_MANIFEST) as fh:
        val_manifest = json.load(fh)

    src_paths = plan_payload["src_paths"]
    missing_src = [p for p in src_paths if not os.path.exists(p)]
    expected_staged = train_manifest["n_shards"] + val_manifest["n_shards"]

    if missing_src:
        # Sources already gone (resume after partial swap).  Only OK if
        # staging is complete; otherwise we can't reconstruct.
        n_staged = 0
        for d in (STAGING_TRAIN, STAGING_VAL):
            if os.path.isdir(d):
                n_staged += len(
                    [f for f in os.listdir(d) if f.endswith(".tar")]
                )
        if n_staged != expected_staged:
            raise FileNotFoundError(
                f"sources missing ({len(missing_src)} gone) and staging "
                f"incomplete ({n_staged}/{expected_staged}). Aborting."
            )
        print("[repack] sources gone but staging complete; jumping to swap")
    else:
        print(f"[repack] indexing {len(src_paths)} source shards...")
        src_index = things_manifest.index_source_shards(src_paths)
        print(f"[repack] indexed {len(src_index)} image_ids")

        os.makedirs(STAGING_TRAIN, exist_ok=True)
        os.makedirs(STAGING_VAL, exist_ok=True)

        for split_name, manifest, staging in (
            ("train", train_manifest, STAGING_TRAIN),
            ("val", val_manifest, STAGING_VAL),
        ):
            for shard_id, info in manifest["shards"].items():
                out_path = os.path.join(staging, f"{shard_id}.tar")
                if os.path.exists(out_path):
                    print(f"[repack] {split_name}/{shard_id}: exists, skip")
                    continue
                things_manifest.write_shard_from_locations(
                    out_path, info["image_ids"], src_index
                )
                print(
                    f"[repack] {split_name}/{shard_id}: "
                    f"{info['n_images']} imgs, "
                    f"{os.path.getsize(out_path) / 1e6:.1f} MB"
                )
        project_volume.commit()

    # Validate staging is complete before destructive ops.
    n_train_staged = len(
        [f for f in os.listdir(STAGING_TRAIN) if f.endswith(".tar")]
    )
    n_val_staged = len(
        [f for f in os.listdir(STAGING_VAL) if f.endswith(".tar")]
    )
    if n_train_staged != train_manifest["n_shards"]:
        raise RuntimeError(
            f"staged {n_train_staged} train tars, manifest expects "
            f"{train_manifest['n_shards']}"
        )
    if n_val_staged != val_manifest["n_shards"]:
        raise RuntimeError(
            f"staged {n_val_staged} val tars, manifest expects "
            f"{val_manifest['n_shards']}"
        )
    print(f"[repack] staging OK: {n_train_staged} train + {n_val_staged} val shards")

    # Swap into place.
    print("[repack] swapping into place...")
    surviving_sources = [p for p in src_paths if os.path.exists(p)]
    for p in surviving_sources:
        os.remove(p)
        print(f"[repack]   rm {p}")
    os.makedirs(DST_TRAIN_RGB, exist_ok=True)
    os.makedirs(DST_VAL_RGB, exist_ok=True)
    for f in sorted(os.listdir(STAGING_TRAIN)):
        os.replace(
            os.path.join(STAGING_TRAIN, f), os.path.join(DST_TRAIN_RGB, f)
        )
    for f in sorted(os.listdir(STAGING_VAL)):
        os.replace(
            os.path.join(STAGING_VAL, f), os.path.join(DST_VAL_RGB, f)
        )
    shutil.rmtree(STAGING_ROOT, ignore_errors=True)
    project_volume.commit()

    return {
        "train_n_shards": n_train_staged,
        "val_n_shards": n_val_staged,
        "sources_removed": len(surviving_sources),
    }


@app.function(
    image=repack_image,
    volumes={PROJECT_MOUNT: project_volume},
    cpu=2.0,
    memory=4 * 1024,
    timeout=60 * 20,
)
def verify_remote() -> dict:
    """Phase 3: confirm on-disk train/val matches manifests."""
    with open(CATALOG_PATH) as fh:
        catalog = json.load(fh)
    with open(TRAIN_MANIFEST) as fh:
        train_manifest = json.load(fh)
    with open(VAL_MANIFEST) as fh:
        val_manifest = json.load(fh)
    print(f"[verify] catalog: {catalog['n_images']} images")
    print(
        f"[verify] train manifest: {train_manifest['n_shards']} shards, "
        f"{train_manifest['n_images']} images"
    )
    print(
        f"[verify] val manifest:   {val_manifest['n_shards']} shards, "
        f"{val_manifest['n_images']} images"
    )

    for split_name, manifest, dst_dir in (
        ("train", train_manifest, DST_TRAIN_RGB),
        ("val", val_manifest, DST_VAL_RGB),
    ):
        on_disk = sorted(f for f in os.listdir(dst_dir) if f.endswith(".tar"))
        expected = [f"{s}.tar" for s in sorted(manifest["shards"].keys())]
        if on_disk != expected:
            raise RuntimeError(
                f"{split_name} dir mismatch: on_disk={on_disk[:5]}..., "
                f"expected={expected[:5]}..."
            )
        # Spot-check first + last shard per split.
        sample_ids = sorted(manifest["shards"].keys())
        for shard_id in [sample_ids[0], sample_ids[-1]]:
            p = os.path.join(dst_dir, f"{shard_id}.tar")
            actual = things_manifest.extract_shard_contents(p)
            expected_ids = set(manifest["shards"][shard_id]["image_ids"])
            if set(actual.keys()) != expected_ids:
                missing = list(expected_ids - set(actual.keys()))[:3]
                extra = list(set(actual.keys()) - expected_ids)[:3]
                raise RuntimeError(
                    f"{split_name}/{shard_id}: missing={missing}, extra={extra}"
                )
            print(f"[verify] {split_name}/{shard_id}: OK ({len(actual)} images)")
    print("[verify] all checks passed")
    return {
        "train_n_shards": train_manifest["n_shards"],
        "val_n_shards": val_manifest["n_shards"],
        "n_images": catalog["n_images"],
    }


@app.local_entrypoint()
def plan():
    summary = plan_remote.remote()
    print("\n[plan] summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


@app.local_entrypoint()
def repack():
    summary = repack_remote.remote()
    print("\n[repack] summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


@app.local_entrypoint()
def verify():
    summary = verify_remote.remote()
    print("\n[verify] summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
