"""Deep audit of repacked THINGS 4M catalog-slot layout on the project Volume.

Run from modal/:
    modal run modal_audit_things_4m.py::audit
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from collections import Counter, defaultdict

import modal

import things_manifest

app = modal.App("audit-things-4m")
project_volume = modal.Volume.from_name("project")
PROJECT = "/project"

audit_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .add_local_python_source("things_manifest")
)

THINGS_SPLIT = f"{PROJECT}/data/things_split.json"
MODALITIES = ("tok_rgb", "tok_depth", "tok_meg", "tok_eeg", "meg_mask", "eeg_mask")
SLOTS = [f"shard_{i:03d}" for i in range(27)]
MEG_SENTINEL = (1, 16, 8, 4)
EEG_SENTINEL = (1, 17)


def _dst(split: str, modality: str) -> str:
    return f"{PROJECT}/data/{split}/things/{modality}"


def _tar_keys(path: str) -> list[str]:
    with tarfile.open(path, "r") as tar:
        return sorted(m.name[:-4] for m in tar if m.name.endswith(".npy"))


def _load_npy(tar_path: str, image_id: str):
    import numpy as np

    with tarfile.open(tar_path, "r") as tar:
        m = tar.getmember(f"{image_id}.npy")
        f = tar.extractfile(m)
        assert f is not None
        return np.load(io.BytesIO(f.read()), allow_pickle=False)


@app.function(
    image=audit_image,
    volumes={PROJECT: project_volume},
    cpu=4.0,
    memory=16 * 1024,
    timeout=60 * 30,
)
def audit() -> dict:
    import numpy as np

    split = json.loads(open(THINGS_SPLIT).read())
    train_ids = set(split["train_image_ids"])
    val_ids = set(split["val_image_ids"])
    catalog = train_ids | val_ids

    report: dict = {
        "split": {
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_catalog": len(catalog),
            "expected_train": split["n_train"],
            "expected_val": split["n_val"],
        },
        "layout": {},
        "slot_assignment": {"errors": []},
        "cross_modality": {"errors": []},
        "content": {},
        "mask_consistency": {"errors": []},
        "summary": {},
    }

    for split_name, expected in (("train", train_ids), ("val", val_ids)):
        expected_pack = things_manifest.pack_by_catalog_slot(expected)
        seen: set[str] = set()
        split_info: dict = {"modalities": {}, "shard_counts": {}}

        for mod in MODALITIES:
            mod_dir = _dst(split_name, mod)
            tars = sorted(f for f in os.listdir(mod_dir) if f.endswith(".tar"))
            split_info["modalities"][mod] = {
                "n_tars": len(tars),
                "tar_names": tars[:3] + (["..."] if len(tars) > 3 else []),
                "last_tar": tars[-1] if tars else None,
            }
            if tars != [f"{s}.tar" for s in SLOTS]:
                report["layout"].setdefault("errors", []).append(
                    f"{split_name}/{mod}: expected 27 shard_000..026.tar, got {len(tars)}"
                )

        for slot in SLOTS:
            split_info["shard_counts"][slot] = {}
            keys_per_mod: dict[str, list[str]] = {}
            for mod in MODALITIES:
                p = os.path.join(_dst(split_name, mod), f"{slot}.tar")
                if not os.path.isfile(p):
                    report["layout"].setdefault("errors", []).append(
                        f"missing {p}"
                    )
                    continue
                keys = _tar_keys(p)
                keys_per_mod[mod] = keys
                split_info["shard_counts"][slot][mod] = len(keys)

            if not keys_per_mod:
                continue

            ref = next(iter(keys_per_mod.values()))
            for mod, keys in keys_per_mod.items():
                if keys != ref:
                    report["cross_modality"]["errors"].append(
                        f"{split_name}/{slot}: {mod} keys != tok_rgb"
                    )

            expected_ids = expected_pack.get(slot, [])
            if ref != sorted(expected_ids):
                report["cross_modality"]["errors"].append(
                    f"{split_name}/{slot}: keys != things_split pack ({len(ref)} vs {len(expected_ids)})"
                )

            for image_id in ref:
                if things_manifest.catalog_slot_id(image_id) != slot:
                    report["slot_assignment"]["errors"].append(
                        f"{image_id} in {split_name}/{slot} but slot should be "
                        f"{things_manifest.catalog_slot_id(image_id)}"
                    )
            seen.update(ref)

        if seen != expected:
            report["cross_modality"]["errors"].append(
                f"{split_name}: union of keys ({len(seen)}) != split ({len(expected)})"
            )

        report["layout"][split_name] = split_info

    # Content sampling: every modality on a few representative ids
    samples = {
        "train": ["000000001", "000001000", "000001001", "000026107"],
        "val": sorted(val_ids)[:2],
    }
    content: dict = {}
    for split_name, ids in samples.items():
        content[split_name] = {}
        for image_id in ids:
            if image_id not in (train_ids if split_name == "train" else val_ids):
                continue
            slot = things_manifest.catalog_slot_id(image_id)
            row: dict = {"slot": slot}
            for mod in MODALITIES:
                p = os.path.join(_dst(split_name, mod), f"{slot}.tar")
                arr = _load_npy(p, image_id)
                row[mod] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            content[split_name][image_id] = row
    report["content"]["samples"] = content

    # Full content scan on shard_000 train (spot-check all arrays)
    shape_counts: dict[str, Counter] = {m: Counter() for m in MODALITIES}
    dtype_counts: dict[str, Counter] = {m: Counter() for m in MODALITIES}
    meg_trial_counts: Counter = Counter()
    eeg_trial_counts: Counter = Counter()
    rgb_bad = depth_bad = 0
    meg_mask_bad = eeg_mask_bad = 0
    meg_sentinel_count = meg_real_count = 0
    eeg_sentinel_count = eeg_real_count = 0

    for split_name, expected in (("train", train_ids), ("val", val_ids)):
        expected_pack = things_manifest.pack_by_catalog_slot(expected)
        for slot, image_ids in sorted(expected_pack.items()):
            paths = {m: os.path.join(_dst(split_name, m), f"{slot}.tar") for m in MODALITIES}
            for image_id in image_ids:
                rgb = _load_npy(paths["tok_rgb"], image_id)
                depth = _load_npy(paths["tok_depth"], image_id)
                meg = _load_npy(paths["tok_meg"], image_id)
                eeg = _load_npy(paths["tok_eeg"], image_id)
                mm = _load_npy(paths["meg_mask"], image_id)
                em = _load_npy(paths["eeg_mask"], image_id)

                shape_counts["tok_rgb"][tuple(rgb.shape)] += 1
                shape_counts["tok_depth"][tuple(depth.shape)] += 1
                shape_counts["tok_meg"][tuple(meg.shape)] += 1
                shape_counts["tok_eeg"][tuple(eeg.shape)] += 1
                shape_counts["meg_mask"][tuple(mm.shape)] += 1
                shape_counts["eeg_mask"][tuple(em.shape)] += 1
                dtype_counts["tok_rgb"][str(rgb.dtype)] += 1
                dtype_counts["tok_depth"][str(depth.dtype)] += 1
                dtype_counts["tok_meg"][str(meg.dtype)] += 1
                dtype_counts["tok_eeg"][str(eeg.dtype)] += 1
                dtype_counts["meg_mask"][str(mm.dtype)] += 1
                dtype_counts["eeg_mask"][str(em.dtype)] += 1

                if rgb.shape != (196,) or rgb.dtype != np.int16:
                    rgb_bad += 1
                if depth.shape != (196,) or depth.dtype != np.int16:
                    depth_bad += 1
                if mm.shape != (1,) or mm.dtype != np.uint8 or int(mm[0]) not in (0, 1):
                    meg_mask_bad += 1
                if em.shape != (1,) or em.dtype != np.uint8 or int(em[0]) not in (0, 1):
                    eeg_mask_bad += 1

                is_meg_sentinel = tuple(meg.shape) == MEG_SENTINEL and meg.dtype == np.int16
                if is_meg_sentinel and np.all(meg == -1):
                    meg_sentinel_count += 1
                    if int(mm[0]) != 0:
                        report["mask_consistency"]["errors"].append(
                            f"{split_name}/{image_id}: meg sentinel but mask={int(mm[0])}"
                        )
                elif meg.ndim == 3 and meg.shape[1:] == (16, 8, 4) and meg.dtype == np.int16:
                    meg_real_count += 1
                    meg_trial_counts[meg.shape[0]] += 1
                    if int(mm[0]) != 1:
                        report["mask_consistency"]["errors"].append(
                            f"{split_name}/{image_id}: meg real but mask={int(mm[0])}"
                        )
                else:
                    report["mask_consistency"]["errors"].append(
                        f"{split_name}/{image_id}: bad meg shape/dtype {meg.shape} {meg.dtype}"
                    )

                is_eeg_sentinel = tuple(eeg.shape) == EEG_SENTINEL and eeg.dtype == np.int16
                if is_eeg_sentinel and np.all(eeg == -1):
                    eeg_sentinel_count += 1
                    if int(em[0]) != 0:
                        report["mask_consistency"]["errors"].append(
                            f"{split_name}/{image_id}: eeg sentinel but mask={int(em[0])}"
                        )
                elif eeg.ndim == 2 and eeg.shape[1] == 17 and eeg.dtype == np.int16:
                    eeg_real_count += 1
                    eeg_trial_counts[eeg.shape[0]] += 1
                    if int(em[0]) != 1:
                        report["mask_consistency"]["errors"].append(
                            f"{split_name}/{image_id}: eeg real but mask={int(em[0])}"
                        )
                else:
                    report["mask_consistency"]["errors"].append(
                        f"{split_name}/{image_id}: bad eeg shape/dtype {eeg.shape} {eeg.dtype}"
                    )

    report["content"]["shape_counts"] = {k: dict(v) for k, v in shape_counts.items()}
    report["content"]["dtype_counts"] = {k: dict(v) for k, v in dtype_counts.items()}
    report["content"]["meg_trial_hist_top10"] = meg_trial_counts.most_common(10)
    report["content"]["eeg_trial_hist_top10"] = eeg_trial_counts.most_common(10)
    report["content"]["bad_counts"] = {
        "rgb_bad": rgb_bad,
        "depth_bad": depth_bad,
        "meg_mask_bad": meg_mask_bad,
        "eeg_mask_bad": eeg_mask_bad,
    }
    report["summary"] = {
        "meg_real": meg_real_count,
        "meg_sentinel": meg_sentinel_count,
        "eeg_real": eeg_real_count,
        "eeg_sentinel": eeg_sentinel_count,
        "total_images": meg_real_count + meg_sentinel_count,
        "n_errors": sum(
            len(report[k].get("errors", []))
            for k in ("layout", "slot_assignment", "cross_modality", "mask_consistency")
        ),
        "ok": sum(
            len(report[k].get("errors", []))
            for k in ("layout", "slot_assignment", "cross_modality", "mask_consistency")
        )
        == 0,
    }

    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main():
    audit.remote()
