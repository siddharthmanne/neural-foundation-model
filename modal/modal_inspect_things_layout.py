"""Inspect current THINGS modality layouts on the project Volume.

Run from modal/:
    modal run modal_inspect_things_layout.py::inspect
"""

from __future__ import annotations

import io
import json
import os
import tarfile

import modal

app = modal.App("inspect-things-layout")
project_volume = modal.Volume.from_name("project")
PROJECT = "/project"

inspect_image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")


def _sample_tar(path: str, n: int = 8) -> list[str]:
    if not os.path.isfile(path):
        return [f"<missing: {path}>"]
    names: list[str] = []
    with tarfile.open(path, "r") as tar:
        for member in tar:
            names.append(member.name)
            if len(names) >= n:
                break
    return names


def _count_tar(path: str) -> int:
    if not os.path.isfile(path):
        return -1
    with tarfile.open(path, "r") as tar:
        return sum(1 for _ in tar)


def _inspect_npy_member(tar_path: str, member_name: str) -> str:
    import numpy as np

    with tarfile.open(tar_path, "r") as tar:
        m = tar.getmember(member_name)
        f = tar.extractfile(m)
        if f is None:
            return "extract failed"
        arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
        return f"shape={arr.shape} dtype={arr.dtype}"


@app.function(
    image=inspect_image,
    volumes={PROJECT: project_volume},
    timeout=60 * 10,
)
def inspect() -> dict:
    out: dict = {}

    splits = {
        "train": f"{PROJECT}/data/train/things",
        "val": f"{PROJECT}/data/val/things",
    }
    modalities = ["rgb", "tok_meg", "tok_eeg", "tok_rgb@224", "tok_depth@224"]

    for split, root in splits.items():
        out[split] = {}
        for mod in modalities:
            mod_dir = os.path.join(root, mod)
            if not os.path.isdir(mod_dir):
                out[split][mod] = {"status": "missing"}
                continue
            entries = sorted(os.listdir(mod_dir))
            tars = [e for e in entries if e.endswith(".tar")]
            loose = [e for e in entries if not e.endswith(".tar")]
            info: dict = {
                "n_tars": len(tars),
                "n_loose": len(loose),
                "tar_names_sample": tars[:3] + (["..."] if len(tars) > 3 else []),
            }
            if tars:
                first = os.path.join(mod_dir, tars[0])
                last = os.path.join(mod_dir, tars[-1])
                info["first_tar"] = tars[0]
                info["last_tar"] = tars[-1]
                info["first_tar_n_members"] = _count_tar(first)
                info["first_tar_members_sample"] = _sample_tar(first, 6)
                info["last_tar_n_members"] = _count_tar(last)
                if info["first_tar_members_sample"] and not info[
                    "first_tar_members_sample"
                ][0].startswith("<missing"):
                    m0 = info["first_tar_members_sample"][0]
                    if m0.endswith(".npy") or ".npy" in m0 or m0.endswith(".jpg"):
                        try:
                            info["first_member_array"] = _inspect_npy_member(
                                first, m0
                            )
                        except Exception as e:
                            info["first_member_array"] = f"err: {e}"
            if loose:
                info["loose_sample"] = sorted(loose)[:6]
            out[split][mod] = info
            print(f"\n[{split}/{mod}] {json.dumps(info, indent=2)}")

    # Manifest / split summaries
    for path in (
        f"{PROJECT}/data/things_split.json",
        f"{PROJECT}/data/things_catalog.json",
        f"{PROJECT}/data/train/things_manifest.json",
        f"{PROJECT}/data/val/things_manifest.json",
    ):
        if os.path.isfile(path):
            payload = json.loads(open(path).read())
            if "train_image_ids" in payload:
                print(
                    f"\n[things_split] train={payload['n_train']} "
                    f"val={payload['n_val']} policy={payload.get('policy')}"
                )
            elif "n_images" in payload:
                print(f"\n[{os.path.basename(path)}] n_images={payload['n_images']}")
            elif "n_shards" in payload:
                print(
                    f"\n[{os.path.basename(path)}] "
                    f"split={payload.get('split')} "
                    f"shards={payload['n_shards']} imgs={payload['n_images']}"
                )

    return out


@app.local_entrypoint()
def main():
    inspect.remote()
