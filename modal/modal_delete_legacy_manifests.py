"""Remove outdated train/val manifest JSONs from the project Volume.

Run from modal/:
    modal run modal_delete_legacy_manifests.py::delete
"""

from __future__ import annotations

import os

import modal

app = modal.App("delete-legacy-manifests")
project_volume = modal.Volume.from_name("project")
PROJECT = "/project"

LEGACY_MANIFESTS = (
    f"{PROJECT}/data/train/things_manifest.json",
    f"{PROJECT}/data/val/things_manifest.json",
    f"{PROJECT}/data/train/things_meg_manifest.json",
    f"{PROJECT}/data/val/things_meg_manifest.json",
    f"{PROJECT}/data/train/things_eeg_manifest.json",
    f"{PROJECT}/data/val/things_eeg_manifest.json",
)


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={PROJECT: project_volume},
    timeout=60 * 5,
)
def list_remaining() -> list[str]:
    hits: list[str] = []
    for root, _dirs, files in os.walk(f"{PROJECT}/data"):
        for fname in files:
            if "manifest" in fname.lower():
                hits.append(os.path.join(root, fname))
    for path in sorted(hits):
        print(path)
    return hits


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={PROJECT: project_volume},
    timeout=60 * 5,
)
def delete() -> dict:
    removed: list[str] = []
    missing: list[str] = []
    for path in LEGACY_MANIFESTS:
        if os.path.isfile(path):
            os.remove(path)
            removed.append(path)
            print(f"[delete] removed {path}")
        else:
            missing.append(path)
            print(f"[delete] not found (skip) {path}")
    project_volume.commit()
    return {"removed": removed, "missing": missing}


@app.local_entrypoint()
def main():
    print(delete.remote())


@app.local_entrypoint()
def list_manifests():
    print(list_remaining.remote())
