"""Sync modal/data/README.md to /project/data/README.md on the project Volume.

Run from modal/:
    modal run modal_sync_data_readme.py::sync
"""

from __future__ import annotations

import shutil

import modal

app = modal.App("sync-data-readme")
project_volume = modal.Volume.from_name("project")
PROJECT = "/project"

sync_image = modal.Image.debian_slim(python_version="3.11").add_local_file(
    "data/README.md",
    remote_path="/_sync/README.md",
)


@app.function(
    image=sync_image,
    volumes={PROJECT: project_volume},
    timeout=60 * 2,
)
def sync() -> dict:
    dst = f"{PROJECT}/data/README.md"
    shutil.copyfile("/_sync/README.md", dst)
    project_volume.commit()
    print(f"[sync] wrote {dst}")
    return {"path": dst, "ok": True}


@app.local_entrypoint()
def main():
    print(sync.remote())
