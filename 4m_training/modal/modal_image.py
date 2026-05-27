"""Shared Modal image for 4M training.

This file defines the container environment used by all modal_*.py scripts.
Edit this once when you change deps or paths — copy modal_train.py for new jobs.

Rebuild triggers (slow):
  - Changing pip/apt packages below
  - Changing Python version

Does NOT rebuild when you edit repo code (configs, train_4m.py, etc.):
  - Repo is mounted at container start via ``add_local_dir`` (no ``copy=True``)
  - ``fourm`` is installed editable once per container at runtime, not in the image
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# Where your repo appears inside the Modal container.
REPO = "/opt/repo"

# This file is at <repo>/4m_training/modal/modal_image.py.
_MODAL_DIR = Path(__file__).resolve().parent
_TRAINING_DIR = _MODAL_DIR.parent            # 4m_training
_REPO_ROOT = _TRAINING_DIR.parent            # repo root (mounted into the container)

# Single source of truth for paths: lib/repo_paths.py (edit there, not here).
sys.path.insert(0, str(_TRAINING_DIR / "lib"))
import repo_paths  # noqa: E402

PROJECT_VOLUME_NAME = repo_paths.PROJECT_VOLUME_NAME   # your Modal volume name
PROJECT_MOUNT = "/project"                             # mount path is fixed

# Step 1: build a base image with system + Python packages.
# Changing anything here triggers a slow image rebuild.
# Stick to Python 3.10 — stock 4M breaks on 3.11+ (random.sample on dict views).
_deps_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy<2",
        "torch",
        "torchvision",
        "webdataset",
        "pyyaml",
        "tokenizers",
        "einops",
        "timm",
        "opencv-python-headless",
    )
)

# Your local 4M checkout — taken from repo_paths (set ML4M_DIR_OVERRIDE there).
ML4M_LOCAL = repo_paths.ML4M_DIR

# Where 4M lands in the container, and whether it needs its own mount.
try:
    _rel = ML4M_LOCAL.relative_to(_REPO_ROOT)        # 4M lives inside the repo (default)
    ML4M_CONTAINER = f"{REPO}/{_rel.as_posix()}"
    _ml4m_outside_repo = False
except ValueError:                                   # 4M is elsewhere — mount it separately
    ML4M_CONTAINER = "/opt/ml-4m"
    _ml4m_outside_repo = True

# Step 2: attach your local repo to the container at /opt/repo.
# Syncs at container start — code edits on your machine show up without rebuilding.
train_image = _deps_image.add_local_dir(str(_REPO_ROOT), remote_path=REPO)
if _ml4m_outside_repo:
    train_image = train_image.add_local_dir(str(ML4M_LOCAL), remote_path=ML4M_CONTAINER)

# Marker file so we only pip-install fourm once per container.
_FOURM_READY = "/tmp/.fourm_editable_ok"


def ensure_fourm() -> None:
    """Make the 4M package importable inside the container."""
    if os.path.isfile(_FOURM_READY):
        return  # already installed on this container
    # Editable install from wherever 4M was mounted (inside the repo, or its own mount).
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", ML4M_CONTAINER],
        check=True,
    )
    Path(_FOURM_READY).write_text("ok\n")


def training_env() -> dict[str, str]:
    """Environment variables passed to the training subprocess."""
    return {
        **os.environ,
        # So Python finds our library modules (4m_training/lib) and upstream 4M.
        "PYTHONPATH": f"{REPO}/4m_training/lib:{ML4M_CONTAINER}",
        # Tell the in-container subprocess (repo_paths) where 4M is.
        "FOURM_ML4M_DIR": ML4M_CONTAINER,
        # 4M always uses DDP; these fake a single-GPU run.
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }
