"""Shared Modal image for 4M training.

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

REPO = "/opt/repo"
_REPO_ROOT = "/home/users/liubr/projects/neural-image-foundation/neural-foundation-model/" #Path(__file__).resolve().parent.parent

# Python 3.10, not 3.11+: stock 4M's decoder forward calls
# random.sample(mod_dict.items(), ...) (fourm/models/fm.py), and random.sample
# rejected dict views/sets starting in 3.11. 4M's pyproject says ">=3.8" but is
# only exercised on 3.8-3.10. Bumping Python crashes training, not import/dryrun.
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

train_image = _deps_image.add_local_dir(str(_REPO_ROOT), remote_path=REPO)

_FOURM_READY = "/tmp/.fourm_editable_ok"


def ensure_fourm() -> None:
    """Editable-install fourm once per container (fast after first call in that container)."""
    if os.path.isfile(_FOURM_READY):
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}/ml-4m"],
        check=True,
    )
    Path(_FOURM_READY).write_text("ok\n")


def training_env() -> dict[str, str]:
    """Env for single-GPU runs (4M trainer always wraps DDP; needs torchrun-style vars)."""
    return {
        **os.environ,
        "PYTHONPATH": f"{REPO}/4m_training/lib:{REPO}/4m_training:{REPO}/ml-4m",
        "FOURM_ML4M_DIR": f"{REPO}/ml-4m",
        "PYTHONUNBUFFERED": "1",
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }
