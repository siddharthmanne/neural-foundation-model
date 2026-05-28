"""Modal GPU wrapper for ``train_4m.py`` — the entry point for real training runs.

This is a thin launcher: it starts a Modal container and runs train_4m.py inside it.
Shared setup (deps, repo mount, pip install 4M) lives in modal_image.py.

Run from repo root (this is THE training command)::

    modal run 4m_training/modal/modal_train.py --config 4m_training/configs/4m_things_main.yaml

The shipped ``4m_things_main.yaml`` / ``4m_things_data.yaml`` train vision PLUS the neural
modalities (``tok_meg_rvq0..3`` + ``tok_eeg``) symmetrically — brain signals are both encoder
context and reconstruction targets, as a regularizer. ``find_unused_params: true`` is already
in the main YAML (required because a head can get 0 targets on a step), so no extra flag is
needed. See notes/4m_neural_modality_design.md.

Dryrun on the volume (no GPU, no training — fast config/data check)::

    modal run 4m_training/modal/modal_train.py --dryrun --config 4m_training/configs/4m_things_data.yaml

Before paying for a GPU, prove the neural heads learn (every head's loss must descend)::

    modal run 4m_training/modal/modal_smoke_train.py --case neural_heads_descend

Image rebuilds only when ``modal_image.py`` pip/apt deps change, not when you edit YAML or Python.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import importlib.util
import modal


def _load_modal_image():
    """Load shared image config — works locally and inside Modal containers."""
    for path in (
        __import__("pathlib").Path("/opt/repo/4m_training/modal/_modal_load.py"),
        __import__("pathlib").Path(__file__).resolve().parent / "_modal_load.py",
    ):
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_modal_load", path)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        return loader.load_modal_image()
    raise ImportError("_modal_load.py not found (expected /opt/repo/4m_training on Modal)")


# Pull shared settings from modal_image.py (paths, image, env, ensure_fourm).
_mi = _load_modal_image()
REPO = _mi.REPO
ensure_fourm = _mi.ensure_fourm
train_image = _mi.train_image
training_env = _mi.training_env

# Modal app name (shows up in the Modal dashboard).
app = modal.App("train-4m-things")

# Persistent storage for data shards and checkpoints (name configurable in modal_image.py).
project_volume = modal.Volume.from_name(_mi.PROJECT_VOLUME_NAME)
PROJECT = _mi.PROJECT_MOUNT


def _run_train(config: str, dryrun: bool, n_batches: int) -> None:
    """Install 4M if needed, then run train_4m.py as a subprocess."""
    ensure_fourm()

    # Config paths in YAML are relative to repo root.
    cfg = config
    if not os.path.isabs(cfg):
        cfg = os.path.join(REPO, cfg)

    # This is the actual command that runs inside the container.
    cmd = [
        sys.executable,
        os.path.join(REPO, "4m_training/lib/train_4m.py"),
        "dryrun" if dryrun else "train",
        "--config",
        cfg,
    ]
    if dryrun:
        cmd.extend(["--n-batches", str(n_batches)])

    env = training_env()
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},  # mount volume at /project
    gpu="L40S",
    timeout=60 * 60 * 24,
    memory=64 * 1024,
)
def train(config: str = "4m_training/configs/4m_things_main.yaml") -> None:
    """Full GPU training job."""
    _run_train(config, dryrun=False, n_batches=4)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    timeout=60 * 30,
    memory=32 * 1024,
    # No gpu= — runs on CPU only (cheaper sanity check).
)
def dryrun_job(
    config: str = "4m_training/configs/4m_things_data.yaml",
    n_batches: int = 4,
) -> None:
    """CPU-only dataloader smoke test — use this before paying for GPU train."""
    _run_train(config, dryrun=True, n_batches=n_batches)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    gpu="T4",
    timeout=60 * 60,
    memory=32 * 1024,
)
def validate_job(
    config: str = "4m_training/configs/4m_things_main.yaml",
    tasks: str = "4m_training/configs/4m_things_val_tasks.yaml",
    select: str = "",
    checkpoint: str = "",
    n_batches: int = 0,
) -> None:
    """Run named validation tasks on the project volume (optionally from a checkpoint)."""
    ensure_fourm()
    cmd = [
        sys.executable, os.path.join(REPO, "4m_training/validate_4m.py"),
        "--config", os.path.join(REPO, config),
        "--tasks", os.path.join(REPO, tasks),
        "--n-batches", str(n_batches),
    ]
    if select:
        cmd += ["--select", select]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=training_env())


@app.local_entrypoint()
def main(
    config: str = "4m_training/configs/4m_things_main.yaml",
    dryrun: bool = False,
    validate: bool = False,
    tasks: str = "4m_training/configs/4m_things_val_tasks.yaml",
    select: str = "",
    checkpoint: str = "",
    n_batches: int = 4,
) -> None:
    """Called by `modal run` on your laptop — dispatches to a remote Modal function.

    Default (no flags) = full GPU training on the main config. Examples::
        modal run 4m_training/modal/modal_train.py --config 4m_training/configs/4m_things_main.yaml
        modal run 4m_training/modal/modal_train.py --dryrun --config 4m_training/configs/4m_things_data.yaml
        modal run 4m_training/modal/modal_train.py --validate --select rgb2depth,anyany_neural
        modal run 4m_training/modal/modal_train.py --validate --checkpoint /project/runs/.../checkpoint-last.pth
    """
    # Flags must not be named after a Modal function (e.g. `dryrun_job`/`validate_job`).
    if validate:
        validate_job.remote(config=config, tasks=tasks, select=select,
                            checkpoint=checkpoint, n_batches=n_batches)
    elif dryrun:
        dryrun_job.remote(config=config, n_batches=n_batches)
    else:
        train.remote(config=config)
