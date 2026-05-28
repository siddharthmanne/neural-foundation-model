"""Modal GPU wrapper for ``train_4m.py``.

Run from repo root::

    modal run 4m_training/modal_train.py --config 4m_training/configs/4m_things_main.yaml

Dryrun on the volume (no GPU, no training — fast config/data check)::

    modal run 4m_training/modal_train.py --dryrun --config 4m_training/configs/4m_things_data.yaml

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
    for path in (
        __import__("pathlib").Path("/opt/repo/4m_training/_modal_load.py"),
        __import__("pathlib").Path(__file__).resolve().parent / "_modal_load.py",
    ):
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_modal_load", path)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        return loader.load_modal_image()
    raise ImportError("_modal_load.py not found (expected /opt/repo/4m_training on Modal)")


_mi = _load_modal_image()
REPO = _mi.REPO
ensure_fourm = _mi.ensure_fourm
train_image = _mi.train_image
training_env = _mi.training_env

app = modal.App("train-4m-things")
project_volume = modal.Volume.from_name("project")
PROJECT = "/project"


def _run_train(config: str, dryrun: bool, n_batches: int) -> None:
    ensure_fourm()
    # cfg = config
    # if not os.path.isabs(cfg):
    #     cfg = os.path.join(REPO, cfg)

    # cmd = [
    #     sys.executable,
    #     os.path.join(REPO, "4m_training/train_4m.py"),
    #     "dryrun" if dryrun else "train",
    #     "--config",
    #     cfg,
    # ]
    cmd = [
        sys.executable,
        os.path.join(REPO, "ml-4m/run_scaling_experiment_2.py"),
        "--mode", "sweep",
    ]
    if dryrun:
        cmd.append("--test_run")

    env = training_env()
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    gpu="A100",
    timeout=60 * 60 * 24,
    memory=64 * 1024,
)
def train(config: str = "4m_training/configs/4m_things_main.yaml") -> None:
    _run_train(config, dryrun=False, n_batches=4)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    timeout=60 * 30,
    memory=32 * 1024,
)
def dryrun_job(
    config: str = "4m_training/configs/4m_things_data.yaml",
    n_batches: int = 4,
) -> None:
    """CPU-only dataloader smoke test — use this before paying for GPU train."""
    _run_train(config, dryrun=True, n_batches=n_batches)


@app.local_entrypoint()
def main(
    config: str = "4m_training/configs/4m_things_main.yaml",
    dryrun: bool = False,
    n_batches: int = 4,
) -> None:
    # NB: the bool flag `dryrun` must not share a name with a Modal function, or
    # `<flag>.remote(...)` resolves to the bool. Function is `dryrun_job`.
    if dryrun:
        print("dryrun")
        dryrun_job.remote(config=config, n_batches=n_batches)
    else:
        train.remote(config=config)
