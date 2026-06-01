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


def _run_train(dryrun: bool, condition: str = "rgb_only", large_gpu: bool = False) -> None:
    ensure_fourm()
    cmd = [
        sys.executable,
        os.path.join(REPO, "ml-4m/run_scaling_experiment_2.py"),
        "--mode", "sweep",
        "--condition", condition,
    ]
    if dryrun:
        cmd.append("--test_run")
    if large_gpu:
        cmd.append("--large_gpu")

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
def train(condition: str = "rgb_only", large_gpu: bool = False) -> None:
    _run_train(dryrun=False, condition=condition, large_gpu=large_gpu)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    memory=64 * 1024,
)
def train_large(condition: str = "rgb_only") -> None:
    """80GB A100 variant — use for dim=512 re-runs with bs=512."""
    _run_train(dryrun=False, condition=condition, large_gpu=True)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    timeout=60 * 30,
    memory=32 * 1024,
)
def dryrun_job(condition: str = "rgb_only") -> None:
    """CPU-only dataloader smoke test — use this before paying for GPU train."""
    _run_train(dryrun=True, condition=condition)


def _run_analysis(condition: str | None, mode: str, loss_type: str = "rgb") -> None:
    """Run collect or fit mode of run_scaling_experiment_2.py.

    condition=None processes all conditions found under sweep_base_dir.
    """
    ensure_fourm()
    cmd = [
        sys.executable,
        os.path.join(REPO, "ml-4m/run_scaling_experiment_2.py"),
        "--mode", mode,
        "--loss_type", loss_type,
    ]
    if condition:
        cmd.extend(["--condition", condition])
    env = training_env()
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    timeout=60 * 30,
    memory=16 * 1024,
)
def collect_job(condition: str = "") -> None:
    """Parse training logs and write results.json. Empty condition = all conditions."""
    _run_analysis(condition or None, "collect")


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    timeout=60 * 30,
    memory=16 * 1024,
)
def fit_job(condition: str = "", loss_type: str = "rgb") -> None:
    """Fit Chinchilla scaling law. Empty condition = all conditions."""
    _run_analysis(condition or None, "fit", loss_type)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    timeout=60 * 30,
    memory=16 * 1024,
)
def plot_job(
    sweep_dir: str = "/project/data/scaling_sweep",
    conditions: str = "",
    control: str = "rgb_only",
    treatment: str = "",
) -> None:
    """Plot per-condition and comparison plots from the scaling sweep.

    Plots are saved to {sweep_dir}/{condition}/plots/ per condition and
    {sweep_dir}/plots/ for cross-condition comparisons.
    conditions: comma-separated list to limit which conditions are plotted;
    empty string means auto-discover all conditions under sweep_dir.
    treatment: specific treatment condition for gap plot; empty = all non-control conditions.
    """
    ensure_fourm()
    cmd = [
        sys.executable,
        os.path.join(REPO, "ml-4m/plot_training.py"),
        "--sweep_dir", sweep_dir,
        "--control", control,
    ]
    if treatment:
        cmd.extend(["--treatment", treatment])
    if conditions:
        cmd.extend(["--conditions", conditions])
    env = training_env()
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


@app.local_entrypoint()
def main(
    condition: str = "",
    dryrun: bool = False,
    large_gpu: bool = False,
    mode: str = "train",
    loss_type: str = "rgb",
    treatment: str = "",
) -> None:
    """
    mode: train | dryrun | collect | fit | plot

    --condition defaults to "" which means all conditions for collect/fit/plot,
    and is required for train/dryrun.
    --treatment for plot mode: specific treatment condition for gap plot;
    empty = auto-generate gap plots for all non-control conditions.

    Examples:
      modal run 4m_training/modal_train.py --condition pixel_meg
      modal run 4m_training/modal_train.py --condition rgb_only_pure_all2all
      modal run 4m_training/modal_train.py --mode collect
      modal run 4m_training/modal_train.py --mode collect --condition pixel_meg
      modal run 4m_training/modal_train.py --mode fit --loss_type depth
      modal run 4m_training/modal_train.py --mode fit --condition rgb_only --loss_type depth
      modal run 4m_training/modal_train.py --mode plot
      modal run 4m_training/modal_train.py --mode plot --treatment pixel_meg
      modal run 4m_training/modal_train.py --large_gpu --condition pixel_meg
    """
    if mode == "collect":
        collect_job.remote(condition=condition)
    elif mode == "fit":
        fit_job.remote(condition=condition, loss_type=loss_type)
    elif mode == "plot":
        plot_job.remote(conditions=condition, treatment=treatment)
    elif dryrun:
        print("dryrun")
        dryrun_job.remote(condition=condition)
    elif large_gpu:
        print("launching on A100-80GB with --large_gpu batch sizes")
        train_large.remote(condition=condition)
    else:
        train.remote(condition=condition, large_gpu=large_gpu)
