"""Modal GPU smoke tests using production-style *_main.yaml + *_data.yaml configs.

From repo root::

    modal run 4m_training/modal/modal_smoke_train.py --case prod_things
    modal run 4m_training/modal/modal_smoke_train.py --case all

Prove the neural decoding heads learn on GPU (every head's loss must descend)::

    modal run 4m_training/modal/modal_smoke_train.py --case neural_heads_descend

Fast checks without GPU train (no image rebuild on code edits)::

    modal run 4m_training/modal/modal_train.py --dryrun --config 4m_training/configs/4m_smoke_things_data.yaml
    modal run 4m_training/modal/modal_smoke_train.py --case probe
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import importlib.util
import modal

def _load_modal_image():
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


_mi = _load_modal_image()
REPO = _mi.REPO
ensure_fourm = _mi.ensure_fourm
train_image = _mi.train_image
training_env = _mi.training_env

app = modal.App("train-4m-smoke")
project_volume = modal.Volume.from_name(_mi.PROJECT_VOLUME_NAME)
PROJECT = _mi.PROJECT_MOUNT

SMOKE_CASES: dict[str, tuple[str, list[str]]] = {
    "prod_things": ("4m_training/configs/4m_smoke_things_main.yaml", []),
    "loss_decreases": ("4m_training/configs/4m_smoke_things_loss_main.yaml", []),
    "prod_things_neural_in": (
        "4m_training/configs/4m_smoke_things_main.yaml",
        ["--data_config", "4m_training/configs/4m_smoke_things_neural_in_data.yaml"],
    ),
    # Neural FROM VISION: the loss-tuned main config + the neural-output data config, so the
    # MEG (4 RVQ heads) / EEG reconstruction losses can be watched descending on GPU.
    # --find_unused_params: with 7 output heads and a stochastic target budget, some heads
    # get 0 targets on a given step -> their params produce no grad -> DDP needs this.
    "loss_decreases_neural_out": (
        "4m_training/configs/4m_smoke_things_loss_main.yaml",
        [
            "--data_config",
            "4m_training/configs/4m_smoke_things_neural_out_data.yaml",
            "--find_unused_params",
        ],
    ),
    # Neural SYMMETRIC (the production design): neural is encoder input AND decoder target
    # in one config. Proves the real DDP trainer trains the neural heads on real shards.
    "loss_decreases_neural_symmetric": (
        "4m_training/configs/4m_smoke_things_loss_main.yaml",
        [
            "--data_config",
            "4m_training/configs/4m_smoke_things_neural_symmetric_data.yaml",
            "--find_unused_params",
        ],
    ),
    "prod_cc12m": ("4m_training/configs/4m_smoke_cc12m_main.yaml", []),
    # In-loop validation: trains 1 epoch, then runs the full named-task suite on the
    # live model and logs per-task loss (proves in-train validation works end-to-end).
    "inloop_val": ("4m_training/configs/4m_smoke_inloop_val_main.yaml", []),
}


def _run_train(main_config_rel: str, extra_argv: list[str]) -> None:
    ensure_fourm()
    cfg = os.path.join(REPO, main_config_rel)
    cmd = [
        sys.executable,
        os.path.join(REPO, "4m_training/lib/train_4m.py"),
        "train",
        "--config",
        cfg,
    ]
    if extra_argv:
        cmd.extend(["--", *extra_argv])
    env = {**training_env(), "CUDA_VISIBLE_DEVICES": "0"}
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


@app.function(image=train_image, volumes={PROJECT: project_volume}, timeout=60 * 10)
def probe_volume() -> None:
    """CPU path check on the project volume."""
    for base in (
        f"{PROJECT}/data/train/things/tok_rgb",
        f"{PROJECT}/data/train/cc12m/crop_settings",
        f"{PROJECT}/data/train/cc12m/tok_rgb@224",
    ):
        print(f"exists {base}:", os.path.isdir(base), flush=True)
        if os.path.isdir(base):
            tars = sorted(f for f in os.listdir(base) if f.endswith(".tar"))[:3]
            print(f"  sample tars: {tars}", flush=True)


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    gpu="T4",
    timeout=60 * 15,
    memory=32 * 1024,
)
def neural_heads_descend(steps: int = 80) -> str:
    """PROOF: on GPU, every neural decoding head's loss descends during training.

    Runs the one-batch overfit harness (overfit_smoke.py), where neural is symmetric so the
    4 MEG RVQ heads + EEG + vision are ALL decoder targets. The harness asserts each
    modality's loss drops below its init (~ln(vocab)); a non-descending head -> nonzero exit
    -> this function fails. This is the end-to-end proof that the neural heads receive
    gradient and learn. See notes/4m_neural_modality_design.md.
    """
    ensure_fourm()
    cmd = [
        sys.executable,
        os.path.join(REPO, "4m_training/overfit_smoke.py"),
        "--steps", str(steps),
        "--device", "cuda",
    ]
    env = {**training_env(), "CUDA_VISIBLE_DEVICES": "0"}
    print("running:", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=REPO, env=env)
    return f"OK neural_heads_descend ({steps} steps, GPU) in {time.time() - t0:.1f}s"


@app.function(
    image=train_image,
    volumes={PROJECT: project_volume},
    gpu="T4",
    timeout=60 * 15,
    memory=32 * 1024,
)
def smoke_one(case: str) -> str:
    if case not in SMOKE_CASES:
        raise ValueError(f"unknown case {case!r}; choose from {list(SMOKE_CASES)}")
    main_cfg, extra = SMOKE_CASES[case]
    t0 = time.time()
    _run_train(main_cfg, extra)
    msg = f"OK {case} in {time.time() - t0:.1f}s"
    print(msg, flush=True)
    return msg


@app.local_entrypoint()
def main(case: str = "all") -> None:
    if case == "probe":
        probe_volume.remote()
        return
    if case == "neural_heads_descend":
        print(neural_heads_descend.remote())
        return
    if case == "all":
        print("\n".join(smoke_one.remote(name) for name in SMOKE_CASES))
    else:
        print(smoke_one.remote(case))
