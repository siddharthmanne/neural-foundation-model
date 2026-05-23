"""Load BrainTokenizer from the pinned OpenTSLab/BrainOmni submodule."""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

_BRAINOMNI_ON_PATH = False


def _ensure_brainomni_importable(repo_path: str) -> None:
    """Add BrainOmni to sys.path and stub deepspeed for inference-only imports."""
    global _BRAINOMNI_ON_PATH
    repo_path = os.path.abspath(repo_path)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    if "deepspeed" not in sys.modules:
        import types

        class _ReduceOp:
            SUM = "sum"

        def _noop(*args, **kwargs):
            return None

        comm = types.ModuleType("deepspeed.comm")
        comm.is_initialized = lambda: False
        comm.get_world_size = lambda: 1
        comm.all_reduce = _noop
        comm.broadcast = _noop
        comm.ReduceOp = _ReduceOp

        stub = types.ModuleType("deepspeed")
        stub.comm = comm
        sys.modules["deepspeed"] = stub
        sys.modules["deepspeed.comm"] = comm
    _BRAINOMNI_ON_PATH = True


def load_braintokenizer(
    ckpt_dir: str,
    brainomni_repo: str,
    device: str = "cpu",
    eval_mode: bool = True,
    codebook_size: int | None = None,
) -> "nn.Module":
    """Instantiate BrainTokenizer and load ``BrainTokenizer.pt`` weights.

    If ``codebook_size`` differs from the checkpoint, quantizer weights are
    reinitialized (``strict=False``); encoder/decoder/sensor weights still load.
    """
    import torch

    _ensure_brainomni_importable(brainomni_repo)
    from braintokenizer.model import BrainTokenizer

    cfg_path = os.path.join(ckpt_dir, "model_cfg.json")
    weights_path = os.path.join(ckpt_dir, "BrainTokenizer.pt")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing model_cfg.json at {cfg_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Missing BrainTokenizer.pt at {weights_path}. "
            "Download from https://huggingface.co/OpenTSLab/BrainOmni"
        )

    with open(cfg_path) as f:
        model_cfg: dict[str, Any] = json.load(f)

    ckpt_codebook_size = int(model_cfg["codebook_size"])
    if codebook_size is not None and codebook_size != ckpt_codebook_size:
        model_cfg["codebook_size"] = codebook_size

    model = BrainTokenizer(**model_cfg)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if codebook_size is not None and codebook_size != ckpt_codebook_size:
        filtered = {}
        skipped = []
        for key, value in state.items():
            if key.startswith("quantizer."):
                skipped.append(key)
                continue
            if key in model.state_dict() and model.state_dict()[key].shape == value.shape:
                filtered[key] = value
            elif key in model.state_dict():
                skipped.append(key)
            else:
                skipped.append(key)
        model.load_state_dict(filtered, strict=False)
        print(
            f"[brainomni] loaded {len(filtered)} tensors; "
            f"skipped {len(skipped)} (codebook {ckpt_codebook_size}->{codebook_size})"
        )
    else:
        model.load_state_dict(state, strict=True)
    model.to(device)
    if eval_mode:
        model.eval()
    return model
