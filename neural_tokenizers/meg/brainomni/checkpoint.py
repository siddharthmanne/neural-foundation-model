"""Ensure BrainTokenizer checkpoint is present (local or Modal)."""

from __future__ import annotations

import os


def default_ckpt_dir(brainomni_root: str) -> str:
    return os.path.join(brainomni_root, "ckpt_collection", "braintokenizer")


def ensure_braintokenizer_ckpt(brainomni_root: str) -> str:
    """Return ckpt dir, downloading from HuggingFace if weights are missing."""
    ckpt_dir = default_ckpt_dir(brainomni_root)
    weights = os.path.join(ckpt_dir, "BrainTokenizer.pt")
    cfg = os.path.join(ckpt_dir, "model_cfg.json")
    if os.path.isfile(weights) and os.path.isfile(cfg):
        return ckpt_dir

    os.makedirs(ckpt_dir, exist_ok=True)
    from huggingface_hub import hf_hub_download

    for name in ("BrainTokenizer.pt", "model_cfg.json"):
        path = hf_hub_download(
            repo_id="OpenTSLab/BrainOmni",
            filename=f"braintokenizer/{name}",
            local_dir=os.path.join(brainomni_root, "ckpt_collection"),
        )
        print(f"[brainomni] downloaded {path}")
    return ckpt_dir


def resolve_ckpt_dir(payload_ckpt_dir: str | None, brainomni_root: str) -> str:
    """Map config ckpt_dir to an absolute path and ensure weights exist."""
    if payload_ckpt_dir:
        if os.path.isabs(payload_ckpt_dir):
            ckpt = payload_ckpt_dir
        elif payload_ckpt_dir.startswith("external/BrainOmni"):
            ckpt = os.path.join(
                brainomni_root,
                payload_ckpt_dir.split("external/BrainOmni/", 1)[-1],
            )
        else:
            ckpt = os.path.join(brainomni_root, payload_ckpt_dir)
    else:
        ckpt = default_ckpt_dir(brainomni_root)

    if not os.path.isfile(os.path.join(ckpt, "BrainTokenizer.pt")):
        ensure_braintokenizer_ckpt(brainomni_root)
    return ckpt
