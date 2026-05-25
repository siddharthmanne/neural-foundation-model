"""BrainOmni run helpers — config dataclass lives in meg_config.py."""

from __future__ import annotations

import os

from ..meg_config import BRAINOMNI_DEFAULT, BrainOmniConfig

__all__ = ["BRAINOMNI_DEFAULT", "BrainOmniConfig", "default_ckpt_dir", "run_slug"]


def run_slug(cfg: BrainOmniConfig | None = None, stage: str = "3a") -> str:
    """Versioned run directory name under meg/brainomni/runs/."""
    cfg = cfg or BRAINOMNI_DEFAULT
    return (
        f"V{cfg.codebook_size}_rvq{cfg.num_quantizers}"
        f"_win{cfg.window_length}_sf{int(cfg.target_sfreq_hz)}"
        f"_{stage}"
    )


def default_ckpt_dir() -> str:
    """Resolve checkpoint directory relative to repo root."""
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(repo_root, BRAINOMNI_DEFAULT.ckpt_dir)
