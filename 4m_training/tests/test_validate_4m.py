"""Validation runner: each task masks/predicts only its own out_domains.

Builds synthetic val shards and runs two contrasting tasks, asserting the loss
is computed on the right modalities — cross-modal tasks must NOT incur loss on
their input modality. Guards the per-task masking semantics of validate_4m.py.
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural_constants import (
    EEG_TRIAL_SHAPE,
    EEG_VOCAB_SIZE,
    MEG_TRIAL_SHAPE,
    MEG_VOCAB_SIZE,
    TOK_DEPTH_VOCAB_SIZE,
    TOK_RGB_TOKENS_PER_IMAGE,
    TOK_RGB_VOCAB_SIZE,
)
from validate_4m import run_validation


def _npy(arr: np.ndarray) -> bytes:
    b = io.BytesIO(); np.save(b, arr, allow_pickle=False); return b.getvalue()


def _tar(path: Path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as t:
        for key, data in entries:
            info = tarfile.TarInfo(name=f"{key}.npy"); info.size = len(data)
            t.addfile(info, io.BytesIO(data))


def _build_val(root: Path, n: int = 6) -> None:
    rng = np.random.default_rng(0)
    ids = [f"{i:09d}" for i in range(n)]
    mods = {
        "tok_rgb": lambda: rng.integers(0, TOK_RGB_VOCAB_SIZE, (TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16),
        "tok_depth": lambda: rng.integers(0, TOK_DEPTH_VOCAB_SIZE, (TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16),
        "tok_meg": lambda: rng.integers(0, MEG_VOCAB_SIZE, (2, *MEG_TRIAL_SHAPE), dtype=np.int16),
        "tok_eeg": lambda: rng.integers(0, EEG_VOCAB_SIZE, (2, *EEG_TRIAL_SHAPE), dtype=np.int16),
        "meg_mask": lambda: np.array([1], dtype=np.uint8),
        "eeg_mask": lambda: np.array([1], dtype=np.uint8),
    }
    for mod, gen in mods.items():
        _tar(root / mod / "shard_000.tar", [(i, _npy(gen())) for i in ids])


@pytest.fixture
def tasks_cfg(tmp_path: Path) -> dict:
    root = tmp_path / "val"
    _build_val(root)
    return {
        "val_root": str(root),
        "shards": "shard_{000..000}.tar",
        "main_augment_domain": "tok_rgb",
        "fixed_eval_input_tokens": 64,
        "fixed_eval_target_tokens": 64,
        "tasks": {
            "anyany_neural": {
                "modalities": ["tok_rgb", "tok_depth", "tok_meg", "tok_eeg", "meg_mask", "eeg_mask"],
                "in_domains": "tok_rgb-tok_depth-tok_meg-tok_eeg",
                "out_domains": "tok_rgb-tok_depth",
            },
            "rgb2depth": {
                "modalities": ["tok_rgb", "tok_depth"],
                "in_domains": "tok_rgb",
                "out_domains": "tok_depth",
            },
            "depth2rgb": {
                "modalities": ["tok_rgb", "tok_depth"],
                "in_domains": "tok_depth",
                "out_domains": "tok_rgb",
            },
        },
    }


_MAIN = {"model": "fm_tiny_6e_6d_swiglu_nobias", "input_size": 224, "loss_type": "mod"}


def test_cross_modal_tasks_only_score_their_target(tasks_cfg):
    res = run_validation(_MAIN, tasks_cfg, select=["rgb2depth", "depth2rgb"],
                         checkpoint=None, device="cpu", batch_size=2, n_batches=1)
    # rgb2depth predicts depth only — RGB is input, must not contribute loss.
    assert res["rgb2depth"]["tok_depth_loss"] > 0
    assert res["rgb2depth"].get("tok_rgb_loss", 0.0) == 0.0
    # depth2rgb is the mirror image.
    assert res["depth2rgb"]["tok_rgb_loss"] > 0
    assert res["depth2rgb"].get("tok_depth_loss", 0.0) == 0.0


def test_anyany_scores_both_vision_modalities(tasks_cfg):
    res = run_validation(_MAIN, tasks_cfg, select=["anyany_neural"],
                         checkpoint=None, device="cpu", batch_size=2, n_batches=1)
    stats = res["anyany_neural"]
    assert stats["tok_rgb_loss"] > 0 and stats["tok_depth_loss"] > 0
    # neural is input-only: it never appears as a predicted-modality loss.
    assert "tok_meg_loss" not in stats and "tok_eeg_loss" not in stats


def test_unknown_task_raises(tasks_cfg):
    with pytest.raises(SystemExit):
        run_validation(_MAIN, tasks_cfg, select=["nope"], checkpoint=None,
                       device="cpu", batch_size=2, n_batches=1)
