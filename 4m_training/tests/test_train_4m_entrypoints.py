"""Tests for train_4m.py CLI plumbing that the suite never exercised.

Two regressions hide here:
  * ``train ... -- --data_config X`` forwarded the ``--`` separator to the 4M
    trainer, which rejects it as an unrecognized argument.
  * ``dryrun`` passed ``num_workers=0`` to the stock mixture loader, which
    computes ``epoch_size // (num_gpus * num_workers * batch_size)`` and so
    divided by zero before yielding a single batch.
"""

from __future__ import annotations

import io
import sys
import tarfile
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from repo_paths import REPO_ROOT, TRAINING_DIR

sys.path.insert(0, str(TRAINING_DIR))

import fourm_neural_modalities  # noqa: F401
from fourm_dataloader import patch_pretrain_utils
from train_4m import _build_modality_info, _clean_extra_argv


class TestCleanExtraArgv:
    def test_strips_leading_separator(self):
        assert _clean_extra_argv(["--", "--data_config", "X"]) == ["--data_config", "X"]

    def test_passes_through_without_separator(self):
        assert _clean_extra_argv(["--data_config", "X"]) == ["--data_config", "X"]

    def test_empty(self):
        assert _clean_extra_argv([]) == []

    def test_only_strips_first_separator(self):
        # A literal value-level "--" after real args is preserved.
        assert _clean_extra_argv(["--", "--a", "1", "--"]) == ["--a", "1", "--"]


# --- num_workers regression (build-only; no worker spawn) ---


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _make_tar(path: Path, entries: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for key, data in entries:
            info = tarfile.TarInfo(name=f"{key}.npy")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _rgb_only_shard(root: Path) -> str:
    from neural_constants import TOK_RGB_TOKENS_PER_IMAGE, TOK_RGB_VOCAB_SIZE

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, TOK_RGB_VOCAB_SIZE, (TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16)
    _make_tar(root / "tok_rgb" / "shard_000.tar", [("000000001", _npy_bytes(rgb))])
    return f"{root}/[tok_rgb]/shard_{{000..000}}.tar"


def _build_loader(data_path: str, num_workers: int):
    from tokenizers import Tokenizer
    from fourm.data.pretrain_utils import get_train_dataloader, setup_sampling_mod_info

    patch_pretrain_utils()
    ds_cfg = {
        "type": "multimodal",
        "use_wds": True,
        "data_path": data_path,
        "in_domains": "tok_rgb",
        "out_domains": "tok_rgb",
        "main_augment_domain": "tok_rgb",
        "tok_train_aug": False,
        "input_alphas": "1.0",
        "target_alphas": "1.0",
        "aligned_captions": False,
    }
    full = _build_modality_info(["tok_rgb"], input_size=224)
    mod_info, weights = setup_sampling_mod_info(ds_cfg, full)
    tok = Tokenizer.from_file(
        str(REPO_ROOT / "external/ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json")
    )
    return get_train_dataloader(
        dataset_config=ds_cfg, modality_info=mod_info, sampling_weights=weights,
        text_tokenizer=tok, input_size=224, num_input_tokens=64, num_target_tokens=64,
        min_input_tokens=None, min_target_tokens=None, num_tasks=1,
        num_workers=num_workers, dataset_batch_size=2, epoch_size=8,
    )


class TestTrainLoaderWorkers:
    def test_zero_workers_raises(self, tmp_path: Path):
        """Documents why dryrun must not pass num_workers=0 (stock loader divides by it)."""
        data_path = _rgb_only_shard(tmp_path / "things")
        with pytest.raises(ZeroDivisionError):
            _build_loader(data_path, num_workers=0)

    def test_one_worker_builds(self, tmp_path: Path):
        data_path = _rgb_only_shard(tmp_path / "things")
        loader = _build_loader(data_path, num_workers=1)
        assert loader is not None
