"""Tests for fourm_dataloader patches (rename without crop_settings, train loader)."""

from __future__ import annotations

import io
import sys
import tarfile
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import webdataset as wds
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourm_neural_modalities  # noqa: F401
from fourm.data.pretrain_utils import get_train_dataloader, setup_sampling_mod_info
from neural_constants import (
    MEG_TRIAL_SHAPE,
    THINGS_IMAGE_SIZE,
    TOK_RGB_TOKENS_PER_IMAGE,
)
from fourm.data.unified_datasets import (
    filter_metadata,
    map as keyless_map,
    multi_tarfile_samples,
    remove_extensions,
    tok_to_int64,
    wds_decoder,
)
from fourm_dataloader import _rename_modalities, patch_pretrain_utils, unpatch_pretrain_utils
from repo_paths import REPO_ROOT as _REPO
from train_4m import _build_modality_info
_TOK = _REPO / "external/ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json"


def _make_tar(path: Path, entries: list[tuple[str, str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for key, ext, data in entries:
            info = tarfile.TarInfo(name=f"{key}.{ext}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


@pytest.fixture
def things_shard_root(tmp_path: Path) -> Path:
    root = tmp_path / "things"
    arr_rgb = np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16)
    arr_depth = np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16)
    meg = np.full((2, *MEG_TRIAL_SHAPE), 1, dtype=np.int16)
    _make_tar(
        root / "tok_rgb/shard_000.tar",
        [("000000001", "npy", _npy_bytes(arr_rgb))],
    )
    _make_tar(
        root / "tok_depth/shard_000.tar",
        [("000000001", "npy", _npy_bytes(arr_depth))],
    )
    _make_tar(
        root / "tok_meg/shard_000.tar",
        [("000000001", "npy", _npy_bytes(meg))],
    )
    _make_tar(
        root / "meg_mask/shard_000.tar",
        [("000000001", "npy", _npy_bytes(np.array([1], dtype=np.uint8)))],
    )
    return root


def test_log_print_freq_override(capsys) -> None:
    """set_log_print_freq overrides the stock trainer's hardcoded log_every interval."""
    from fourm.utils.logger import MetricLogger
    import fourm_dataloader as fdl

    fdl.patch_pretrain_utils()
    try:
        fdl.set_log_print_freq(2)  # print often
        list(MetricLogger().log_every(range(20), print_freq=999, iter_len=20, header="HDR"))
        often = capsys.readouterr().out.count("HDR")

        fdl.set_log_print_freq(10)  # print rarely
        list(MetricLogger().log_every(range(20), print_freq=999, iter_len=20, header="HDR"))
        rarely = capsys.readouterr().out.count("HDR")
    finally:
        fdl.set_log_print_freq(None)
        unpatch_pretrain_utils()
    # The passed print_freq=999 is ignored in favor of our override, so freq=2 logs more.
    assert often > rarely, f"override not applied: freq2={often} freq10={rarely}"


def test_rename_modalities_skips_missing_crop_settings() -> None:
    sample = {"tok_rgb": 1, "tok_depth": 2}
    paths = {
        "tok_rgb": "tok_rgb",
        "tok_depth": "tok_depth",
        "crop_settings": "crop_settings",
    }
    out = _rename_modalities(sample, paths)
    assert out == {"tok_rgb": 1, "tok_depth": 2}
    assert "crop_settings" not in out


def test_train_dataloader_one_batch(things_shard_root: Path) -> None:
    patch_pretrain_utils()
    try:
        root = things_shard_root
        dataset_config = {
            "type": "multimodal",
            "use_wds": True,
            "data_path": (
                f"{root}/[tok_rgb,tok_depth,tok_meg,meg_mask]/shard_{{000..000}}.tar"
            ),
            # tok_meg folder fans out to the 4 symmetric RVQ modalities (both in + out).
            "in_domains": "tok_rgb-tok_depth-tok_meg_rvq0-tok_meg_rvq1-tok_meg_rvq2-tok_meg_rvq3",
            "out_domains": "tok_rgb-tok_depth-tok_meg_rvq0-tok_meg_rvq1-tok_meg_rvq2-tok_meg_rvq3",
            "main_augment_domain": "tok_rgb",
            "tok_train_aug": False,
            "input_alphas": "1.0",
            "target_alphas": "1.0",
            "aligned_captions": False,
            "wds_n_repeats": 1,
            "wds_shuffle_buffer_tar": 10,
            "wds_shuffle_buffer_repeat": 10,
        }
        in_d = sorted(dataset_config["in_domains"].split("-"))
        out_d = sorted(dataset_config["out_domains"].split("-"))
        all_d = sorted(set(in_d) | set(out_d))
        mod_full = _build_modality_info(all_d, input_size=THINGS_IMAGE_SIZE)
        mod_info, weights = setup_sampling_mod_info(dataset_config, mod_full)
        text_tokenizer = Tokenizer.from_file(str(_TOK))

        loader = get_train_dataloader(
            dataset_config=dataset_config,
            modality_info=mod_info,
            sampling_weights=weights,
            text_tokenizer=text_tokenizer,
            input_size=THINGS_IMAGE_SIZE,
            num_input_tokens=64,
            num_target_tokens=64,
            min_input_tokens=None,
            min_target_tokens=None,
            num_tasks=1,
            num_workers=0,  # avoid pickling patched augmenter across processes
            dataset_batch_size=2,
            epoch_size=None,
        )
        batch = next(iter(loader))
        assert "tok_rgb" in batch
        assert "tensor" in batch["tok_rgb"]
    finally:
        unpatch_pretrain_utils()
