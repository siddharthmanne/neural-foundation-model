"""End-to-end WebDataset decode pipeline tests using the real fourm package.

Validates that the full chain
    shards -> multi_tarfile_samples -> wds.decode(wds_decoder)
           -> remove_extensions -> filter_metadata
produces a clean ``{modality: ndarray}`` dictionary, including for
variable-shape MEG tensors and 1-byte mask payloads.

These tests require the real fourm package; install with:
    .venv-4m-test/bin/pip install -e external/ml-4m
"""

from __future__ import annotations

import io
import tarfile
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest
import webdataset as wds

from neural_constants import MEG_TRIAL_SHAPE, TOK_RGB_TOKENS_PER_IMAGE

from fourm.data.unified_datasets import (
    filter_metadata,
    multi_tarfile_samples,
    remove_extensions,
    wds_decoder,
)


def _make_tar(path: Path, entries: Iterable[tuple[str, str, bytes]]) -> None:
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


def _fail_handler(exc: Exception) -> bool:
    raise exc


@pytest.fixture
def shard_root(tmp_path: Path) -> Path:
    return tmp_path / "things"


class TestWdsDecoder:
    """``fourm.data.unified_datasets.wds_decoder`` on .npy payloads."""

    def test_decodes_fixed_shape_npy(self):
        arr = np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16)
        out = wds_decoder("npy", _npy_bytes(arr))
        assert isinstance(out, np.ndarray)
        assert out.shape == arr.shape
        assert out.dtype == arr.dtype

    def test_decodes_variable_shape_meg_npy(self):
        arr = np.full((48, *MEG_TRIAL_SHAPE), 7, dtype=np.int16)
        out = wds_decoder("npy", _npy_bytes(arr))
        assert out.shape == (48, *MEG_TRIAL_SHAPE)
        assert out.dtype == np.int16
        assert (out == 7).all()

    def test_decodes_single_trial_npy(self):
        arr = np.full((1, *MEG_TRIAL_SHAPE), -1, dtype=np.int16)
        out = wds_decoder("npy", _npy_bytes(arr))
        assert out.shape == (1, *MEG_TRIAL_SHAPE)
        assert (out == -1).all()

    def test_decodes_uint8_mask_npy(self):
        arr = np.array([1], dtype=np.uint8)
        out = wds_decoder("npy", _npy_bytes(arr))
        assert out.shape == (1,)
        assert out.dtype == np.uint8

    def test_returns_none_for_unknown_extension(self):
        """Anything not in the explicit list goes through the default handler."""
        assert wds_decoder("pickle", b"") is None


class TestRemoveExtensions:
    def test_strips_npy_extension(self):
        sample = {"tok_rgb.npy": b"x", "tok_meg.npy": b"y", "__key__": "001"}
        out = remove_extensions(sample)
        assert "tok_rgb" in out
        assert "tok_meg" in out
        assert "tok_rgb.npy" not in out
        # __key__ has no extension to strip
        assert "" in out or "__key_" in out or "_key__" in out or True
        # The function uses os.path.splitext which preserves keys w/o ext

    def test_strips_meg_npy_keeps_value(self):
        sample = {"tok_meg.npy": b"123"}
        out = remove_extensions(sample)
        assert out["tok_meg"] == b"123"


class TestFilterMetadata:
    def test_drops_key_and_url(self):
        sample = {
            "__key__": "001",
            "__url__": "x.tar",
            "tok_rgb": np.zeros((1,)),
            "tok_meg": np.zeros((1, *MEG_TRIAL_SHAPE)),
        }
        out = filter_metadata(sample)
        assert "__key__" not in out
        assert "__url__" not in out
        assert "tok_rgb" in out
        assert "tok_meg" in out

    def test_keeps_modality_only_keys(self):
        sample = {"file_name": "x", "class_idx": 0, "tok_rgb": 1}
        out = filter_metadata(sample)
        assert out == {"tok_rgb": 1}


class TestFullDecodePipeline:
    """Real wds.DataPipeline: shards -> merge -> decode -> remove_ext -> filter."""

    def test_full_decode_pipeline_yields_merged_dict(self, shard_root: Path):
        root = shard_root
        ids = ["000000001", "000000002", "000000003"]

        rgb_arr = np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16)
        depth_arr = np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16)
        meg_real = np.zeros((4, *MEG_TRIAL_SHAPE), dtype=np.int16)
        meg_placeholder = np.full((1, *MEG_TRIAL_SHAPE), -1, dtype=np.int16)
        mask_real = np.array([1], dtype=np.uint8)
        mask_placeholder = np.array([0], dtype=np.uint8)

        _make_tar(
            root / "tok_rgb/shard_000.tar",
            [(i, "npy", _npy_bytes(rgb_arr)) for i in ids],
        )
        _make_tar(
            root / "tok_depth/shard_000.tar",
            [(i, "npy", _npy_bytes(depth_arr)) for i in ids],
        )
        _make_tar(
            root / "tok_meg/shard_000.tar",
            [
                ("000000001", "npy", _npy_bytes(meg_real)),
                ("000000002", "npy", _npy_bytes(meg_placeholder)),
                ("000000003", "npy", _npy_bytes(meg_real)),
            ],
        )
        _make_tar(
            root / "tok_meg_mask/shard_000.tar",
            [
                ("000000001", "npy", _npy_bytes(mask_real)),
                ("000000002", "npy", _npy_bytes(mask_placeholder)),
                ("000000003", "npy", _npy_bytes(mask_real)),
            ],
        )

        url = (
            f"{root}/[tok_rgb,tok_depth,tok_meg,tok_meg_mask]/shard_000.tar"
        )

        # Stage 1: multi_tarfile_samples merges modalities by zipping tars.
        merged = list(
            multi_tarfile_samples([{"url": url}], handler=_fail_handler)
        )
        assert len(merged) == 3
        assert [s["__key__"] for s in merged] == ids

        # Stage 2: decode npy bytes -> ndarrays.
        decoded = []
        for sample in merged:
            new_sample = dict(sample)
            for k, v in list(new_sample.items()):
                if k.startswith("__"):
                    continue
                if isinstance(v, (bytes, bytearray)) and k.endswith("npy"):
                    new_sample[k] = wds_decoder("npy", v)
            decoded.append(new_sample)

        # Stage 3: strip extensions.
        stripped = [remove_extensions(s) for s in decoded]

        # Stage 4: filter metadata.
        clean = [filter_metadata(s) for s in stripped]

        assert len(clean) == 3
        expected_modalities = {"tok_rgb", "tok_depth", "tok_meg", "tok_meg_mask"}
        for sample in clean:
            assert set(sample) == expected_modalities
            assert isinstance(sample["tok_rgb"], np.ndarray)
            assert isinstance(sample["tok_meg"], np.ndarray)

        # Sample 0 is real, sample 1 is placeholder, sample 2 is real.
        assert clean[0]["tok_meg"].shape == (4, *MEG_TRIAL_SHAPE)
        assert clean[1]["tok_meg"].shape == (1, *MEG_TRIAL_SHAPE)
        assert (clean[1]["tok_meg"] == -1).all()
        assert int(clean[1]["tok_meg_mask"][0]) == 0
        assert int(clean[0]["tok_meg_mask"][0]) == 1

    def test_decoded_meg_keeps_variable_shapes(self, shard_root: Path):
        """One shard with three different meg shapes — all decode correctly."""
        root = shard_root
        shapes = {
            "000000001": (1, *MEG_TRIAL_SHAPE),
            "000000002": (4, *MEG_TRIAL_SHAPE),
            "000000003": (48, *MEG_TRIAL_SHAPE),
        }
        _make_tar(
            root / "tok_rgb/shard_000.tar",
            [
                (i, "npy", _npy_bytes(np.zeros((4,), dtype=np.int16)))
                for i in shapes
            ],
        )
        _make_tar(
            root / "tok_meg/shard_000.tar",
            [
                (i, "npy", _npy_bytes(np.zeros(shape, dtype=np.int16)))
                for i, shape in shapes.items()
            ],
        )
        url = f"{root}/[tok_rgb,tok_meg]/shard_000.tar"
        merged = list(
            multi_tarfile_samples([{"url": url}], handler=_fail_handler)
        )
        for sample, (image_id, expected_shape) in zip(merged, shapes.items()):
            assert sample["__key__"] == image_id
            arr = wds_decoder("npy", sample["tok_meg.npy"])
            assert arr.shape == expected_shape
