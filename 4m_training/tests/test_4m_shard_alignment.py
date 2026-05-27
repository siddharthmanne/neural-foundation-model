"""Verify 4M WebDataset shard alignment rules with synthetic tars.

Uses the real ``multi_tarfile_samples`` implementation from ml-4m
(``fourm/data/unified_datasets.py``) and real ``webdataset`` tar parsing.

Run (from repo root):
    .venv-4m-test/bin/pip install webdataset braceexpand numpy
    .venv-4m-test/bin/python -m pytest modal/test_4m_shard_alignment.py -v

Or:
    .venv-4m-test/bin/python modal/test_4m_shard_alignment.py
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest
import webdataset as wds

from neural_constants import MEG_TRIAL_SHAPE, MEG_TOKENS_PER_TRIAL, TOK_RGB_TOKENS_PER_IMAGE

from fourm.data.unified_datasets import multi_tarfile_samples


def _fail_handler(exc: Exception) -> bool:
    """Re-raise instead of 4M's default warn-and-continue."""
    raise exc


def _make_tar(path: Path, entries: Iterable[tuple[str, str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for key, ext, data in entries:
            info = tarfile.TarInfo(name=f"{key}.{ext}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def shard_root(tmp_path: Path) -> Path:
    return tmp_path / "things"


def _merge(url: str) -> list[dict]:
    return list(
        multi_tarfile_samples([{"url": url}], handler=_fail_handler)
    )


class TestDenseModalityAlignment:
    """rgb / depth / tok_rgb must be 1:1 when loaded together via 4M."""

    def test_catalog_slot_holes_ok_if_all_dense_modalities_match(
        self, shard_root: Path
    ):
        """Train shard with val 'holes' works IF every zipped modality skips the same ids."""
        root = shard_root
        _make_tar(
            root / "tok_rgb/shard_000.tar",
            [("000000001", "npy", b"r1"), ("000000003", "npy", b"r3")],
        )
        _make_tar(
            root / "tok_depth/shard_000.tar",
            [("000000001", "npy", b"d1"), ("000000003", "npy", b"d3")],
        )
        samples = _merge(f"{root}/[tok_rgb,tok_depth]/shard_000.tar")
        assert [s["__key__"] for s in samples] == ["000000001", "000000003"]

    def test_key_mismatch_raises(self, shard_root: Path):
        root = shard_root
        _make_tar(
            root / "tok_rgb/shard_001.tar",
            [("000000001", "npy", b"r1"), ("000000002", "npy", b"r2")],
        )
        _make_tar(
            root / "tok_depth/shard_001.tar",
            [("000000001", "npy", b"d1"), ("000000003", "npy", b"d3")],
        )
        with pytest.raises(ValueError, match="Divergence detected"):
            _merge(f"{root}/[tok_rgb,tok_depth]/shard_001.tar")

    def test_count_mismatch_truncates_silently(self, shard_root: Path):
        """zip() stops at the shorter tar — extra rgb rows are never seen."""
        root = shard_root
        _make_tar(
            root / "tok_rgb/shard_002.tar",
            [("000000001", "npy", b"r1"), ("000000002", "npy", b"r2")],
        )
        _make_tar(root / "tok_depth/shard_002.tar", [("000000001", "npy", b"d1")])
        samples = _merge(f"{root}/[tok_rgb,tok_depth]/shard_002.tar")
        assert [s["__key__"] for s in samples] == ["000000001"]


class TestNeuralModalityAlignment:
    """MEG/EEG only align with rgb when zipped in the same 4M data_path."""

    def test_multitrial_meg_keys_do_not_match_rgb(self, shard_root: Path):
        root = shard_root
        _make_tar(
            root / "tok_rgb/shard_003.tar",
            [("000000001", "npy", b"r1"), ("000000002", "npy", b"r2")],
        )
        _make_tar(
            root / "tok_meg/shard_003.tar",
            [
                ("000000001_P1_t0", "meg.npy", b"m11"),
                ("000000001_P2_t0", "meg.npy", b"m12"),
                ("000000002_P1_t0", "meg.npy", b"m21"),
            ],
        )
        with pytest.raises(ValueError, match="Divergence detected"):
            _merge(f"{root}/[tok_rgb,tok_meg]/shard_003.tar")

    def test_sparse_meg_truncates_without_error(self, shard_root: Path):
        """Missing meg rows do NOT raise — zip silently drops tail rgb samples."""
        root = shard_root
        _make_tar(
            root / "tok_rgb/shard_004.tar",
            [
                ("000000001", "npy", b"r1"),
                ("000000002", "npy", b"r2"),
                ("000000003", "npy", b"r3"),
            ],
        )
        _make_tar(
            root / "tok_meg/shard_004.tar",
            [("000000001", "meg.npy", b"m1"), ("000000002", "meg.npy", b"m2")],
        )
        samples = _merge(f"{root}/[tok_rgb,tok_meg]/shard_004.tar")
        assert [s["__key__"] for s in samples] == ["000000001", "000000002"]

    def test_meg_1to1_with_rgb_when_same_keys_same_order(
        self, shard_root: Path
    ):
        """Only case where tok_meg can share a 4M data_path with tok_rgb."""
        root = shard_root
        _make_tar(
            root / "tok_rgb/shard_005.tar",
            [("000000001", "npy", b"r1"), ("000000002", "npy", b"r2")],
        )
        _make_tar(
            root / "tok_meg/shard_005.tar",
            [("000000001", "meg.npy", b"m1"), ("000000002", "meg.npy", b"m2")],
        )
        samples = _merge(f"{root}/[tok_rgb,tok_meg]/shard_005.tar")
        assert [s["__key__"] for s in samples] == ["000000001", "000000002"]
        assert "tok_rgb.npy" in samples[0]
        assert "meg.npy" in samples[0]  # multi-dot ext kept as-is (check_dots)

    def test_meg_wds_keys_are_not_image_ids(self, shard_root: Path):
        root = shard_root
        _make_tar(
            root / "tok_meg/shard_006.tar",
            [("000000001_P1_t0", "meg.npy", b"x")],
        )
        sample = next(
            iter(
                wds.WebDataset(
                    str(root / "tok_meg/shard_006.tar"), shardshuffle=False
                )
            )
        )
        assert sample["__key__"] == "000000001_P1_t0"

    def test_one_trial_per_image_uses_image_id_key(self, shard_root: Path):
        root = shard_root
        _make_tar(
            root / "tok_meg/shard_007.tar",
            [("000000001", "meg.npy", b"x")],
        )
        sample = next(
            iter(
                wds.WebDataset(
                    str(root / "tok_meg/shard_007.tar"), shardshuffle=False
                )
            )
        )
        assert sample["__key__"] == "000000001"


class TestPlaceholderStrategy:
    """Verify the 'one .npy per image_id with sentinel for missing MEG' scheme.

    This is the low-friction path for jointly training 4M on
    rgb + depth + tok_rgb + tok_depth + tok_meg + tok_eeg in ONE data_path.

    Rules:
      - Every image_id in rgb appears in tok_meg/tok_eeg as well.
      - Real MEG: real bytes. Missing MEG: tiny sentinel npy.
      - Filename pattern: ``<image_id>.npy`` for all modalities.
    """

    def test_all_modalities_same_keys_placeholders_keep_zip_aligned(
        self, shard_root: Path
    ):
        import numpy as np

        root = shard_root
        ids = ["000000001", "000000002", "000000003", "000000004"]

        def npy_bytes(arr):
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            return buf.getvalue()

        # Dense rgb / depth: real data for every image.
        _make_tar(
            root / "tok_rgb/shard_000.tar",
            [(i, "npy", npy_bytes(np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16))) for i in ids],
        )
        _make_tar(
            root / "tok_depth/shard_000.tar",
            [(i, "npy", npy_bytes(np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16))) for i in ids],
        )

        # MEG: real for 1,3; placeholder for 2,4.
        real_meg = np.full(MEG_TRIAL_SHAPE, 7, dtype=np.int16)
        placeholder_meg = np.full(MEG_TRIAL_SHAPE, -1, dtype=np.int16)  # sentinel
        _make_tar(
            root / "tok_meg/shard_000.tar",
            [
                ("000000001", "npy", npy_bytes(real_meg)),
                ("000000002", "npy", npy_bytes(placeholder_meg)),
                ("000000003", "npy", npy_bytes(real_meg)),
                ("000000004", "npy", npy_bytes(placeholder_meg)),
            ],
        )

        samples = _merge(
            f"{root}/[tok_rgb,tok_depth,tok_meg]/shard_000.tar"
        )
        assert [s["__key__"] for s in samples] == ids
        assert len(samples) == 4

        # Each merged sample has 3 modality-prefixed npy entries.
        s0 = samples[0]
        expected_keys = {"tok_rgb.npy", "tok_depth.npy", "tok_meg.npy"}
        assert expected_keys.issubset(set(s0)), (
            f"missing modality keys; got {sorted(s0)}"
        )

        # Sentinel detection: load the tok_meg array and check first element.
        for i, sample in enumerate(samples):
            arr = np.load(io.BytesIO(sample["tok_meg.npy"]), allow_pickle=False)
            is_placeholder = bool((arr == -1).all())
            assert is_placeholder == (ids[i] in {"000000002", "000000004"})

    def test_multitrial_stacked_inside_one_npy(self, shard_root: Path):
        """One .npy per image with shape (n_trials, 16, 8, 4) still aligns."""
        import numpy as np

        root = shard_root

        def npy_bytes(arr):
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            return buf.getvalue()

        # img 1 has 4 trials, img 2 has 1 trial — variable shape per file is fine
        # because 4M's zip only checks __key__, not array shape.
        meg_4_trials = np.zeros((4, *MEG_TRIAL_SHAPE), dtype=np.int16)
        meg_1_trial = np.zeros((1, *MEG_TRIAL_SHAPE), dtype=np.int16)

        _make_tar(
            root / "tok_rgb/shard_000.tar",
            [
                ("000000001", "npy", npy_bytes(np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16))),
                ("000000002", "npy", npy_bytes(np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16))),
            ],
        )
        _make_tar(
            root / "tok_meg/shard_000.tar",
            [
                ("000000001", "npy", npy_bytes(meg_4_trials)),
                ("000000002", "npy", npy_bytes(meg_1_trial)),
            ],
        )
        samples = _merge(f"{root}/[tok_rgb,tok_meg]/shard_000.tar")
        assert [s["__key__"] for s in samples] == ["000000001", "000000002"]


def _catalog_slot(image_id: str, images_per_shard: int = 1000) -> str:
    """Reference implementation of catalog-slot ownership."""
    n = (int(image_id) - 1) // images_per_shard
    return f"shard_{n:03d}"


def _npy_bytes(arr):
    import numpy as np  # local import to avoid top-level dep for early tests

    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


class TestCatalogSlotMath:
    """Pure-Python math for ``shard_index = (int(image_id) - 1) // 1000``.

    THINGS image ids are 1-indexed (lowest is 000000001).
    """

    @pytest.mark.parametrize(
        "image_id,expected",
        [
            ("000000001", "shard_000"),
            ("000000999", "shard_000"),
            ("000001000", "shard_000"),
            ("000001001", "shard_001"),
            ("000005432", "shard_005"),
            ("000026000", "shard_025"),
            ("000026001", "shard_026"),
            ("000026107", "shard_026"),
        ],
    )
    def test_boundary_ids(self, image_id, expected):
        assert _catalog_slot(image_id) == expected

    def test_first_id_lands_in_shard_000(self):
        assert _catalog_slot("000000001") == "shard_000"

    def test_last_thousand_id_stays_in_shard_000(self):
        assert _catalog_slot("000001000") == "shard_000"

    def test_first_id_of_shard_001(self):
        assert _catalog_slot("000001001") == "shard_001"


class TestSevenModalityZip:
    """Brace expansion + zip with the full Option 1 modality list."""

    def test_seven_modality_bracket_expansion(self, shard_root: Path):
        import numpy as np

        root = shard_root
        ids = ["000000001", "000000002", "000000003"]
        # NOTE: meg_mask/eeg_mask deliberately omit the ``tok_`` prefix so
        # 4M's tok_to_int64 stage doesn't cast our uint8 presence flag.
        modality_payloads = {
            "tok_rgb": np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16),
            "tok_depth": np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16),
            "tok_meg": np.zeros((4, *MEG_TRIAL_SHAPE), dtype=np.int16),
            "tok_eeg": np.zeros((4, 17, 250), dtype=np.int16),
            "meg_mask": np.array([1], dtype=np.uint8),
            "eeg_mask": np.array([1], dtype=np.uint8),
            "crop_settings": np.zeros((10, 5), dtype=np.float32),
        }
        for mod_name, payload in modality_payloads.items():
            _make_tar(
                root / f"{mod_name}/shard_000.tar",
                [(i, "npy", _npy_bytes(payload)) for i in ids],
            )

        url = (
            f"{root}/[tok_rgb,tok_depth,tok_meg,tok_eeg,"
            f"meg_mask,eeg_mask,crop_settings]/shard_000.tar"
        )
        samples = _merge(url)
        assert [s["__key__"] for s in samples] == ids

        expected_payload_keys = {f"{m}.npy" for m in modality_payloads}
        for sample in samples:
            assert expected_payload_keys.issubset(set(sample)), (
                f"missing modality keys; got {sorted(sample)}"
            )


class TestVariableShapeAndPlaceholders:
    """Mixed real/placeholder MEG with realistic trial counts."""

    def test_variable_shape_meg_with_placeholders_mixed(self, shard_root: Path):
        import numpy as np

        root = shard_root
        ids = ["000000001", "000000002", "000000003", "000000004"]

        # Shape per id: 1-trial exp image, 48-trial test image,
        # placeholder (sentinel), 4-trial exp image.
        arrays = {
            "000000001": np.zeros((1, *MEG_TRIAL_SHAPE), dtype=np.int16),
            "000000002": np.zeros((48, *MEG_TRIAL_SHAPE), dtype=np.int16),
            "000000003": np.full((1, *MEG_TRIAL_SHAPE), -1, dtype=np.int16),
            "000000004": np.zeros((4, *MEG_TRIAL_SHAPE), dtype=np.int16),
        }
        _make_tar(
            root / "tok_rgb/shard_000.tar",
            [(i, "npy", _npy_bytes(np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16))) for i in ids],
        )
        _make_tar(
            root / "tok_meg/shard_000.tar",
            [(i, "npy", _npy_bytes(arrays[i])) for i in ids],
        )

        samples = _merge(f"{root}/[tok_rgb,tok_meg]/shard_000.tar")
        assert [s["__key__"] for s in samples] == ids

        for sample, image_id in zip(samples, ids):
            arr = np.load(io.BytesIO(sample["tok_meg.npy"]), allow_pickle=False)
            assert arr.shape == arrays[image_id].shape
            is_sentinel = bool(arr.shape[0] == 1 and (arr == -1).all())
            assert is_sentinel == (image_id == "000000003")

    def test_meg_mask_modality_byte_payload(self, shard_root: Path):
        """meg_mask carries a uint8 0/1 flag per image. Name omits ``tok_``
        prefix so 4M's tok_to_int64 stage doesn't cast it.
        """
        import numpy as np

        root = shard_root
        ids = ["000000001", "000000002"]
        _make_tar(
            root / "tok_rgb/shard_000.tar",
            [(i, "npy", _npy_bytes(np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16))) for i in ids],
        )
        _make_tar(
            root / "meg_mask/shard_000.tar",
            [
                ("000000001", "npy", _npy_bytes(np.array([1], dtype=np.uint8))),
                ("000000002", "npy", _npy_bytes(np.array([0], dtype=np.uint8))),
            ],
        )
        samples = _merge(f"{root}/[tok_rgb,meg_mask]/shard_000.tar")
        assert [s["__key__"] for s in samples] == ids
        for sample, expected in zip(samples, [1, 0]):
            arr = np.load(
                io.BytesIO(sample["meg_mask.npy"]), allow_pickle=False
            )
            assert arr.shape == (1,)
            assert arr.dtype == np.uint8
            assert int(arr[0]) == expected


class TestShardEdgeCases:
    """Empty and partial shards."""

    def test_empty_val_shard_is_valid(self, shard_root: Path):
        """An empty tar still zips cleanly — yields 0 samples, no error."""
        import numpy as np

        root = shard_root
        for mod in ("tok_rgb", "tok_meg"):
            _make_tar(root / f"{mod}/shard_005.tar", [])

        samples = _merge(f"{root}/[tok_rgb,tok_meg]/shard_005.tar")
        assert samples == []

    def test_partial_last_shard(self, shard_root: Path):
        """Last catalog slot has 107 ids — fewer than 1000 — and still zips."""
        import numpy as np

        root = shard_root
        ids = [f"{i:09d}" for i in range(26001, 26108)]  # 107 ids
        assert len(ids) == 107

        _make_tar(
            root / "tok_rgb/shard_026.tar",
            [(i, "npy", _npy_bytes(np.zeros((4,), dtype=np.int16))) for i in ids],
        )
        _make_tar(
            root / "tok_meg/shard_026.tar",
            [
                (i, "npy", _npy_bytes(np.zeros((1, *MEG_TRIAL_SHAPE), dtype=np.int16)))
                for i in ids
            ],
        )
        samples = _merge(f"{root}/[tok_rgb,tok_meg]/shard_026.tar")
        assert len(samples) == 107
        assert samples[0]["__key__"] == "000026001"
        assert samples[-1]["__key__"] == "000026107"


def _print_summary() -> None:
    """Run scenarios and print JSON for manual inspection."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "things"
        scenarios = []

        def run(label: str, url: str, expect_ok: bool):
            try:
                samples = _merge(url)
                ok = expect_ok
                payload = {
                    "scenario": label,
                    "expect_ok": expect_ok,
                    "ok": ok,
                    "n": len(samples),
                    "keys": [s["__key__"] for s in samples],
                }
            except ValueError as e:
                payload = {
                    "scenario": label,
                    "expect_ok": expect_ok,
                    "ok": not expect_ok,
                    "error": str(e)[:120],
                }
            scenarios.append(payload)

        _make_tar(
            root / "tok_rgb/shard_000.tar",
            [("000000001", "npy", b"r1"), ("000000003", "npy", b"r3")],
        )
        _make_tar(
            root / "tok_depth/shard_000.tar",
            [("000000001", "npy", b"d1"), ("000000003", "npy", b"d3")],
        )
        run(
            "dense_holes_aligned",
            f"{root}/[tok_rgb,tok_depth]/shard_000.tar",
            True,
        )

        _make_tar(
            root / "tok_rgb/shard_001.tar",
            [("000000001", "npy", b"r1"), ("000000002", "npy", b"r2")],
        )
        _make_tar(
            root / "tok_depth/shard_001.tar",
            [("000000001", "npy", b"d1"), ("000000003", "npy", b"d3")],
        )
        run(
            "dense_key_mismatch",
            f"{root}/[tok_rgb,tok_depth]/shard_001.tar",
            False,
        )

        _make_tar(
            root / "tok_rgb/shard_002.tar",
            [("000000001", "npy", b"r1"), ("000000002", "npy", b"r2")],
        )
        _make_tar(root / "tok_depth/shard_002.tar", [("000000001", "npy", b"d1")])
        run(
            "dense_count_mismatch",
            f"{root}/[tok_rgb,tok_depth]/shard_002.tar",
            True,
        )

        _make_tar(
            root / "tok_rgb/shard_003.tar",
            [("000000001", "npy", b"r1"), ("000000002", "npy", b"r2")],
        )
        _make_tar(
            root / "tok_meg/shard_003.tar",
            [
                ("000000001_P1_t0", "meg.npy", b"m11"),
                ("000000001_P2_t0", "meg.npy", b"m12"),
                ("000000002_P1_t0", "meg.npy", b"m21"),
            ],
        )
        run(
            "rgb_plus_multitrial_meg",
            f"{root}/[tok_rgb,tok_meg]/shard_003.tar",
            False,
        )

        _make_tar(
            root / "tok_rgb/shard_004.tar",
            [
                ("000000001", "npy", b"r1"),
                ("000000002", "npy", b"r2"),
                ("000000003", "npy", b"r3"),
            ],
        )
        _make_tar(
            root / "tok_meg/shard_004.tar",
            [("000000001", "meg.npy", b"m1"), ("000000002", "meg.npy", b"m2")],
        )
        run(
            "rgb_plus_sparse_meg",
            f"{root}/[tok_rgb,tok_meg]/shard_004.tar",
            True,
        )

        _make_tar(
            root / "tok_rgb/shard_005.tar",
            [("000000001", "npy", b"r1"), ("000000002", "npy", b"r2")],
        )
        _make_tar(
            root / "tok_meg/shard_005.tar",
            [("000000001", "meg.npy", b"m1"), ("000000002", "meg.npy", b"m2")],
        )
        run(
            "rgb_plus_meg_1to1",
            f"{root}/[tok_rgb,tok_meg]/shard_005.tar",
            True,
        )

        print(json.dumps(scenarios, indent=2))


if __name__ == "__main__":
    _print_summary()
