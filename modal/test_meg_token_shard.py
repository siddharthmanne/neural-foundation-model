"""Laptop tests for meg_token_shard.py (no Modal, no MNE)."""

from __future__ import annotations

import numpy as np
import pytest

from meg_token_shard import (
    EXPECTED_TOKEN_DTYPE,
    EXPECTED_TOKEN_SHAPE,
    MegEntry,
    build_image_id_to_shard,
    group_entries_by_shard,
    meg_filename,
    parse_meg_filename,
    read_meg_shard_tar,
    write_meg_shard_tar,
)


def _rand_tokens(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 512, size=EXPECTED_TOKEN_SHAPE, dtype=EXPECTED_TOKEN_DTYPE)


# ---------- filename convention ----------

def test_meg_filename_roundtrip():
    fn = meg_filename("000000001", "P1", 0)
    assert fn == "000000001_P1_t0.meg.npy"
    parsed = parse_meg_filename(fn)
    assert parsed == ("000000001", "P1", 0)


def test_meg_filename_larger_trial_idx():
    fn = meg_filename("000026107", "P4", 11)
    assert parse_meg_filename(fn) == ("000026107", "P4", 11)


@pytest.mark.parametrize(
    "image_id",
    ["1", "0001", "000_000_001", "abc", "1234567890"],
)
def test_meg_filename_rejects_bad_image_id(image_id):
    with pytest.raises(ValueError):
        meg_filename(image_id, "P1", 0)


def test_meg_filename_rejects_bad_subject():
    with pytest.raises(ValueError):
        meg_filename("000000001", "subjA", 0)


def test_meg_filename_rejects_negative_trial():
    with pytest.raises(ValueError):
        meg_filename("000000001", "P1", -1)


@pytest.mark.parametrize(
    "bad",
    [
        "000000001.jpg",                          # wrong ext
        "000000001_P1_t0.npy",                    # missing .meg
        "00000001_P1_t0.meg.npy",                 # 8-digit id
        "000000001_p1_t0.meg.npy",                # lowercase p
        "000000001_P1_0.meg.npy",                 # missing t prefix
        "000000001_P1_t.meg.npy",                 # missing trial number
    ],
)
def test_parse_meg_filename_rejects(bad):
    assert parse_meg_filename(bad) is None


# ---------- entry validation ----------

def test_meg_entry_rejects_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        MegEntry("000000001", "P1", 0, np.zeros((16, 8), dtype=EXPECTED_TOKEN_DTYPE))


def test_meg_entry_filename_property():
    e = MegEntry("000000123", "P2", 4, _rand_tokens())
    assert e.filename == "000000123_P2_t4.meg.npy"


# ---------- planning ----------

def _fake_manifest(shards: dict[str, list[str]]) -> dict:
    return {
        "split": "train",
        "n_shards": len(shards),
        "n_images": sum(len(v) for v in shards.values()),
        "shards_subpath": "things/meg",
        "shards": {
            sid: {"n_images": len(ids), "image_ids": ids} for sid, ids in shards.items()
        },
    }


def test_build_image_id_to_shard_basic():
    manifest = _fake_manifest(
        {"shard_000": ["000000001", "000000002"], "shard_001": ["000000003"]}
    )
    inv = build_image_id_to_shard(manifest)
    assert inv == {"000000001": "shard_000", "000000002": "shard_000", "000000003": "shard_001"}


def test_build_image_id_to_shard_rejects_duplicate():
    manifest = _fake_manifest(
        {"shard_000": ["000000001"], "shard_001": ["000000001"]}
    )
    with pytest.raises(ValueError, match="multiple shards"):
        build_image_id_to_shard(manifest)


def test_group_entries_skips_other_split():
    manifest = _fake_manifest({"shard_000": ["000000001"]})
    inv = build_image_id_to_shard(manifest)
    entries = [
        MegEntry("000000001", "P1", 0, _rand_tokens(1)),
        MegEntry("000000999", "P1", 0, _rand_tokens(2)),  # NOT in this split
    ]
    grouped = group_entries_by_shard(entries, inv)
    assert set(grouped) == {"shard_000"}
    assert len(grouped["shard_000"]) == 1
    assert grouped["shard_000"][0].image_id == "000000001"


def test_group_entries_rejects_duplicate_within_shard():
    manifest = _fake_manifest({"shard_000": ["000000001"]})
    inv = build_image_id_to_shard(manifest)
    entries = [
        MegEntry("000000001", "P1", 0, _rand_tokens(1)),
        MegEntry("000000001", "P1", 0, _rand_tokens(2)),  # same filename
    ]
    with pytest.raises(ValueError, match="duplicate"):
        group_entries_by_shard(entries, inv)


def test_group_entries_handles_multiple_trials_per_image():
    """Test image has up to 12 repeats per subject — make sure those land
    as 12 separate entries with distinct trial_idx."""
    manifest = _fake_manifest({"shard_000": ["000000005"]})
    inv = build_image_id_to_shard(manifest)
    entries = [
        MegEntry("000000005", "P1", i, _rand_tokens(i)) for i in range(12)
    ]
    grouped = group_entries_by_shard(entries, inv)
    assert len(grouped["shard_000"]) == 12
    assert len({e.trial_idx for e in grouped["shard_000"]}) == 12


# ---------- tar I/O round trip ----------

def test_write_then_read_shard(tmp_path):
    entries = [
        MegEntry("000000001", "P1", 0, _rand_tokens(1)),
        MegEntry("000000001", "P2", 0, _rand_tokens(2)),
        MegEntry("000000002", "P1", 0, _rand_tokens(3)),
    ]
    p = tmp_path / "shard_000.tar"
    write_meg_shard_tar(str(p), entries)

    got = read_meg_shard_tar(str(p))
    assert set(got.keys()) == {
        "000000001_P1_t0.meg.npy",
        "000000001_P2_t0.meg.npy",
        "000000002_P1_t0.meg.npy",
    }
    for e in entries:
        np.testing.assert_array_equal(got[e.filename], e.tokens)
        assert got[e.filename].dtype == EXPECTED_TOKEN_DTYPE
        assert got[e.filename].shape == EXPECTED_TOKEN_SHAPE


def test_write_uses_atomic_rename(tmp_path):
    entries = [MegEntry("000000001", "P1", 0, _rand_tokens())]
    p = tmp_path / "shard_000.tar"
    write_meg_shard_tar(str(p), entries)
    assert p.exists()
    assert not (tmp_path / "shard_000.tar.tmp").exists()


def test_serialize_rejects_out_of_int16_range():
    """If something upstream produces tokens outside int16 range, fail
    rather than silently truncating."""
    bad = np.full(EXPECTED_TOKEN_SHAPE, 70000, dtype=np.int32)
    with pytest.raises(ValueError, match="int16 range"):
        from meg_token_shard import _serialize_tokens
        _serialize_tokens(bad)


def test_full_pipeline_disjoint_splits(tmp_path):
    """End-to-end mini: 4 images, 2 shards per split, mixed trial counts."""
    train_manifest = _fake_manifest(
        {"shard_000": ["000000001", "000000002"]}
    )
    val_manifest = _fake_manifest(
        {"shard_000": ["000000003"]}
    )
    entries = [
        MegEntry("000000001", "P1", 0, _rand_tokens(10)),
        MegEntry("000000001", "P2", 0, _rand_tokens(11)),
        MegEntry("000000002", "P1", 0, _rand_tokens(12)),
        MegEntry("000000003", "P1", 0, _rand_tokens(13)),
        MegEntry("000000003", "P1", 1, _rand_tokens(14)),  # test repeat
        MegEntry("000000999", "P1", 0, _rand_tokens(15)),  # not in any split
    ]

    train_inv = build_image_id_to_shard(train_manifest)
    val_inv = build_image_id_to_shard(val_manifest)
    train_grouped = group_entries_by_shard(entries, train_inv)
    val_grouped = group_entries_by_shard(entries, val_inv)

    assert sum(len(v) for v in train_grouped.values()) == 3
    assert sum(len(v) for v in val_grouped.values()) == 2

    # Write + read back each.
    train_path = tmp_path / "train_shard_000.tar"
    val_path = tmp_path / "val_shard_000.tar"
    write_meg_shard_tar(str(train_path), train_grouped["shard_000"])
    write_meg_shard_tar(str(val_path), val_grouped["shard_000"])

    train_back = read_meg_shard_tar(str(train_path))
    val_back = read_meg_shard_tar(str(val_path))

    # Disjoint image_id coverage across splits.
    train_ids = {parse_meg_filename(n)[0] for n in train_back}
    val_ids = {parse_meg_filename(n)[0] for n in val_back}
    assert train_ids == {"000000001", "000000002"}
    assert val_ids == {"000000003"}
    assert train_ids.isdisjoint(val_ids)
