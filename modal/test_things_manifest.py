"""Laptop-runnable unit tests for things_manifest.py.

No Modal, no network. Uses synthetic 5-shard fixtures so the full
read→split→repack pipeline can be validated without touching the project
Volume. Run from the inner repo root:

    cd neural-foundation-model && pytest modal/test_things_manifest.py -v
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from things_manifest import (
    build_catalog,
    build_meg_coverage_payload,
    build_split_manifest,
    build_things_split_payload,
    extract_shard_contents,
    image_level_split,
    index_source_shards,
    load_coverage_json,
    neural_intersection_split,
    pack_into_shards,
    parse_eeg_coverage,
    write_shard_from_locations,
)


def _make_shard(path: Path, items: dict[str, str]) -> None:
    """Write a tar at `path` with paired <id>.jpg + <id>.txt entries."""
    with tarfile.open(path, "w") as tar:
        for image_id, filename in items.items():
            txt = filename.encode("utf-8")
            info = tarfile.TarInfo(f"{image_id}.txt")
            info.size = len(txt)
            tar.addfile(info, io.BytesIO(txt))
            jpg = b"\xff\xd8\xff\xe0FAKE" + image_id.encode()
            info = tarfile.TarInfo(f"{image_id}.jpg")
            info.size = len(jpg)
            tar.addfile(info, io.BytesIO(jpg))


def test_extract_shard_contents_roundtrip(tmp_path):
    p = tmp_path / "shard_000.tar"
    items = {
        "000000001": "aardvark_01b.jpg",
        "000000002": "aardvark_02s.jpg",
        "000000003": "abacus_01b.jpg",
    }
    _make_shard(p, items)
    assert extract_shard_contents(str(p)) == items


def test_build_catalog_sorts_and_counts():
    cat = build_catalog(
        {"000000003": "c.jpg", "000000001": "a.jpg", "000000002": "b.jpg"}
    )
    assert cat["n_images"] == 3
    assert list(cat["image_id_to_filename"].keys()) == [
        "000000001",
        "000000002",
        "000000003",
    ]
    assert "id_format" in cat


def test_image_level_split_deterministic_and_disjoint():
    ids = [f"{i:09d}" for i in range(1000)]
    t1, v1 = image_level_split(ids, val_frac=0.15, seed=0)
    t2, v2 = image_level_split(ids, val_frac=0.15, seed=0)
    assert t1 == t2 and v1 == v2
    assert set(t1).isdisjoint(set(v1))
    assert set(t1) | set(v1) == set(ids)
    assert abs(len(v1) / len(ids) - 0.15) < 0.005


def test_image_level_split_different_seeds_give_different_vals():
    ids = [f"{i:09d}" for i in range(1000)]
    _, v0 = image_level_split(ids, val_frac=0.15, seed=0)
    _, v1 = image_level_split(ids, val_frac=0.15, seed=1)
    assert set(v0) != set(v1)


def test_image_level_split_rejects_bad_frac():
    with pytest.raises(ValueError):
        image_level_split(["a"], val_frac=0.0)
    with pytest.raises(ValueError):
        image_level_split(["a"], val_frac=1.0)


def test_load_coverage_json_from_image_ids():
    assert load_coverage_json({"image_ids": ["000000001", "000000002"]}) == {
        "000000001",
        "000000002",
    }


def test_load_coverage_json_from_eeg_intersection():
    assert load_coverage_json(
        {"image_ids_intersection": ["000000003", "000000001"]}
    ) == {"000000001", "000000003"}


def test_load_coverage_json_prefers_union_over_intersection():
    payload = {
        "image_ids_union": ["000000001", "000000002"],
        "image_ids_intersection": ["000000001"],
    }
    assert load_coverage_json(payload) == {"000000001", "000000002"}


def test_parse_eeg_coverage_counts():
    parsed = parse_eeg_coverage(
        {
            "image_ids_eeg1": ["000000001", "000000002"],
            "image_ids_eeg2": ["000000002", "000000003"],
            "image_ids_intersection": ["000000002"],
            "image_ids_union": ["000000001", "000000002", "000000003"],
        }
    )
    assert parsed["n_eeg1"] == 2
    assert parsed["n_eeg2"] == 2
    assert parsed["n_eeg_intersection"] == 1
    assert parsed["n_eeg_union"] == 3


def test_build_meg_coverage_payload_deduplicates():
    payload = build_meg_coverage_payload(
        {"1": "000000001", "2": "000000001", "3": "000000002"}
    )
    assert payload["n_image_ids"] == 2
    assert payload["image_ids"] == ["000000001", "000000002"]


def test_neural_intersection_split_val_subset_of_pool():
    catalog = [f"{i:09d}" for i in range(100)]
    meg = {f"{i:09d}" for i in range(80)}
    eeg = {f"{i:09d}" for i in range(10, 90)}

    train, val, val_pool, stats = neural_intersection_split(
        catalog, meg, eeg, val_frac=0.20, seed=0
    )
    assert set(train).isdisjoint(set(val))
    assert set(train) | set(val) == set(catalog)
    assert set(val).issubset(set(val_pool))
    assert set(val_pool) == meg & eeg & set(catalog)
    assert stats["n_val"] == round(len(val_pool) * 0.20)
    assert stats["n_intersection"] == len(val_pool)
    assert len(catalog) - len(val) == stats["n_train"]


def test_neural_intersection_split_deterministic():
    catalog = [f"{i:09d}" for i in range(200)]
    meg = set(catalog)
    eeg = set(catalog)
    a = neural_intersection_split(catalog, meg, eeg, val_frac=0.20, seed=0)
    b = neural_intersection_split(catalog, meg, eeg, val_frac=0.20, seed=0)
    assert a[:3] == b[:3]


def test_build_things_split_payload_schema():
    catalog = [f"{i:09d}" for i in range(50)]
    meg = set(catalog)
    eeg_parsed = parse_eeg_coverage(
        {
            "image_ids_eeg1": catalog,
            "image_ids_eeg2": catalog,
            "image_ids_intersection": catalog,
            "image_ids_union": catalog,
        }
    )
    payload = build_things_split_payload(
        catalog, meg, eeg_parsed, val_frac=0.20, seed=0
    )
    assert payload["policy"] == "image_level_val_from_neural_intersection"
    assert payload["val_frac"] == 0.20
    assert payload["eeg"]["n_eeg1"] == 50
    assert payload["eeg"]["n_intersection"] == 50
    assert "n_eeg" not in payload
    assert "val_pool_image_ids" not in payload
    assert "intersection_image_ids" in payload
    assert len(payload["intersection_image_ids"]) == payload["n_intersection"]
    assert "proposed_shard_layout" not in payload
    assert "legacy_references" in payload
    assert set(payload["train_image_ids"]) | set(payload["val_image_ids"]) == set(
        catalog
    )


def test_pack_into_shards_sorts_and_handles_partial():
    ids = [f"{i:09d}" for i in range(2150)]
    shards = pack_into_shards(ids, images_per_shard=1000)
    assert len(shards) == 3
    assert [len(s) for s in shards] == [1000, 1000, 150]
    assert shards[0][0] == "000000000"
    assert shards[2][-1] == "000002149"


def test_build_split_manifest_schema():
    shards = [["000000001", "000000002"], ["000000003"]]
    m = build_split_manifest("train", shards, "things/rgb")
    assert m["split"] == "train"
    assert m["n_shards"] == 2
    assert m["n_images"] == 3
    assert m["shards_subpath"] == "things/rgb"
    assert m["shards"]["shard_000"]["n_images"] == 2
    assert m["shards"]["shard_001"]["image_ids"] == ["000000003"]


def test_index_source_shards_rejects_duplicate(tmp_path):
    a = tmp_path / "shard_000.tar"
    b = tmp_path / "shard_001.tar"
    _make_shard(a, {"000000001": "x.jpg"})
    _make_shard(b, {"000000001": "y.jpg"})
    with pytest.raises(ValueError, match="found in both"):
        index_source_shards([str(a), str(b)])


def test_full_repack_pipeline_no_data_loss(tmp_path):
    """End-to-end: 5 src shards → split → repack → every (id, filename, jpg)
    triple round-trips. This is the critical invariant for the Modal job.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    id_to_fn: dict[str, str] = {}
    id_to_jpg: dict[str, bytes] = {}
    for s in range(5):
        items = {}
        for j in range(100):
            image_id = f"{s * 100 + j:09d}"
            filename = f"concept{s:02d}_{j:02d}.jpg"
            items[image_id] = filename
            id_to_fn[image_id] = filename
            id_to_jpg[image_id] = b"\xff\xd8\xff\xe0FAKE" + image_id.encode()
        _make_shard(src_dir / f"shard_{s:03d}.tar", items)

    cat = build_catalog(id_to_fn)
    assert cat["n_images"] == 500

    train_ids, val_ids = image_level_split(id_to_fn.keys(), val_frac=0.20, seed=0)
    train_shards = pack_into_shards(train_ids, images_per_shard=200)
    val_shards = pack_into_shards(val_ids, images_per_shard=200)
    assert sum(len(s) for s in train_shards) == 400
    assert sum(len(s) for s in val_shards) == 100

    src_paths = [str(src_dir / f"shard_{s:03d}.tar") for s in range(5)]
    src_index = index_source_shards(src_paths)

    out_train = tmp_path / "out_train"
    out_val = tmp_path / "out_val"
    out_train.mkdir()
    out_val.mkdir()
    for n, ids in enumerate(train_shards):
        write_shard_from_locations(
            str(out_train / f"shard_{n:03d}.tar"), ids, src_index
        )
    for n, ids in enumerate(val_shards):
        write_shard_from_locations(
            str(out_val / f"shard_{n:03d}.tar"), ids, src_index
        )

    seen_fn: dict[str, str] = {}
    seen_jpg: dict[str, bytes] = {}
    for tar_path in list(out_train.iterdir()) + list(out_val.iterdir()):
        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                if member.name.endswith(".txt"):
                    iid = member.name[:-4]
                    assert iid not in seen_fn, f"duplicate txt {iid}"
                    seen_fn[iid] = tar.extractfile(member).read().decode()
                elif member.name.endswith(".jpg"):
                    iid = member.name[:-4]
                    assert iid not in seen_jpg, f"duplicate jpg {iid}"
                    seen_jpg[iid] = tar.extractfile(member).read()
    assert seen_fn == id_to_fn
    assert seen_jpg == id_to_jpg


# ---------------------------------------------------------------------------
# Catalog-slot ownership tests (proposed Option 1 packing scheme)
# ---------------------------------------------------------------------------
#
# Reference impl lives in-test until things_manifest.py is updated.
# This lets us validate the math + structure without breaking the existing
# `pack_into_shards` sequential packer used by the legacy MEG repack job.


def _catalog_slot_id(image_id: str, images_per_shard: int = 1000) -> str:
    """``shard_NNN`` for an image_id using catalog-slot ownership.

    THINGS ids are 1-indexed (lowest is 000000001), so we subtract 1
    before integer-dividing.
    """
    return f"shard_{(int(image_id) - 1) // images_per_shard:03d}"


def _pack_by_catalog_slot(
    image_ids, images_per_shard: int = 1000
) -> dict[str, list[str]]:
    """Group image_ids by catalog slot. Returns {shard_id: [ids...]}.

    Output ids per shard are sorted ascending. Shards with no matching
    ids in the input are simply absent from the returned dict.
    """
    grouped: dict[str, list[str]] = {}
    for image_id in sorted(set(image_ids)):
        slot = _catalog_slot_id(image_id, images_per_shard)
        grouped.setdefault(slot, []).append(image_id)
    return grouped


class TestCatalogSlotPureFunction:
    def test_first_id_maps_to_shard_000(self):
        assert _catalog_slot_id("000000001") == "shard_000"

    def test_last_id_of_shard_000(self):
        assert _catalog_slot_id("000001000") == "shard_000"

    def test_first_id_of_shard_001(self):
        assert _catalog_slot_id("000001001") == "shard_001"

    @pytest.mark.parametrize(
        "image_id,expected_shard",
        [
            ("000005432", "shard_005"),
            ("000010000", "shard_009"),  # id 10000 -> (10000-1)//1000 = 9
            ("000010001", "shard_010"),
            ("000026000", "shard_025"),
            ("000026001", "shard_026"),
            ("000026107", "shard_026"),
        ],
    )
    def test_other_boundaries(self, image_id, expected_shard):
        assert _catalog_slot_id(image_id) == expected_shard

    def test_custom_images_per_shard(self):
        assert _catalog_slot_id("000000500", images_per_shard=250) == "shard_001"
        assert _catalog_slot_id("000000251", images_per_shard=250) == "shard_001"
        assert _catalog_slot_id("000000250", images_per_shard=250) == "shard_000"


class TestCatalogSlotPackingTrainVal:
    """Verify that train + val tars in the same shard_NNN come from one slot."""

    def test_catalog_slot_packs_train_with_holes(self):
        """Catalog has ids 1..2000, val = {5, 12, 1500}; train shard_001 omits 1500."""
        catalog = [f"{i:09d}" for i in range(1, 2001)]
        val = {"000000005", "000000012", "000001500"}
        train_ids = [i for i in catalog if i not in val]

        train_packs = _pack_by_catalog_slot(train_ids)
        assert set(train_packs) == {"shard_000", "shard_001"}
        assert "000000005" not in train_packs["shard_000"]
        assert "000000012" not in train_packs["shard_000"]
        assert "000001500" not in train_packs["shard_001"]
        # Sanity: 998 train ids in shard_000, 999 train ids in shard_001.
        assert len(train_packs["shard_000"]) == 998
        assert len(train_packs["shard_001"]) == 999

    def test_catalog_slot_packs_val_only_val_ids(self):
        val = ["000000005", "000000012", "000001500"]
        val_packs = _pack_by_catalog_slot(val)
        assert set(val_packs) == {"shard_000", "shard_001"}
        assert val_packs["shard_000"] == ["000000005", "000000012"]
        assert val_packs["shard_001"] == ["000001500"]

    def test_train_val_share_same_shard_ids(self):
        """Same slot index appears in train and val keys."""
        catalog = [f"{i:09d}" for i in range(1, 2001)]
        val = {"000000005", "000001500"}
        train_ids = [i for i in catalog if i not in val]
        train_packs = _pack_by_catalog_slot(train_ids)
        val_packs = _pack_by_catalog_slot(val)
        # Both splits use the same shard naming scheme.
        assert set(train_packs) >= set(val_packs)


class TestEmptyAndPartialShards:
    def test_some_slots_have_empty_val_shard(self):
        """A slot where every id is train -> no entry in val packs."""
        catalog = [f"{i:09d}" for i in range(1, 3001)]
        # Put all val ids in slot 0 only.
        val = {"000000010", "000000020"}
        train_ids = [i for i in catalog if i not in val]

        train_packs = _pack_by_catalog_slot(train_ids)
        val_packs = _pack_by_catalog_slot(val)

        assert set(train_packs) == {"shard_000", "shard_001", "shard_002"}
        assert set(val_packs) == {"shard_000"}
        # Slots 1 and 2 are train-only.
        assert "shard_001" not in val_packs
        assert "shard_002" not in val_packs

    def test_partial_last_shard(self):
        """The last catalog slot may contain fewer than images_per_shard ids."""
        # Real catalog has 26107 ids -> shard_026 has 107 ids.
        last_slot_ids = [f"{i:09d}" for i in range(26001, 26108)]
        packed = _pack_by_catalog_slot(last_slot_ids)
        assert list(packed) == ["shard_026"]
        assert len(packed["shard_026"]) == 107
        assert packed["shard_026"][0] == "000026001"
        assert packed["shard_026"][-1] == "000026107"


class TestCatalogSlotAgainstThingsSplit:
    """End-to-end sanity using the same logic as `things_split.json` would drive."""

    def test_full_catalog_partition(self):
        """Every catalog id lands in exactly one slot."""
        catalog = [f"{i:09d}" for i in range(1, 26108)]
        packed = _pack_by_catalog_slot(catalog)
        assert sum(len(v) for v in packed.values()) == len(catalog)
        assert set().union(*packed.values()) == set(catalog)
        # 27 shards total.
        assert len(packed) == 27

    def test_split_disjoint_and_complete(self):
        """Train ∪ val == catalog AND train ∩ val == ∅ across all shards."""
        catalog = [f"{i:09d}" for i in range(1, 5001)]
        val = {f"{i:09d}" for i in range(100, 4900, 137)}  # ~35 ids spread
        train = set(catalog) - val

        train_packs = _pack_by_catalog_slot(train)
        val_packs = _pack_by_catalog_slot(val)

        all_train = set().union(*train_packs.values())
        all_val = set().union(*val_packs.values())

        assert all_train == train
        assert all_val == val
        assert all_train.isdisjoint(all_val)
        assert all_train | all_val == set(catalog)


def test_no_tmp_files_left_behind(tmp_path):
    """write_shard_from_locations must os.replace the .tmp into place."""
    src = tmp_path / "shard_000.tar"
    _make_shard(src, {"000000001": "x.jpg"})
    idx = index_source_shards([str(src)])
    out = tmp_path / "out.tar"
    write_shard_from_locations(str(out), ["000000001"], idx)
    assert out.exists()
    assert not (tmp_path / "out.tar.tmp").exists()
