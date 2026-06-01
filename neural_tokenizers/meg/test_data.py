"""Unit tests for label mappings in meg/data.py.

These tests do not require MNE or the THINGS-MEG .fif files — they exercise
the JSON-only parts of `data.py` (ConceptMapping, SuperordinateMapping). MNE
is only imported lazily by the .fif loaders, so this file runs anywhere
pandas/numpy/torch are available.

Run:
    cd neural_tokenizers && pytest meg/test_data.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import torch

from meg.data import (
    ANIMATE_SUPER_NAMES,
    AnimacyMapping,
    ConceptMapping,
    SuperordinateMapping,
    average_trials_by_image,
)


# ---------- ConceptMapping ------------------------------------------------


def test_concept_mapping_from_json_builds_dense_index():
    payload = {"image_id_to_concept_id": {"100": 5, "200": 5, "300": 9, "400": 12}}
    cm = ConceptMapping.from_json(payload)
    # Three unique concepts → dense indices [0, 1, 2] in sorted order.
    assert cm.n_concepts == 3
    np.testing.assert_array_equal(cm.concept_ids, np.array([5, 9, 12]))


def test_concept_mapping_encode_marks_unknown_invalid():
    payload = {"image_id_to_concept_id": {"1": 100, "2": 200}}
    cm = ConceptMapping.from_json(payload)
    image_ids = np.array([1, 999, 2], dtype=np.int64)
    labels, valid = cm.encode(image_ids)
    assert valid.tolist() == [True, False, True]
    # 100 → dense 0, 200 → dense 1
    assert labels[0] == 0 and labels[2] == 1


# ---------- SuperordinateMapping -----------------------------------------


@pytest.fixture
def super_payload():
    """27-category payload with a few concepts mapped, one multi-membership."""
    names = [
        "animal", "body part", "clothing", "container", "device", "drink",
        "food", "fruit", "furniture", "home decor", "kitchen appliance",
        "kitchen tool", "medical equipment", "musical instrument", "office",
        "personal hygiene", "plant", "sports equipment", "tool", "toy",
        "vegetable", "vehicle", "weapon", "headwear", "jewelry", "footwear",
        "outdoor",
    ]
    return {
        "concept_id_to_superordinate_index": {
            "1": 0,    # concept 1 → animal
            "2": 6,    # concept 2 → food
            "3": 21,   # concept 3 → vehicle
        },
        "category_names": names,
        "n_categories": 27,
        "n_concepts_with_label": 3,
        "n_concepts_multi_membership": 1,
        "multi_membership": {"4": ["food", "fruit"]},
    }


def test_super_mapping_from_json_validates_27_categories(super_payload):
    sm = SuperordinateMapping.from_json(super_payload)
    assert sm.n_categories == 27
    assert len(sm.category_names) == 27


def test_super_mapping_rejects_wrong_category_count(super_payload):
    bad = dict(super_payload, n_categories=10, category_names=["x"] * 10)
    with pytest.raises(ValueError, match="27 categories"):
        SuperordinateMapping.from_json(bad)


def test_super_mapping_encode_drops_multi_membership_concepts(super_payload):
    sm = SuperordinateMapping.from_json(super_payload)
    cm = ConceptMapping.from_json(
        {"image_id_to_concept_id": {"10": 1, "20": 2, "30": 4, "40": 99}}
    )
    image_ids = np.array([10, 20, 30, 40], dtype=np.int64)
    labels, valid = sm.encode_image_ids(image_ids, cm)
    # 10→concept 1→super 0 (valid)
    # 20→concept 2→super 6 (valid)
    # 30→concept 4→multi-membership, excluded from super map (invalid)
    # 40→concept 99→unknown in concept_map (invalid)
    assert valid.tolist() == [True, True, False, False]
    assert labels[0] == 0
    assert labels[1] == 6
    assert labels[2] == -1
    assert labels[3] == -1


def test_super_mapping_labels_always_in_range(super_payload):
    sm = SuperordinateMapping.from_json(super_payload)
    cm = ConceptMapping.from_json(
        {"image_id_to_concept_id": {"1": 1, "2": 2, "3": 3}}
    )
    image_ids = np.array([1, 2, 3], dtype=np.int64)
    labels, valid = sm.encode_image_ids(image_ids, cm)
    valid_labels = labels[valid]
    assert (valid_labels >= 0).all()
    assert (valid_labels < 27).all()


def test_super_mapping_round_trip_through_disk(super_payload, tmp_path):
    path = tmp_path / "super.json"
    path.write_text(json.dumps(super_payload))
    sm = SuperordinateMapping.load(path)
    assert sm.n_categories == 27
    assert sm.concept_id_to_super[1] == 0


# ---------- AnimacyMapping ------------------------------------------------


@pytest.fixture
def animacy_super_payload():
    """Minimal 27-cat payload that includes animal/bird/insect at known indices."""
    # category_names list MUST include the animate three at SOME positions.
    # Order matches the alphabetical sort that the real downloader produces.
    names = [
        "animal", "bird", "body part", "clothing", "clothing accessory",
        "container", "dessert", "drink", "electronic device", "food",
        "fruit", "furniture", "home decor", "insect", "kitchen appliance",
        "kitchen tool", "medical equipment", "musical instrument",
        "office supply", "part of car", "plant", "sports equipment",
        "tool", "toy", "vegetable", "vehicle", "weapon",
    ]
    # animal=0, bird=1, insect=13
    return {
        "concept_id_to_superordinate_index": {
            "1": 0,    # animal     → animate
            "2": 1,    # bird       → animate
            "3": 13,   # insect     → animate
            "4": 9,    # food       → inanimate
            "5": 25,   # vehicle    → inanimate
        },
        "category_names": names,
        "n_categories": 27,
        "n_concepts_with_label": 5,
        "n_concepts_multi_membership": 0,
        "multi_membership": {},
    }


def test_animacy_mapping_resolves_animate_indices(animacy_super_payload):
    sm = SuperordinateMapping.from_json(animacy_super_payload)
    am = AnimacyMapping.from_super_map(sm)
    # animal=0, bird=1, insect=13 by the alphabetical sort
    assert am.animate_super_indices == frozenset({0, 1, 13})


def test_animacy_mapping_rejects_unknown_animate_name(animacy_super_payload):
    sm = SuperordinateMapping.from_json(animacy_super_payload)
    with pytest.raises(ValueError, match="not found in superordinate"):
        AnimacyMapping.from_super_map(sm, animate_super_names=frozenset({"dragon"}))


def test_animacy_encode_binary_output(animacy_super_payload):
    sm = SuperordinateMapping.from_json(animacy_super_payload)
    am = AnimacyMapping.from_super_map(sm)
    cm = ConceptMapping.from_json(
        {"image_id_to_concept_id": {"10": 1, "20": 2, "30": 3, "40": 4, "50": 5, "60": 99}}
    )
    image_ids = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
    labels, valid = am.encode_image_ids(image_ids, cm)
    # 10→animal=animate, 20→bird=animate, 30→insect=animate, 40→food=inanim, 50→vehicle=inanim, 60→unknown
    assert valid.tolist() == [True, True, True, True, True, False]
    assert labels[:5].tolist() == [1, 1, 1, 0, 0]


def test_animate_set_matches_documented_strict_definition():
    """Sanity check the strict animate set hasn't drifted silently."""
    assert ANIMATE_SUPER_NAMES == frozenset({"animal", "bird", "insect"})


# ---------- average_trials_by_image --------------------------------------


def test_average_trials_collapses_groups():
    """Three trials for image 7, two for image 3 → 2 output rows, means correct."""
    X = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],   # image 7
            [[3.0, 3.0], [3.0, 3.0]],   # image 7
            [[5.0, 5.0], [5.0, 5.0]],   # image 7  → mean 3.0
            [[2.0, 2.0], [2.0, 2.0]],   # image 3
            [[4.0, 4.0], [4.0, 4.0]],   # image 3  → mean 3.0
        ]
    )
    image_ids = np.array([7, 7, 7, 3, 3], dtype=np.int64)
    X_avg, ids_unique = average_trials_by_image(X, image_ids)
    # np.unique returns sorted unique ids → [3, 7]
    np.testing.assert_array_equal(ids_unique, np.array([3, 7]))
    assert X_avg.shape == (2, 2, 2)
    # image 3 mean = 3, image 7 mean = 3 (coincidence in this fixture)
    assert torch.allclose(X_avg[0], torch.full((2, 2), 3.0))
    assert torch.allclose(X_avg[1], torch.full((2, 2), 3.0))


def test_average_trials_preserves_singleton_groups():
    """Single-trial image → averaged tensor row equals the original trial."""
    X = torch.randn(3, 4, 8)
    image_ids = np.array([1, 2, 3], dtype=np.int64)
    X_avg, ids_unique = average_trials_by_image(X, image_ids)
    np.testing.assert_array_equal(ids_unique, np.array([1, 2, 3]))
    assert torch.allclose(X_avg, X.to(torch.float32), atol=1e-5)


def test_average_trials_rejects_length_mismatch():
    X = torch.randn(5, 4, 8)
    image_ids = np.array([1, 2, 3], dtype=np.int64)  # wrong length
    with pytest.raises(ValueError, match="must match image_ids length"):
        average_trials_by_image(X, image_ids)


def test_average_trials_accepts_numpy_input():
    """Helper should accept numpy arrays directly (not require torch)."""
    X = np.array([[[1.0]], [[3.0]]])  # (2, 1, 1)
    image_ids = np.array([5, 5], dtype=np.int64)
    X_avg, ids_unique = average_trials_by_image(X, image_ids)
    assert X_avg.shape == (1, 1, 1)
    assert torch.allclose(X_avg[0, 0, 0], torch.tensor(2.0))
