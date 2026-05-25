"""Unit tests for splits.py.

The most important guarantee here is the cross-phase invariant: Phase 1 (μ)
and Phase 2 (Cho2026) must evaluate on identical test trials, even though
they call `split_by_image` with different val_frac. The first test below is
the one that backstops that guarantee — if it ever fails, the leaderboard
comparison across phases breaks silently.

Run:
    cd neural_tokenizers && pytest meg/test_splits.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from meg import SplitDefaults
from meg.splits import Splits, split_by_image


# ----------------------------- fixtures -----------------------------------


@pytest.fixture
def image_ids_per_trial() -> np.ndarray:
    """Realistic shape: ~22k unique image IDs spread across ~27k trials.

    THINGS has ~22,449 unique images and ~27,048 trials per subject — most
    images appear once, some appear twice. We mimic that distribution at
    smaller scale so tests stay fast.
    """
    rng = np.random.default_rng(0)
    n_images = 1000
    n_trials = 1200
    image_pool = np.arange(n_images)
    chosen = rng.choice(image_pool, size=n_trials, replace=True)
    return chosen


@pytest.fixture
def mu_defaults() -> SplitDefaults:
    return SplitDefaults(train_frac=0.90, val_frac=0.0, test_frac=0.10, seed=0)


@pytest.fixture
def learnable_defaults() -> SplitDefaults:
    return SplitDefaults(train_frac=0.80, val_frac=0.10, test_frac=0.10, seed=0)


# ----------------------------- the critical invariant ---------------------


def test_test_set_identical_across_val_fractions(image_ids_per_trial, mu_defaults, learnable_defaults):
    """Same (seed, test_frac), different val_frac → SAME test image IDs.

    This is what makes the §5 leaderboard comparable across Phase 1 and
    Phase 2+. If this test fails, mu-transform and Cho2026 will be graded
    on different held-out images and we'll silently get an unfair comparison.
    """
    s_mu = split_by_image(image_ids_per_trial, mu_defaults)
    s_learn = split_by_image(image_ids_per_trial, learnable_defaults)
    np.testing.assert_array_equal(s_mu.image_ids_test, s_learn.image_ids_test)


def test_test_set_changes_with_seed(image_ids_per_trial, mu_defaults):
    """Same params except seed → different test set. Sanity check on the rng path."""
    seed_a = split_by_image(image_ids_per_trial, mu_defaults)
    seed_b = split_by_image(
        image_ids_per_trial,
        SplitDefaults(train_frac=0.90, val_frac=0.0, test_frac=0.10, seed=42),
    )
    assert not np.array_equal(seed_a.image_ids_test, seed_b.image_ids_test)


# ----------------------------- by-image invariants ------------------------


def test_image_partitions_are_disjoint(image_ids_per_trial, learnable_defaults):
    """No image appears in more than one of train/val/test."""
    s = split_by_image(image_ids_per_trial, learnable_defaults)
    train_imgs = set(image_ids_per_trial[s.train].tolist())
    val_imgs = set(image_ids_per_trial[s.val].tolist())
    test_imgs = set(image_ids_per_trial[s.test].tolist())
    assert train_imgs.isdisjoint(val_imgs)
    assert train_imgs.isdisjoint(test_imgs)
    assert val_imgs.isdisjoint(test_imgs)


def test_no_trial_appears_twice(image_ids_per_trial, learnable_defaults):
    """Trial indices are unique across the three sets — no leakage."""
    s = split_by_image(image_ids_per_trial, learnable_defaults)
    combined = np.concatenate([s.train, s.val, s.test])
    assert len(combined) == len(set(combined.tolist()))


def test_all_trials_covered(image_ids_per_trial, learnable_defaults):
    """Every trial ends up in exactly one of train / val / test."""
    s = split_by_image(image_ids_per_trial, learnable_defaults)
    combined = np.sort(np.concatenate([s.train, s.val, s.test]))
    np.testing.assert_array_equal(combined, np.arange(len(image_ids_per_trial)))


# ----------------------------- fraction sanity ----------------------------


def test_split_sizes_roughly_match_requested_fractions(image_ids_per_trial, learnable_defaults):
    """Trial counts won't exactly match 80/10/10 (images repeat unevenly), but
    they should be within a few percent."""
    s = split_by_image(image_ids_per_trial, learnable_defaults)
    n = len(image_ids_per_trial)
    fracs = {k: v / n for k, v in s.sizes().items()}
    assert abs(fracs["train"] - 0.80) < 0.05
    assert abs(fracs["val"] - 0.10) < 0.05
    assert abs(fracs["test"] - 0.10) < 0.05


def test_val_empty_when_val_frac_zero(image_ids_per_trial, mu_defaults):
    s = split_by_image(image_ids_per_trial, mu_defaults)
    assert len(s.val) == 0
    assert s.val.dtype == np.int64


# ----------------------------- edge cases ---------------------------------


def test_raises_when_no_train_images_left(image_ids_per_trial):
    bad = SplitDefaults(train_frac=0.0, val_frac=0.5, test_frac=0.5, seed=0)
    with pytest.raises(ValueError, match="no train"):
        split_by_image(image_ids_per_trial, bad)


def test_rejects_2d_input():
    bad_input = np.zeros((10, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="1-D"):
        split_by_image(bad_input, SplitDefaults(0.8, 0.1, 0.1, 0))


def test_deterministic_with_same_seed(image_ids_per_trial, learnable_defaults):
    s1 = split_by_image(image_ids_per_trial, learnable_defaults)
    s2 = split_by_image(image_ids_per_trial, learnable_defaults)
    for f in ("train", "val", "test", "image_ids_test"):
        np.testing.assert_array_equal(getattr(s1, f), getattr(s2, f))
