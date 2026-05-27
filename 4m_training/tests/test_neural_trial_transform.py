"""Unit tests for ``neural_trial_transform`` (MEG + EEG trial sampling)."""

from __future__ import annotations

import numpy as np
import pytest

from neural_constants import MEG_GRID_SHAPE
from neural_trial_transform import (
    EEG_TRIAL_SHAPE,
    EEG_TOKENS_PER_TRIAL,
    MEG_TRIAL_SHAPE,
    SENTINEL_VALUE,
    EegTrialSampleTransform,
    MegTrialSampleTransform,
    is_placeholder,
)


def _real_meg(n_trials: int, fill: int = 7) -> np.ndarray:
    arr = np.full((n_trials, *MEG_TRIAL_SHAPE), fill, dtype=np.int16)
    for t in range(n_trials):
        arr[t] = fill + t
    return arr


def _placeholder_meg() -> np.ndarray:
    return np.full((1, *MEG_TRIAL_SHAPE), SENTINEL_VALUE, dtype=np.int16)


def _real_eeg(n_trials: int = 3) -> np.ndarray:
    arr = np.full((n_trials, *EEG_TRIAL_SHAPE), 5, dtype=np.int16)
    for t in range(n_trials):
        arr[t] = 5 + t
    return arr


def _placeholder_eeg() -> np.ndarray:
    return np.full((1, *EEG_TRIAL_SHAPE), SENTINEL_VALUE, dtype=np.int16)


class TestMegPlaceholder:
    def test_sentinel_detected(self):
        assert is_placeholder(_placeholder_meg(), MEG_TRIAL_SHAPE)

    def test_real_trial_not_detected(self):
        assert not is_placeholder(_real_meg(1), MEG_TRIAL_SHAPE)

    def test_zeros_not_detected(self):
        zeros = np.zeros((1, *MEG_TRIAL_SHAPE), dtype=np.int16)
        assert not is_placeholder(zeros, MEG_TRIAL_SHAPE)

    def test_wrong_shape_not_detected(self):
        wrong = np.full((2, *MEG_TRIAL_SHAPE), SENTINEL_VALUE, dtype=np.int16)
        assert not is_placeholder(wrong, MEG_TRIAL_SHAPE)


class TestMegTrialSampleTransform:
    def test_picks_one_trial_from_stack(self):
        arr = _real_meg(4)
        transform = MegTrialSampleTransform(training=True, seed=0)
        tokens, valid = transform(arr)
        assert tokens.shape == MEG_GRID_SHAPE  # (128, 4) = cells x RVQ layers
        assert tokens.dtype == np.int32
        assert valid is True
        candidates = [arr[t].astype(np.int32).reshape(MEG_GRID_SHAPE) for t in range(4)]
        assert any(np.array_equal(tokens, c) for c in candidates)

    def test_sentinel_returns_zero_tokens_and_false_mask(self):
        transform = MegTrialSampleTransform(training=True, seed=0)
        tokens, valid = transform(_placeholder_meg())
        assert tokens.shape == MEG_GRID_SHAPE
        assert (tokens == 0).all()
        assert valid is False

    def test_deterministic_in_eval_mode(self):
        arr = _real_meg(4)
        t_eval = MegTrialSampleTransform(training=False)
        first, _ = t_eval(arr)
        for _ in range(5):
            other, _ = t_eval(arr)
            assert np.array_equal(first, other)
        assert np.array_equal(first, arr[0].astype(np.int32).reshape(MEG_GRID_SHAPE))

    def test_random_across_calls_in_train_mode(self):
        transform = MegTrialSampleTransform(training=True, seed=42)
        seen = set()
        for _ in range(200):
            tokens, _ = transform(_real_meg(8))
            seen.add(int(tokens[0, 0]))
        assert len(seen) >= 4

    def test_handles_single_trial_image(self):
        arr = _real_meg(1)
        tokens, valid = MegTrialSampleTransform(training=True, seed=0)(arr)
        assert valid is True
        assert np.array_equal(tokens, arr[0].astype(np.int32).reshape(MEG_GRID_SHAPE))

    def test_handles_48_trial_test_image(self):
        tokens, valid = MegTrialSampleTransform(training=True, seed=0)(_real_meg(48))
        assert valid is True
        assert tokens.shape == MEG_GRID_SHAPE

    def test_rejects_wrong_inner_shape(self):
        arr = np.zeros((4, 32, 8, 4), dtype=np.int16)
        with pytest.raises(ValueError, match=r"shape"):
            MegTrialSampleTransform()(arr)

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError):
            MegTrialSampleTransform()(np.zeros(MEG_TRIAL_SHAPE, dtype=np.int16))

    def test_seed_makes_training_reproducible(self):
        arr = _real_meg(8)
        a = MegTrialSampleTransform(training=True, seed=123)
        b = MegTrialSampleTransform(training=True, seed=123)
        for _ in range(10):
            t_a, _ = a(arr)
            t_b, _ = b(arr)
            assert np.array_equal(t_a, t_b)


class TestEegTrialSampleTransform:
    def test_output_shape(self):
        tokens, valid = EegTrialSampleTransform(training=False)(_real_eeg(4))
        assert tokens.shape == (EEG_TOKENS_PER_TRIAL,)
        assert valid is True

    def test_eval_uses_trial_zero(self):
        tokens, _ = EegTrialSampleTransform(training=False)(_real_eeg(4))
        assert tokens[0] == 5

    def test_train_random_trial(self):
        tx = EegTrialSampleTransform(training=True, seed=42)
        seen = set()
        for _ in range(30):
            tokens, _ = tx(_real_eeg(8))
            seen.add(int(tokens[0]))
        assert len(seen) > 1

    def test_placeholder_returns_zeros_and_invalid(self):
        tokens, valid = EegTrialSampleTransform(training=True)(_placeholder_eeg())
        assert valid is False
        assert (tokens == 0).all()

    def test_is_placeholder_eeg(self):
        assert is_placeholder(_placeholder_eeg(), EEG_TRIAL_SHAPE)
        assert not is_placeholder(_real_eeg(1), EEG_TRIAL_SHAPE)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            EegTrialSampleTransform()(np.zeros((2, 16), dtype=np.int16))
