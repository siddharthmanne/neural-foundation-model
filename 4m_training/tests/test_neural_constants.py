"""Sanity checks for ``neural_constants`` derived values."""

from __future__ import annotations

from neural_constants import (
    EEG_CODE_MAX,
    EEG_TOKENS_PER_TRIAL,
    EEG_TRIAL_SHAPE,
    EEG_VOCAB_SIZE,
    MEG_CODE_MAX,
    MEG_TOKENS_PER_TRIAL,
    MEG_TRIAL_SHAPE,
    MEG_VOCAB_SIZE,
    THINGS_CENTER_CROP,
    THINGS_IMAGE_SIZE,
    THINGS_PATCH_SIZE,
    TOK_DEPTH_VOCAB_SIZE,
    TOK_RGB_TOKENS_PER_IMAGE,
    TOK_RGB_VOCAB_SIZE,
    pretoken_grid_num_tokens,
)


def test_meg_geometry():
    assert MEG_TOKENS_PER_TRIAL == 512
    assert MEG_VOCAB_SIZE == MEG_TOKENS_PER_TRIAL
    assert MEG_CODE_MAX == MEG_VOCAB_SIZE - 1
    assert MEG_TRIAL_SHAPE == (16, 8, 4)


def test_eeg_geometry():
    assert EEG_TOKENS_PER_TRIAL == 17
    assert EEG_CODE_MAX == EEG_VOCAB_SIZE - 1
    assert EEG_TRIAL_SHAPE == (17,)


def test_vision_geometry_matches_stock_4m_224():
    assert TOK_RGB_TOKENS_PER_IMAGE == pretoken_grid_num_tokens(
        THINGS_IMAGE_SIZE, THINGS_PATCH_SIZE
    )
    assert TOK_RGB_TOKENS_PER_IMAGE == 196
    assert TOK_RGB_VOCAB_SIZE == 16384
    assert TOK_DEPTH_VOCAB_SIZE == 8192


def test_things_center_crop():
    top, left, h, w, flip = THINGS_CENTER_CROP
    assert (top, left, flip) == (0, 0, 0)
    assert (h, w) == (THINGS_IMAGE_SIZE, THINGS_IMAGE_SIZE)
