"""Unit tests for BrainOmni preprocessing and adapter."""

from __future__ import annotations

import os

import pytest
import torch

from meg import MEG_DATA
from meg.brainomni.config import BRAINOMNI_DEFAULT
from meg.brainomni.preprocess import (
    inverse_preprocess,
    pad_to_window,
    preprocess_for_braintokenizer,
    resample_time,
    zscore_per_trial,
)


def test_resample_time_shape():
    x = torch.randn(2, 10, 281)
    out = resample_time(x, 200.0, 256.0)
    expected_t = int(round(281 * 256 / 200))
    assert out.shape == (2, 10, expected_t)


def test_zscore_per_trial_unit_variance():
    x = torch.randn(4, 8, 100) * 5 + 3
    normed, mean, std = zscore_per_trial(x)
    assert torch.allclose(normed.mean(dim=-1), torch.zeros(4, 8), atol=1e-5)
    assert torch.allclose(normed.std(dim=-1), torch.ones(4, 8), atol=1e-4)


def test_pad_to_window_right():
    x = torch.randn(1, 5, 360)
    padded, valid = pad_to_window(x, 512, pad_side="right")
    assert padded.shape[-1] == 512
    assert valid == 360
    assert torch.all(padded[..., 360:] == 0)


def test_preprocess_roundtrip_shape():
    x = torch.randn(3, MEG_DATA.n_channels, MEG_DATA.n_timepoints)
    x_pad, state, mask = preprocess_for_braintokenizer(x)
    assert x_pad.shape == (3, MEG_DATA.n_channels, BRAINOMNI_DEFAULT.window_length)
    assert mask.shape == x_pad.shape
    assert mask[..., : state.valid_len].eq(1).all()
    assert mask[..., state.valid_len :].eq(0).all()

    # Fake recon at window length
    x_rec = x_pad.clone()
    out = inverse_preprocess(x_rec, state)
    assert out.shape == x.shape


@pytest.mark.skipif(
    not os.path.isfile(
        os.path.join(
            "..",
            "external/BrainOmni/ckpt_collection/braintokenizer/BrainTokenizer.pt",
        )
    ),
    reason="BrainTokenizer checkpoint not downloaded",
)
def test_braintokenizer_roundtrip():
    from meg.brainomni import BrainOmniTokenizer

    tok = BrainOmniTokenizer.from_checkpoint(device="cpu")
    x = torch.randn(2, MEG_DATA.n_channels, MEG_DATA.n_timepoints)
    tokens = tok.tokenize(x)
    assert tokens.shape == (2, 16, 8, 4)
    assert tokens.max() < tok.codebook_size
    x_hat = tok.decode_tokens(tokens)
    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()
