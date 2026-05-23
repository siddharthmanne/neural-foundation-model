"""Uniform binning of [-1, 1] floats into integer token IDs.

Companded values land in [-1, 1] (modulo clipping numerics, which we re-clamp
to be safe). We split that interval into V equal-width bins and emit the bin
index as the token ID.

Symmetric: bin centers are {-1 + (i + 0.5) * 2/V} for i in [0, V).

Inverse mapping (`dequantize`) returns the bin center, which is what the
decoder's inverse-μ-law expects.
"""

from __future__ import annotations

import torch


def quantize(x_companded: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Map (B, C, T) floats in [-1, 1] to (B, C, T) long token IDs in [0, V).

    Values outside [-1, 1] are clamped first — this handles tiny numerical
    drift from the encoder's float arithmetic without silently producing
    out-of-range tokens.
    """
    _assert_vocab_size(vocab_size)
    x = x_companded.clamp(-1.0, 1.0)
    # Map [-1, 1] -> [0, V) via affine: idx = floor((x + 1) / 2 * V), clamped.
    half_V = 0.5 * vocab_size
    idx = (x + 1.0) * half_V
    return idx.floor().clamp_(0, vocab_size - 1).to(torch.long)


def dequantize(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Map token IDs back to bin-center floats in [-1, 1].

    bin_center(i) = -1 + (i + 0.5) * 2 / V
    """
    _assert_vocab_size(vocab_size)
    bin_width = 2.0 / vocab_size
    return (tokens.to(torch.float32) + 0.5) * bin_width - 1.0


def bin_centers(vocab_size: int, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    """The V bin centers as a 1-D tensor of shape (V,). Useful for plots / tests."""
    _assert_vocab_size(vocab_size)
    bin_width = 2.0 / vocab_size
    idx = torch.arange(vocab_size, dtype=dtype, device=device)
    return (idx + 0.5) * bin_width - 1.0


def _assert_vocab_size(vocab_size: int) -> None:
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
