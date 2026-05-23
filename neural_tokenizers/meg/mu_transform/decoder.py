"""μ-transform decoder: bin centers → inverse μ-law → inverse max-abs scale.

We do NOT undo clipping — that's lossy by construction; any value clipped at
encode time is gone. The decoder's job is to map a token ID back to the
amplitude that best represents its bin in the original signal scale.

Inverse μ-law:
    F^-1(y) = sgn(y) * ((1 + μ)^|y| - 1) / μ
"""

from __future__ import annotations

import torch

from .calibration import MuCalibration
from .quantizer import dequantize


def decode(tokens: torch.Tensor, calib: MuCalibration) -> torch.Tensor:
    """Map (B, C, T) long tokens back to (B, C, T) float on the input scale.

    Args:
        tokens: long tensor in [0, V), shape (B, C, T).
        calib:  MuCalibration used at encode time.

    Returns:
        x_hat: (B, C, T) float on the same device/dtype family as calib tensors.
    """
    _assert_shape(tokens, calib)
    device = tokens.device
    dtype = calib.scaler.dtype

    # 1) Token -> bin center in [-1, 1], on the token's device.
    y = dequantize(tokens, calib.vocab_size).to(device=device, dtype=dtype)

    # 2) Inverse μ-law expansion back to [-1, 1] linear amplitude.
    mu = float(calib.mu)
    expanded = (torch.pow(1.0 + mu, y.abs()) - 1.0) / mu
    x_norm = torch.sign(y) * expanded

    # 3) Undo max-abs scaling: x_hat = x_norm * s.
    s = calib.scaler.to(device).view(1, -1, 1)
    return x_norm * s


def _assert_shape(tokens: torch.Tensor, calib: MuCalibration) -> None:
    if tokens.ndim != 3:
        raise ValueError(f"expected (B, C, T) tokens; got shape {tuple(tokens.shape)}")
    C = tokens.shape[1]
    if calib.scaler.numel() != C:
        raise ValueError(
            f"calibration has {calib.scaler.numel()} channels, tokens have {C}"
        )
