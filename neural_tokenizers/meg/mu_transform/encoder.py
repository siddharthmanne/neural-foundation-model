"""μ-transform encoder: clip → max-abs scale → μ-law compand.

Stateless transformation parameterized by a `MuCalibration` (the per-channel
clip thresholds and scaler). The encoder does NOT quantize — that's quantizer.py.
Splitting "compress amplitude" from "discretize" keeps each piece short and
test-isolated:
  * encoder maps (B, C, T) float → (B, C, T) float in [-1, 1]
  * quantizer maps (B, C, T) float in [-1, 1] → (B, C, T) long in [0, V)

All ops are pure torch and run on whatever device the input tensor lives on.
"""

from __future__ import annotations

import torch

from .calibration import MuCalibration


def encode(x: torch.Tensor, calib: MuCalibration) -> torch.Tensor:
    """Apply the three encoder steps and return a tensor in [-1, 1].

    Args:
        x:     (B, C, T) float tensor on any device.
        calib: fitted MuCalibration (per-channel clip + scaler, μ).

    Returns:
        x_companded: (B, C, T) float in [-1, 1], same device/dtype as `x`.
    """
    _assert_shape(x, calib)

    # 1) Clip to per-channel [q_lo, q_hi]. Broadcast: (1, C, 1).
    lo = calib.clip_lo.to(x.device).view(1, -1, 1)
    hi = calib.clip_hi.to(x.device).view(1, -1, 1)
    x_clipped = torch.clamp(x, min=lo, max=hi)

    # 2) Max-abs scale to [-1, 1] using the per-channel `s`.
    s = calib.scaler.to(x.device).view(1, -1, 1)
    x_scaled = x_clipped / s.clamp_min(1e-12)

    # 3) μ-law compand: sgn(x) * log(1 + μ|x|) / log(1 + μ).
    mu = float(calib.mu)
    abs_x = x_scaled.abs()
    log_term = torch.log1p(mu * abs_x) / torch.log(torch.tensor(1.0 + mu, dtype=x.dtype, device=x.device))
    return torch.sign(x_scaled) * log_term


def _assert_shape(x: torch.Tensor, calib: MuCalibration) -> None:
    if x.ndim != 3:
        raise ValueError(f"expected (B, C, T); got shape {tuple(x.shape)}")
    C = x.shape[1]
    if calib.clip_lo.numel() != C:
        raise ValueError(
            f"calibration has {calib.clip_lo.numel()} channels, input has {C}"
        )
