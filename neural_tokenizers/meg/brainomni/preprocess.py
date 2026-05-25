"""THINGS-MEG → BrainTokenizer preprocessing.

Pipeline (plan § Length alignment):
  1. Resample 200 Hz → 256 Hz along time axis.
  2. Per-trial per-channel z-score (matches BrainOmni sample-level norm).
  3. Zero-pad post-stimulus tail to ``window_length`` (512).
  4. Inverse on decode: trim pad, resample 256 → 200 Hz, re-apply stored scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import BrainOmniConfig, BRAINOMNI_DEFAULT


@dataclass
class PreprocessState:
    """Per-batch normalization stats for inverse transform."""

    mean: torch.Tensor   # (B, C, 1)
    std: torch.Tensor    # (B, C, 1)
    valid_len: int       # samples at target_sfreq before padding


def resample_time(
    x: torch.Tensor,
    source_hz: float,
    target_hz: float,
) -> torch.Tensor:
    """Linear resample ``(B, C, T)`` along the last axis."""
    if source_hz == target_hz:
        return x
    b, c, t_in = x.shape
    t_out = int(round(t_in * target_hz / source_hz))
    flat = x.reshape(b * c, 1, t_in)
    out = F.interpolate(flat, size=t_out, mode="linear", align_corners=False)
    return out.reshape(b, c, t_out)


def zscore_per_trial(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-mean unit-variance per channel per trial. Returns (x_norm, mean, std)."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mean) / std, mean, std


def pad_to_window(
    x: torch.Tensor,
    window_length: int,
    pad_side: str = "right",
) -> tuple[torch.Tensor, int]:
    """Pad time axis to ``window_length``. Returns (padded, valid_len)."""
    valid_len = x.shape[-1]
    if valid_len >= window_length:
        return x[..., :window_length], window_length
    pad = window_length - valid_len
    if pad_side == "right":
        padding = (0, pad)
    elif pad_side == "center":
        left = pad // 2
        padding = (left, pad - left)
    else:
        raise ValueError(f"pad_side must be right|center, got {pad_side!r}")
    return F.pad(x, padding), valid_len


def build_valid_mask(
    batch_size: int,
    n_channels: int,
    window_length: int,
    valid_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Mask for loss: 1 on real samples, 0 on padded tail. Shape (B, C, window_length)."""
    mask = torch.zeros(window_length, device=device, dtype=dtype)
    mask[:valid_len] = 1.0
    return mask.view(1, 1, -1).expand(batch_size, n_channels, -1)


def preprocess_for_braintokenizer(
    x: torch.Tensor,
    cfg: BrainOmniConfig = BRAINOMNI_DEFAULT,
    pad_side: str = "right",
) -> tuple[torch.Tensor, PreprocessState, torch.Tensor]:
    """Full forward preprocess.

    Args:
        x: (B, C, T) float @ source_sfreq_hz.

    Returns:
        x_pad:   (B, C, window_length) @ target_sfreq_hz, z-scored + padded.
        state:   stats for inverse transform.
        mask:    (B, C, window_length) valid-sample mask.
    """
    x_rs = resample_time(x, cfg.source_sfreq_hz, cfg.target_sfreq_hz)
    x_norm, mean, std = zscore_per_trial(x_rs)
    x_pad, valid_len = pad_to_window(x_norm, cfg.window_length, pad_side=pad_side)
    state = PreprocessState(mean=mean, std=std, valid_len=valid_len)
    mask = build_valid_mask(
        x_pad.shape[0],
        x_pad.shape[1],
        cfg.window_length,
        valid_len,
        device=x_pad.device,
    )
    return x_pad, state, mask


def inverse_preprocess(
    x_rec: torch.Tensor,
    state: PreprocessState,
    cfg: BrainOmniConfig = BRAINOMNI_DEFAULT,
    target_t: int | None = None,
) -> torch.Tensor:
    """Map reconstructed BrainTokenizer output back to original THINGS shape.

    Args:
        x_rec: (B, C, N, L) or (B, C, window_length) normalized recon.
        state: from ``preprocess_for_braintokenizer``.
        target_t: output time length (default MEG_DATA.n_timepoints).

    Returns:
        (B, C, target_t) float on the input amplitude scale.
    """
    from ..meg_config import MEG_DATA

    target_t = target_t or MEG_DATA.n_timepoints

    if x_rec.dim() == 4:
        # Single window per trial: N=1
        x_rec = x_rec.squeeze(2)
    x_rec = x_rec[..., : state.valid_len]
    x_denorm = x_rec * state.std + state.mean
    x_out = resample_time(x_denorm, cfg.target_sfreq_hz, cfg.source_sfreq_hz)
    if x_out.shape[-1] != target_t:
        if x_out.shape[-1] > target_t:
            x_out = x_out[..., :target_t]
        else:
            x_out = F.pad(x_out, (0, target_t - x_out.shape[-1]))
    return x_out
