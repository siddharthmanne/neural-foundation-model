"""μ-transform calibration: fit per-channel clip percentiles + scaler.

There is no gradient descent here — this is the "fit" step that produces the
dataset-dependent constants the μ-transform needs:

  * clip_lo[c], clip_hi[c]   per-channel percentile clip thresholds
  * scaler[c]                 per-channel max-abs scaler s, derived from
                              the clip thresholds (a clipped signal can't
                              exceed max(|q_lo|, |q_hi|) by construction)

These are saved as a small JSON sidecar and reloaded at inference. Treat
the JSON as a versioned tokenizer "checkpoint" (per meg/CLAUDE.md §10).

Why we derive `s` from the clip thresholds instead of computing a separate
max|x|: after clipping, max|x| is exactly max(|q_lo|, |q_hi|). Two birds,
one pass.

Why fit on a subsample: per-channel quantile estimates at 0.5% / 99.5%
converge in ~1k trials × 281 samples = ~280k samples per channel. The full
train split (~24k trials × 281 ≈ 7M samples) is overkill. Subsampling lets
us do one GPU torch.quantile call instead of streaming-histogram code.

All math is torch — runs on whatever device the input tensor lives on
(CPU for tests, GPU on Modal).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import torch

from ..meg_config import MuTransformConfig


@dataclass
class MuCalibration:
    """Fitted parameters for the μ-transform.

    Tensor fields are 1-D on CPU; the encoder/decoder move them to the input
    device on demand. Storing on CPU lets us pickle/JSON the calibration
    portably without an explicit device dance.

    Attributes:
        clip_lo:    (C,) per-channel lower clip threshold (raw amplitude units).
        clip_hi:    (C,) per-channel upper clip threshold.
        scaler:     (C,) per-channel max-abs scaler `s`.
        mu:         companding parameter (a float, not per-channel).
        vocab_size: V (a float-friendly int).
        clip_lo_pct, clip_hi_pct: percentiles used at fit time (for the JSON
                    record so we can audit later).
        channel_mode: "per_channel" or "global" — also recorded for audit.
    """

    clip_lo: torch.Tensor
    clip_hi: torch.Tensor
    scaler: torch.Tensor
    mu: float
    vocab_size: int
    clip_lo_pct: float
    clip_hi_pct: float
    channel_mode: str

    @property
    def n_channels(self) -> int:
        return int(self.clip_lo.numel())

    # ---- IO ----------------------------------------------------------

    def to_json(self) -> dict:
        """Serializable dict. Tensors become lists; everything else passes through."""
        return {
            "clip_lo": self.clip_lo.tolist(),
            "clip_hi": self.clip_hi.tolist(),
            "scaler": self.scaler.tolist(),
            "mu": float(self.mu),
            "vocab_size": int(self.vocab_size),
            "clip_lo_pct": float(self.clip_lo_pct),
            "clip_hi_pct": float(self.clip_hi_pct),
            "channel_mode": str(self.channel_mode),
        }

    @classmethod
    def from_json(cls, d: dict) -> "MuCalibration":
        return cls(
            clip_lo=torch.tensor(d["clip_lo"], dtype=torch.float32),
            clip_hi=torch.tensor(d["clip_hi"], dtype=torch.float32),
            scaler=torch.tensor(d["scaler"], dtype=torch.float32),
            mu=float(d["mu"]),
            vocab_size=int(d["vocab_size"]),
            clip_lo_pct=float(d["clip_lo_pct"]),
            clip_hi_pct=float(d["clip_hi_pct"]),
            channel_mode=str(d["channel_mode"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MuCalibration":
        return cls.from_json(json.loads(Path(path).read_text()))


# ---------- fit -----------------------------------------------------------

@torch.no_grad()
def fit_calibration(
    X_sample: torch.Tensor,
    config: MuTransformConfig,
) -> MuCalibration:
    """Estimate clip thresholds + per-channel scaler from a subsample.

    Args:
        X_sample: (N, C, T) float tensor — a uniformly-drawn subsample of
                  trials from the *training* split. Put it on the device
                  you want the quantile reduction to run on (typically GPU
                  on Modal, CPU in tests).
        config:   MuTransformConfig (μ, V, percentiles, channel_mode).

    Returns:
        MuCalibration with CPU tensors.

    Notes on quantile computation:
      - per_channel: flatten over (batch, time) per channel → quantile per ch.
      - global:      flatten everything → single scalar quantile broadcast
                     across all channels.
    """
    if X_sample.ndim != 3:
        raise ValueError(f"X_sample must be (N, C, T); got {tuple(X_sample.shape)}")
    if X_sample.shape[0] == 0:
        raise ValueError("X_sample is empty — cannot fit calibration on 0 trials.")

    N, C, T = X_sample.shape
    q_lo = config.clip_lo_pct / 100.0
    q_hi = config.clip_hi_pct / 100.0

    if config.channel_mode == "per_channel":
        # (C, N*T) — torch.quantile reduces over the last dim by default.
        flat = X_sample.permute(1, 0, 2).reshape(C, N * T).contiguous()
        clip_lo, clip_hi = _safe_quantile(flat, q_lo, q_hi)
    elif config.channel_mode == "global":
        flat = X_sample.reshape(-1)
        lo_g, hi_g = _safe_quantile(flat.unsqueeze(0), q_lo, q_hi)
        clip_lo = lo_g.expand(C).clone()
        clip_hi = hi_g.expand(C).clone()
    else:
        raise ValueError(f"unknown channel_mode {config.channel_mode!r}")

    # After clipping, |x| <= max(|q_lo|, |q_hi|), so that's exactly the scaler.
    scaler = torch.maximum(clip_lo.abs(), clip_hi.abs()).clamp_min(1e-12)

    return MuCalibration(
        clip_lo=clip_lo.cpu().to(torch.float32),
        clip_hi=clip_hi.cpu().to(torch.float32),
        scaler=scaler.cpu().to(torch.float32),
        mu=config.mu,
        vocab_size=config.vocab_size,
        clip_lo_pct=config.clip_lo_pct,
        clip_hi_pct=config.clip_hi_pct,
        channel_mode=config.channel_mode,
    )


def _safe_quantile(
    flat: torch.Tensor,
    q_lo: float,
    q_hi: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row quantile that handles torch.quantile's 16M-sample limit.

    `torch.quantile` errors above ~16M elements per reduction dim on GPU.
    For pooled THINGS-MEG calibration (2k trials × 281 ≈ 562k samples per
    channel) we're well under that, but the helper is defensive: if a row
    exceeds the limit we sub-sample it down before reducing.
    """
    MAX_PER_ROW = 16_000_000 - 1
    if flat.shape[-1] > MAX_PER_ROW:
        idx = torch.randperm(flat.shape[-1], device=flat.device)[:MAX_PER_ROW]
        flat = flat[:, idx]
    q = torch.tensor([q_lo, q_hi], dtype=flat.dtype, device=flat.device)
    out = torch.quantile(flat, q, dim=-1)   # (2, n_rows)
    return out[0].contiguous(), out[1].contiguous()
