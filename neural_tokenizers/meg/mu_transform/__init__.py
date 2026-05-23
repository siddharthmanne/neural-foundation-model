"""μ-transform tokenizer (Phase 1).

Public surface:
    MuTransformTokenizer  — Tokenizer-protocol class
    MuCalibration         — fitted parameters (per-channel clip + scaler + μ + V)
    fit_calibration       — estimate MuCalibration from a subsample
    run_slug              — deterministic short slug for a config (used for
                            per-run output subdirs so sweeps don't overwrite)
"""

from .calibration import MuCalibration, fit_calibration
from .tokenizer import MuTransformTokenizer


def run_slug(mu: float, vocab_size: int, clip_lo_pct: float, clip_hi_pct: float,
             channel_mode: str, seed: int) -> str:
    """Stable short name for a (config, seed) tuple.

    Format: V<vocab>_mu<mu>_clip<lo>-<hi>_<mode>_s<seed>
        e.g. V256_mu255_clip0.5-99.5_per_channel_s0

    Used to put each calibration/eval pair in its own subdir so sweeps don't
    clobber each other. Short enough for filesystem paths; human-readable
    so you can eyeball what config a directory came from.
    """
    return (
        f"V{vocab_size}_mu{int(mu) if float(mu).is_integer() else mu}"
        f"_clip{clip_lo_pct}-{clip_hi_pct}"
        f"_{channel_mode}_s{seed}"
    )


__all__ = ["MuCalibration", "MuTransformTokenizer", "fit_calibration", "run_slug"]
