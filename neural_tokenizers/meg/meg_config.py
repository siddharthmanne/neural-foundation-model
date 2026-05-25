"""MEG-wide constants and per-phase tokenizer defaults.

All magic numbers for the MEG tokenizers live here (parent §7 "no magic
constants" rule). Each tokenizer phase imports its own dataclass from this
module and never hardcodes literals.

Why dataclasses instead of one big constants module:
  - `MEGDataSpec` is fixed by the dataset (verified on 2026-05-22 — see
    meg/CLAUDE.md §3). It does NOT change between phases.
  - `MuTransformConfig` is Phase 1 — μ/V/percentiles can be swept per §9.
  - `EvalDefaults` is shared eval-harness knobs (PSD range, bands) that
    `EvalConfig` from the evaluation package gets populated from.

Per-phase config is intentionally a dataclass not a yaml: it's small,
type-checkable, and importable directly into tests without a yaml loader.
The Cho 2026 yaml configs in `external/` are reference material, not runtime
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------- THINGS-MEG dataset spec (verified 2026-05-22) -----------------
#
# Source of truth: re-run modal/modal_inspect_things_meg.py if any preprocessing
# changes. The numbers below correspond to /project/data/things-meg/preprocessed/.

@dataclass(frozen=True)
class MEGDataSpec:
    """Immutable description of the preprocessed THINGS-MEG data layout."""

    n_channels: int = 271
    n_timepoints: int = 281
    sfreq_hz: float = 200.0
    tmin_s: float = -0.100
    tmax_s: float = 1.300
    n_trials_per_subject: int = 27048
    subjects: tuple[str, ...] = ("P1", "P2", "P3", "P4")
    data_dir: str = "/project/data/things-meg/preprocessed"

    @property
    def trial_shape(self) -> tuple[int, int]:
        """(channels, time) per trial."""
        return (self.n_channels, self.n_timepoints)


MEG_DATA = MEGDataSpec()


# ---------- Phase 1: μ-transform tokenizer config -------------------------
#
# Cho 2026 reference:
#   external/Cho2026_Tokenizer/models/tokenizer/mu_transform/config.yml
#   (mu=255, n_tokens=256, normalization=max_abs).
# Their clip percentiles are implicit in the paper; we default to a symmetric
# 0.5 / 99.5 split. Per-channel scaler is the strong prior in meg/CLAUDE.md §5.

@dataclass(frozen=True)
class MuTransformConfig:
    """All knobs for the μ-transform tokenizer.

    Fields:
        mu:           companding parameter (paper default: 255).
        vocab_size:   number of uniform bins in [-1, 1] (paper default: 256).
        clip_lo_pct:  lower clip percentile in (0, 100).
        clip_hi_pct:  upper clip percentile.
        channel_mode: "per_channel" or "global" max-abs scaler.

    Why these defaults: μ=255, V=256 reproduce the paper baseline. Per-channel
    is safer than global because inter-sensor amplitude variance is real even
    after preprocessing (CLAUDE.md §5 caveat).
    """

    mu: float = 255.0
    vocab_size: int = 256
    clip_lo_pct: float = 0.5
    clip_hi_pct: float = 99.5
    channel_mode: str = "per_channel"   # or "global"

    def __post_init__(self):
        if self.channel_mode not in ("per_channel", "global"):
            raise ValueError(
                f"channel_mode must be per_channel|global, got {self.channel_mode!r}"
            )
        if not (0.0 < self.clip_lo_pct < self.clip_hi_pct < 100.0):
            raise ValueError(
                f"need 0 < clip_lo_pct < clip_hi_pct < 100, got "
                f"({self.clip_lo_pct}, {self.clip_hi_pct})"
            )
        if self.vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {self.vocab_size}")
        if self.mu <= 0:
            raise ValueError(f"mu must be > 0, got {self.mu}")


MU_TRANSFORM_DEFAULT = MuTransformConfig()


# ---------- Eval-harness defaults (passed into EvalConfig) ----------------

MEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 80.0),
}


@dataclass(frozen=True)
class EvalDefaults:
    """Defaults for the §5 harness when applied to MEG.

    Used by the Modal entrypoint to construct an EvalConfig — keeping these
    here means the harness stays modality-agnostic and all MEG-specific knobs
    live in meg_config.py.
    """

    sample_rate_hz: float = MEG_DATA.sfreq_hz
    bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(MEG_BANDS)
    )
    psd_nperseg: int = 128       # ~0.64 s window → multiple windows fit in 1.4 s trial
    probe_epochs: int = 200
    probe_top_k: tuple[int, ...] = (1, 5)
    probe_test_frac: float = 0.2


EVAL_DEFAULTS = EvalDefaults()


# ---------- Split policy defaults -----------------------------------------
#
# Phase-1 split is by image_id, not by trial — see meg/CLAUDE.md §9. When the
# team-wide things_split.json is finalized, only `splits.py::split_by_image`'s
# internals change; the (train/val/test) interface every downstream caller
# uses stays the same.
#
# Why two SplitDefaults variants:
#   - μ-transform (Phase 1) is non-learnable; it needs only calibration data
#     + held-out eval data. val_frac=0.0 saves 10% of trials for calibration.
#   - Cho2026 (Phase 2) is a learned AE; it needs a val split for early
#     stopping / lr scheduling. val_frac=0.10.
#
# Critical invariant maintained by splits.py: the TEST SET is determined by
# (seed, test_frac) alone — independent of val_frac. So Phase 1 and Phase 2
# evaluate on identical trials, and the §5 leaderboard is apples-to-apples.

@dataclass(frozen=True)
class SplitDefaults:
    train_frac: float
    val_frac: float
    test_frac: float
    seed: int = 0

    def __post_init__(self):
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")
        for name, v in (("train_frac", self.train_frac), ("val_frac", self.val_frac),
                        ("test_frac", self.test_frac)):
            if v < 0.0 or v > 1.0:
                raise ValueError(f"{name}={v} must be in [0, 1]")


# Phase 1: μ-transform — no val.
MU_SPLIT_DEFAULTS = SplitDefaults(train_frac=0.90, val_frac=0.0, test_frac=0.10, seed=0)

# Phase 2+: learnable tokenizers — val for early stopping.
LEARNABLE_SPLIT_DEFAULTS = SplitDefaults(
    train_frac=0.80, val_frac=0.10, test_frac=0.10, seed=0
)


# ---------- Phase 3: BrainOmni BrainTokenizer config ----------------------

@dataclass(frozen=True)
class BrainOmniConfig:
    """BrainTokenizer adapter knobs — see meg/brainomni/."""

    source_sfreq_hz: float = MEG_DATA.sfreq_hz
    target_sfreq_hz: float = 256.0
    window_length: int = 512
    overlap_ratio: float = 0.0
    codebook_size: int = 512
    num_quantizers: int = 4
    n_latent_sources: int = 16
    ckpt_dir: str = "external/BrainOmni/ckpt_collection/braintokenizer"
    brainomni_repo: str = "external/BrainOmni"

    @property
    def resampled_n_timepoints(self) -> int:
        ratio = self.target_sfreq_hz / self.source_sfreq_hz
        return int(round(MEG_DATA.n_timepoints * ratio))

    @property
    def pad_width(self) -> int:
        return self.window_length - self.resampled_n_timepoints


BRAINOMNI_DEFAULT = BrainOmniConfig()

# Reconstruction eval compares decode output back at THINGS rate (200 Hz).
BRAINOMNI_EVAL_DEFAULTS = EvalDefaults(
    sample_rate_hz=MEG_DATA.sfreq_hz,
    psd_nperseg=EVAL_DEFAULTS.psd_nperseg,
)
