"""4M modality registration entries for MEG tokenizers.

BrainOmni uses a different token grid than sample-level μ-transform / Cho2026.
Register as a dedicated modality so 4M sees structured multi-codebook seq input.
"""

from __future__ import annotations

from ..meg_config import BRAINOMNI_DEFAULT, MEG_DATA, MU_TRANSFORM_DEFAULT

# Sample-level tokenizers (Phase 1 μ-transform, Phase 2 Cho2026 target shape).
MEG_TOKENS_MU_TRANSFORM = {
    "name": "meg_tokens",
    "type": "seq",
    "num_channels": MEG_DATA.n_channels,
    "max_length": MEG_DATA.n_timepoints,
    "vocab_size": MU_TRANSFORM_DEFAULT.vocab_size,
    "num_codebooks": 1,
    "sample_rate_hz": MEG_DATA.sfreq_hz,
    "description": "Per-channel sample-level μ-transform tokens (271 × 281).",
}

# BrainOmni BrainTokenizer (Phase 3): latent-source × temporal × RVQ layers.
# SEANet ratios [8,4,2] compress 512 samples → 8 temporal tokens per window.
MEG_TOKENS_BRAINOMNI = {
    "name": "meg_tokens_brainomni",
    "type": "seq",
    "num_channels": BRAINOMNI_DEFAULT.n_latent_sources,
    "max_length": 8,
    "vocab_size": BRAINOMNI_DEFAULT.codebook_size,
    "num_codebooks": BRAINOMNI_DEFAULT.num_quantizers,
    "sample_rate_hz": BRAINOMNI_DEFAULT.target_sfreq_hz,
    "window_samples": BRAINOMNI_DEFAULT.window_length,
    "description": (
        "BrainOmni RVQ tokens (C'=16 latent sources, T_win=8, Q=4 codebooks). "
        "Input trials resampled 200→256 Hz and padded to 512 samples before tokenize."
    ),
}

MODALITY_INFO_MEG: dict[str, dict] = {
    "meg_tokens": MEG_TOKENS_MU_TRANSFORM,
    "meg_tokens_brainomni": MEG_TOKENS_BRAINOMNI,
}
