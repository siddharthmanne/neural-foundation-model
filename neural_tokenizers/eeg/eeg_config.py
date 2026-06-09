"""EEG-wide constants and eval defaults.

Mirrors neural_tokenizers/meg/meg_config.py for the EEG track. All magic
numbers for the EEG tokenizer track live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 75.0),
}


@dataclass(frozen=True)
class EEGDataSpec:
    """Immutable description of the LaBraM-tokenized THINGS-EEG data layout."""

    n_channels: int = 17
    sfreq_hz_input: float = 100.0      # THINGS-EEG2 preprocessed sampling rate
    sfreq_hz_model: float = 200.0      # LaBraM model rate (after upsampling)
    token_vocab_size: int = 8192
    embed_dim: int = 64

    subjects_eeg2: tuple[str, ...] = tuple(f"sub-{i:02d}" for i in range(1, 11))
    subjects_eeg1: tuple[str, ...] = tuple(f"sub-{i:02d}" for i in range(1, 51))

    cache_root: str = "/project/data/things-eeg/tokens/labram"
    default_slug: str = "V8192_d64_ch17_sr200_train-eeg1+2_e5"
    checkpoint_path: str = (
        "/project/checkpoints/eeg/labram/"
        "V8192_d64_ch17_sr200_train-eeg1+2_e5/checkpoint.pt"
    )


EEG_DATA = EEGDataSpec()


@dataclass(frozen=True)
class EvalDefaults:
    """Defaults for the §5 harness when applied to EEG.

    Used by modal_eeg_eval.py to construct an EvalConfig. Keeping these here
    means the harness stays modality-agnostic.
    """

    sample_rate_hz: float = EEG_DATA.sfreq_hz_input
    bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(EEG_BANDS)
    )
    psd_nperseg: int = 50             # 0.5 s window @ 100 Hz — fits in 1 s trial
    probe_epochs: int = 200
    probe_top_k: tuple[int, ...] = (1, 5)
    probe_test_frac: float = 0.2
    probe_n_folds: int = 5
    probe_rvq_layers: tuple = (None,)  # single codebook, no RVQ layers
    probe_class_weighted: bool = True
    probe_classifier: str = "linear"
    probe_mlp_hidden: int = 256
    probe_mlp_dropout: float = 0.5
    probe_cnn_hidden: int = 64


EVAL_DEFAULTS = EvalDefaults()
