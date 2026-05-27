"""Register neural modalities with 4M at import time.

Import this module BEFORE any 4M script that reads MODALITY_INFO or builds
a dataloader. Mutates ``MODALITY_INFO`` and ``MODALITY_TRANSFORMS`` in place.
"""

from __future__ import annotations

import copy
from functools import partial

from fourm.data import modality_info as _mi
from fourm.models.encoder_embeddings import SequenceEncoderEmbedding
from fourm.utils import generate_uint15_hash

from fourm_neural_embeddings import MegEncoderEmbedding
from fourm_neural_transforms import (
    EegTokTransform,
    MaskFlagTransform,
    MegTokTransform,
)
from neural_constants import (
    EEG_TOKENS_PER_TRIAL,
    EEG_VOCAB_SIZE,
    MEG_N_RVQ,
    MEG_N_SOURCES,
    MEG_N_TIME,
    MEG_POSITIONS_PER_TRIAL,
    MEG_VOCAB_SIZE,
)
from things_augmenter import ThingsTokTransform

# Re-export for callers that import vocab sizes from this module.
__all__ = [
    "EEG_VOCAB_SIZE",
    "MEG_VOCAB_SIZE",
    "register",
]

# MEG/EEG are INPUT-ONLY (never targets) — encoder embedding only, no decoder.
# ``type: seq_token`` keeps the input masking on the image_mask path (see
# PresenceAwareUnifiedMasking); its autoregressive decoder branch is never reached
# because neural mods stay out of out_domains. See notes/4m_neural_modality_design.md.
_NEURAL_MODALITY_INFO = {
    "tok_meg": {
        "vocab_size": MEG_VOCAB_SIZE,
        "encoder_embedding": partial(
            MegEncoderEmbedding,
            vocab_size=MEG_VOCAB_SIZE,
            n_rvq=MEG_N_RVQ,
            n_sources=MEG_N_SOURCES,
            n_time=MEG_N_TIME,
        ),
        "min_tokens": 0,
        "max_tokens": MEG_POSITIONS_PER_TRIAL,  # 128 grid cells (RVQ summed per cell)
        "type": "seq_token",
        "id": generate_uint15_hash("tok_meg"),
        "path": "tok_meg",
        "pretokenized": True,
    },
    "tok_eeg": {
        "vocab_size": EEG_VOCAB_SIZE,
        "encoder_embedding": partial(
            SequenceEncoderEmbedding,
            vocab_size=EEG_VOCAB_SIZE,
            max_length=EEG_TOKENS_PER_TRIAL,
            padding_idx=None,  # code 0 is a real LaBraM code; absence is presence-masked
        ),
        "min_tokens": 0,
        "max_tokens": EEG_TOKENS_PER_TRIAL,
        "type": "seq_token",
        "id": generate_uint15_hash("tok_eeg"),
        "path": "tok_eeg",
        "pretokenized": True,
    },
    "meg_mask": {
        "type": "meta",
        "id": generate_uint15_hash("meg_mask"),
        "path": "meg_mask",
    },
    "eeg_mask": {
        "type": "meta",
        "id": generate_uint15_hash("eeg_mask"),
        "path": "eeg_mask",
    },
}

_REGISTERED_TRAINING: bool | None = None


def register(training: bool = True) -> None:
    """Inject MEG/EEG modalities into 4M's global registries.

    Always updates ``MODALITY_TRANSFORMS`` for tok_meg/tok_eeg so train vs val
    trial sampling can be switched by calling ``register(training=False)``.
    """
    global _REGISTERED_TRAINING

    for name, info in _NEURAL_MODALITY_INFO.items():
        _mi.MODALITY_INFO.setdefault(name, info)

    # THINGS on-disk folders use tok_rgb / tok_depth; stock 4M registers @224 names.
    for alias, src in (("tok_rgb", "tok_rgb@224"), ("tok_depth", "tok_depth@224")):
        if alias not in _mi.MODALITY_INFO and src in _mi.MODALITY_INFO:
            info = copy.deepcopy(_mi.MODALITY_INFO[src])
            info["path"] = alias
            _mi.MODALITY_INFO[alias] = info

    # THINGS vision shards are flat (196,) with no augmentation axis; stock
    # TokTransform would index that axis and collapse the grid to one token.
    # ThingsTokTransform falls back to stock behaviour for (n_augs, n_tokens),
    # so the shared "tok_rgb" transform key stays correct for CC12M too.
    for vision_mod in ("tok_rgb", "tok_depth"):
        _mi.MODALITY_TRANSFORMS[vision_mod] = ThingsTokTransform()

    _mi.MODALITY_TRANSFORMS["tok_meg"] = MegTokTransform(training=training)
    _mi.MODALITY_TRANSFORMS["tok_eeg"] = EegTokTransform(training=training)
    _mi.MODALITY_TRANSFORMS.setdefault("meg_mask", MaskFlagTransform())
    _mi.MODALITY_TRANSFORMS.setdefault("eeg_mask", MaskFlagTransform())
    _REGISTERED_TRAINING = training


register()
