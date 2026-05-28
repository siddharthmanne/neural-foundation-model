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

from fourm_neural_embeddings import (
    EegDecoderEmbedding,
    MegEncoderEmbedding,
    MegRVQDecoderEmbedding,
)
from fourm_neural_transforms import (
    EegTokTransform,
    MaskFlagTransform,
    MegTokTransform,
    NeuralTargetTransform,
)
from neural_constants import (
    EEG_CODE_MAX,
    EEG_OUT_MODALITY,
    EEG_OUT_SOURCE_PATH,
    EEG_TOKENS_PER_TRIAL,
    EEG_VOCAB_SIZE,
    MEG_CODE_MAX,
    MEG_N_RVQ,
    MEG_N_SOURCES,
    MEG_N_TIME,
    MEG_OUT_SOURCE_PATH,
    MEG_POSITIONS_PER_TRIAL,
    MEG_RVQ_OUT_MODALITIES,
    MEG_VOCAB_SIZE,
    NEURAL_GRID_TYPE,
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

# MEG/EEG OUTPUT (target) modalities — predicted as a reconstruction regularizer.
# They use ``type: neural_grid`` so 4M routes them through the PARALLEL decoder branch
# (not seq_token's autoregressive path) and the trainer's square ``max_tokens`` rule for
# ``img`` leaves their non-square grids alone. Each reads an existing on-disk folder via
# ``path`` (no repack); the rename seam splits one sampled trial into these heads.
# MEG: one head per RVQ layer (layer-specific 512-vocab codebooks) over the 16x8 grid.
for _q, _meg_mod in enumerate(MEG_RVQ_OUT_MODALITIES):
    _NEURAL_MODALITY_INFO[_meg_mod] = {
        "vocab_size": MEG_VOCAB_SIZE,
        "encoder_embedding": None,  # output-only
        "decoder_embedding": partial(
            MegRVQDecoderEmbedding,
            vocab_size=MEG_VOCAB_SIZE,
            n_sources=MEG_N_SOURCES,
            n_time=MEG_N_TIME,
        ),
        "min_tokens": 0,
        "max_tokens": MEG_POSITIONS_PER_TRIAL,  # 128 grid cells
        "type": NEURAL_GRID_TYPE,
        "id": generate_uint15_hash(_meg_mod),
        "path": MEG_OUT_SOURCE_PATH,  # all four read the tok_meg folder
        "pretokenized": True,
    }
# EEG: a single codebook -> one head over the 17-token sequence.
_NEURAL_MODALITY_INFO[EEG_OUT_MODALITY] = {
    "vocab_size": EEG_VOCAB_SIZE,
    "encoder_embedding": None,  # output-only
    "decoder_embedding": partial(
        EegDecoderEmbedding,
        vocab_size=EEG_VOCAB_SIZE,
        max_length=EEG_TOKENS_PER_TRIAL,
    ),
    "min_tokens": 0,
    "max_tokens": EEG_TOKENS_PER_TRIAL,
    "type": NEURAL_GRID_TYPE,
    "id": generate_uint15_hash(EEG_OUT_MODALITY),
    "path": EEG_OUT_SOURCE_PATH,  # reads the tok_eeg folder
    "pretokenized": True,
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

    # OUTPUT modalities: the trial pick + RVQ split already happened in the rename seam,
    # so these transforms just clip to the head's vocab. (Training-mode trial sampling for
    # the split is configured on NeuralTargetSplitter, not here.)
    for meg_mod in MEG_RVQ_OUT_MODALITIES:
        _mi.MODALITY_TRANSFORMS[meg_mod] = NeuralTargetTransform(code_max=MEG_CODE_MAX)
    _mi.MODALITY_TRANSFORMS[EEG_OUT_MODALITY] = NeuralTargetTransform(code_max=EEG_CODE_MAX)
    _REGISTERED_TRAINING = training


register()
