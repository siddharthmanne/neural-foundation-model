"""Register neural modalities with 4M at import time.

Import this module BEFORE any 4M script that reads MODALITY_INFO or builds
a dataloader. Mutates ``MODALITY_INFO`` and ``MODALITY_TRANSFORMS`` in place.

MEG/EEG are **symmetric**: each is one modality used on both the encoder (input) and
decoder (target) side. 4M's masked prediction partitions a modality's cells into input
vs target on each step (disjoint), so neural acts as a reconstruction regularizer with no
input->target leakage. MEG is 4 modalities (one per RVQ layer, ``tok_meg_rvq0..3``); EEG is
one (``tok_eeg``). All carry ``type: neural_grid`` (parallel decoder branch) plus both an
encoder and a decoder embedding. See ``notes/4m_neural_modality_design.md``.
"""

from __future__ import annotations

import copy
from functools import partial

from fourm.data import modality_info as _mi
from fourm.utils import generate_uint15_hash

from fourm_neural_embeddings import (
    EegDecoderEmbedding,
    EegEncoderEmbedding,
    MegRVQDecoderEmbedding,
    MegRVQEncoderEmbedding,
)
from fourm_neural_transforms import (
    MaskFlagTransform,
    NeuralTargetTransform,
)
from neural_constants import (
    EEG_CODE_MAX,
    EEG_MODALITY,
    EEG_TOKENS_PER_TRIAL,
    EEG_VOCAB_SIZE,
    MEG_CODE_MAX,
    MEG_N_SOURCES,
    MEG_N_TIME,
    MEG_POSITIONS_PER_TRIAL,
    MEG_RVQ_MODALITIES,
    MEG_SOURCE_PATH,
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

_NEURAL_MODALITY_INFO: dict[str, dict] = {
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

# MEG: one symmetric modality per RVQ layer (layer-specific 512-vocab codebooks) over the
# 16x8 grid. Both embeddings present -> usable in in_domains AND out_domains. All four read
# the on-disk ``tok_meg`` folder via ``path``; the rename seam splits one sampled trial.
for _meg_mod in MEG_RVQ_MODALITIES:
    _NEURAL_MODALITY_INFO[_meg_mod] = {
        "vocab_size": MEG_VOCAB_SIZE,
        "encoder_embedding": partial(
            MegRVQEncoderEmbedding,
            vocab_size=MEG_VOCAB_SIZE,
            n_sources=MEG_N_SOURCES,
            n_time=MEG_N_TIME,
        ),
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
        "path": MEG_SOURCE_PATH,  # all four read the tok_meg folder
        "pretokenized": True,
    }
# EEG: a single codebook -> one symmetric modality over the 17-token sequence. Its name
# doubles as the on-disk folder, so ``path`` defaults to the modality name.
_NEURAL_MODALITY_INFO[EEG_MODALITY] = {
    "vocab_size": EEG_VOCAB_SIZE,
    "encoder_embedding": partial(
        EegEncoderEmbedding,
        vocab_size=EEG_VOCAB_SIZE,
        max_length=EEG_TOKENS_PER_TRIAL,
    ),
    "decoder_embedding": partial(
        EegDecoderEmbedding,
        vocab_size=EEG_VOCAB_SIZE,
        max_length=EEG_TOKENS_PER_TRIAL,
    ),
    "min_tokens": 0,
    "max_tokens": EEG_TOKENS_PER_TRIAL,
    "type": NEURAL_GRID_TYPE,
    "id": generate_uint15_hash(EEG_MODALITY),
    "path": EEG_MODALITY,
    "pretokenized": True,
}

_REGISTERED_TRAINING: bool | None = None


def register(training: bool = True) -> None:
    """Inject MEG/EEG modalities into 4M's global registries.

    Neural transforms are passthrough-clip (the per-sample trial pick + RVQ split happen
    once in the dataloader rename seam via ``NeuralTargetSplitter``), so train vs val trial
    sampling is configured on the splitter, not here. ``training`` is kept for API symmetry.
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

    # Neural modalities: the trial pick + RVQ split already happened in the rename seam,
    # so these transforms just clip to the modality's vocab and cast to int64.
    for meg_mod in MEG_RVQ_MODALITIES:
        _mi.MODALITY_TRANSFORMS[meg_mod] = NeuralTargetTransform(code_max=MEG_CODE_MAX)
    _mi.MODALITY_TRANSFORMS[EEG_MODALITY] = NeuralTargetTransform(code_max=EEG_CODE_MAX)
    _mi.MODALITY_TRANSFORMS.setdefault("meg_mask", MaskFlagTransform())
    _mi.MODALITY_TRANSFORMS.setdefault("eeg_mask", MaskFlagTransform())
    _REGISTERED_TRAINING = training


register()
