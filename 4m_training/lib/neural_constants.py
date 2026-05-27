"""Single source of truth for THINGS + neural modality geometry and vocab sizes.

Import from here instead of scattering literals (512, 8192, 196, 224, …) across
transforms, masking, tests, and demos. Values match stock 4M ``tok_*@224`` entries
and the BrainOmni / LaBraM tokenizers documented in ``modal/data/README.md``.
"""

from __future__ import annotations

# --- MEG (BrainOmni V512_rvq4_win512_sf256) ---
# On-disk trial is (C', W, N_q): 16 latent source vars x 8 temporal steps x 4 RVQ layers.
# See notes/4m_neural_modality_design.md for why each axis is treated the way it is.
MEG_VOCAB_SIZE: int = 512            # per-RVQ-layer codebook size
MEG_N_SOURCES: int = 16              # C': latent source variables (spatial, unordered)
MEG_N_TIME: int = 8                  # W: temporal latent steps (ordered)
MEG_N_RVQ: int = 4                   # N_q: residual quantizer layers (collapsed by summation)
MEG_TRIAL_SHAPE: tuple[int, int, int] = (MEG_N_SOURCES, MEG_N_TIME, MEG_N_RVQ)
# After collapsing the trial to a (source*time) grid of cells, each cell keeps its N_q codes.
MEG_POSITIONS_PER_TRIAL: int = MEG_N_SOURCES * MEG_N_TIME   # 128 grid cells = encoder tokens
MEG_GRID_SHAPE: tuple[int, int] = (MEG_POSITIONS_PER_TRIAL, MEG_N_RVQ)  # (128, 4)
MEG_TOKENS_PER_TRIAL: int = MEG_N_SOURCES * MEG_N_TIME * MEG_N_RVQ      # 512 raw codes/trial
MEG_CODE_MAX: int = MEG_VOCAB_SIZE - 1

# --- EEG (LaBraM V8192) ---
EEG_VOCAB_SIZE: int = 8192
EEG_TRIAL_SHAPE: tuple[int] = (17,)
EEG_TOKENS_PER_TRIAL: int = EEG_TRIAL_SHAPE[0]
EEG_CODE_MAX: int = EEG_VOCAB_SIZE - 1

# --- THINGS vision pretokens (stock ``tok_rgb@224`` / ``tok_depth@224``) ---
THINGS_IMAGE_SIZE: int = 224
THINGS_PATCH_SIZE: int = 16
TOK_RGB_VOCAB_SIZE: int = 16384
TOK_DEPTH_VOCAB_SIZE: int = 8192
TOK_RGB_CODE_MAX: int = TOK_RGB_VOCAB_SIZE - 1
TOK_DEPTH_CODE_MAX: int = TOK_DEPTH_VOCAB_SIZE - 1


def pretoken_grid_num_tokens(image_size: int, patch_size: int) -> int:
    """Number of patch tokens for a square pretokenized image grid."""
    return (image_size // patch_size) ** 2


TOK_RGB_TOKENS_PER_IMAGE: int = pretoken_grid_num_tokens(
    THINGS_IMAGE_SIZE, THINGS_PATCH_SIZE
)

# --- On-disk sentinel / placeholder semantics ---
NEURAL_SENTINEL_VALUE: int = -1
NEURAL_PLACEHOLDER_CODE: int = 0

# Center crop for pretokenized THINGS when ``crop_settings/`` is absent: top, left, h, w, flip
THINGS_CENTER_CROP: tuple[int, int, int, int, int] = (
    0,
    0,
    THINGS_IMAGE_SIZE,
    THINGS_IMAGE_SIZE,
    0,
)
