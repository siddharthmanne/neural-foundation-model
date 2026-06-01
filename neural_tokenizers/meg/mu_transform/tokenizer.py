"""MuTransformTokenizer — Phase 1 MEG tokenizer, Tokenizer-protocol compliant.

Composes encoder + quantizer + decoder behind the protocol from
evaluation/protocol.py:

    tokenize(x)        : (B, C, T) float -> (B, C, T) long
    decode_tokens(tok) : (B, C, T) long  -> (B, C, T) float

We keep tokens as (B, C, T) rather than flattening because:
  1. The 4M modality registration declares
     `num_channels=271, max_length=281, type='seq'` (meg/CLAUDE.md §2) —
     so the per-channel/time factorization matters to 4M.
  2. The §5 harness handles arbitrary trailing token shape via reshape, so
     this costs nothing.

No learnable parameters; the only state is the MuCalibration object.
"""

from __future__ import annotations

import torch

from .calibration import MuCalibration
from .decoder import decode as _decode
from .encoder import encode as _encode
from .quantizer import dequantize as _dequantize, quantize as _quantize


class MuTransformTokenizer:
    """Phase 1 MEG tokenizer satisfying evaluation.protocol.Tokenizer.

    Attributes:
        calibration: fitted MuCalibration.
        codebook_size: vocab_size from the calibration (the protocol field).

    Example:
        >>> from .calibration import MuCalibration
        >>> calib = MuCalibration.load("calibration.json")
        >>> tok = MuTransformTokenizer(calib)
        >>> tokens = tok.tokenize(x)         # (B, C, T) long
        >>> x_hat = tok.decode_tokens(tokens)  # (B, C, T) float
    """

    def __init__(self, calibration: MuCalibration):
        self.calibration = calibration
        self.codebook_size: int = int(calibration.vocab_size)

    # ---- Tokenizer protocol ------------------------------------------

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) float -> (B, C, T) long token IDs in [0, V)."""
        companded = _encode(x, self.calibration)
        return _quantize(companded, self.calibration.vocab_size)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, C, T) long -> (B, C, T) float on the input amplitude scale."""
        return _decode(tokens, self.calibration)

    # ---- Optional probe-featurization hook ---------------------------
    #
    # μ-transform is a position-preserving per-(channel, time-sample) codec:
    # each of the 271×281 = 76,151 tokens encodes one signal sample. The
    # default bag-of-codes featurization in the probe would collapse that
    # spatial-temporal structure into a 256-bin histogram and discard the
    # information μ-transform faithfully preserved. We override it to map
    # each token to its bin-center value and treat the flattened sequence
    # as features — essentially the raw (companded) signal, which is what
    # 4M's embedding layer would learn to recover from these tokens.
    #
    # Output shape (B, 1, n_channels * n_timepoints): the L=1 dim absorbs
    # the harness's mean-pool, so the (channel × time) features reach the
    # linear head intact. `layers` is accepted only for signature parity
    # with multi-codebook tokenizers; μ-transform has a single codebook.

    @torch.no_grad()
    def tokens_to_embedding(
        self, tokens: torch.Tensor, layers: tuple[int, ...] | None = None
    ) -> torch.Tensor:
        del layers  # μ-transform has one codebook; no layer subset to apply
        centers = _dequantize(tokens, self.codebook_size)
        return centers.reshape(centers.shape[0], 1, -1)
