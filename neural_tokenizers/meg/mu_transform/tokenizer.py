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
from .quantizer import quantize as _quantize


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
