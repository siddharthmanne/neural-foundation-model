"""Encoder + decoder embeddings for SYMMETRIC neural modalities.

MEG (BrainOmni) tokens are a ``16 x 8`` spatiotemporal grid with 4 RVQ codes per cell.
Each RVQ layer is registered as its **own** modality (``tok_meg_rvqN``, vocab 512, 128-cell
grid) used on **both** the encoder and decoder side, so 4M's masked prediction makes each
cell either an encoder input or a decoder target on a step — never both (leak-free). EEG
(LaBraM) is a single 1D 17-token sequence: one modality (``tok_eeg``) on both sides.

Each modality therefore needs an **encoder** embedding (ids -> ``x``, ``emb``) and a
**decoder** embedding (``forward_embed`` + ``forward_logits``). Both sides share their
positional layout — axial (learned-source + sincos-time) for MEG, sincos-1D for EEG — via
the ``_AxialPositions`` / ``_SincosPositions`` mixins, so the encoder and decoder cannot
drift apart. See ``notes/4m_neural_modality_design.md``.

The encoder modules mirror ``ImageTokenEncoderEmbedding``'s ``init`` / ``forward`` contract;
the decoder modules mirror ``ImageTokenDecoderEmbedding``'s ``forward_embed`` /
``forward_logits`` (fixed per-cell positions; parallel decoding).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from einops import repeat
from fourm.models.fm_utils import build_1d_sincos_posemb

from neural_constants import (
    EEG_TOKENS_PER_TRIAL,
    EEG_VOCAB_SIZE,
    MEG_N_SOURCES,
    MEG_N_TIME,
    MEG_VOCAB_SIZE,
)


# ---------------------------------------------------------------------------
# Positional layout mixins — shared by the encoder AND decoder of a modality so
# the two sides always agree on how a cell's position is encoded.
# ---------------------------------------------------------------------------


class _AxialPositions:
    """MEG: a **learned** embedding for the unordered sources + **sincos** for ordered time.

    Per-cell position = ``source_emb[s] + time_emb[t]`` where cell ``p`` (row-major over the
    ``n_sources x n_time`` grid) maps to ``source = p // n_time``, ``time = p % n_time``.
    """

    n_sources: int
    n_time: int
    n_positions: int

    def _build_positions(self, dim_tokens: int, init_std: float) -> None:
        # No inherent order over latent sources -> learned; time is ordered -> sincos.
        self.source_emb = nn.Parameter(torch.zeros(self.n_sources, dim_tokens))
        nn.init.normal_(self.source_emb, std=init_std)
        time_emb = build_1d_sincos_posemb(max_len=self.n_time, embed_dim=dim_tokens)[0]
        self.register_buffer("time_emb", time_emb)  # (n_time, D)
        cells = torch.arange(self.n_positions)
        self.register_buffer("src_idx", cells // self.n_time)
        self.register_buffer("time_idx", cells % self.n_time)

    def _position_embedding(self) -> torch.Tensor:
        return self.source_emb[self.src_idx] + self.time_emb[self.time_idx]

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"source_emb"}


class _SincosPositions:
    """EEG: fixed 1D sincos positions over the ``n_positions``-token sequence."""

    n_positions: int

    def _build_positions(self, dim_tokens: int, init_std: float) -> None:
        pos = build_1d_sincos_posemb(max_len=self.n_positions, embed_dim=dim_tokens)[0]
        self.register_buffer("pos_emb", pos)  # (n_positions, D)

    def _position_embedding(self) -> torch.Tensor:
        return self.pos_emb

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()


# ---------------------------------------------------------------------------
# Encoder / decoder bases — the embed mechanics; positions come from a mixin.
# ---------------------------------------------------------------------------


class _NeuralGridEncoderEmbedding(nn.Module):
    """Embed ``(B, P)`` single-codebook ids -> ``d['x']``, ``d['emb']`` (parallel grid input).

    Subclasses set ``self.vocab_size`` / ``self.n_positions`` (+ axis sizes for the position
    mixin) in ``__init__`` then call ``self.init(dim_tokens)`` (4M calls it lazily otherwise).
    """

    vocab_size: int
    n_positions: int

    def init(self, dim_tokens: int = 768, init_std: float = 0.02) -> None:
        """Build dimension-dependent parameters (called once when 4M is set up)."""
        self.dim_tokens = dim_tokens
        # No padding_idx: code 0 is a real token; absent neural is presence-masked upstream.
        self.token_emb = nn.Embedding(self.vocab_size, dim_tokens)
        self.mod_emb = nn.Parameter(torch.zeros(1, 1, dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)
        self._build_positions(dim_tokens, init_std)

    def forward(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """``d['tensor']``: ``(B, P)`` ids -> adds ``d['x']`` and ``d['emb']``."""
        assert self.dim_tokens is not None, "Call init(dim_tokens) before forward."
        ids = d["tensor"]
        B = ids.shape[0]
        ids = ids.reshape(B, -1)
        x = self.token_emb(ids)
        x_emb = repeat(
            self._position_embedding().unsqueeze(0) + self.mod_emb, "() n d -> b n d", b=B
        )
        d["x"] = x
        d["emb"] = x_emb
        return d


class _NeuralGridDecoderEmbedding(nn.Module):
    """Parallel decoder head for a fixed-size neural token grid.

    Mirrors ``ImageTokenDecoderEmbedding``'s ``forward_embed`` / ``forward_logits`` contract
    so 4M routes these through its **parallel** decoder branch: target ids are embedded
    (``d['x']``), per-cell positional + modality embeddings are added (``d['emb']``), and
    ``forward_logits`` projects decoder outputs to vocab logits. The decoder replaces
    ``d['x']`` with its mask token, so the model must *predict* each target — no leakage.

    Subclasses set ``self.vocab_size`` / ``self.n_positions`` (+ axis sizes) in ``__init__``
    then call ``self.init(dim_tokens)`` (4M calls it lazily otherwise).
    """

    vocab_size: int
    n_positions: int

    def init(self, dim_tokens: int = 768, init_std: float = 0.02) -> None:
        self.dim_tokens = dim_tokens
        # No padding_idx: code 0 is a real token; absence is handled by presence masking.
        self.token_emb = nn.Embedding(self.vocab_size, dim_tokens)
        self.to_logits = nn.Linear(dim_tokens, self.vocab_size, bias=False)
        self.mod_emb = nn.Parameter(torch.zeros(1, 1, dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)
        self._build_positions(dim_tokens, init_std)

    def forward_embed(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert self.dim_tokens is not None, "Call init(dim_tokens) before forward_embed."
        ids = d["tensor"]
        B = ids.shape[0]
        ids = ids.reshape(B, -1)
        x = self.token_emb(ids)
        x_emb = repeat(
            self._position_embedding().unsqueeze(0) + self.mod_emb, "() n d -> b n d", b=B
        )
        d["x"] = x
        d["emb"] = x_emb
        d["ids"] = ids
        return d

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_logits(x)


# ---------------------------------------------------------------------------
# Concrete MEG / EEG embeddings (one encoder + one decoder per modality).
# Position mixin first in the bases so its layout methods win the MRO.
# ---------------------------------------------------------------------------


class MegRVQEncoderEmbedding(_AxialPositions, _NeuralGridEncoderEmbedding):
    """Encoder for one MEG RVQ layer: ``(B, 128)`` ids of a 16x8 grid, vocab 512, axial pos."""

    def __init__(
        self,
        vocab_size: int = MEG_VOCAB_SIZE,
        n_sources: int = MEG_N_SOURCES,
        n_time: int = MEG_N_TIME,
        dim_tokens: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_sources = n_sources
        self.n_time = n_time
        self.n_positions = n_sources * n_time
        self.dim_tokens = dim_tokens
        if dim_tokens is not None:
            self.init(dim_tokens)


class MegRVQDecoderEmbedding(_AxialPositions, _NeuralGridDecoderEmbedding):
    """Decoder head for one MEG RVQ layer: ``(B, 128)`` ids of a 16x8 grid, vocab 512.

    Positions mirror ``MegRVQEncoderEmbedding`` (shared ``_AxialPositions`` mixin). Four of
    these (one per RVQ layer) reconstruct a MEG token in parallel.
    """

    def __init__(
        self,
        vocab_size: int = MEG_VOCAB_SIZE,
        n_sources: int = MEG_N_SOURCES,
        n_time: int = MEG_N_TIME,
        dim_tokens: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_sources = n_sources
        self.n_time = n_time
        self.n_positions = n_sources * n_time
        self.dim_tokens = dim_tokens
        if dim_tokens is not None:
            self.init(dim_tokens)


class EegEncoderEmbedding(_SincosPositions, _NeuralGridEncoderEmbedding):
    """Encoder for EEG: ``(B, 17)`` ids of a 1D sequence, vocab 8192, sincos positions."""

    def __init__(
        self,
        vocab_size: int = EEG_VOCAB_SIZE,
        max_length: int = EEG_TOKENS_PER_TRIAL,
        dim_tokens: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_positions = max_length
        self.dim_tokens = dim_tokens
        if dim_tokens is not None:
            self.init(dim_tokens)


class EegDecoderEmbedding(_SincosPositions, _NeuralGridDecoderEmbedding):
    """Decoder head for EEG: ``(B, 17)`` ids of a 1D sequence, vocab 8192, sincos positions."""

    def __init__(
        self,
        vocab_size: int = EEG_VOCAB_SIZE,
        max_length: int = EEG_TOKENS_PER_TRIAL,
        dim_tokens: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_positions = max_length
        self.dim_tokens = dim_tokens
        if dim_tokens is not None:
            self.init(dim_tokens)
