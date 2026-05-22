"""Stub tokenizers with known properties — used to verify the harness.

Three stubs covering different failure modes:

  RandomTokenizer            uniform random tokens, random decoder
      -> high perplexity, low channel_corr, probe near chance
  ConstantTokenizer          one token forever, zero decoder
      -> perplexity=1, dead_code_fraction=1-1/V, mean_run_length=L
  InformativeStubTokenizer   quantize per-trial mean into one code, identity-ish decode
      -> probe well above chance (deterministic class -> code mapping)

None of these is a real tokenizer. They exist only so test_tokenizer.py can
make assertions about harness outputs whose ground truth we can derive on
paper.
"""

from __future__ import annotations

import torch


class RandomTokenizer:
    """Emit uniformly random tokens; decode to random noise of the right shape.

    Used as the lower bound for the probe (the harness compares the real
    tokenizer's probe accuracy against random-tokens accuracy).
    """

    def __init__(
        self,
        codebook_size: int = 64,
        seq_len: int = 8,
        signal_shape: tuple[int, int] = (4, 16),
        seed: int = 0,
    ):
        self.codebook_size = codebook_size
        self.seq_len = seq_len
        self.signal_shape = signal_shape
        self._gen = torch.Generator().manual_seed(seed)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return torch.randint(
            0, self.codebook_size, (B, self.seq_len), generator=self._gen, dtype=torch.long
        )

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        C, T = self.signal_shape
        return torch.randn(B, C, T, generator=self._gen)


class ConstantTokenizer:
    """Emit the same token for every position. Tests the §5.4 collapse detector."""

    def __init__(
        self,
        codebook_size: int = 64,
        seq_len: int = 8,
        signal_shape: tuple[int, int] = (4, 16),
        token: int = 0,
    ):
        self.codebook_size = codebook_size
        self.seq_len = seq_len
        self.signal_shape = signal_shape
        self.token = token

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return torch.full((B, self.seq_len), self.token, dtype=torch.long)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        C, T = self.signal_shape
        return torch.zeros(B, C, T)


class InformativeStubTokenizer:
    """Class-deterministic encoder: same trial -> same tokens, different classes
    -> different tokens. Decoder is uninformative.

    Used to verify the probe can detect a tokenizer whose tokens carry class
    information, even when reconstruction is poor. Specifically, we hash the
    per-channel mean of x into a codebook index, so similar trials map to
    similar codes and the linear probe can pick up the signal.
    """

    def __init__(
        self,
        codebook_size: int = 64,
        seq_len: int = 8,
        signal_shape: tuple[int, int] = (4, 16),
    ):
        self.codebook_size = codebook_size
        self.seq_len = seq_len
        self.signal_shape = signal_shape

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        chunks = x.reshape(B, -1).chunk(self.seq_len, dim=1)
        means = torch.stack([c.mean(dim=1) for c in chunks], dim=1)
        lo, hi = means.min(), means.max()
        normed = (means - lo) / (hi - lo + 1e-6)
        return (normed * (self.codebook_size - 1)).round().long().clamp(
            0, self.codebook_size - 1
        )

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        C, T = self.signal_shape
        return torch.zeros(B, C, T)
