"""Pytest tests for the §5.5 model-free retrieval probe.

The retrieval probe answers "does the feature geometry place same-class
trials nearer than other-class trials?" without training any classifier.
Tests are structured around three classes of inputs:

  1. Synthetic features with KNOWN class structure → precision@K must be
     well above chance.
  2. Random features → precision@K must sit at balanced-chance.
  3. Degenerate cases (token sets saturate) → Jaccard correctly skipped.

Run:
    cd neural_tokenizers && pytest test_retrieval.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from evaluation import EvalConfig, compute_retrieval_metrics
from stubs import InformativeStubTokenizer, RandomTokenizer


@pytest.fixture
def config() -> EvalConfig:
    return EvalConfig(
        sample_rate_hz=100.0,
        batch_size=16,
        seed=0,
        probe_top_k=(1, 5),
    )


def test_retrieval_random_at_balanced_chance(config):
    """Random tokenizer + random labels → balanced precision@K ≈ 1/n_classes."""
    torch.manual_seed(0)
    B, C, T = 90, 4, 16
    n_classes = 3
    signal = torch.randn(B, C, T)
    labels = torch.randint(0, n_classes, (B,))
    tok = RandomTokenizer(codebook_size=16, seq_len=4, signal_shape=(C, T))
    result = compute_retrieval_metrics(tok, signal, labels, config)
    chance = 1.0 / n_classes
    # All three feature sets should be near chance with some sampling noise.
    for key in ("prec@1_tokens", "prec@1_raw", "prec@1_random"):
        assert key in result.values
        assert abs(result.values[key] - chance) < 0.25


def test_retrieval_informative_features_beat_chance():
    """An informative stub (signal → class → distinct codes) should put same-
    class trials nearest each other → precision@K > chance by a wide margin.

    Note: shuffle class order so each batch contains all classes. The
    InformativeStubTokenizer normalizes per-batch using batch-wide min/max,
    so a sorted-by-class layout would make every batch single-class and
    collapse the normalization. This is a quirk of the stub, not of the
    retrieval probe — real tokenizers are stateless across batches.
    """
    torch.manual_seed(7)
    B, C, T = 120, 4, 16
    n_classes = 3
    classes = torch.arange(n_classes).repeat_interleave(B // n_classes)
    perm = torch.randperm(B)
    classes = classes[perm]
    means = torch.linspace(-3.0, 3.0, n_classes)
    signal = means[classes].view(-1, 1, 1) + 0.05 * torch.randn(B, C, T)
    tok = InformativeStubTokenizer(codebook_size=16, seq_len=4, signal_shape=(C, T))
    cfg = EvalConfig(sample_rate_hz=100.0, batch_size=B, seed=0, probe_top_k=(1,))
    result = compute_retrieval_metrics(tok, signal, classes, cfg)
    chance = 1.0 / n_classes
    # Tokens carry the signal: precision well above chance.
    assert result.values["prec@1_tokens"] > chance + 0.20
    # Random tokens are still at chance.
    assert abs(result.values["prec@1_random"] - chance) < 0.20


def test_retrieval_jaccard_emitted_for_sparse_codes(config):
    """Random codes used sparsely per trial (small seq_len, large vocab) →
    Jaccard precision is reported as a separate key."""
    torch.manual_seed(0)
    B, C, T = 64, 4, 16
    signal = torch.randn(B, C, T)
    labels = torch.randint(0, 3, (B,))
    tok = RandomTokenizer(codebook_size=128, seq_len=4, signal_shape=(C, T))
    result = compute_retrieval_metrics(tok, signal, labels, config)
    # seq_len=4 tokens × from vocab=128 → trial uses ≤4/128 = 3% of vocab.
    # Sparse — Jaccard should fire.
    assert "prec@1_tokens_jaccard" in result.values


def test_retrieval_jaccard_skipped_for_saturated_codes(config):
    """A tokenizer whose every trial uses most of the vocab (μ-transform
    regime) → Jaccard would degenerate; the heuristic must skip it."""

    class DenseTokensTokenizer:
        codebook_size = 8

        def tokenize(self, x):
            # Every trial uses every code many times → set saturated.
            B = x.shape[0]
            return torch.arange(8).repeat(B, 100)  # (B, 800), all codes present

        def decode_tokens(self, tokens):
            return torch.zeros(tokens.shape[0], 4, 16)

    tok = DenseTokensTokenizer()
    B = 30
    signal = torch.randn(B, 4, 16)
    labels = torch.randint(0, 3, (B,))
    result = compute_retrieval_metrics(tok, signal, labels, config)
    # Cosine retrieval keys present, Jaccard one absent.
    assert "prec@1_tokens" in result.values
    assert "prec@1_tokens_jaccard" not in result.values


def test_retrieval_balanced_precision_is_balanced():
    """With heavily imbalanced classes and random features, balanced
    precision@K should sit near 1/n_classes — NOT near the majority-class
    frequency. This is the analog of balanced accuracy in §5.3."""
    torch.manual_seed(0)
    B = 200
    classes = torch.cat([
        torch.zeros(int(B * 0.8), dtype=torch.long),
        torch.ones(int(B * 0.15), dtype=torch.long),
        torch.full((int(B * 0.05),), 2, dtype=torch.long),
    ])
    signal = torch.randn(B, 4, 16)
    tok = RandomTokenizer(codebook_size=16, seq_len=4, signal_shape=(4, 16))
    cfg = EvalConfig(sample_rate_hz=100.0, batch_size=16, seed=0, probe_top_k=(1,))
    result = compute_retrieval_metrics(tok, signal, classes, cfg)
    # With balanced precision, random features → ~1/3 = 0.33; UN-balanced
    # would give ~0.8 from majority dominance.
    assert result.values["prec@1_random"] < 0.55  # well below majority floor


def test_retrieval_precision_values_in_unit_interval(config):
    torch.manual_seed(0)
    B, C, T = 50, 4, 16
    signal = torch.randn(B, C, T)
    labels = torch.randint(0, 4, (B,))
    tok = RandomTokenizer(codebook_size=16, seq_len=4, signal_shape=(C, T))
    result = compute_retrieval_metrics(tok, signal, labels, config)
    for k, v in result.values.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0, 1]"
        assert math.isfinite(v)
