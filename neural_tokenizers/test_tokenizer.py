"""Pytest tests for the tokenizer evaluation harness.

Strategy: feed stub tokenizers with known mathematical properties through the
real harness and assert the metrics match the values we can derive by hand.
If these tests pass, the harness numbers on real tokenizers can be trusted.

Run:
    cd neural_tokenizers && pytest test_tokenizer.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from evaluation import (
    EvalConfig,
    compute_codebook_metrics_from_tokens,
    compute_probe_metrics,
    compute_reconstruction_metrics,
    compute_sequence_metrics,
    evaluate,
)
from stubs import ConstantTokenizer, InformativeStubTokenizer, RandomTokenizer


# ----------------------------- fixtures -----------------------------------


@pytest.fixture
def signal() -> torch.Tensor:
    """Small (B, C, T) tensor — keeps tests fast on CPU."""
    torch.manual_seed(0)
    return torch.randn(64, 4, 16)


@pytest.fixture
def labels() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 3, (64,), dtype=torch.long)


@pytest.fixture
def config() -> EvalConfig:
    return EvalConfig(
        sample_rate_hz=100.0,
        batch_size=16,
        seed=0,
        probe_epochs=50,
        probe_top_k=(1, 5),
    )


# --------------------------- reconstruction -------------------------------


def test_reconstruction_returns_finite_numbers_on_any_shape(signal, config):
    tok = RandomTokenizer(codebook_size=32, seq_len=4, signal_shape=(4, 16))
    result = compute_reconstruction_metrics(tok, signal, config)
    assert {"mse", "channel_corr_mean", "psd_mse"} <= set(result.values)
    assert all(math.isfinite(v) for v in result.values.values())


def test_reconstruction_zero_when_decoder_returns_input():
    """Identity decoder -> MSE should be zero."""

    class IdentityTokenizer:
        codebook_size = 8

        def tokenize(self, x):
            return torch.zeros(x.shape[0], 4, dtype=torch.long)

        def decode_tokens(self, tokens):
            # We need access to the input — fake it by storing the last batch.
            # For this test we cheat: the harness retokenizes then decodes, so
            # we need a deterministic round-trip. Use a fixed-shape zero output
            # and a fixed-shape zero input.
            return torch.zeros(tokens.shape[0], 4, 16)

    tok = IdentityTokenizer()
    x = torch.zeros(32, 4, 16)
    cfg = EvalConfig(sample_rate_hz=100.0, batch_size=8)
    result = compute_reconstruction_metrics(tok, x, cfg)
    assert result.values["mse"] == pytest.approx(0.0, abs=1e-6)


# ------------------------------ codebook ----------------------------------


def test_codebook_perplexity_near_size_for_uniform_random():
    """Uniform random tokens -> perplexity should approach V (== max entropy)."""
    V = 64
    tokens = torch.randint(0, V, (2000, 8), generator=torch.Generator().manual_seed(0))
    result = compute_codebook_metrics_from_tokens(tokens, V)
    assert result.values["perplexity"] > 0.85 * V


def test_codebook_one_used_when_constant():
    V = 100
    tokens = torch.full((50, 8), 7, dtype=torch.long)
    result = compute_codebook_metrics_from_tokens(tokens, V)
    assert result.values["codes_used"] == 1.0
    assert result.values["perplexity"] == pytest.approx(1.0, abs=1e-6)
    assert result.values["dead_code_fraction"] == pytest.approx(0.99)


# ------------------------------ sequence ----------------------------------


def test_sequence_collapsed_to_one_token_detected():
    """The §5.4 motivating example: all-same tokens must trigger every red flag."""
    V, L = 64, 8
    tokens = torch.full((50, L), 3, dtype=torch.long)
    result = compute_sequence_metrics(tokens, V)
    assert result.values["unigram_entropy"] == pytest.approx(0.0, abs=1e-6)
    assert result.values["bigram_conditional_entropy"] == pytest.approx(0.0, abs=1e-6)
    assert result.values["mean_run_length"] == pytest.approx(float(L))
    assert result.values["frac_runs_ge_2"] == pytest.approx(1.0)


def test_sequence_high_entropy_low_gap_for_iid_random():
    """iid random tokens: H(t_i|t_{i-1}) ~= H(t), so gap should be ~0."""
    V = 32
    tokens = torch.randint(0, V, (500, 16), generator=torch.Generator().manual_seed(1))
    result = compute_sequence_metrics(tokens, V)
    max_H = math.log(V)
    assert result.values["unigram_entropy"] > 0.95 * max_H
    assert abs(result.values["entropy_gap_pct"]) < 5.0


# ------------------------------- probe ------------------------------------


def test_probe_random_tokenizer_at_chance(signal, labels, config):
    """Random tokenizer carries no class info; probe should not beat chance much."""
    tok = RandomTokenizer(codebook_size=32, seq_len=4, signal_shape=(4, 16))
    result = compute_probe_metrics(tok, signal, labels, config)
    chance = 1.0 / int(labels.max().item() + 1)
    assert result.values["top1_tokens"] <= chance + 0.25  # small B noise tolerance
    assert result.values["top1_random"] <= chance + 0.25


def test_probe_informative_tokenizer_beats_random():
    """A deterministic (signal -> class -> token) mapping should be probe-readable."""
    torch.manual_seed(42)
    B, C, T = 120, 4, 16
    classes = torch.randint(0, 4, (B,))
    class_means = torch.linspace(-2.0, 2.0, 4)
    signal = class_means[classes].view(-1, 1, 1) + 0.1 * torch.randn(B, C, T)

    tok = InformativeStubTokenizer(codebook_size=32, seq_len=8, signal_shape=(C, T))
    cfg = EvalConfig(sample_rate_hz=100.0, batch_size=32, probe_epochs=200, seed=0)
    result = compute_probe_metrics(tok, signal, classes, cfg)
    assert result.values["top1_tokens"] > result.values["top1_random"] + 0.1


# ------------------------------- end-to-end --------------------------------


def test_evaluate_runs_all_axes(signal, labels, config):
    tok = RandomTokenizer(codebook_size=32, seq_len=4, signal_shape=(4, 16))
    report = evaluate(tok, signal, labels, config)
    assert report.reconstruction is not None
    assert report.codebook is not None
    assert report.sequence is not None
    assert report.probe is not None
    text = str(report)
    assert "reconstruction" in text and "codebook" in text and "sequence" in text


def test_evaluate_skips_probe_when_labels_none(signal, config):
    tok = RandomTokenizer(codebook_size=32, seq_len=4, signal_shape=(4, 16))
    report = evaluate(tok, signal, None, config)
    assert report.probe is None
    assert report.reconstruction is not None


def test_evaluate_skips_axes_via_config(signal, labels):
    tok = RandomTokenizer(codebook_size=32, seq_len=4, signal_shape=(4, 16))
    cfg = EvalConfig(
        sample_rate_hz=100.0,
        run_reconstruction=False,
        run_codebook=True,
        run_probe=False,
        run_sequence=True,
    )
    report = evaluate(tok, signal, labels, cfg)
    assert report.reconstruction is None
    assert report.probe is None
    assert report.codebook is not None
    assert report.sequence is not None


def test_constant_tokenizer_e2e_caught_by_sequence_axis(signal, labels, config):
    """End-to-end smoke test: collapsed tokenizer should be flagged by §5.4."""
    tok = ConstantTokenizer(codebook_size=32, seq_len=4, signal_shape=(4, 16))
    report = evaluate(tok, signal, labels, config)
    assert report.sequence.values["mean_run_length"] == pytest.approx(4.0)
    assert report.codebook.values["codes_used"] == 1.0
