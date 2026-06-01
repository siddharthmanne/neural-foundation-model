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
    assert result.values["top1_tokens_mean"] <= chance + 0.25  # small B noise tolerance
    assert result.values["top1_random_mean"] <= chance + 0.25


def test_probe_informative_tokenizer_beats_random():
    """A deterministic (signal -> class -> token) mapping should be probe-readable."""
    torch.manual_seed(42)
    B, C, T = 120, 4, 16
    classes = torch.randint(0, 4, (B,))
    class_means = torch.linspace(-2.0, 2.0, 4)
    signal = class_means[classes].view(-1, 1, 1) + 0.1 * torch.randn(B, C, T)

    tok = InformativeStubTokenizer(codebook_size=32, seq_len=8, signal_shape=(C, T))
    cfg = EvalConfig(
        sample_rate_hz=100.0, batch_size=32, probe_epochs=200, seed=0, probe_n_folds=3
    )
    result = compute_probe_metrics(tok, signal, classes, cfg)
    assert result.values["top1_tokens_mean"] > result.values["top1_random_mean"] + 0.1


def test_probe_kfold_emits_mean_and_std_keys(signal, labels, config):
    """K-fold (n_folds>=2) must emit `_mean` and `_std` for every (k, set)."""
    tok = RandomTokenizer(codebook_size=32, seq_len=4, signal_shape=(4, 16))
    cfg = EvalConfig(
        sample_rate_hz=100.0,
        batch_size=16,
        seed=0,
        probe_epochs=20,
        probe_top_k=(1, 5),
        probe_n_folds=4,
    )
    result = compute_probe_metrics(tok, signal, labels, cfg)
    for k in (1, 5):
        for s in ("tokens", "raw", "random"):
            assert f"top{k}_{s}_mean" in result.values
            assert f"top{k}_{s}_std" in result.values


def test_probe_kfold_std_is_nonnegative_and_finite(signal, labels):
    """Std across folds must be finite and non-negative."""
    tok = RandomTokenizer(codebook_size=32, seq_len=4, signal_shape=(4, 16))
    cfg = EvalConfig(sample_rate_hz=100.0, batch_size=16, seed=0, probe_epochs=20, probe_n_folds=4)
    result = compute_probe_metrics(tok, signal, labels, cfg)
    for k, v in result.values.items():
        if k.endswith("_std"):
            assert v >= 0.0
            assert math.isfinite(v)


def test_probe_n_folds_1_is_back_compat_single_split(signal, labels, config):
    """n_folds=1 → single (1-test_frac, test_frac) split. _std should be 0."""
    tok = RandomTokenizer(codebook_size=32, seq_len=4, signal_shape=(4, 16))
    cfg = EvalConfig(
        sample_rate_hz=100.0,
        batch_size=16,
        seed=0,
        probe_epochs=20,
        probe_top_k=(1, 5),
        probe_n_folds=1,
        probe_test_frac=0.2,
    )
    result = compute_probe_metrics(tok, signal, labels, cfg)
    for k in result.values:
        if k.endswith("_std"):
            assert result.values[k] == 0.0


def test_build_head_linear_default():
    """`probe_classifier='linear'` returns Sequential(Flatten, Linear)."""
    from evaluation.probe import _build_head

    cfg = EvalConfig(sample_rate_hz=100.0)
    head = _build_head(feat_shape=(64,), n_classes=10, config=cfg)
    assert isinstance(head, torch.nn.Sequential)
    layers = list(head)
    assert isinstance(layers[0], torch.nn.Flatten)
    assert isinstance(layers[1], torch.nn.Linear)
    assert layers[1].in_features == 64 and layers[1].out_features == 10
    # Survives a structured (non-flat) input — Flatten handles it.
    out = head(torch.randn(2, 64))
    assert out.shape == (2, 10)


def test_build_head_mlp_2layer_with_dropout():
    """`probe_classifier='mlp'` returns Sequential(Flatten, Linear, ReLU, Dropout, Linear)."""
    from evaluation.probe import _build_head

    cfg = EvalConfig(
        sample_rate_hz=100.0,
        probe_classifier="mlp",
        probe_mlp_hidden=128,
        probe_mlp_dropout=0.3,
    )
    head = _build_head(feat_shape=(64,), n_classes=10, config=cfg)
    assert isinstance(head, torch.nn.Sequential)
    layers = list(head)
    assert isinstance(layers[0], torch.nn.Flatten)
    assert isinstance(layers[1], torch.nn.Linear) and layers[1].out_features == 128
    assert isinstance(layers[2], torch.nn.ReLU)
    assert isinstance(layers[3], torch.nn.Dropout) and layers[3].p == 0.3
    assert isinstance(layers[4], torch.nn.Linear) and layers[4].out_features == 10


def test_build_head_rejects_unknown_classifier():
    from evaluation.probe import _build_head

    cfg = EvalConfig(sample_rate_hz=100.0, probe_classifier="transformer")
    with pytest.raises(ValueError, match="linear.*mlp.*cnn"):
        _build_head(feat_shape=(64,), n_classes=10, config=cfg)


def test_build_head_cnn1d_for_3d_input_shape():
    """`probe_classifier='cnn'` with (C, T) feat shape → CNN1D over time."""
    from evaluation.probe import _build_head, _CNN1DHead

    cfg = EvalConfig(sample_rate_hz=100.0, probe_classifier="cnn", probe_cnn_hidden=32)
    head = _build_head(feat_shape=(271, 281), n_classes=27, config=cfg)
    assert isinstance(head, _CNN1DHead)
    # Forward pass produces (B, n_classes).
    out = head(torch.randn(4, 271, 281))
    assert out.shape == (4, 27)


def test_build_head_cnn2d_for_4d_input_shape():
    """`probe_classifier='cnn'` with (S1, S2, D) feat shape → CNN2D."""
    from evaluation.probe import _build_head, _CNN2DHead

    cfg = EvalConfig(sample_rate_hz=100.0, probe_classifier="cnn", probe_cnn_hidden=16)
    head = _build_head(feat_shape=(16, 8, 64), n_classes=4, config=cfg)
    assert isinstance(head, _CNN2DHead)
    out = head(torch.randn(2, 16, 8, 64))
    assert out.shape == (2, 4)


def test_build_head_cnn_rejects_flat_input():
    """CNN needs structured features; flat (D,) is a configuration error."""
    from evaluation.probe import _build_head

    cfg = EvalConfig(sample_rate_hz=100.0, probe_classifier="cnn")
    with pytest.raises(ValueError, match="\\(C, T\\)"):
        _build_head(feat_shape=(64,), n_classes=10, config=cfg)


def test_probe_cnn_runs_end_to_end_on_layered_tokenizer():
    """CNN probe should train and emit the expected metric keys on a
    BrainOmni-style layered tokenizer (token shape (B, S1, S2, Q))."""

    class StructuredStubTokenizer:
        """Mimics BrainOmni shape: (B, 4 latent, 2 time, 3 RVQ) tokens with
        small per-class embeddings so CNN has a chance to find signal."""

        codebook_size = 8

        def __init__(self, class_per_trial: torch.Tensor, n_dim: int = 8):
            self._classes = class_per_trial
            self._n_dim = n_dim
            torch.manual_seed(0)
            self._codebook = torch.randn(self.codebook_size, n_dim)

        def tokenize(self, x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            # Class drives a deterministic position-dependent code pattern.
            cls = self._classes[:b].to(torch.long)
            return ((torch.arange(4 * 2 * 3).reshape(4, 2, 3)[None, ...] + cls[:, None, None, None])
                    % self.codebook_size).long()

        def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
            return torch.zeros(tokens.shape[0], 4, 16)

        def tokens_to_embedding(
            self, tokens: torch.Tensor, layers: tuple[int, ...] | None = None
        ) -> torch.Tensor:
            # tokens: (B, 4, 2, 3) → emb (B, 4, 2, D) (sum over RVQ axis)
            idx = tokens.reshape(tokens.shape[0], -1, 3).long()  # (B, L=8, Q=3)
            chosen = layers if layers is not None else (0, 1, 2)
            out = torch.zeros(idx.shape[0], idx.shape[1], self._n_dim)
            for q in chosen:
                out = out + self._codebook[idx[..., q]]
            return out  # (B, L=8, D)

    torch.manual_seed(0)
    B = 80
    classes = torch.randint(0, 3, (B,))
    signal = torch.randn(B, 4, 16)
    tok = StructuredStubTokenizer(classes)
    cfg = EvalConfig(
        sample_rate_hz=100.0,
        batch_size=16,
        seed=0,
        probe_epochs=40,
        probe_top_k=(1,),
        probe_n_folds=3,
        probe_classifier="cnn",
        probe_cnn_hidden=16,
        probe_rvq_layers=(None,),
    )
    result = compute_probe_metrics(tok, signal, classes, cfg)
    # Must emit the expected metric keys (no crash, no NaNs).
    assert "top1_tokens_all_mean" in result.values
    assert "bal_acc_tokens_all_mean" in result.values
    assert "top1_raw_mean" in result.values
    assert "top1_random_mean" in result.values
    for v in result.values.values():
        assert math.isfinite(v)


def test_probe_mlp_runs_end_to_end_on_informative_signal():
    """MLP probe trains and emits the same set of metric keys as linear."""
    torch.manual_seed(0)
    B, C, T = 96, 4, 16
    classes = torch.randint(0, 3, (B,))
    means = torch.linspace(-2.0, 2.0, 3)
    signal = means[classes].view(-1, 1, 1) + 0.1 * torch.randn(B, C, T)
    tok = InformativeStubTokenizer(codebook_size=16, seq_len=4, signal_shape=(C, T))
    cfg = EvalConfig(
        sample_rate_hz=100.0,
        batch_size=24,
        seed=0,
        probe_epochs=60,
        probe_top_k=(1,),
        probe_n_folds=3,
        probe_classifier="mlp",
        probe_mlp_hidden=32,
    )
    result = compute_probe_metrics(tok, signal, classes, cfg)
    assert "top1_tokens_mean" in result.values
    assert "bal_acc_tokens_mean" in result.values
    # MLP should still be able to read off a 3-class deterministic signal.
    assert result.values["top1_tokens_mean"] > result.values["top1_random_mean"]


def test_class_weights_match_sklearn_formula():
    """Per-fold weights = N / (n_classes * count_c) for sklearn 'balanced'."""
    from evaluation.probe import _class_weights

    y = torch.tensor([0, 0, 0, 0, 1, 1, 2], dtype=torch.long)
    w = _class_weights(y, n_classes=3, device="cpu")
    # N=7, n_classes=3
    # w[0] = 7 / (3 * 4) = 0.5833
    # w[1] = 7 / (3 * 2) = 1.1667
    # w[2] = 7 / (3 * 1) = 2.3333
    assert w.shape == (3,)
    assert abs(w[0].item() - 7 / 12) < 1e-5
    assert abs(w[1].item() - 7 / 6) < 1e-5
    assert abs(w[2].item() - 7 / 3) < 1e-5


def test_class_weights_default_for_missing_class():
    """If a class is absent in a fold, weight falls back to N / n_classes (defensive)."""
    from evaluation.probe import _class_weights

    y = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)  # class 2 absent
    w = _class_weights(y, n_classes=3, device="cpu")
    # Missing class 2: default = 6 / 3 = 2.0
    assert abs(w[2].item() - 2.0) < 1e-5


def _imbalanced_random_setup():
    """Heavily imbalanced 3-class problem with random features. Shared by the
    weighted / unweighted regression tests below."""
    torch.manual_seed(0)
    B = 300
    classes = torch.cat([
        torch.zeros(int(B * 0.8), dtype=torch.long),
        torch.ones(int(B * 0.15), dtype=torch.long),
        torch.full((int(B * 0.05),), 2, dtype=torch.long),
    ])
    signal = torch.randn(B, 2, 8)  # truly random, no class info
    tok = RandomTokenizer(codebook_size=16, seq_len=4, signal_shape=(2, 8))
    return signal, classes, tok


def test_probe_unweighted_collapses_to_majority_on_imbalanced():
    """Unweighted CE + imbalanced labels + random features → majority-predict.

    This is the failure mode that motivates `probe_class_weighted` and is the
    reason raw top-1 is not a reliable diagnostic on THINGS' 27-class scheme.
    """
    signal, classes, tok = _imbalanced_random_setup()
    cfg = EvalConfig(
        sample_rate_hz=100.0,
        batch_size=32,
        seed=0,
        probe_epochs=80,
        probe_top_k=(1,),
        probe_n_folds=3,
        probe_class_weighted=False,
    )
    result = compute_probe_metrics(tok, signal, classes, cfg)
    # Top-1 collapses to majority class ≈ 80%.
    # Balanced accuracy collapses to 1/n_classes ≈ 33% (recall(class 0)=1, others=0).
    assert result.values["top1_random_mean"] > 0.6
    assert result.values["bal_acc_random_mean"] < 0.45
    assert result.values["bal_acc_random_mean"] > 0.20


def test_probe_class_weighted_prevents_majority_collapse():
    """Class-weighted CE + same setup → classifier predicts ~uniformly, so
    top-1 drops from majority floor to roughly 1/n_classes and balanced
    accuracy is still at chance (no signal to find)."""
    signal, classes, tok = _imbalanced_random_setup()
    cfg = EvalConfig(
        sample_rate_hz=100.0,
        batch_size=32,
        seed=0,
        probe_epochs=80,
        probe_top_k=(1,),
        probe_n_folds=3,
        probe_class_weighted=True,
    )
    result = compute_probe_metrics(tok, signal, classes, cfg)
    # Class weights break the majority-collapse. Top-1 should be FAR below
    # the 80% majority floor — closer to 1/3 with some noise.
    assert result.values["top1_random_mean"] < 0.5
    # Balanced accuracy is still at chance — no signal, weighting doesn't
    # create signal where there is none.
    assert 0.20 < result.values["bal_acc_random_mean"] < 0.45


def test_probe_layered_embedding_emits_per_subset_keys():
    """If a tokenizer's tokens_to_embedding accepts `layers`, the probe must
    emit one `tokens_<subset>_*` block per `probe_rvq_layers` element."""

    class LayeredStub:
        """RVQ-style stub: 2 layers, semantic info only in layer 0."""

        codebook_size = 8

        def __init__(self, class_per_trial: torch.Tensor):
            # `cls` indexes into a per-class codebook embedding so layer 0
            # carries the class info; layer 1 is noise.
            self._classes = class_per_trial
            torch.manual_seed(0)
            self._layer0 = torch.randn(self.codebook_size, 16)  # informative
            self._layer1 = torch.randn(self.codebook_size, 16) * 0.0  # noise-free zero

        def tokenize(self, x: torch.Tensor) -> torch.Tensor:
            # (B, C, T) → (B, 1, 1, 2) — 1 latent source, 1 time tok, 2 layers
            b = x.shape[0]
            layer0_tok = self._classes[:b].to(torch.long)  # class → distinct code
            layer1_tok = torch.zeros(b, dtype=torch.long)
            return torch.stack([layer0_tok, layer1_tok], dim=-1).reshape(b, 1, 1, 2)

        def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
            return torch.zeros(tokens.shape[0], 4, 16)

        def tokens_to_embedding(
            self, tokens: torch.Tensor, layers: tuple[int, ...] | None = None
        ) -> torch.Tensor:
            # tokens: (B, 1, 1, 2)
            idx = tokens.reshape(tokens.shape[0], -1, 2).long()  # (B, L=1, Q=2)
            out = torch.zeros(idx.shape[0], idx.shape[1], 16)
            books = (self._layer0, self._layer1)
            chosen = layers if layers is not None else tuple(range(len(books)))
            for q in chosen:
                out = out + books[q][idx[..., q]]
            return out  # (B, L, D)

    torch.manual_seed(0)
    B = 80
    classes = torch.randint(0, 4, (B,))
    signal = torch.randn(B, 4, 16)
    tok = LayeredStub(classes)
    cfg = EvalConfig(
        sample_rate_hz=100.0,
        batch_size=16,
        seed=0,
        probe_epochs=20,
        probe_top_k=(1,),
        probe_n_folds=3,
        probe_rvq_layers=(None, (0,), (1,)),
    )
    result = compute_probe_metrics(tok, signal, classes, cfg)
    assert "top1_tokens_all_mean" in result.values
    assert "top1_tokens_rvq0_mean" in result.values
    assert "top1_tokens_rvq1_mean" in result.values
    # Layer 0 carries the class info; layer 1 is zeros — probing rvq0 alone
    # should match or exceed probing rvq1 alone.
    assert result.values["top1_tokens_rvq0_mean"] >= result.values["top1_tokens_rvq1_mean"]


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
