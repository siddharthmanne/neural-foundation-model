"""Unit tests for the μ-transform tokenizer.

Strategy: write tests that pin mathematical properties (idempotence,
round-trip error bounds, output ranges) so failures point at a real bug,
not at incidental numerical noise. Tests use small synthetic tensors that
match the THINGS-MEG shape — (B, 271, 281) at MEG_DATA defaults — but
with a tiny B so the suite runs in seconds on CPU.

Run:
    cd neural_tokenizers && pytest meg/mu_transform/test_mu_transform.py -v
"""

from __future__ import annotations

import json

import pytest
import torch

from evaluation import EvalConfig, evaluate
from meg import (
    MEG_DATA,
    MU_TRANSFORM_DEFAULT,
    MuCalibration,
    MuTransformConfig,
    MuTransformTokenizer,
    fit_calibration,
)
from meg.mu_transform import encoder as enc_mod
from meg.mu_transform import quantizer as q_mod


# ----------------------------- fixtures -----------------------------------


@pytest.fixture
def small_shape() -> tuple[int, int, int]:
    """Use a small B so the suite stays fast; full MEG (C, T) keeps it realistic."""
    return (8, MEG_DATA.n_channels, MEG_DATA.n_timepoints)


@pytest.fixture
def synthetic_meg(small_shape) -> torch.Tensor:
    """Synthetic MEG-like signal: per-channel scaled Gaussian + occasional outliers."""
    torch.manual_seed(0)
    B, C, T = small_shape
    x = torch.randn(B, C, T)
    # Inter-channel amplitude variance: scale each channel by a random factor in
    # [0.5, 2.0] — mirrors the real per-sensor variance the per-channel scaler
    # is designed to absorb.
    scale = 0.5 + 1.5 * torch.rand(C)
    x = x * scale.view(1, C, 1)
    # Sprinkle outliers (1 in 1000 samples) — the clip percentiles should remove these.
    mask = torch.rand(B, C, T) < 1e-3
    x[mask] = x[mask] + 50.0 * torch.sign(torch.randn(int(mask.sum())))
    return x


@pytest.fixture
def calib(synthetic_meg) -> MuCalibration:
    return fit_calibration(synthetic_meg, MU_TRANSFORM_DEFAULT)


# ----------------------------- calibration --------------------------------


def test_calibration_shapes(calib):
    C = MEG_DATA.n_channels
    assert calib.clip_lo.shape == (C,)
    assert calib.clip_hi.shape == (C,)
    assert calib.scaler.shape == (C,)
    assert calib.vocab_size == MU_TRANSFORM_DEFAULT.vocab_size


def test_calibration_clip_thresholds_ordered(calib):
    """Each channel's lo < hi (sanity); scaler = max(|lo|, |hi|)."""
    assert torch.all(calib.clip_lo < calib.clip_hi)
    expected_s = torch.maximum(calib.clip_lo.abs(), calib.clip_hi.abs())
    assert torch.allclose(calib.scaler, expected_s)


def test_calibration_global_mode_gives_shared_thresholds():
    torch.manual_seed(1)
    x = torch.randn(16, MEG_DATA.n_channels, MEG_DATA.n_timepoints)
    cfg = MuTransformConfig(channel_mode="global")
    cal = fit_calibration(x, cfg)
    assert torch.allclose(cal.clip_lo, cal.clip_lo[0].expand_as(cal.clip_lo))
    assert torch.allclose(cal.clip_hi, cal.clip_hi[0].expand_as(cal.clip_hi))


def test_calibration_json_roundtrip(calib, tmp_path):
    p = tmp_path / "calibration.json"
    calib.save(p)
    loaded = MuCalibration.load(p)
    assert torch.allclose(loaded.clip_lo, calib.clip_lo)
    assert torch.allclose(loaded.clip_hi, calib.clip_hi)
    assert torch.allclose(loaded.scaler, calib.scaler)
    assert loaded.mu == calib.mu
    assert loaded.vocab_size == calib.vocab_size
    # JSON should be human-readable, not a base64 blob.
    raw = json.loads(p.read_text())
    assert isinstance(raw["clip_lo"], list)


def test_calibration_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        fit_calibration(torch.empty(0, MEG_DATA.n_channels, MEG_DATA.n_timepoints), MU_TRANSFORM_DEFAULT)


# ----------------------------- encoder ------------------------------------


def test_encoder_output_in_unit_range(synthetic_meg, calib):
    y = enc_mod.encode(synthetic_meg, calib)
    assert y.shape == synthetic_meg.shape
    # The encoder clips first, so output must lie strictly in [-1, 1].
    assert y.min() >= -1.0 - 1e-6
    assert y.max() <= 1.0 + 1e-6


def test_encoder_preserves_sign(calib):
    """μ-law is sign-preserving: sgn(F(x)) == sgn(x) for any x in [-1, 1]."""
    torch.manual_seed(0)
    x = torch.randn(4, MEG_DATA.n_channels, MEG_DATA.n_timepoints)
    y = enc_mod.encode(x, calib)
    # Allow zeros to match either way (sgn(0) == 0).
    nonzero = x.abs() > 1e-6
    assert torch.all(torch.sign(y[nonzero]) == torch.sign(x[nonzero]))


# ----------------------------- quantizer ----------------------------------


def test_quantize_token_range():
    """Every output token id must lie in [0, V)."""
    V = 256
    x = torch.linspace(-1.5, 1.5, 4096).reshape(1, 1, -1)  # values outside [-1,1] too
    tokens = q_mod.quantize(x, V)
    assert tokens.dtype == torch.long
    assert tokens.min().item() >= 0
    assert tokens.max().item() < V


def test_quantize_dequantize_within_bin_width():
    """|x - dequantize(quantize(x))| <= bin_width = 2/V (after clamping to [-1,1])."""
    V = 256
    bin_width = 2.0 / V
    x = torch.linspace(-1.0, 1.0, 1000).reshape(1, 1, -1)
    tokens = q_mod.quantize(x, V)
    x_back = q_mod.dequantize(tokens, V)
    err = (x_back - x).abs()
    assert err.max().item() <= bin_width + 1e-6


def test_bin_centers_are_evenly_spaced():
    V = 256
    centers = q_mod.bin_centers(V)
    assert centers.shape == (V,)
    diffs = centers[1:] - centers[:-1]
    assert torch.allclose(diffs, torch.full_like(diffs, 2.0 / V))


# ----------------------------- tokenizer (round-trip) ---------------------


def test_tokenizer_protocol_shapes(synthetic_meg, calib):
    tok = MuTransformTokenizer(calib)
    tokens = tok.tokenize(synthetic_meg)
    assert tokens.shape == synthetic_meg.shape       # (B, C, T)
    assert tokens.dtype == torch.long
    assert tokens.min().item() >= 0
    assert tokens.max().item() < tok.codebook_size

    x_hat = tok.decode_tokens(tokens)
    assert x_hat.shape == synthetic_meg.shape
    assert x_hat.dtype == calib.scaler.dtype


def test_roundtrip_error_bounded(synthetic_meg, calib):
    """Round-trip error on non-clipped values should be <= scaler * bin_width."""
    tok = MuTransformTokenizer(calib)
    x_hat = tok.decode_tokens(tok.tokenize(synthetic_meg))

    # Restrict to samples that were NOT clipped, so the only loss is quantization.
    lo = calib.clip_lo.view(1, -1, 1)
    hi = calib.clip_hi.view(1, -1, 1)
    not_clipped = (synthetic_meg >= lo) & (synthetic_meg <= hi)
    err = (x_hat - synthetic_meg).abs()
    # Per-channel max permissible: bin_width(=2/V) * scaler_c, after μ-law expansion
    # the actual max is a bit larger near |x|=s; give a 3x slack for safety.
    bin_amp = (2.0 / calib.vocab_size) * calib.scaler.view(1, -1, 1)
    assert (err[not_clipped] <= 3.0 * bin_amp.expand_as(err)[not_clipped]).all()


def test_tokenizer_runs_on_gpu_if_available(synthetic_meg, calib):
    """Confirm device-agnostic ops: same tokens whether on CPU or CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    tok = MuTransformTokenizer(calib)
    tokens_cpu = tok.tokenize(synthetic_meg)
    tokens_gpu = tok.tokenize(synthetic_meg.cuda()).cpu()
    assert torch.equal(tokens_cpu, tokens_gpu)


# ----------------------------- harness e2e --------------------------------


def test_tokens_to_embedding_preserves_position_as_bin_centers(calib):
    """Token IDs round-trip to bin centers in [-1, 1] in the (B, 1, C*T)
    shape the §5 probe expects. Crucial for near-lossless codecs — bag-of-codes
    would discard the spatial-temporal structure μ-transform faithfully preserves.
    """
    tok = MuTransformTokenizer(calib)
    B, C, T = 4, MEG_DATA.n_channels, MEG_DATA.n_timepoints
    tokens = torch.randint(0, tok.codebook_size, (B, C, T), dtype=torch.long)
    emb = tok.tokens_to_embedding(tokens)
    assert emb.shape == (B, 1, C * T)
    assert emb.dtype == torch.float32
    assert emb.min() >= -1.0 and emb.max() < 1.0
    V = tok.codebook_size
    assert abs(emb.max().item() - (1.0 - 1.0 / V)) < 1e-5


def test_mu_passes_eval_harness_smoke(synthetic_meg, calib):
    """End-to-end smoke: harness runs on MuTransformTokenizer and produces
    finite numbers for every axis. Doesn't assert quality (that's what the
    Modal run on real data is for) — just that we satisfy the protocol."""
    tok = MuTransformTokenizer(calib)
    B = synthetic_meg.shape[0]
    labels = torch.randint(0, 3, (B,), dtype=torch.long)
    cfg = EvalConfig(
        sample_rate_hz=MEG_DATA.sfreq_hz,
        batch_size=4,
        seed=0,
        psd_nperseg=128,
        probe_epochs=30,
        probe_test_frac=0.25,
    )
    report = evaluate(tok, synthetic_meg, labels, cfg)
    for axis in (report.reconstruction, report.codebook, report.sequence, report.probe):
        assert axis is not None
        for v in axis.values.values():
            assert torch.isfinite(torch.tensor(v)).item()
