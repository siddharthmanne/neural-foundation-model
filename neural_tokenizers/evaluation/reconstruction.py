"""§5.1 Reconstruction fidelity.

Round-trip the signal through tokenize -> decode_tokens and measure:
  - mse:               time-domain mean squared error
  - channel_corr_mean: mean Pearson r across channels and batch
  - psd_mse:           MSE between Welch power spectra of x and x_hat

The PSD metric is the most diagnostic single number: neural signal lives in
frequency bands, so a tokenizer can have low time-domain MSE while still
scrambling band power. Always read the spectral number alongside the MSE.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .protocol import EvalConfig, MetricResult, Tokenizer


@torch.no_grad()
def compute_reconstruction_metrics(
    tokenizer: Tokenizer,
    signal: torch.Tensor,
    config: EvalConfig,
) -> MetricResult:
    """Compute the three reconstruction-fidelity numbers over the full signal."""
    _assert_signal_shape(signal)

    weighted_sums = {"mse": 0.0, "corr": 0.0, "psd_mse": 0.0}
    n_total = 0

    for batch in _iter_batches(signal, config.batch_size):
        x = batch.to(config.device)
        tokens = tokenizer.tokenize(x)
        x_hat = tokenizer.decode_tokens(tokens)

        b = x.shape[0]
        weighted_sums["mse"] += _mse(x, x_hat) * b
        weighted_sums["corr"] += _channel_corr(x, x_hat) * b
        weighted_sums["psd_mse"] += _psd_mse(x, x_hat, config) * b
        n_total += b

    return MetricResult(
        name="reconstruction",
        values={
            "mse": weighted_sums["mse"] / n_total,
            "channel_corr_mean": weighted_sums["corr"] / n_total,
            "psd_mse": weighted_sums["psd_mse"] / n_total,
        },
    )


def _assert_signal_shape(signal: torch.Tensor) -> None:
    if signal.ndim != 3:
        raise ValueError(
            f"signal must have shape (B, C, T); got {tuple(signal.shape)}"
        )


def _iter_batches(signal: torch.Tensor, batch_size: int):
    for start in range(0, signal.shape[0], batch_size):
        yield signal[start : start + batch_size]


def _mse(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    return F.mse_loss(x_hat, x).item()


def _channel_corr(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Mean Pearson r over (batch, channel)."""
    x_c = x - x.mean(dim=-1, keepdim=True)
    h_c = x_hat - x_hat.mean(dim=-1, keepdim=True)
    num = (x_c * h_c).sum(dim=-1)
    den = torch.sqrt((x_c * x_c).sum(dim=-1) * (h_c * h_c).sum(dim=-1)).clamp_min(1e-12)
    return (num / den).mean().item()


def _psd_mse(x: torch.Tensor, x_hat: torch.Tensor, config: EvalConfig) -> float:
    psd_x = _welch_psd(x, config)
    psd_h = _welch_psd(x_hat, config)
    return F.mse_loss(psd_h, psd_x).item()


def _welch_psd(x: torch.Tensor, config: EvalConfig) -> torch.Tensor:
    """Welch-style PSD via STFT: window the signal, FFT each window, average.

    Returns (B, C, F) power spectra. Falls back to a single FFT if the signal
    is shorter than the requested window — this is what we want for short
    THINGS-EEG epochs (~100 samples).
    """
    B, C, T = x.shape
    nperseg = config.psd_nperseg or min(256, T)
    if T < nperseg:
        nperseg = T

    flat = x.reshape(B * C, T)
    window = torch.hann_window(nperseg, device=x.device, dtype=x.dtype)

    if T == nperseg:
        spec = torch.fft.rfft(flat * window, n=nperseg)
        psd = (spec.abs() ** 2)
    else:
        hop = max(nperseg // 2, 1)
        spec = torch.stft(
            flat,
            n_fft=nperseg,
            hop_length=hop,
            win_length=nperseg,
            window=window,
            center=False,
            return_complex=True,
        )
        psd = (spec.abs() ** 2).mean(dim=-1)

    return psd.reshape(B, C, -1)
