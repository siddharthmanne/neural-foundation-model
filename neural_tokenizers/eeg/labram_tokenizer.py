"""
LaBraMTokenizer: wraps a finetuned LaBraM vqnsp checkpoint in the Tokenizer
protocol expected by neural_tokenizers/evaluation/.

Only importable inside a Modal container where LaBraM is cloned at /LaBraM
(or wherever LABRAM_ROOT points, default /LaBraM).

Protocol shapes:
    tokenize(x: (B, 17, 100)) -> (B, 17) long   [one code per channel, 8192 vocab]
    decode_tokens((B, 17))     -> (B, 17, 100)   [17 channels, 100 timepoints @ 100 Hz]
    codebook_size              -> 8192

Input EEG is at 100 Hz (THINGS-EEG2 preprocessed format). The LaBraM model
expects 200 Hz, so we upsample before encoding and downsample after decoding.
"""

from __future__ import annotations

import os
import sys
from typing import ClassVar

import scipy.signal
import torch
import torch.nn.functional as F


LABRAM_ROOT = os.environ.get("LABRAM_ROOT", "/LaBraM")

# Add LaBraM to sys.path so its relative imports (modeling_vqnsp, utils, etc.) resolve.
if LABRAM_ROOT not in sys.path:
    sys.path.insert(0, LABRAM_ROOT)

# THINGS-EEG2 posterior/occipital channels in Gifford's preprocessing order.
# Used as the default ch_names when no override is provided. The actual order
# is written to the HDF5 chOrder attr during conversion and should match this.
THINGS_EEG2_CH_NAMES = [
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "PZ",
    "PO3", "PO4", "PO7", "PO8", "POZ",
    "O1", "O2", "OZ",
]


class LaBraMTokenizer:
    """Wraps the LaBraM VQNSP model to satisfy the neural_tokenizers Tokenizer protocol."""

    codebook_size: ClassVar[int] = 8192

    def __init__(self, ckpt_path: str, ch_names: list[str] | None = None, device: str = "cpu"):
        if ch_names is None:
            ch_names = THINGS_EEG2_CH_NAMES
        self.device = torch.device(device)
        self.model = _load_vqnsp(ckpt_path, self.device)
        # input_chans maps our channel subset to LaBraM's position-embedding table.
        # Entry 0 is the CLS slot; remaining entries are 1-indexed positions in
        # standard_1020 (utils.py:42-57). Must match the order used during training.
        self.input_chans = _build_input_chans(ch_names)

    # ------------------------------------------------------------------
    # Tokenizer protocol
    # ------------------------------------------------------------------

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Encode EEG trials to discrete token IDs.

        Args:
            x: (B, 17, 100) float tensor at 100 Hz.

        Returns:
            (B, 17) long tensor with values in [0, 8192).
        """
        x200 = self._upsample(x).to(self.device)   # (B, 17, 200)
        # Model was trained with 8-patch windows (input_size=1600). Passing A=1
        # means the encoder only uses positions 0-17 of a pos_embed built for 137
        # positions — it was never trained that way and produces degenerate codes.
        # Tile the 1-second trial to 8 identical patches so the encoder sees its
        # native sequence length (B, 17, 8, 200), then take the code for patch 0
        # of each channel.
        # NOTE: the 8 per-channel codes are NOT all identical. Measured on 512
        # trials (diagnose_patch_codes.py, 2026-05-26): 80% of channel-trials
        # have all 8 equal, the rest are usually a 2-way split, and patch 0
        # matches the majority of the 8 in 93% of cases. time_embed makes the
        # same content encode slightly differently per patch position, so patch
        # 0 is a representative pick, not lossless dedup.
        x_nats = x200.unsqueeze(2).expand(-1, -1, 8, -1).contiguous()  # (B, 17, 8, 200)
        with torch.no_grad():
            tokens_all = self.model.get_codebook_indices(
                x_nats, input_chans=self.input_chans
            )  # (B, 136) — 8 codes per channel, ordered [ch0_t0..ch0_t7, ch1_t0..]
            tokens = tokens_all[:, ::8]             # first patch per channel → (B, 17)
        return tokens.long().cpu()

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reconstruct EEG from token IDs.

        Reconstruction path: indices -> codebook embeddings -> LaBraM decoder
        -> (amplitude, phase) spectra -> iFFT -> time domain -> downsample.

        The output is an approximate reconstruction; exact amplitude scale is
        lost because LaBraM's decoder operates on batch-normalized spectra.
        The spectral structure (frequency band power) is preserved.

        Args:
            tokens: (B, 17) long tensor.

        Returns:
            (B, 17, 100) float tensor at 100 Hz.
        """
        tokens = tokens.to(self.device)
        B, N = tokens.shape  # N = num_channels = 17

        with torch.no_grad():
            # Step 1: look up codebook embeddings (B, N, embed_dim)
            quant_emb = _lookup_codebook(self.model, tokens)  # (B, N, embed_dim)

            # Step 2: reshape to (B, embed_dim, N, A) for the decoder.
            # Model was trained with input_size=1600 (A=8 time patches). Decoder
            # checks t == patch_size (1==1) when A=1 → wrong branch → pos_embed
            # shape mismatch. Replicate the single code across 8 time slots so the
            # correct branch (t=8 != patch_size=1) fires.
            quant_hw = quant_emb.permute(0, 2, 1).unsqueeze(-1)   # (B, embed_dim, N, 1)
            quant_hw8 = quant_hw.expand(-1, -1, -1, 8).contiguous()  # (B, embed_dim, N, 8)

            # Step 3: decode -> (amplitude_spec, phase_spec), each (B, N*8, 200)
            # Output ordering is H-major (H=channels, W=time): [ch0_t0..ch0_t7, ch1_t0..].
            # We only need the first time patch per channel (indices 0, 8, 16, ...).
            rec_amp, rec_phase = self.model.decode(quant_hw8, input_chans=self.input_chans)
            rec_amp   = rec_amp[:, ::8, :]    # (B, N, 200)
            rec_phase = rec_phase[:, ::8, :]  # (B, N, 200)

            # Step 4: reconstruct time domain via inverse FFT.
            # NOTE: rec_amp is LaBraM's std-normalized amplitude spectrum
            # (zero-mean, so it can be negative) — NOT a true non-negative
            # magnitude. Using it directly in r*exp(i*phase) makes this a
            # qualitative inverse only. Time-domain MSE against the input is a
            # unit/scale mismatch and is meaningless for this tokenizer; use
            # channel_corr (scale-invariant) as the reconstruction metric of
            # record. Do not "fix" the magnitude and then trust the MSE.
            x_complex = rec_amp * torch.exp(1j * rec_phase.float())
            x_time200 = torch.fft.irfft(x_complex, n=200, dim=-1)  # (B, 17, 200)

        return self._downsample(x_time200.cpu()).to(self.device)  # (B, 17, 100)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _upsample(x: torch.Tensor) -> torch.Tensor:
        arr = x.detach().cpu().numpy()
        return torch.tensor(
            scipy.signal.resample_poly(arr, up=2, down=1, axis=-1),
            dtype=torch.float32,
        )

    @staticmethod
    def _downsample(x: torch.Tensor) -> torch.Tensor:
        arr = x.detach().cpu().numpy()
        return torch.tensor(
            scipy.signal.resample_poly(arr, up=1, down=2, axis=-1),
            dtype=torch.float32,
        )

    @classmethod
    def from_volume(
        cls,
        ckpt_path: str,
        ch_names: list[str] | None = None,
        device: str = "cpu",
    ) -> "LaBraMTokenizer":
        return cls(ckpt_path=ckpt_path, ch_names=ch_names, device=device)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _build_input_chans(ch_names: list[str]) -> list[int]:
    """Map channel names to LaBraM's position-embedding indices.

    Entry 0 is the CLS slot. Remaining entries are (1 + 0-based index in
    standard_1020), matching utils.py:713-717.
    """
    from utils import standard_1020  # available when LABRAM_ROOT is in sys.path
    return [0] + [standard_1020.index(ch.upper()) + 1 for ch in ch_names]


def _load_vqnsp(ckpt_path: str, device: torch.device):
    """Load a VQNSP model from a checkpoint, rebuilding the architecture via timm."""
    import timm
    import modeling_vqnsp  # noqa: F401 — registers LaBraM models with timm

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Our finetune pipeline strips optimizer/scaler; raw LaBraM checkpoints have 'model' key.
    state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    args = ckpt.get("args", None)

    # Determine hyperparams: prefer saved args, fall back to our finetune defaults.
    if args is not None and hasattr(args, "codebook_n_emd"):
        n_code = args.codebook_n_emd
        code_dim = getattr(args, "codebook_emd_dim", 64)
        eeg_size = getattr(args, "input_size", 200)
    else:
        # vqnsp.pth was pretrained with input_size=1600; use that as the safe default.
        n_code, code_dim, eeg_size = 8192, 64, 1600

    # 'vqnsp_encoder_base_decoder_3x200x12' is registered by @register_model
    # in modeling_vqnsp.py and is the model used in LaBraM's default training.
    model = timm.create_model(
        "vqnsp_encoder_base_decoder_3x200x12",
        pretrained=False,
        as_tokenzer=False,
        EEG_size=eeg_size,
        n_code=n_code,
        code_dim=code_dim,
        decay=0.99,
        quantize_kmeans_init=True,
    )
    # strict=False is intentional: the checkpoint carries EMA tracking buffers
    # and amplitude/phase task layers we don't run at inference. But it would
    # also silently tolerate a wrong/half-saved checkpoint and load random
    # encoder/quantizer weights, producing garbage tokens with no error. Fail
    # loudly if any load-bearing weight (encoder or codebook) is missing.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    load_bearing_missing = [
        k for k in missing
        if k.startswith(("encoder.", "quantize.embedding.weight", "patch_embed."))
    ]
    if load_bearing_missing:
        raise RuntimeError(
            f"Checkpoint at {ckpt_path} is missing load-bearing weights "
            f"{load_bearing_missing[:8]}... ({len(load_bearing_missing)} total). "
            "Wrong file or a corrupt save — refusing to tokenize with random weights."
        )
    print(f"[labram] loaded {ckpt_path}: "
          f"{len(missing)} missing (non-load-bearing), {len(unexpected)} unexpected keys")
    model.eval()
    return model.to(device)


def _lookup_codebook(model, tokens: torch.Tensor) -> torch.Tensor:
    """Look up codebook embeddings for token indices.

    LaBraM uses NormEMAVectorQuantizer; the embedding table lives at
    model.quantize.embedding (an nn.Embedding) or model.quantize.embedding.weight
    (a Parameter). Try both.

    Returns: (B, N, embed_dim) float tensor.
    """
    quantizer = model.quantize

    if hasattr(quantizer, "embedding") and isinstance(quantizer.embedding, torch.nn.Embedding):
        return quantizer.embedding(tokens)  # (B, N, embed_dim)

    if hasattr(quantizer, "embedding") and hasattr(quantizer.embedding, "weight"):
        return F.embedding(tokens, quantizer.embedding.weight)

    if hasattr(quantizer, "embed") and isinstance(quantizer.embed, torch.nn.Embedding):
        return quantizer.embed(tokens)

    # Last resort: try looking for any nn.Embedding sub-module in quantizer
    for name, mod in quantizer.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            return mod(tokens)

    raise AttributeError(
        "Cannot find embedding table in model.quantize. "
        "Inspect `model.quantize` manually and update _lookup_codebook."
    )
