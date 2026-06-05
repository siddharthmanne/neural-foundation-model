"""Lightweight eval-only adapter for the LaBraM EEG tokenizer.

At eval time the tokens are already in the npz cache, so we only need the
8192×64 codebook embedding table — NOT the full 94 MB LaBraM model. Loading
just the embedding table keeps the Modal container CPU-only and cheap (~2 MB
vs 94 MB, no LaBraM Python modules required).

Satisfies the Tokenizer protocol in neural_tokenizers/evaluation/protocol.py:
  - tokenize(x)              → tokens  (pass-through for pre-cached tokens)
  - decode_tokens(tokens)    → raises NotImplementedError (reconstruction skipped)
  - tokens_to_embedding(tok) → (B, N, D) dense embeddings from codebook
  - codebook_size            → 8192

Use LaBraMTokenizer (labram_tokenizer.py) when you need the full
encode/decode pipeline — it requires LaBraM on sys.path and a GPU is
recommended.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .eeg_config import EEG_DATA


class LaBraMEvalAdapter:
    """Eval-only wrapper: loads only the VQ codebook embedding table."""

    codebook_size: int = EEG_DATA.token_vocab_size   # 8192
    embed_dim: int = EEG_DATA.embed_dim              # 64

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.embedding = _load_embedding_table(ckpt_path, self.codebook_size, self.embed_dim)
        self.embedding.to(self.device)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Pass-through for pre-cached token IDs.

        Accepts:
          (B, N)    long/float — token IDs per position
          (B, N, 1) float      — token IDs with a dummy time axis, as
                                 produced by the evaluate() harness when it
                                 treats (B, N, 1) as the raw signal.
        """
        if x.ndim == 3 and x.shape[-1] == 1:
            return x.squeeze(-1).long()
        return x.long()

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Reconstruction is not available in eval-only mode. "
            "This adapter loads only the embedding table. "
            "Use LaBraMTokenizer for full encode/decode."
        )

    def tokens_to_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Look up codebook embeddings for pre-cached token IDs.

        Args:
            tokens: (B, N) long tensor — e.g. (B, 17) for 17-channel EEG.

        Returns:
            (B, N, 64) float tensor of codebook embeddings.
        """
        return self.embedding(tokens.long().to(self.device))

    @classmethod
    def from_config(cls, payload: dict) -> "LaBraMEvalAdapter":
        """Factory used by modal_eeg_eval.py dispatch table."""
        return cls(
            ckpt_path=payload["ckpt_path"],
            device=payload.get("device", "cpu"),
        )


def _load_embedding_table(
    ckpt_path: str,
    codebook_size: int,
    embed_dim: int,
) -> nn.Embedding:
    """Extract the VQ codebook embedding weight from a LaBraM checkpoint.

    The quantizer embedding lives at 'quantize.embedding.weight' (shape
    (8192, 64)). We load nothing else — no encoder, no decoder, no EMA
    buffers — keeping peak memory below 10 MB.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt

    # Locate the embedding weight. Exclude EMA tracking buffers (embed_avg,
    # cluster_size) that share the 'quantize' prefix.
    candidates = [
        k for k in state_dict
        if "quantize" in k
        and "embedding" in k
        and k.endswith(".weight")
        and "avg" not in k
        and "cluster" not in k
        and "ema" not in k.lower()
    ]
    if not candidates:
        quant_keys = [k for k in state_dict if "quantize" in k]
        raise KeyError(
            f"Cannot find embedding weight in {ckpt_path}. "
            f"quantize keys present: {quant_keys[:10]}"
        )
    emb_key = min(candidates, key=len)   # prefer the shortest key if ambiguous
    emb_weight = state_dict[emb_key]

    if tuple(emb_weight.shape) != (codebook_size, embed_dim):
        raise ValueError(
            f"Expected embedding shape ({codebook_size}, {embed_dim}), "
            f"got {tuple(emb_weight.shape)} at key {emb_key!r}"
        )
    print(f"[labram_adapter] loaded codebook {tuple(emb_weight.shape)} from {emb_key!r}")
    return nn.Embedding.from_pretrained(emb_weight.float(), freeze=True)
