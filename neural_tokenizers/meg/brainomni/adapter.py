"""BrainOmniTokenizer — Phase 3 adapter implementing evaluation.protocol.Tokenizer."""

from __future__ import annotations

from typing import Any

import torch
from einops import rearrange

from ..meg_config import BRAINOMNI_DEFAULT, BrainOmniConfig
from .config import default_ckpt_dir
from .load import load_braintokenizer
from .preprocess import PreprocessState, inverse_preprocess, preprocess_for_braintokenizer
from .sensor_metadata import SensorMetadata, load_things_meg_sensor_metadata


def _decode_rvq_indices(rvq, indices: torch.Tensor, dim: int) -> torch.Tensor:
    """Map RVQ indices ``(..., Q)`` → ``(..., dim)`` by summing layer decodes."""
    *leading, q = indices.shape
    flat = indices.reshape(-1, q)
    out = torch.zeros(flat.shape[0], dim, device=indices.device, dtype=torch.float32)
    for qi, vq_layer in enumerate(rvq.layers):
        layer_idx = flat[:, qi].long()
        out = out + vq_layer.decode(layer_idx)
    return out.reshape(*leading, dim)


class BrainOmniTokenizer:
    """Wraps OpenTSLab BrainTokenizer for THINGS-MEG.

    Token shape after ``tokenize``: ``(B, C', T_win, Q)`` where C'=16 latent
    sources, T_win≈8 temporal tokens per 512-sample window, Q=4 RVQ layers.

    ``decode_tokens`` requires a preceding ``tokenize`` on the same batch so
    per-trial z-score stats are available for inverse preprocessing (the §5
    harness calls them back-to-back per batch).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sensor_meta: SensorMetadata,
        cfg: BrainOmniConfig = BRAINOMNI_DEFAULT,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.sensor_meta = sensor_meta
        self.cfg = cfg
        self.device = torch.device(device)
        self.codebook_size: int = cfg.codebook_size
        self._preprocess_states: list[PreprocessState] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str | None = None,
        brainomni_repo: str | None = None,
        cfg: BrainOmniConfig = BRAINOMNI_DEFAULT,
        sensor_meta: SensorMetadata | None = None,
        device: str = "cpu",
    ) -> "BrainOmniTokenizer":
        ckpt_dir = ckpt_dir or default_ckpt_dir()
        brainomni_repo = brainomni_repo or cfg.brainomni_repo
        repo_abs = brainomni_repo
        if not brainomni_repo.startswith("/"):
            import os

            here = os.path.dirname(__file__)
            repo_abs = os.path.abspath(os.path.join(here, "..", "..", "..", brainomni_repo))
        model = load_braintokenizer(ckpt_dir, repo_abs, device=device)
        if sensor_meta is None:
            try:
                sensor_meta = load_things_meg_sensor_metadata()
            except FileNotFoundError:
                # Synthetic fallback for unit tests without .fif data.
                from ..meg_config import MEG_DATA

                c = MEG_DATA.n_channels
                sensor_meta = SensorMetadata(
                    pos=torch.zeros(c, 6),
                    sensor_type=torch.ones(c, dtype=torch.long),
                )
        return cls(model, sensor_meta, cfg=cfg, device=device)

    def _preprocess_batch(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        x_pad, state, mask = preprocess_for_braintokenizer(x, self.cfg)
        pos, sensor_type = self.sensor_meta.batch(x_pad.shape[0], self.device)
        self._preprocess_states = [state]
        return x_pad, pos, sensor_type, mask

    @torch.no_grad()
    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) @ 200 Hz → (B, C', T_win, Q) long token IDs."""
        self.model.eval()
        x_pad, pos, sensor_type, _ = self._preprocess_batch(x)
        _, indices = self.model.tokenize(
            x_pad, pos, sensor_type, overlap_ratio=self.cfg.overlap_ratio
        )
        # BrainTokenizer returns (B, C'=16, T_win=8, Q=4).
        return indices.long()

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, C', T_win, Q) → (B, C, T) @ 200 Hz."""
        if self._preprocess_states is None:
            raise RuntimeError(
                "decode_tokens requires a preceding tokenize() on the same batch "
                "for z-score inversion."
            )
        state = self._preprocess_states[0]
        self.model.eval()

        b = tokens.shape[0]
        pos, sensor_type = self.sensor_meta.batch(b, self.device)
        dim = self.model.n_dim
        feature = _decode_rvq_indices(self.model.quantizer.rvq, tokens, dim)
        feature = rearrange(feature, "b c t d -> b c 1 t d")
        sensor_embedding = self.model.sensor_embed(pos, sensor_type)
        x_rec = self.model.decoder(feature, sensor_embedding)
        x_hat = inverse_preprocess(x_rec, state, self.cfg)
        self._preprocess_states = None
        return x_hat

    @torch.no_grad()
    def tokens_to_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """RVQ codebook embeddings for the linear probe. Shape (B, L, D)."""
        dim = self.model.n_dim
        emb = _decode_rvq_indices(self.model.quantizer.rvq, tokens, dim)
        return emb.reshape(tokens.shape[0], -1, dim)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Direct round-trip via BrainTokenizer.visualize (for debugging)."""
        self.model.eval()
        x_pad, pos, sensor_type, _ = self._preprocess_batch(x)
        out = self.model.visualize(x_pad, pos, sensor_type)
        state = self._preprocess_states[0]
        return inverse_preprocess(out["x_rec"], state, self.cfg)


def build_tokenizer_from_payload(payload: dict[str, Any]) -> BrainOmniTokenizer:
    """Factory for Modal eval — payload from config.json."""
    cfg_dict = payload.get("config", {})
    base = {f: getattr(BRAINOMNI_DEFAULT, f) for f in BRAINOMNI_DEFAULT.__dataclass_fields__}
    base.update(cfg_dict)
    cfg = BrainOmniConfig(**base)
    return BrainOmniTokenizer.from_checkpoint(
        ckpt_dir=payload.get("ckpt_dir", default_ckpt_dir()),
        brainomni_repo=payload.get("brainomni_repo", cfg.brainomni_repo),
        cfg=cfg,
        device=payload.get("device", "cpu"),
    )
