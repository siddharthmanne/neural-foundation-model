"""Training utilities for BrainTokenizer finetune on THINGS-MEG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..meg_config import BRAINOMNI_DEFAULT, BrainOmniConfig

FinetuneMode = Literal["adapt", "full", "rvq_only"]


@dataclass(frozen=True)
class FinetuneConfig:
    """Finetune hyperparameters tuned for THINGS-MEG domain shift.

    ``adapt`` (default): freeze SEANet conv stacks pretrained on 656 h resting
    MEG; train sensor layout, cross-attention latent mapping, RVQ codebooks,
    and cross-attention decoder path. This is conservative given ~42 h of
    event-related finetune data vs massive resting pretraining.

    ``full``: unfreeze everything (higher overfit / catastrophic-forgetting risk).

    ``rvq_only``: freeze all conv/attn weights; only RVQ EMA codebook updates
    during forward (no gradient-based params — diagnostic mode).
    """

    mode: FinetuneMode = "adapt"
    lr: float = 1e-5
    codebook_lr: float = 3e-5
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    batch_size: int = 32
    epochs: int = 10
    # Early stopping: break out of the epoch loop if `val_loss` hasn't beaten
    # the running best in `patience` consecutive epochs. `min_epochs` bounds
    # how early we can stop — at least this many epochs run regardless, to
    # absorb cold-start volatility. Used by Experiment 2 (averaged-MEG
    # finetune) where 18k samples × 959k params gives lower
    # samples-per-param and overfitting is a real risk. Set patience<=0 to
    # disable.
    patience: int = 2
    min_epochs: int = 3


FINETUNE_DEFAULT = FinetuneConfig()


def apply_finetune_mode(model, mode: FinetuneMode = "adapt") -> dict[str, int]:
    """Set ``requires_grad`` flags. Returns trainable/total param counts."""
    for p in model.parameters():
        p.requires_grad = True

    if mode in ("adapt", "rvq_only"):
        for p in model.encoder.seanet_encoder.parameters():
            p.requires_grad = False
        for p in model.decoder.seanet_decoder.parameters():
            p.requires_grad = False

    if mode == "rvq_only":
        for _name, p in model.named_parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


def build_optimizer_groups(
    model,
    ft: FinetuneConfig = FINETUNE_DEFAULT,
) -> list[dict]:
    """AdamW param groups mirroring BrainOmni's norm/codebook split."""
    normal, no_decay, codebook = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "norm" in name or name == "sensor_embed.sensor_embedding_layer.weight":
            no_decay.append(p)
        elif "quantizer" in name:
            codebook.append(p)
        else:
            normal.append(p)
    groups = []
    if normal:
        groups.append({"params": normal, "lr": ft.lr, "weight_decay": ft.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": ft.lr, "weight_decay": 0.0})
    if codebook:
        groups.append({"params": codebook, "lr": ft.codebook_lr, "weight_decay": 0.0})
    return groups


def _expand_mask_to_unfold(
    mask: torch.Tensor,
    n_windows: int,
    window_length: int,
) -> torch.Tensor:
    if n_windows == 1:
        return mask.unsqueeze(2)
    return mask.unsqueeze(2).expand(-1, -1, n_windows, -1)


def compute_braintokenizer_loss(
    model,
    x_pad: torch.Tensor,
    pos: torch.Tensor,
    sensor_type: torch.Tensor,
    valid_mask: torch.Tensor,
    cfg: BrainOmniConfig = BRAINOMNI_DEFAULT,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward + padding-aware losses (BrainOmni pretrain recipe)."""
    from model_utils.loss import get_frequency_domain_loss, get_pcc

    x_unfold = model.unfold(x_pad, overlap_ratio=cfg.overlap_ratio)
    sensor_embedding = model.sensor_embed(pos, sensor_type)
    feature = model.encoder(x_unfold, sensor_embedding)
    feature, _indices, commitment_loss = model.quantizer(feature)
    x_rec = model.decoder(feature, sensor_embedding)

    x_tgt = model.norm_target(x_unfold)
    x_rec_f = x_rec.float()
    mask_u = _expand_mask_to_unfold(
        valid_mask, x_unfold.shape[2], cfg.window_length
    ).to(x_rec_f.dtype)

    diff = (x_rec_f - x_tgt) ** 2
    time_loss = (diff * mask_u).sum() / mask_u.sum().clamp_min(1.0)
    pcc = get_pcc(x_rec_f * mask_u, x_tgt * mask_u)
    amp_loss, phase_loss = get_frequency_domain_loss(
        x_rec_f * mask_u, x_tgt * mask_u
    )

    total = (
        time_loss
        + torch.exp(-pcc)
        + commitment_loss
        + amp_loss
        + 0.5 * phase_loss
    )
    metrics = {
        "time_loss": time_loss.detach(),
        "pcc": pcc.detach(),
        "commitment_loss": commitment_loss.detach(),
        "amp_loss": amp_loss.detach(),
        "phase_loss": phase_loss.detach(),
        "total": total.detach(),
    }
    return total, metrics
