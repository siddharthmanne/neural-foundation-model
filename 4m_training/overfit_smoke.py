"""Overfit-one-batch sanity check — the minimum bar before any real training.

A model that cannot drive the loss down on a single fixed batch has a broken
forward/backward path, and no amount of data or compute will fix that. This
harness builds ``fm_tiny``, takes ONE batch with every modality present, and
runs a handful of optimizer steps on that same batch. If wiring is correct,
*each* modality's loss falls monotonically toward zero.

This is not training — it is a unit-level "does gradient flow to every head"
test. Runs on CPU in seconds; also exposed as a Modal entrypoint for GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Library modules live in lib/; put it on sys.path so the flat imports below resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

from repo_paths import TEXT_TOKENIZER  # noqa: E402

import fourm_neural_modalities  # noqa: F401,E402 — register modalities + transforms
from fourm_dataloader import patch_pretrain_utils  # noqa: E402
from neural_constants import (  # noqa: E402
    EEG_TRIAL_SHAPE,
    EEG_VOCAB_SIZE,
    MEG_TRIAL_SHAPE,
    MEG_VOCAB_SIZE,
    THINGS_IMAGE_SIZE,
    TOK_DEPTH_VOCAB_SIZE,
    TOK_RGB_TOKENS_PER_IMAGE,
    TOK_RGB_VOCAB_SIZE,
)

patch_pretrain_utils()

# Neural mods are input-only: encoded but never predicted. Vision tokens are the
# targets whose loss must descend. See notes/4m_neural_modality_design.md.
_IN_DOMAINS = ["tok_rgb", "tok_depth", "tok_meg", "tok_eeg"]
_OUT_DOMAINS = ["tok_rgb", "tok_depth"]


def _decoded_sample(rng: np.random.Generator) -> dict:
    """One image with every modality present (mask=1) and in-vocab tokens."""
    return {
        "tok_rgb": rng.integers(0, TOK_RGB_VOCAB_SIZE, (TOK_RGB_TOKENS_PER_IMAGE,)).astype(np.int64),
        "tok_depth": rng.integers(0, TOK_DEPTH_VOCAB_SIZE, (TOK_RGB_TOKENS_PER_IMAGE,)).astype(np.int64),
        "tok_meg": rng.integers(0, MEG_VOCAB_SIZE, (2, *MEG_TRIAL_SHAPE)).astype(np.int64),
        "tok_eeg": rng.integers(0, EEG_VOCAB_SIZE, (2, *EEG_TRIAL_SHAPE)).astype(np.int64),
        "meg_mask": np.array([1], dtype=np.int64),
        "eeg_mask": np.array([1], dtype=np.int64),
        "__key__": "overfit",
    }


def _build_model(modality_info: dict, model_name: str):
    """Encoder embeddings for all inputs; decoder embeddings only for vision targets."""
    from fourm.utils import create_model

    def _instantiate(mod: str, key: str):
        info = modality_info[mod]
        is_img = info["type"] == "img"
        kw = dict(patch_size=info.get("patch_size", 16), image_size=THINGS_IMAGE_SIZE)
        return info[key](**(kw if is_img else {}))

    enc = {mod: _instantiate(mod, "encoder_embedding") for mod in _IN_DOMAINS}
    dec = {mod: _instantiate(mod, "decoder_embedding") for mod in _OUT_DOMAINS}
    return create_model(
        model_name, encoder_embeddings=enc, decoder_embeddings=dec,
        modality_info=modality_info, num_register_tokens=0,
    )


def _fixed_batch(batch_size: int, input_tokens: int, target_tokens: int, seed: int):
    """Build one collated batch; retry seeds until every modality has targets."""
    from fourm.data.modality_info import MODALITY_TRANSFORMS
    from fourm.data.modality_transforms import IdentityTransform, UnifiedDataTransform
    from fourm.data.pretrain_utils import setup_sampling_mod_info
    from fourm.data.unified_datasets import default_collate
    from neural_masking import PresenceAwareUnifiedMasking
    from things_augmenter import ThingsImageAugmenter
    from train_4m import _build_modality_info

    ds_cfg = {
        "in_domains": "-".join(sorted(_IN_DOMAINS)),
        "out_domains": "-".join(sorted(_OUT_DOMAINS)),
        "input_alphas": "-".join("1.0" for _ in _IN_DOMAINS),
        "target_alphas": "-".join("1.0" for _ in _OUT_DOMAINS),
    }
    from tokenizers import Tokenizer

    text_tokenizer = Tokenizer.from_file(str(TEXT_TOKENIZER))
    full = _build_modality_info(_IN_DOMAINS, input_size=THINGS_IMAGE_SIZE)
    mask_info, _ = setup_sampling_mod_info(ds_cfg, full)

    transforms = dict(MODALITY_TRANSFORMS)
    transforms["__key__"] = IdentityTransform()
    udt = UnifiedDataTransform(
        transforms_dict=transforms,
        image_augmenter=ThingsImageAugmenter(
            target_size=THINGS_IMAGE_SIZE, no_aug=True, main_domain="tok_rgb"
        ),
    )

    for attempt in range(seed, seed + 50):
        torch.manual_seed(attempt)
        masker = PresenceAwareUnifiedMasking(
            modality_info=mask_info, text_tokenizer=text_tokenizer,
            input_tokens_range=(input_tokens, input_tokens),
            target_tokens_range=(target_tokens, target_tokens),
        )
        rng = np.random.default_rng(attempt)
        samples = [masker(udt(_decoded_sample(rng))) for _ in range(batch_size)]
        batch = default_collate(samples)
        targets = {m: int((~batch[m]["target_mask"]).sum()) for m in _OUT_DOMAINS}
        if all(v > 0 for v in targets.values()):
            return batch, full, mask_info, targets
    raise RuntimeError(f"could not build a batch with targets for all out-domains: {targets}")


def run_overfit(
    steps: int = 150,
    batch_size: int = 2,
    input_tokens: int = 48,
    target_tokens: int = 48,
    lr: float = 2e-3,
    model_name: str = "fm_tiny_6e_6d_swiglu_nobias",
    seed: int = 0,
    device: str | None = None,
    log_every: int = 25,
) -> dict[str, list[float]]:
    """Run ``steps`` optimizer steps on one fixed batch; return per-modality loss history.

    A small target budget on a tiny batch lets ``fm_tiny`` drive every modality's
    loss toward zero. ``model.eval()`` disables dropout so the descent is clean and
    the run is deterministic (the only per-step variation, 4M's modality shuffle, is
    loss-invariant); we still optimize all parameters.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch, model_info, _, targets = _fixed_batch(batch_size, input_tokens, target_tokens, seed)
    print(f"fixed batch target tokens per modality: {targets}", flush=True)

    model = _build_model(model_info, model_name).to(device).eval()
    batch = {m: {k: v.to(device) for k, v in d.items()} for m, d in batch.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    # Give the encoder/decoder a generous token cap so the top-k selection never
    # truncates a modality's (few) tokens — otherwise that modality's loss reads 0
    # and looks "stuck" when it simply never reached a head.
    total_targets = sum(targets.values())
    encode_cap = total_targets + input_tokens
    decode_cap = total_targets + target_tokens

    # Targets (vision) are what the loss is computed on; MEG/EEG flow through the
    # encoder only. mod_loss is keyed by out-domains.
    history: dict[str, list[float]] = {m: [] for m in _OUT_DOMAINS}
    history["total"] = []
    for step in range(steps):
        opt.zero_grad()
        loss, mod_loss = model(
            batch, num_encoder_tokens=encode_cap,
            num_decoder_tokens=decode_cap, loss_type="mod",
        )
        loss.backward()
        opt.step()
        history["total"].append(loss.item())
        for m in _OUT_DOMAINS:
            history[m].append(mod_loss[m].mean().item())
        if step % log_every == 0 or step == steps - 1:
            parts = "  ".join(f"{m}={history[m][-1]:6.3f}" for m in _OUT_DOMAINS)
            print(f"step {step:3d}  total={loss.item():6.3f}  {parts}", flush=True)
    return history


def assert_decreasing(history: dict[str, list[float]], min_drop: float = 0.5) -> None:
    """Each predicted (out-domain) modality's loss must end meaningfully lower."""
    for mod in _OUT_DOMAINS:
        first = min(history[mod][:3])
        last = min(history[mod][-3:])
        assert last < first - min_drop, (
            f"{mod}: loss did not decrease (start≈{first:.3f}, end≈{last:.3f})"
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    hist = run_overfit(steps=args.steps, lr=args.lr, device=args.device)
    assert_decreasing(hist)
    print("\nOVERFIT OK — vision targets descended; MEG/EEG flowed through the encoder.")
