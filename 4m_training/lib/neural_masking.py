"""Masking wrapper that skips Dirichlet budget for absent MEG/EEG samples.

When ``meg_mask`` or ``eeg_mask`` is 0 (sentinel placeholder on disk), force
token budgets for the corresponding ``tok_*`` modality to zero so 4M's
``sequence_token_mask`` produces no decoder logits and loss is skipped.
"""

from __future__ import annotations

import random
from typing import Any

import torch
from fourm.data.masking import UnifiedMasking
from fourm.data.modality_transforms import get_transform_key

from neural_constants import (
    EEG_MODALITY,
    MEG_RVQ_MODALITIES,
    NEURAL_GRID_TYPE,
)

# Maps token modality -> presence-flag key loaded from shard tars. The four MEG RVQ
# modalities share ``meg_mask``; EEG gates on ``eeg_mask`` (many modalities -> one mask).
PRESENCE_FLAGS: dict[str, str] = {
    **{mod: "meg_mask" for mod in MEG_RVQ_MODALITIES},
    EEG_MODALITY: "eeg_mask",
}


def _read_presence_flag(value: Any) -> bool:
    """Return True when neural data is present (mask == 1)."""
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return bool(int(value.reshape(-1)[0].item()))
    return bool(int(value.reshape(-1)[0]))


def extract_presence_flags(mod_dict: dict) -> dict[str, bool]:
    """Pop ``meg_mask`` / ``eeg_mask`` from ``mod_dict`` and return presence map.

    Several token modalities may share one flag (e.g. the four ``tok_meg_rvq*`` heads
    all gate on ``meg_mask``), so read every flag before removing the unique flag keys.
    """
    presence: dict[str, bool] = {}
    for tok_mod, flag_mod in PRESENCE_FLAGS.items():
        presence[tok_mod] = (
            _read_presence_flag(mod_dict[flag_mod]) if flag_mod in mod_dict else True
        )
    for flag_mod in set(PRESENCE_FLAGS.values()):
        mod_dict.pop(flag_mod, None)
    return presence


def zero_absent_budgets(
    modality_info: dict,
    input_budget: list[int],
    target_budget: list[int | None],
    presence: dict[str, bool],
) -> tuple[list[int], list[int | None]]:
    """Set input/target budgets to 0 for token modalities marked absent."""
    input_budget = list(input_budget)
    target_budget = list(target_budget)
    for i, mod_name in enumerate(modality_info.keys()):
        if mod_name in presence and not presence[mod_name]:
            input_budget[i] = 0
            if target_budget[i] is not None:
                target_budget[i] = 0
    return input_budget, target_budget


class PresenceAwareUnifiedMasking(UnifiedMasking):
    """``UnifiedMasking`` that respects per-sample MEG/EEG presence flags."""

    def __call__(self, mod_dict):  # noqa: D102 — mirrors upstream signature
        presence = extract_presence_flags(mod_dict)

        if self.sampling_weights is not None:
            dir_idx = torch.multinomial(self.sampling_weights, 1).item()
        else:
            dir_idx = random.randint(0, self.num_dirichlets - 1)

        num_input_tokens = random.randint(*self.input_tokens_range)
        num_target_tokens = (
            random.randint(*self.target_tokens_range)
            if self.target_tokens_range is not None
            else None
        )

        input_token_budget = self.input_token_budget(num_input_tokens, dir_idx)

        if num_target_tokens is not None:
            target_token_budget = self.target_token_budget(
                input_token_budget, num_target_tokens, dir_idx
            )
        else:
            target_token_budget = [None] * self.num_modalities

        input_token_budget, target_token_budget = zero_absent_budgets(
            self.modality_info,
            input_token_budget,
            target_token_budget,
            presence,
        )

        masked_mod_dict = {}
        for (mod_name, mod_info), input_budget, target_budget in zip(
            self.modality_info.items(),
            input_token_budget,
            target_token_budget,
        ):
            mod_type = mod_info["type"]
            mod_name_load = (
                mod_name if mod_name in mod_dict else get_transform_key(mod_name)
            )
            if mod_type == "img":
                masked_mod_dict[mod_name] = self.image_mask(
                    mod_dict[mod_name_load],
                    mod_info["max_tokens"],
                    input_budget,
                    target_budget,
                )
            elif mod_type == "seq":
                keep_scheme = (
                    "random"
                    if ("keep" not in mod_info)
                    else mod_info["keep"][dir_idx]
                )
                masked_mod_dict[mod_name] = self.sequence_mask(
                    mod_dict[mod_name_load],
                    mod_info["max_tokens"],
                    input_budget,
                    target_budget,
                    keep_scheme,
                )
            elif mod_type == NEURAL_GRID_TYPE:
                # Symmetric neural modalities: parallel grid masking (scatter), like img.
                # image_mask takes BOTH budgets and partitions cells into input vs target
                # (disjoint), so a cell is never both an encoder input and a decoder target
                # -> no input->target leakage. The decoder branch in fm.cat_decoder_tensors
                # treats any non-seq type as parallel, so targets are per-cell, not shifted.
                masked_mod_dict[mod_name] = self.image_mask(
                    mod_dict[mod_name_load],
                    mod_info["max_tokens"],
                    input_budget,
                    target_budget,
                )
            elif mod_type == "seq_token":
                keep_scheme = (
                    "random"
                    if ("keep" not in mod_info)
                    else mod_info["keep"][dir_idx]
                )
                vocab_offset = mod_info.get("vocab_offset", 0)
                masked_mod_dict[mod_name] = self.sequence_token_mask(
                    mod_dict[mod_name_load],
                    mod_info["max_tokens"],
                    input_budget,
                    target_budget,
                    keep_scheme,
                    vocab_offset=vocab_offset,
                )
            elif mod_type == "seq_emb":
                keep_scheme = (
                    "random"
                    if ("keep" not in mod_info)
                    else mod_info["keep"][dir_idx]
                )
                masked_mod_dict[mod_name] = self.sequence_emb_mask_span(
                    mod_dict[mod_name_load],
                    mod_info["max_tokens"],
                    input_budget,
                    target_budget,
                    keep_scheme,
                )
            else:
                raise ValueError(f"Invalid modality type: {mod_type}")

        return masked_mod_dict
