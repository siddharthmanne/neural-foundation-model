"""Placeholder-correct accounting of the training tokens a model has seen.

Stock 4M logs a *closed-form* tokens-seen every epoch (run_training_4m.py:643-645)::

    input_tokens_seen_b = (epoch+1) * steps * (batch/accum) * num_input_tokens / 1e9

which assumes every sample contributes the full ``num_input_tokens`` /
``num_target_tokens`` budget. For THINGS that overcounts: placeholder MEG/EEG
samples get their Dirichlet budget zeroed by ``neural_masking.zero_absent_budgets``
and the dropped tokens are NOT redistributed, so such samples truly contribute
fewer tokens — and the dropped amount is random per sample, so there is no
closed-form correction.

We instead count the *actual* selected tokens straight from the per-modality masks
4M produces. A placeholder modality has ``input_budget == 0`` so ``image_mask``
selects no cells: ``(~input_mask).sum() == 0``. The masks already encode the
zeroing, so counting from them is placeholder-correct with no special-casing, and
works uniformly for image / neural-grid / sequence modalities.

Wiring (no edits to ``external/ml-4m``):

  * A forward-pre-hook on the live model sums ``~input_mask`` / ``~target_mask`` on
    every TRAINING forward (gated on ``module.training``, so eval / in-loop-val
    forwards are ignored). It accumulates on-device and syncs once per epoch.
  * ``make_accounting_train_one_epoch`` wraps the stock ``train_one_epoch``: it
    lazily registers the hook on the first epoch, merges ``[tokens] *_seen`` into
    the returned stats (→ ``log.txt``), and calls ``on_epoch_end(totals)``.
  * ``install_token_accounting`` stashes the running totals onto ``trainer.args`` so
    stock ``save_model`` serializes them into each checkpoint; the live accountant
    is returned so in-loop validation can report tokens-seen too.

The standalone ``validate_4m.py`` reads the per-checkpoint totals back with
``read_tokens_seen``; ``resolve_resume_ckpt`` mirrors stock ``auto_load_model`` so
the count continues across ``auto_resume`` rather than restarting at zero.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Callable

import torch

# Per-epoch log keys for the merged tokens-seen stats (land in log.txt / wandb).
# Public so in-loop validation can surface the same numbers without re-deriving them.
TOKEN_STAT_KEYS = {
    "input": "[tokens] input_seen",
    "target": "[tokens] target_seen",
    "total": "[tokens] total_seen",
}


def format_tokens_seen(tokens: dict[str, int] | None) -> str:
    """One-line, comma-grouped summary of tokens seen — shared by the standalone and
    in-loop validators so both report the count identically."""
    if not tokens:
        return "tokens seen (training): unknown (no token metadata in checkpoint)"
    return (
        f"tokens seen (training): input={tokens['input']:,} "
        f"target={tokens['target']:,} total={tokens['total']:,}"
    )


class TokenAccountant:
    """Accumulate the actual encoder-input and decoder-target tokens a model trains on.

    Counting reads the masks rather than the budgets, so placeholder MEG/EEG samples
    (zeroed budget → no cells selected) contribute nothing automatically. Accumulation
    happens on the mask's device to avoid a host sync every step; ``totals`` syncs once
    (and all-reduces across ranks under DDP).
    """

    def __init__(self, input_seed: int = 0, target_seed: int = 0):
        self._input_seed = int(input_seed)
        self._target_seed = int(target_seed)
        self._input_dev: torch.Tensor | None = None  # this-run running sum (on device)
        self._target_dev: torch.Tensor | None = None

    @staticmethod
    def _selected(mask: torch.Tensor) -> torch.Tensor:
        """Number of selected tokens. Masks are bool with True == masked-out, so
        ``~mask`` marks the tokens actually fed to the encoder / predicted by the
        decoder; summing counts across the whole batch."""
        return (~mask).sum()

    def _accumulate(self, attr: str, value: torch.Tensor) -> None:
        running = getattr(self, attr)
        if running is None:
            running = torch.zeros((), dtype=torch.long, device=value.device)
            setattr(self, attr, running)
        running += value.long()

    def hook(self, module: Any, inputs: tuple) -> None:
        """forward-pre-hook: count tokens for a single TRAINING forward.

        ``inputs`` is the tuple of positional args 4M calls the model with; the first
        is the ``mod_dict`` of per-modality ``{tensor, input_mask, target_mask, ...}``.
        Eval forwards (``module.training`` is False) are skipped.
        """
        if not getattr(module, "training", False):
            return
        if not inputs:
            return
        mod_dict = inputs[0]
        if not isinstance(mod_dict, dict):
            return
        in_sum: torch.Tensor | None = None
        tgt_sum: torch.Tensor | None = None
        for value in mod_dict.values():
            if not isinstance(value, dict):
                continue
            input_mask = value.get("input_mask")
            if input_mask is not None:
                s = self._selected(input_mask)
                in_sum = s if in_sum is None else in_sum + s
            target_mask = value.get("target_mask")
            if target_mask is not None:
                s = self._selected(target_mask)
                tgt_sum = s if tgt_sum is None else tgt_sum + s
        if in_sum is not None:
            self._accumulate("_input_dev", in_sum)
        if tgt_sum is not None:
            self._accumulate("_target_dev", tgt_sum)

    def _reduced_run_count(self, running: torch.Tensor | None) -> int:
        """Materialize a this-run device counter to a host int, all-reducing across
        ranks first (on a clone, so the local running sum is never double-counted)."""
        if running is None:
            return 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                running = running.clone()
                torch.distributed.all_reduce(running)
        return int(running.item())

    def totals(self) -> dict[str, int]:
        """Cumulative tokens seen = resume seed + this run's reduced counts."""
        inp = self._input_seed + self._reduced_run_count(self._input_dev)
        tgt = self._target_seed + self._reduced_run_count(self._target_dev)
        return {"input": inp, "target": tgt, "total": inp + tgt}


def make_accounting_train_one_epoch(
    orig_train_one_epoch: Callable,
    accountant: TokenAccountant,
    on_epoch_end: Callable[[dict[str, int]], None],
) -> Callable:
    """Wrap ``train_one_epoch`` to count tokens and surface the running totals.

    On the first call the model's forward-pre-hook is registered (the model only
    exists once training starts, so registration is lazy). After each epoch the
    cumulative totals are merged into the returned stats under ``[tokens] *_seen``
    (so they land in ``log.txt`` / wandb) and handed to ``on_epoch_end`` for
    persistence onto the checkpoint. Stock calls this with ``model=...`` as a kwarg.
    """
    state = {"registered": False}

    def wrapped(*args: Any, **kwargs: Any):
        model = kwargs.get("model")
        if model is not None and not state["registered"]:
            model.register_forward_pre_hook(accountant.hook)
            state["registered"] = True
        stats = orig_train_one_epoch(*args, **kwargs)
        totals = accountant.totals()
        on_epoch_end(totals)
        return {**stats, **{TOKEN_STAT_KEYS[k]: v for k, v in totals.items()}}

    return wrapped


def read_tokens_seen(ckpt: Any) -> dict[str, int] | None:
    """Return the ``{input, target, total}`` tokens-seen stashed in a checkpoint, or
    ``None`` if absent (e.g. an external checkpoint or one trained before this change)."""
    if not isinstance(ckpt, dict):
        return None
    args = ckpt.get("args")
    tokens = getattr(args, "tokens_seen", None)
    if isinstance(tokens, dict) and "total" in tokens:
        return tokens
    return None


def resolve_resume_ckpt(args: Any) -> str | None:
    """Path of the checkpoint training will resume from, or ``None`` for a fresh run.

    Mirrors stock ``auto_load_model``: an explicit ``args.resume`` wins; otherwise,
    when ``auto_resume`` is set, pick the highest-epoch ``checkpoint-<N>.pth`` in
    ``output_dir`` (non-numeric names like ``checkpoint-final.pth`` are ignored).
    A ``finetune`` source is intentionally NOT seeded from — that begins a new run.
    """
    resume = getattr(args, "resume", "") or ""
    if resume:
        return resume if os.path.exists(resume) else None
    if getattr(args, "auto_resume", False) and getattr(args, "output_dir", ""):
        latest_epoch = -1
        for ckpt in glob.glob(os.path.join(args.output_dir, "checkpoint-*.pth")):
            tag = ckpt.split("-")[-1].split(".")[0]
            if tag.isdigit():
                latest_epoch = max(int(tag), latest_epoch)
        if latest_epoch >= 0:
            return os.path.join(args.output_dir, f"checkpoint-{latest_epoch}.pth")
    return None


def install_token_accounting(trainer, *, resume_totals: dict[str, int] | None = None):
    """Reassign ``trainer.train_one_epoch`` to the token-counting wrapper.

    Seeds from ``resume_totals`` (read off the resumed checkpoint) so the count is
    cumulative across ``auto_resume``. Each epoch the running totals are stashed onto
    ``trainer.args.tokens_seen`` — the same Namespace stock ``save_model`` serializes,
    so every checkpoint carries the tokens-seen for the model at that point. Returns
    the live accountant so the caller (e.g. in-loop validation) can report totals.
    """
    seed = resume_totals or {}
    accountant = TokenAccountant(
        input_seed=seed.get("input", 0), target_seed=seed.get("target", 0)
    )

    def on_epoch_end(totals: dict[str, int]) -> None:
        if getattr(trainer, "args", None) is not None:
            trainer.args.tokens_seen = totals

    trainer.train_one_epoch = make_accounting_train_one_epoch(
        trainer.train_one_epoch, accountant, on_epoch_end
    )
    return accountant
