"""Placeholder-correct training-token accounting.

The stock 4M loop logs a *closed-form* tokens-seen (run_training_4m.py:643-645)
that assumes every sample contributes the full input/target budget. For THINGS
that overcounts: placeholder MEG/EEG samples get their Dirichlet budget zeroed
(neural_masking.zero_absent_budgets) and NOT redistributed, so they contribute
fewer tokens. We instead count the *actual* selected tokens from the per-modality
masks, which already encode the zeroing.

These tests exercise the pure accounting logic (torch only, no fourm / no model):
the hook's mask counting, the training gate, resume seeding, the train_one_epoch
wrapper's lazy hook registration + stat merging, and the checkpoint read/resolve
helpers. The end-to-end path on a live model is covered by the smoke/integration
suites.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

from token_accounting import (
    TokenAccountant,
    install_token_accounting,
    make_accounting_train_one_epoch,
    read_tokens_seen,
    resolve_resume_ckpt,
)


def _mod(*, training=True):
    """A stand-in for the model the hook is attached to (only `.training` matters)."""
    return types.SimpleNamespace(training=training)


def _selected(input_sel: int, target_sel: int, length: int = 8) -> dict:
    """A per-modality mask dict where `input_sel` cells are visible to the encoder
    and `target_sel` are decoder targets. Masks are bool with True == masked-out,
    so `~mask` marks selected tokens — matching 4M's image_mask convention."""
    input_mask = torch.ones(length, dtype=torch.bool)
    input_mask[:input_sel] = False
    target_mask = torch.ones(length, dtype=torch.bool)
    target_mask[input_sel : input_sel + target_sel] = False
    return {"tensor": torch.zeros(length), "input_mask": input_mask, "target_mask": target_mask}


# ── hook counting ────────────────────────────────────────────────────────────


def test_counts_actual_selected_tokens():
    acc = TokenAccountant()
    mod_dict = {"tok_rgb": _selected(3, 2), "tok_depth": _selected(4, 1)}
    acc.hook(_mod(), (mod_dict,))
    t = acc.totals()
    assert t["input"] == 3 + 4
    assert t["target"] == 2 + 1
    assert t["total"] == t["input"] + t["target"]


def test_placeholder_modality_contributes_zero():
    """A placeholder MEG/EEG modality has budget 0 -> 0 cells selected -> it must
    add nothing, with no special-casing in the accountant."""
    real_only = TokenAccountant()
    real_only.hook(_mod(), ({"tok_rgb": _selected(5, 5)},))

    with_placeholder = TokenAccountant()
    with_placeholder.hook(
        _mod(),
        ({"tok_rgb": _selected(5, 5), "tok_meg_rvq0": _selected(0, 0)},),
    )

    assert with_placeholder.totals() == real_only.totals()


def test_hook_gated_on_training():
    """Validation/eval forwards (model.eval()) must not be counted as tokens trained."""
    acc = TokenAccountant()
    acc.hook(_mod(training=False), ({"tok_rgb": _selected(5, 5)},))
    assert acc.totals() == {"input": 0, "target": 0, "total": 0}


def test_totals_accumulate_across_steps():
    acc = TokenAccountant()
    for _ in range(3):
        acc.hook(_mod(), ({"tok_rgb": _selected(2, 2)},))
    assert acc.totals()["input"] == 6
    assert acc.totals()["target"] == 6


def test_seed_offsets_totals_for_resume():
    """On auto_resume the count must continue from the checkpoint's stored totals."""
    acc = TokenAccountant(input_seed=100, target_seed=40)
    acc.hook(_mod(), ({"tok_rgb": _selected(3, 2)},))
    t = acc.totals()
    assert t["input"] == 103
    assert t["target"] == 42
    assert t["total"] == 145


def test_modality_without_target_mask_counts_input_only():
    acc = TokenAccountant()
    mod_dict = {"caption": {"input_mask": torch.tensor([False, False, True, True])}}
    t = acc.totals() if False else (acc.hook(_mod(), (mod_dict,)) or acc.totals())
    assert t["input"] == 2
    assert t["target"] == 0


# ── train_one_epoch wrapper ──────────────────────────────────────────────────


class _FakeModel:
    """Records forward-pre-hook registrations; the stub epoch fires them to mimic
    training forwards."""

    def __init__(self):
        self.training = True
        self._hooks = []

    def register_forward_pre_hook(self, fn):
        self._hooks.append(fn)

    def forward_once(self, mod_dict):
        for fn in self._hooks:
            fn(self, (mod_dict,))


def test_wrapper_registers_hook_once_and_merges_tokens():
    acc = TokenAccountant()
    model = _FakeModel()
    captured = {}

    def orig_train(**kwargs):
        # Mimic two training forwards within the epoch.
        kwargs["model"].forward_once({"tok_rgb": _selected(3, 2)})
        kwargs["model"].forward_once({"tok_rgb": _selected(1, 4)})
        return {"loss": 1.0}

    wrapped = make_accounting_train_one_epoch(
        orig_train, acc, on_epoch_end=lambda totals: captured.update(totals)
    )

    s0 = wrapped(model=model, device="cpu", epoch=0)
    assert len(model._hooks) == 1  # registered exactly once
    assert s0["loss"] == 1.0  # original stats preserved
    assert s0["[tokens] input_seen"] == 4
    assert s0["[tokens] target_seen"] == 6
    assert s0["[tokens] total_seen"] == 10
    assert captured == {"input": 4, "target": 6, "total": 10}

    # Second epoch: hook is NOT re-registered, totals keep accumulating.
    s1 = wrapped(model=model, device="cpu", epoch=1)
    assert len(model._hooks) == 1
    assert s1["[tokens] input_seen"] == 8


def test_wrapper_without_model_kwarg_is_noop_passthrough():
    acc = TokenAccountant()
    wrapped = make_accounting_train_one_epoch(
        lambda **k: {"loss": 2.0}, acc, on_epoch_end=lambda totals: None
    )
    out = wrapped(device="cpu", epoch=0)
    assert out["loss"] == 2.0
    assert out["[tokens] total_seen"] == 0


def test_install_wraps_train_one_epoch_and_stashes_on_args():
    args = types.SimpleNamespace(output_dir="", resume="", auto_resume=False)
    trainer = types.SimpleNamespace(
        train_one_epoch=lambda **k: {}, args=args
    )
    acc = install_token_accounting(trainer)
    model = _FakeModel()

    def orig_after_install(**kwargs):
        kwargs["model"].forward_once({"tok_rgb": _selected(7, 3, length=12)})
        return {}

    # install_token_accounting wrapped the lambda; rewrap to drive a forward.
    trainer.train_one_epoch = make_accounting_train_one_epoch(
        orig_after_install, acc, on_epoch_end=lambda t: setattr(args, "tokens_seen", t)
    )
    trainer.train_one_epoch(model=model, device="cpu", epoch=0)
    assert args.tokens_seen == {"input": 7, "target": 3, "total": 10}


# ── checkpoint read / resume resolve ─────────────────────────────────────────


def test_read_tokens_seen_present_and_absent():
    ck_with = {"args": types.SimpleNamespace(tokens_seen={"input": 5, "target": 3, "total": 8})}
    assert read_tokens_seen(ck_with) == {"input": 5, "target": 3, "total": 8}

    assert read_tokens_seen({"args": types.SimpleNamespace()}) is None  # no attr
    assert read_tokens_seen({"model": {}}) is None  # no args
    assert read_tokens_seen(None) is None


def test_resolve_resume_ckpt_prefers_explicit_resume(tmp_path):
    explicit = tmp_path / "checkpoint-3.pth"
    explicit.write_bytes(b"x")
    args = types.SimpleNamespace(resume=str(explicit), auto_resume=True, output_dir=str(tmp_path))
    assert resolve_resume_ckpt(args) == str(explicit)


def test_resolve_resume_ckpt_auto_picks_max_epoch(tmp_path):
    for ep in (1, 5, 10, 2):
        (tmp_path / f"checkpoint-{ep}.pth").write_bytes(b"x")
    (tmp_path / "checkpoint-final.pth").write_bytes(b"x")  # non-numeric, ignored
    args = types.SimpleNamespace(resume="", auto_resume=True, output_dir=str(tmp_path))
    assert resolve_resume_ckpt(args) == str(tmp_path / "checkpoint-10.pth")


def test_resolve_resume_ckpt_none_when_fresh(tmp_path):
    args = types.SimpleNamespace(resume="", auto_resume=False, output_dir=str(tmp_path))
    assert resolve_resume_ckpt(args) is None
    args2 = types.SimpleNamespace(resume="", auto_resume=True, output_dir=str(tmp_path))
    assert resolve_resume_ckpt(args2) is None  # empty dir
