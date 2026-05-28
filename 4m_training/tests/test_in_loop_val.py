"""In-loop validation injection: run the named-task suite on the live model
during training, at the eval cadence, merging per-task loss into the trainer's
per-epoch log stats.

These tests exercise the *pure* wrapper logic (no fourm / no real model), which
decides cadence and stat-merging. The fourm-dependent suite (ValidationSuite) is
covered by test_validate_4m.py.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

from in_loop_val import install_in_loop_validation, make_validating_train_one_epoch


def test_suite_runs_at_eval_cadence_and_merges_into_train_stats():
    calls = {"train": 0, "suite": 0}

    def orig_train(**kwargs):
        calls["train"] += 1
        return {"loss": 1.0}

    def suite_fn(model, device):
        calls["suite"] += 1
        return {"[in-loop val: t] loss": 2.0}

    wrapped = make_validating_train_one_epoch(orig_train, suite_fn, eval_freq=2, epochs=4)
    s0 = wrapped(model="m", device="d", epoch=0)   # (0+1)%2 != 0 -> no suite
    s1 = wrapped(model="m", device="d", epoch=1)   # (1+1)%2 == 0 -> suite
    s2 = wrapped(model="m", device="d", epoch=2)   # no suite

    assert calls["train"] == 3
    assert calls["suite"] == 1                     # only at the cadence
    assert "[in-loop val: t] loss" not in s0
    assert s1["[in-loop val: t] loss"] == 2.0      # merged
    assert s1["loss"] == 1.0                        # original train stats preserved
    assert "[in-loop val: t] loss" not in s2


def test_last_epoch_always_validates():
    calls = {"suite": 0}

    def orig_train(**kwargs):
        return {}

    def suite_fn(model, device):
        calls["suite"] += 1
        return {"x": 1.0}

    wrapped = make_validating_train_one_epoch(orig_train, suite_fn, eval_freq=10, epochs=3)
    wrapped(model="m", device="d", epoch=0)        # (0+1)%10 != 0, not last
    wrapped(model="m", device="d", epoch=2)        # last epoch -> suite even off-cadence
    assert calls["suite"] == 1


def test_suite_exception_does_not_kill_training():
    """A val hiccup mid-training must not crash a long run — train stats survive."""

    def orig_train(**kwargs):
        return {"loss": 1.0}

    def bad_suite(model, device):
        raise RuntimeError("loader blew up")

    wrapped = make_validating_train_one_epoch(orig_train, bad_suite, eval_freq=1, epochs=1)
    out = wrapped(model="m", device="d", epoch=0)
    assert out == {"loss": 1.0}                     # train stats intact, no raise


def test_tokens_seen_reported_when_present_in_train_stats(capsys):
    """When token accounting (installed inside this wrapper) has merged tokens-seen
    into the epoch stats, the in-loop val line must report them next to the losses."""

    def orig_train(**kwargs):
        # Mimic the token-accounting wrapper having already merged its keys.
        return {
            "loss": 1.0,
            "[tokens] input_seen": 100,
            "[tokens] target_seen": 23,
            "[tokens] total_seen": 123,
        }

    def suite_fn(model, device):
        return {"[in-loop val: t] loss": 2.0}

    wrapped = make_validating_train_one_epoch(orig_train, suite_fn, eval_freq=1, epochs=1)
    out = wrapped(model="m", device="d", epoch=0)
    printed = capsys.readouterr().out
    assert "total=123" in printed and "input=100" in printed and "target=23" in printed
    # The merged token stats also survive on the returned dict for log.txt.
    assert out["[tokens] total_seen"] == 123


def test_no_tokens_line_when_accounting_absent(capsys):
    """Without token accounting (no [tokens] keys), the wrapper must not invent a line."""

    def orig_train(**kwargs):
        return {"loss": 1.0}

    def suite_fn(model, device):
        return {"[in-loop val: t] loss": 2.0}

    wrapped = make_validating_train_one_epoch(orig_train, suite_fn, eval_freq=1, epochs=1)
    wrapped(model="m", device="d", epoch=0)
    assert "tokens seen" not in capsys.readouterr().out


def test_install_wraps_train_one_epoch():
    calls = {"suite": 0}

    def t1(**kwargs):
        return {}

    def suite_fn(model, device):
        calls["suite"] += 1
        return {"[in-loop val: t] loss": 9.0}

    trainer = types.SimpleNamespace(train_one_epoch=t1, evaluate=lambda *a, **k: {})
    install_in_loop_validation(trainer, suite_fn, eval_freq=1, epochs=1)

    out = trainer.train_one_epoch(model="m", device="d", epoch=0)
    assert out["[in-loop val: t] loss"] == 9.0
    assert calls["suite"] == 1
