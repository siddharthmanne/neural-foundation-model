"""Run the named validation-task suite on the *live* model during training.

Stock 4M's in-loop eval can only validate the training task (its val loaders
derive masking from the train dataset of the same name), and it only runs at all
when the data config has a ``val:`` section. To also score the named tasks in
``configs/4m_things_val_tasks.yaml`` every ``eval_freq`` epochs — independently
of any ``val:`` section — we wrap a single stock-trainer global:

  * ``train_one_epoch`` → after each epoch, at the eval cadence, run the suite on
                          the just-updated model and merge its per-task losses
                          into the returned train stats (which the trainer writes
                          to ``log.txt`` / wandb).

Hooking the per-epoch train step (always called) rather than ``evaluate`` (only
called with a ``val:`` section) keeps in-loop validation self-contained: it needs
only the ``in_loop_val_tasks`` field. The wrapper factory below is pure (no
fourm) and unit-tested; the fourm-dependent ``build_suite_fn`` reuses
``validate_4m.ValidationSuite`` wholesale — it does not re-implement validation.

Installed from ``train_4m.run_train``, which imports the stock trainer (not
``runpy`` as ``__main__``) so the global can be reassigned before ``main(args)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from token_accounting import TOKEN_STAT_KEYS, format_tokens_seen

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent


def make_validating_train_one_epoch(
    orig_train_one_epoch: Callable,
    suite_fn: Callable[[Any, Any], dict],
    eval_freq: int,
    epochs: int | None,
) -> Callable:
    """Wrap ``train_one_epoch`` to run the suite at the eval cadence.

    After the wrapped epoch, when ``(epoch+1) % eval_freq == 0`` (or it's the
    final epoch), ``suite_fn(model, device)`` runs on the just-updated live model
    and its stats are merged into the returned train stats. Suite errors are
    swallowed with a warning so a val hiccup never kills a long training run.
    Stock calls this as ``train_one_epoch(model=..., device=..., epoch=..., ...)``.
    """

    def wrapped(*args: Any, **kwargs: Any):
        stats = orig_train_one_epoch(*args, **kwargs)
        epoch = kwargs.get("epoch")
        if epoch is None:
            return stats
        at_cadence = (epoch + 1) % eval_freq == 0 or (epochs and epoch + 1 == epochs)
        if at_cadence:
            try:
                suite_stats = suite_fn(kwargs.get("model"), kwargs.get("device"))
                stats = {**stats, **suite_stats}
                summary = "  ".join(
                    f"{k}={v:.4f}" for k, v in suite_stats.items() if k.endswith("] loss")
                )
                if summary:
                    print(f"[in-loop val] epoch {epoch}: {summary}", flush=True)
                # Tokens-seen for the checkpoint being scored. The token-accounting wrapper
                # (installed inside this one) already merged these into `stats`; report them
                # alongside the val losses so each validation moment is labeled with the
                # exact training tokens behind it.
                tokens = {k: stats[v] for k, v in TOKEN_STAT_KEYS.items() if v in stats}
                if tokens:
                    print(f"[in-loop val] epoch {epoch}: {format_tokens_seen(tokens)}", flush=True)
            except Exception as exc:  # noqa: BLE001 — never crash training on val
                print(f"[in-loop val] suite failed at epoch {epoch}: {exc!r}")
        return stats

    return wrapped


def install_in_loop_validation(
    trainer, suite_fn: Callable[[Any, Any], dict], *, eval_freq: int = 1, epochs: int | None = None
) -> None:
    """Reassign the trainer's ``train_one_epoch`` global to the validating wrapper.
    ``main()`` looks it up as a module global at call time, so the patch takes
    effect for the run."""
    trainer.train_one_epoch = make_validating_train_one_epoch(
        trainer.train_one_epoch, suite_fn, eval_freq, epochs
    )


def _dtype_for(args, device):
    """Match the training precision (fp32 on CPU)."""
    import torch

    if device.type == "cpu":
        return torch.float32
    return {
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
    }.get(getattr(args, "dtype", "float32"), torch.float32)


def build_suite_fn(args) -> Callable[[Any, Any], dict]:
    """Build a ``suite(model, device) -> {log_key: value}`` closure from ``args``.

    Thin adapter over ``validate_4m.ValidationSuite`` (the single home for running
    the task suite on a model): it constructs the suite from ``args`` and flattens
    the per-task stats into prefixed keys for the trainer's log. Loaders are built
    once inside the suite and re-iterated each eval epoch, so numbers are
    comparable across epochs and match the standalone validate_4m.py run.
    """
    import sys

    import yaml
    from tokenizers import Tokenizer

    from neural_constants import THINGS_IMAGE_SIZE
    from repo_paths import TEXT_TOKENIZER

    # validate_4m.py lives in 4m_training/ (parent of this lib/); the training entry
    # point only puts lib/ on sys.path, so add it before importing the suite.
    _training_dir = str(_HERE.parent)
    if _training_dir not in sys.path:
        sys.path.insert(0, _training_dir)
    from validate_4m import ValidationSuite

    tasks_path = Path(args.in_loop_val_tasks)
    if not tasks_path.is_absolute():
        tasks_path = (_REPO_ROOT / tasks_path).resolve()
    tasks_cfg = yaml.safe_load(tasks_path.read_text())

    tok_path = getattr(args, "text_tokenizer_path", None)
    tok = Tokenizer.from_file(
        str((_REPO_ROOT / tok_path).resolve() if tok_path else TEXT_TOKENIZER)
    )

    select = getattr(args, "in_loop_val_select", None)
    suite = ValidationSuite(
        tasks_cfg,
        input_size=getattr(args, "input_size", THINGS_IMAGE_SIZE),
        text_tokenizer=tok,
        loss_type=getattr(args, "loss_type", "mod"),
        batch_size=int(getattr(args, "batch_size", 4)),
        n_batches=int(getattr(args, "in_loop_val_n_batches", 4) or 0),
        select=select.split(",") if select else None,
    )

    def suite_fn(model, device) -> dict:
        results = suite.run(model, device, _dtype_for(args, device))
        return {
            f"[in-loop val: {name}] {key}": value
            for name, stats in results.items()
            for key, value in stats.items()
            if key != "n_batches"
        }

    return suite_fn
