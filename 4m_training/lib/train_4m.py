"""Train 4M on THINGS with vision + MEG/EEG neural modalities.

Thin wrapper around 4M's ``run_training_4m.py``. Neural modalities and
dataloaders are registered/patched at import time — no ``external/ml-4m/`` edits.

Modes::

    python 4m_training/train_4m.py demo
    python 4m_training/train_4m.py dryrun --config 4m_training/configs/4m_things_data.yaml
    python 4m_training/train_4m.py train --config 4m_training/configs/4m_things_main.yaml
    python 4m_training/train_4m.py validate --config 4m_training/configs/4m_things_main.yaml
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

# PyTorch 2.6 changed weights_only default to True. 4M checkpoints contain
# argparse.Namespace, numpy scalars, and other types blocked by the new default.
# Checkpoints are local/trusted, so patch torch.load to keep the old behaviour.
_orig_torch_load = torch.load
def _torch_load_unsafe(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _torch_load_unsafe

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Paths come from repo_paths (the single source of truth) — never recompute the repo root
# from __file__, or moving this module (e.g. into lib/) silently breaks every relative path.
import repo_paths  # noqa: E402

_REPO_ROOT = repo_paths.REPO_ROOT

import fourm_neural_modalities  # noqa: F401,E402
from fourm_dataloader import patch_pretrain_utils, set_log_print_freq
from neural_constants import (
    EEG_TRIAL_SHAPE,
    MEG_TRIAL_SHAPE,
    THINGS_IMAGE_SIZE,
    THINGS_PATCH_SIZE,
    TOK_RGB_TOKENS_PER_IMAGE,
    pretoken_grid_num_tokens,
)

patch_pretrain_utils()


def _make_tar(path: Path, entries: list[tuple[str, str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for key, ext, data in entries:
            info = tarfile.TarInfo(name=f"{key}.{ext}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def run_demo() -> None:
    """Build a synthetic shard and walk samples through the 4M decode path."""
    import webdataset as wds
    from fourm.data.unified_datasets import (
        filter_metadata,
        multi_tarfile_samples,
        remove_extensions,
        tok_to_int64,
        wds_decoder,
    )
    from fourm.data.unified_datasets import map as keyless_map

    from fourm_neural_transforms import EegTokTransform, MegTokTransform

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "things"
        ids = ["000000001", "000000002", "000000003"]

        rgb_arr = lambda i: np.full((TOK_RGB_TOKENS_PER_IMAGE,), int(i), dtype=np.int16)
        depth_arr = np.zeros((TOK_RGB_TOKENS_PER_IMAGE,), dtype=np.int16)
        meg_real = np.full((4, *MEG_TRIAL_SHAPE), 7, dtype=np.int16)
        meg_sentinel = np.full((1, *MEG_TRIAL_SHAPE), -1, dtype=np.int16)
        eeg_real = np.full((2, *EEG_TRIAL_SHAPE), 3, dtype=np.int16)
        eeg_sentinel = np.full((1, *EEG_TRIAL_SHAPE), -1, dtype=np.int16)
        mask_real = np.array([1], dtype=np.uint8)
        mask_zero = np.array([0], dtype=np.uint8)

        for mod, arrs in [
            ("tok_rgb", [(i, _npy_bytes(rgb_arr(i))) for i in ids]),
            ("tok_depth", [(i, _npy_bytes(depth_arr)) for i in ids]),
            (
                "tok_meg",
                [
                    ("000000001", _npy_bytes(meg_real)),
                    ("000000002", _npy_bytes(meg_sentinel)),
                    ("000000003", _npy_bytes(meg_real)),
                ],
            ),
            (
                "tok_eeg",
                [
                    ("000000001", _npy_bytes(eeg_real)),
                    ("000000002", _npy_bytes(eeg_sentinel)),
                    ("000000003", _npy_bytes(eeg_real)),
                ],
            ),
            (
                "meg_mask",
                [
                    ("000000001", _npy_bytes(mask_real)),
                    ("000000002", _npy_bytes(mask_zero)),
                    ("000000003", _npy_bytes(mask_real)),
                ],
            ),
            (
                "eeg_mask",
                [
                    ("000000001", _npy_bytes(mask_real)),
                    ("000000002", _npy_bytes(mask_zero)),
                    ("000000003", _npy_bytes(mask_real)),
                ],
            ),
        ]:
            _make_tar(
                root / f"{mod}/shard_000.tar",
                [(k, "npy", b) for k, b in arrs],
            )

        url = (
            f"{root}/[tok_rgb,tok_depth,tok_meg,tok_eeg,meg_mask,eeg_mask]/"
            "shard_000.tar"
        )
        print(f"data_path = {url}\n")

        pipeline = wds.DataPipeline(
            wds.SimpleShardList(url),
            partial(multi_tarfile_samples, handler=lambda e: (_ for _ in ()).throw(e)),
            wds.decode(wds_decoder),
            wds.map(remove_extensions),
            keyless_map(filter_metadata),
            keyless_map(tok_to_int64),
        )

        print("=" * 72)
        print("  4M pipeline output (after decode + filter)")
        print("=" * 72)
        for sample in pipeline:
            shapes = {
                k: getattr(v, "shape", type(v).__name__) for k, v in sample.items()
            }
            print(f"  modalities = {sorted(sample)}")
            print(f"  shapes     = {shapes}")
            print()

        print("=" * 72)
        print("  After trial transforms")
        print("=" * 72)
        meg_tx = MegTokTransform(training=True, seed=0)
        eeg_tx = EegTokTransform(training=True, seed=0)
        for label, arr, tx in [
            ("MEG real", meg_real, meg_tx),
            ("MEG sentinel", meg_sentinel, meg_tx),
            ("EEG real", eeg_real, eeg_tx),
            ("EEG sentinel", eeg_sentinel, eeg_tx),
        ]:
            tokens = tx.preprocess(arr)
            print(f"  {label:>14s}: in={tuple(arr.shape)} -> out={tuple(tokens.shape)}")


def _load_configs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (main_cfg, train_ds_cfg, val_ds_cfg_or_empty). Accepts main or data yaml."""
    import yaml

    config_path = config_path.resolve()
    if not config_path.is_absolute():
        config_path = (_REPO_ROOT / config_path).resolve()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "train" in cfg and "datasets" in cfg.get("train", {}):
        main = {
            "input_size": THINGS_IMAGE_SIZE,
            "num_input_tokens": 256,
            "num_target_tokens": 256,
            "batch_size": 4,
            "epoch_size": 64,
            "text_tokenizer_path": str(repo_paths.TEXT_TOKENIZER),
        }
        train_ds = next(iter(cfg["train"]["datasets"].values()))
        val_ds = {}
        if "val" in cfg and cfg["val"].get("datasets"):
            val_ds = next(iter(cfg["val"]["datasets"].values()))
        return main, train_ds, val_ds

    data_rel = cfg.get("data_config")
    if not data_rel:
        raise ValueError(f"{config_path}: expected data_config or train.datasets")
    data_path = Path(data_rel)
    if not data_path.is_absolute():
        data_path = (_REPO_ROOT / data_path).resolve()
    with open(data_path) as f:
        data_cfg = yaml.safe_load(f)
    train_ds = next(iter(data_cfg["train"]["datasets"].values()))
    val_ds = {}
    if "val" in data_cfg and data_cfg["val"].get("datasets"):
        val_ds = next(iter(data_cfg["val"]["datasets"].values()))
    return cfg, train_ds, val_ds


def run_validate(config_path: Path) -> None:
    from config_validate import validate_config_file

    errors = validate_config_file(config_path, _REPO_ROOT)
    if errors:
        print(f"FAIL {config_path}:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print(f"OK {config_path}")


def _build_modality_info(
    all_domain_names: list[str],
    input_size: int = THINGS_IMAGE_SIZE,
) -> dict:
    import copy

    from fourm.data.modality_info import MODALITY_INFO

    # Model + masking only need in/out token domains, not presence flags.
    token_domains = [d for d in all_domain_names if not d.endswith("_mask")]
    modality_info = {
        mod: copy.deepcopy(MODALITY_INFO[mod])
        for mod in token_domains
        if mod in MODALITY_INFO
    }
    for mod, info in modality_info.items():
        if info.get("type") == "img":
            patch_size = info.get("patch_size", THINGS_PATCH_SIZE)
            info["max_tokens"] = pretoken_grid_num_tokens(input_size, patch_size)
    return modality_info


def _iterate_loader(loader, n_batches: int, label: str) -> None:
    print(f"{label}: iterating {n_batches} batches…\n")
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        print(f"batch {i}: keys = {sorted(batch.keys())}")
        for k, v in batch.items():
            if isinstance(v, dict) and "tensor" in v:
                print(f"  {k:>16s} : tensor shape={tuple(v['tensor'].shape)}")
            else:
                shape = getattr(v, "shape", None)
                print(
                    f"  {k:>16s} : shape={tuple(shape) if shape is not None else type(v).__name__}"
                )
        print()


# Stock mixture loader computes epoch_size // (num_gpus * num_workers * batch_size),
# so the dryrun cannot use 0 workers. One worker keeps it cheap and single-process.
_DRYRUN_NUM_WORKERS = 1


def run_dryrun(config_path: Path, n_batches: int = 4) -> None:
    """Build train + val dataloaders and iterate a few batches."""
    from tokenizers import Tokenizer

    from fourm.data.modality_info import MODALITY_INFO
    from fourm.data.pretrain_utils import get_train_dataloader, get_val_dataloader, setup_sampling_mod_info

    main_cfg, train_ds, val_ds = _load_configs(config_path)
    input_size = main_cfg.get("input_size", THINGS_IMAGE_SIZE)

    in_domains = sorted(train_ds["in_domains"].split("-"))
    out_domains = sorted(train_ds["out_domains"].split("-"))
    all_domains = sorted(set(in_domains) | set(out_domains))
    modality_info_full = _build_modality_info(all_domains, input_size)
    mod_info, sampling_weights = setup_sampling_mod_info(train_ds, modality_info_full)

    tok_path = main_cfg.get("text_tokenizer_path", str(repo_paths.TEXT_TOKENIZER))
    text_tokenizer = Tokenizer.from_file(tok_path)

    train_loader = get_train_dataloader(
        dataset_config=train_ds,
        modality_info=mod_info,
        sampling_weights=sampling_weights,
        text_tokenizer=text_tokenizer,
        input_size=input_size,
        num_input_tokens=main_cfg.get("num_input_tokens", 128),
        num_target_tokens=main_cfg.get("num_target_tokens", 128),
        min_input_tokens=main_cfg.get("min_input_tokens"),
        min_target_tokens=main_cfg.get("min_target_tokens"),
        num_tasks=1,
        num_workers=_DRYRUN_NUM_WORKERS,
        dataset_batch_size=main_cfg.get("batch_size", 4),
        epoch_size=main_cfg.get("epoch_size", max(n_batches * 4, 16)),
    )
    _iterate_loader(train_loader, n_batches, "train dryrun")

    if val_ds.get("data_path"):
        val_loader = get_val_dataloader(
            dataset_config=val_ds,
            dataset_name="things",
            train_configs={"things": train_ds},
            modality_info=mod_info,
            sampling_weights=sampling_weights,
            text_tokenizer=text_tokenizer,
            input_size=input_size,
            num_input_tokens=main_cfg.get("num_input_tokens", 128),
            num_target_tokens=main_cfg.get("num_target_tokens", 128),
            min_input_tokens=main_cfg.get("min_input_tokens"),
            min_target_tokens=main_cfg.get("min_target_tokens"),
            fixed_eval=False,
            fixed_eval_input_tokens=128,
            fixed_eval_target_tokens=128,
            dist_eval=False,
            num_tasks=1,
            num_workers=0,
            batch_size=main_cfg.get("batch_size", 4),
            pin_mem=False,
        )
        _iterate_loader(val_loader, n_batches, "val dryrun")


def _clean_extra_argv(extra_argv: list[str]) -> list[str]:
    """Drop the leading ``--`` separator argparse.REMAINDER keeps.

    ``train --config X -- --data_config Y`` captures ``['--', '--data_config', Y]``.
    The 4M trainer's argparse treats a forwarded ``--`` as a positional and errors
    ("unrecognized arguments"), so strip exactly the one leading separator.
    """
    if extra_argv and extra_argv[0] == "--":
        return extra_argv[1:]
    return extra_argv


def _load_trainer_module(trainer_path: Path):
    """Import the stock trainer as a module (not ``__main__``).

    Running it via ``runpy`` would execute start-to-finish in one shot, leaving
    no seam to patch. Importing it defines its functions while skipping the
    bottom ``if __name__ == '__main__'`` block, so we can reassign module globals
    (``train_one_epoch`` / ``evaluate``) before driving ``main(args)`` ourselves.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_training_4m", trainer_path)
    trainer = importlib.util.module_from_spec(spec)
    sys.modules["run_training_4m"] = trainer
    spec.loader.exec_module(trainer)
    return trainer


def _install_neural_shuffler(trainer) -> None:
    """Wrap train_one_epoch to register a forward-pre-hook that shuffles neural tokens.

    The hook fires only on training forwards. It shuffles tok_meg_* / tok_eeg_* tensor
    values ONLY within samples that have real neural data (identified by having at least
    one unmasked input or target token). CC12M placeholder samples have all tokens masked
    out (budget=0), so they are excluded from the permutation — without this restriction,
    THINGS samples would receive zero-valued placeholder tokens as targets, making
    prediction trivial and invalidating the ablation.

    All rvq levels use the same permutation so cross-codebook coherence within a single
    recording is preserved (just assigned to the wrong image). Masks are untouched:
    the masking direction (pixel→neural vs neural→pixel) stays tied to the original sample.

    This is the null ablation: if training loss still decreases, the model is learning
    statistical regularities in the token distribution, not image-neural correspondence.
    """
    _NEURAL_PREFIXES = ("tok_meg", "tok_eeg")

    def _shuffle_hook(module, inputs):
        if not getattr(module, "training", False):
            return
        if not inputs:
            return
        mod_dict = inputs[0]
        if not isinstance(mod_dict, dict):
            return
        neural_keys = [k for k in mod_dict if k.startswith(_NEURAL_PREFIXES)]
        if not neural_keys:
            return

        # Identify samples with real neural data: placeholder samples have all
        # tokens masked out (input_budget=0), so (~mask).any(dim=1) is False.
        first_entry = mod_dict[neural_keys[0]]
        input_mask  = first_entry.get("input_mask")
        target_mask = first_entry.get("target_mask")
        has_input  = (~input_mask).any(dim=1)  if input_mask  is not None else None
        has_target = (~target_mask).any(dim=1) if target_mask is not None else None
        if has_input is not None and has_target is not None:
            has_neural = has_input | has_target
        elif has_input is not None:
            has_neural = has_input
        elif has_target is not None:
            has_neural = has_target
        else:
            return
        neural_idx = has_neural.nonzero(as_tuple=True)[0]
        if len(neural_idx) <= 1:
            return

        # One permutation shared across all rvq levels.
        perm = torch.randperm(len(neural_idx), device=neural_idx.device)
        src  = neural_idx[perm]
        for key in neural_keys:
            t = mod_dict[key]["tensor"].clone()
            t[neural_idx] = mod_dict[key]["tensor"][src]
            mod_dict[key]["tensor"] = t

    state = {"registered": False}
    orig = trainer.train_one_epoch

    def _wrapped(*args, **kwargs):
        model = kwargs.get("model")
        if model is not None and not state["registered"]:
            model.register_forward_pre_hook(_shuffle_hook)
            state["registered"] = True
        return orig(*args, **kwargs)

    trainer.train_one_epoch = _wrapped


def _drive_trainer_main(trainer, *, shuffle_neural_tokens: bool = False) -> None:
    """Replicate the stock ``__main__`` block (run_training_4m.py:835-847), with an
    opt-in hook to also validate the named-task suite on the live weights."""
    import os

    args = trainer.get_args()

    # Make `args` the trainer module's global. Stock sets it only in its __main__ block;
    # under our import-and-drive path it would be undefined, so (a) our per-epoch wrapper
    # can stash tokens-seen onto the very Namespace stock save_model serializes, and
    # (b) train_one_epoch's bare `args` reference (frozen-model branch) resolves.
    trainer.args = args

    rlimit = trainer.resource.getrlimit(trainer.resource.RLIMIT_NOFILE)
    trainer.resource.setrlimit(trainer.resource.RLIMIT_NOFILE, (args.rlimit, rlimit[1]))
    trainer.utils.setup_run_name(args)
    trainer.utils.setup_s3_args(args)
    if args.output_dir:
        trainer.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Quiet the logs: stock 4M prints a progress line every 10 steps with no knob. Honor the
    # main YAML's optional `print_freq` (set on args via the trainer's set_defaults(**cfg)).
    set_log_print_freq(getattr(args, "print_freq", None))

    # Token accounting (always on): count the ACTUAL masked tokens the model trains on, so
    # placeholder MEG/EEG samples aren't falsely counted (the stock closed-form would —
    # see lib/token_accounting.py). Seed from the resumed checkpoint so the running total
    # stays cumulative across auto_resume. Installed before in-loop val so the per-epoch
    # tokens-seen is already merged into stats when the suite line prints.
    from token_accounting import install_token_accounting, read_tokens_seen, resolve_resume_ckpt

    resume_path = resolve_resume_ckpt(args)
    resume_totals = None
    if resume_path and os.path.exists(resume_path):
        import torch

        resume_totals = read_tokens_seen(
            torch.load(resume_path, map_location="cpu", weights_only=False)
        )
    if resume_totals:
        print(f"[tokens] resuming token count from {resume_path}: {resume_totals}\n")
    install_token_accounting(trainer, resume_totals=resume_totals)

    # Opt-in: shuffle neural tokens across the batch on every training forward.
    # Activated by passing --shuffle_neural_tokens to run_train; stripped from
    # sys.argv before the stock parser runs so it never sees the unknown flag.
    if shuffle_neural_tokens:
        print("[shuffle] Neural token shuffling ENABLED — null ablation mode\n")
        _install_neural_shuffler(trainer)

    # Opt-in: score the named-task suite on the live model every eval_freq epochs
    # (see lib/in_loop_val.py). Absent the `in_loop_val_tasks` YAML field, the
    # launch path is identical to the stock trainer.
    if getattr(args, "in_loop_val_tasks", None):
        from in_loop_val import build_suite_fn, install_in_loop_validation

        print(f"[in-loop val] enabled from {args.in_loop_val_tasks}\n")
        install_in_loop_validation(
            trainer, build_suite_fn(args),
            eval_freq=getattr(args, "eval_freq", 1),
            epochs=getattr(args, "epochs", None),
        )

    trainer.main(args)


def run_train(config_path: Path, extra_argv: list[str]) -> None:
    """Hand off to 4M's official trainer with neural patches active."""
    import os

    extra_argv = _clean_extra_argv(extra_argv)

    # Intercept our custom flags before the stock trainer's argparser sees them.
    # The stock parser rejects unknown arguments, so strip them here and handle
    # them ourselves in _drive_trainer_main.
    shuffle_neural_tokens = "--shuffle_neural_tokens" in extra_argv
    if shuffle_neural_tokens:
        extra_argv = [a for a in extra_argv if a != "--shuffle_neural_tokens"]

    # Stock run_training_4m.py always uses DDP; set single-process vars if missing.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    ml_4m_root = repo_paths.ML4M_DIR
    if not (ml_4m_root / "run_training_4m.py").exists():
        raise FileNotFoundError(
            f"4M trainer not found at {ml_4m_root}. "
            "Install with: pip install -e external/ml-4m"
        )
    sys.path.insert(0, str(ml_4m_root))
    config_path = config_path.resolve()
    if not config_path.is_absolute():
        config_path = (_REPO_ROOT / config_path).resolve()

    # Stock run_training_4m.py defaults --text_tokenizer_path to a cwd-relative path that
    # does not exist outside the 4M checkout (e.g. on Modal). Realize the configs'
    # "auto-derived from FOURM_ML4M_DIR" promise: inject the resolved tokenizer unless the
    # config or caller already provides one. (CLI args override the config's defaults.)
    import yaml

    with open(config_path) as f:
        _main_cfg = yaml.safe_load(f) or {}
    if "--text_tokenizer_path" not in extra_argv and not _main_cfg.get("text_tokenizer_path"):
        extra_argv = [*extra_argv, "--text_tokenizer_path", str(repo_paths.TEXT_TOKENIZER)]

    sys.argv = ["run_training_4m.py", "-c", str(config_path), *extra_argv]
    print(f"handing off to 4M trainer: argv = {sys.argv}\n")
    try:
        trainer = _load_trainer_module(ml_4m_root / "run_training_4m.py")
        _drive_trainer_main(trainer, shuffle_neural_tokens=shuffle_neural_tokens)
    finally:
        # Stock trainer inits the process group but never tears it down; doing it
        # here avoids the "destroy_process_group() was not called" NCCL warning.
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    sub.add_parser("demo", help="Synthetic shard pipeline walkthrough.")

    p_dry = sub.add_parser("dryrun", help="Iterate train + val batches from config.")
    p_dry.add_argument("--config", type=Path, required=True)
    p_dry.add_argument("--n-batches", type=int, default=4)

    p_train = sub.add_parser("train", help="Run 4M run_training_4m.py.")
    p_train.add_argument("--config", type=Path, required=True)
    p_train.add_argument("extra", nargs=argparse.REMAINDER)

    p_val = sub.add_parser("validate", help="Check main/data YAML against 4M (no GPU/data).")
    p_val.add_argument("--config", type=Path, required=True)

    args = p.parse_args()
    if args.mode == "demo":
        run_demo()
    elif args.mode == "dryrun":
        run_dryrun(args.config, args.n_batches)
    elif args.mode == "validate":
        run_validate(args.config)
    elif args.mode == "train":
        run_train(args.config, args.extra or [])


if __name__ == "__main__":
    main()
