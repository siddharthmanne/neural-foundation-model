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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Paths come from repo_paths (the single source of truth) — never recompute the repo root
# from __file__, or moving this module (e.g. into lib/) silently breaks every relative path.
import repo_paths  # noqa: E402

_REPO_ROOT = repo_paths.REPO_ROOT

import fourm_neural_modalities  # noqa: F401,E402
from fourm_dataloader import patch_pretrain_utils
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


def run_train(config_path: Path, extra_argv: list[str]) -> None:
    """Hand off to 4M's official trainer with neural patches active."""
    import os
    import runpy

    extra_argv = _clean_extra_argv(extra_argv)

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
        runpy.run_path(str(ml_4m_root / "run_training_4m.py"), run_name="__main__")
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
