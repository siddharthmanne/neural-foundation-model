"""Validate 4M main + data YAML configs against stock 4M and our neural extensions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import braceexpand
import yaml

from repo_paths import REPO_ROOT as _REPO_ROOT

_PRESENCE_FLAGS = frozenset({"meg_mask", "eeg_mask"})
# ``tok_meg`` is an on-disk FOLDER, not a modality — the neural modalities are the four
# ``tok_meg_rvq0..3`` (which read that folder) and ``tok_eeg``. Map the folder name to a
# helpful hint when it is mistakenly used as a domain. See notes/4m_neural_modality_design.md.
_FOLDER_NOT_MODALITY = {
    "tok_meg": "use tok_meg_rvq0..tok_meg_rvq3 (they read the tok_meg folder)",
}
_TRAIN_TYPES = frozenset({"multimodal"})
_LOSS_TYPES = frozenset({"mod", "token"})


def _resolve(repo_root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p).resolve()


def _modalities_in_bracket(data_path: str) -> list[str]:
    from fourm.data.unified_datasets import extract_modality_names

    m = re.search(r"\[([^\]]+)\]", data_path)
    if not m:
        return []
    return extract_modality_names(f"{{{m.group(1)}}}")


def expand_shard_urls(data_path: str) -> list[str]:
    """Brace-expand a 4M ``data_path``; raises if pattern is invalid."""
    return list(braceexpand.braceexpand(data_path))


def _parse_alphas(value: Any, n_domains: int, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string or null, got {type(value)}")
    parts = value.split("-")
    if len(parts) not in (1, n_domains):
        raise ValueError(
            f"{label} has {len(parts)} value(s) but expected 1 or {n_domains} "
            f"(domains hyphen-separated)"
        )
    for p in parts:
        float(p)


def validate_dataset_config(
    name: str,
    cfg: dict[str, Any],
    repo_root: Path,
    modality_info: dict,
) -> list[str]:
    """Return list of error strings (empty = ok)."""
    errors: list[str] = []

    if cfg.get("type") not in _TRAIN_TYPES:
        errors.append(f"{name}: type must be 'multimodal', got {cfg.get('type')!r}")

    data_path = cfg.get("data_path", "")
    if not data_path:
        errors.append(f"{name}: missing data_path")
        return errors

    if cfg.get("use_wds", True):
        try:
            urls = expand_shard_urls(data_path)
        except Exception as e:
            errors.append(f"{name}: data_path braceexpand failed: {e}")
            return errors

        if any("{" in u or "}" in u for u in urls):
            errors.append(
                f"{name}: data_path has unexpanded braces (use ranges like "
                f"shard_{{000..026}}.tar not shard_{{000}}.tar): {data_path}"
            )
        if not urls:
            errors.append(f"{name}: data_path expanded to zero shard URLs")

    in_domains = sorted(cfg.get("in_domains", "").split("-"))
    out_domains = sorted(cfg.get("out_domains", "").split("-"))
    if not in_domains or not out_domains:
        errors.append(f"{name}: in_domains and out_domains are required")
        return errors

    for dom in in_domains + out_domains:
        if dom in _PRESENCE_FLAGS:
            errors.append(
                f"{name}: {dom} is a presence flag — keep it in data_path "
                f"brackets only, not in_domains/out_domains"
            )
        elif dom in _FOLDER_NOT_MODALITY:
            errors.append(
                f"{name}: {dom!r} is an on-disk folder, not a modality — "
                f"{_FOLDER_NOT_MODALITY[dom]}. See notes/4m_neural_modality_design.md"
            )
        elif dom not in modality_info:
            errors.append(f"{name}: unknown modality {dom!r} (not in MODALITY_INFO)")

    # Neural modalities are symmetric (encoder + decoder embedding), so they may appear in
    # in_domains, out_domains, or both. Any out_domain must have a decoder embedding.
    for dom in out_domains:
        info = modality_info.get(dom)
        if info is not None and info.get("decoder_embedding", "missing") is None:
            errors.append(
                f"{name}: {dom} has no decoder embedding — it cannot be an out_domain"
            )

    bracket_mods = _modalities_in_bracket(data_path)
    train_domains = set(in_domains) | set(out_domains)
    for dom in train_domains:
        if dom in _PRESENCE_FLAGS:
            continue
        # A domain is satisfied if the bracket lists either its own name (stock case,
        # e.g. rgb@224) or its source ``path`` (the four tok_meg_rvq* modalities read the
        # tok_meg folder).
        folder = modality_info.get(dom, {}).get("path", dom) if dom in modality_info else dom
        if dom not in bracket_mods and folder not in bracket_mods:
            suffix = f" (reads folder {folder!r})" if folder != dom else ""
            errors.append(
                f"{name}: {dom}{suffix} in in/out_domains but its folder is not in "
                f"data_path bracket list {bracket_mods}"
            )

    main_aug = cfg.get("main_augment_domain")
    if main_aug and main_aug not in modality_info:
        errors.append(f"{name}: main_augment_domain {main_aug!r} not in MODALITY_INFO")

    try:
        _parse_alphas(cfg.get("input_alphas"), len(in_domains), f"{name}.input_alphas")
        _parse_alphas(cfg.get("target_alphas"), len(out_domains), f"{name}.target_alphas")
    except ValueError as e:
        errors.append(str(e))

    alphas_cfg = cfg.get("alphas_config")
    if alphas_cfg:
        if not _resolve(repo_root, alphas_cfg).is_file():
            errors.append(f"{name}: alphas_config not found: {alphas_cfg}")

    return errors


def validate_data_config(
    data_cfg: dict[str, Any],
    repo_root: Path | None = None,
) -> list[str]:
    repo_root = repo_root or _REPO_ROOT
    import fourm_neural_modalities  # noqa: F401 — register aliases + neural mods

    from fourm.data.modality_info import MODALITY_INFO

    errors: list[str] = []
    train = data_cfg.get("train", {})
    datasets = train.get("datasets", {})
    if not datasets:
        return ["train.datasets is empty"]

    weights = train.get("weights", [1.0])
    if len(weights) != len(datasets):
        errors.append(
            f"train.weights length {len(weights)} != "
            f"number of datasets {len(datasets)}"
        )

    for name, ds in datasets.items():
        errors.extend(validate_dataset_config(name, ds, repo_root, MODALITY_INFO))

    return errors


def _validate_lr_schedule(main_cfg: dict[str, Any]) -> list[str]:
    """Catch LR-schedule footguns before a GPU run (defaults match the stock trainer).

    All keys are optional; the checks only fire if a value would make the trainer abort or
    crash. The stock scheduler choices are 'cosine' and 'inverse_sqrt-<N>'; YAML bypasses
    argparse's ``choices`` (it goes through ``set_defaults``), so we re-check it here.
    """
    errors: list[str] = []

    scheduler = main_cfg.get("scheduler", "cosine")
    if scheduler != "cosine" and not str(scheduler).startswith("inverse_sqrt-"):
        errors.append(
            f"scheduler must be 'cosine' or 'inverse_sqrt-<N>' (e.g. inverse_sqrt-10000), "
            f"got {scheduler!r}"
        )

    # Warmup length comes from epochs OR steps OR tokens; the trainer aborts if all negative.
    warmups = (
        main_cfg.get("warmup_epochs", 10),
        main_cfg.get("warmup_steps", -1),
        main_cfg.get("warmup_tokens", -1),
    )
    if all(w is not None and int(w) < 0 for w in warmups):
        errors.append(
            "set warmup_epochs >= 0 (use 0 to disable warmup) or warmup_steps/warmup_tokens "
            ">= 0 — all-negative makes the trainer abort"
        )

    # Cooldown only matters for inverse_sqrt, but all-negative cooldown_* trips a stock-4M
    # AttributeError (it reads a nonexistent args.lr_schedule), so guard it regardless.
    cd_epochs = main_cfg.get("cooldown_epochs", 10)
    cd_steps = main_cfg.get("cooldown_steps", -1)
    if cd_epochs is not None and cd_steps is not None and int(cd_epochs) < 0 and int(cd_steps) < 0:
        errors.append("keep cooldown_epochs >= 0 or set cooldown_steps >= 0 (all-negative crashes stock 4M)")

    return errors


def validate_main_config(
    main_cfg: dict[str, Any],
    repo_root: Path | None = None,
) -> list[str]:
    repo_root = repo_root or _REPO_ROOT
    errors: list[str] = []

    from fourm.models import fm

    model = main_cfg.get("model", "")
    if model not in fm.__all__:
        errors.append(f"unknown model {model!r}; choose from fourm.models.fm.__all__")

    if main_cfg.get("loss_type") not in _LOSS_TYPES:
        errors.append(f"loss_type must be one of {_LOSS_TYPES}")

    tok_path = main_cfg.get("text_tokenizer_path", "")
    if tok_path and not _resolve(repo_root, tok_path).is_file():
        errors.append(f"text_tokenizer_path not found: {tok_path}")

    num_workers = main_cfg.get("num_workers", 10)
    if num_workers is not None and int(num_workers) < 1:
        errors.append("num_workers must be >= 1 (stock mixture loader divides by it)")

    batch_size = main_cfg.get("batch_size", 0)
    epoch_size = main_cfg.get("epoch_size", 0)
    if batch_size and epoch_size:
        steps = int(epoch_size) // (int(batch_size) * 1)
        if steps < 1:
            errors.append(
                f"epoch_size ({epoch_size}) // batch_size ({batch_size}) must be >= 1"
            )

    errors.extend(_validate_lr_schedule(main_cfg))

    data_config = main_cfg.get("data_config", "")
    if not data_config:
        errors.append("main config missing data_config")
        return errors

    data_path = _resolve(repo_root, data_config)
    if not data_path.is_file():
        errors.append(f"data_config not found: {data_config}")
        return errors

    with open(data_path) as f:
        data_cfg = yaml.safe_load(f)
    errors.extend(validate_data_config(data_cfg, repo_root))
    return errors


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def validate_config_file(
    config_path: Path | str,
    repo_root: Path | None = None,
) -> list[str]:
    """Validate a main yaml (has data_config) or a standalone data yaml."""
    repo_root = repo_root or _REPO_ROOT
    path = _resolve(repo_root, str(config_path))
    cfg = load_yaml(path)
    if "train" in cfg and "datasets" in cfg.get("train", {}):
        return validate_data_config(cfg, repo_root)
    return validate_main_config(cfg, repo_root)
