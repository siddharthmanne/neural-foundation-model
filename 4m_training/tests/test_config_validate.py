"""YAML + 4M registry validation for all shipped training configs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from repo_paths import REPO_ROOT as _REPO, TRAINING_DIR

sys.path.insert(0, str(TRAINING_DIR))

import fourm_neural_modalities  # noqa: F401
from config_validate import (
    expand_shard_urls,
    validate_config_file,
    validate_data_config,
    validate_main_config,
    load_yaml,
)

CONFIG_DIR = TRAINING_DIR / "configs"

MAIN_DATA_PAIRS = [
    ("4m_things_main.yaml", "4m_things_data.yaml"),
    ("4m_smoke_things_main.yaml", "4m_smoke_things_data.yaml"),
    ("4m_smoke_cc12m_main.yaml", "4m_smoke_cc12m_data.yaml"),
]

DATA_ONLY = [
    "4m_smoke_things_neural_in_data.yaml",
    "4m_things_data.yaml",
    "4m_smoke_things_data.yaml",
    "4m_smoke_cc12m_data.yaml",
]


@pytest.mark.parametrize("main_name,data_name", MAIN_DATA_PAIRS)
def test_main_and_data_yaml_pair(main_name: str, data_name: str) -> None:
    main_path = CONFIG_DIR / main_name
    data_path = CONFIG_DIR / data_name
    assert not validate_main_config(load_yaml(main_path), _REPO), main_name
    assert not validate_data_config(load_yaml(data_path), _REPO), data_name


@pytest.mark.parametrize("data_name", DATA_ONLY)
def test_data_yaml_alone(data_name: str) -> None:
    errors = validate_config_file(CONFIG_DIR / data_name, _REPO)
    assert errors == [], f"{data_name}: {errors}"


def test_things_braceexpand_produces_real_shard_names() -> None:
    cfg = load_yaml(CONFIG_DIR / "4m_smoke_things_data.yaml")
    path = cfg["train"]["datasets"]["things"]["data_path"]
    urls = expand_shard_urls(path)
    assert len(urls) == 1
    assert "shard_000.tar" in urls[0]
    assert "{" not in urls[0]


def test_things_full_range_expands_27_shards() -> None:
    cfg = load_yaml(CONFIG_DIR / "4m_things_data.yaml")
    path = cfg["train"]["datasets"]["things"]["data_path"]
    urls = expand_shard_urls(path)
    assert len(urls) == 27
    assert all("shard_0" in u and ".tar" in u for u in urls)


def test_invalid_brace_pattern_detected() -> None:
    cfg = load_yaml(CONFIG_DIR / "4m_smoke_things_data.yaml")
    cfg["train"]["datasets"]["things"]["data_path"] = (
        "/project/data/train/things/[tok_rgb]/shard_{000}.tar"
    )
    errors = validate_data_config(cfg, _REPO)
    assert any("unexpanded braces" in e for e in errors)


def test_meg_mask_in_domains_rejected() -> None:
    cfg = load_yaml(CONFIG_DIR / "4m_smoke_things_data.yaml")
    ds = cfg["train"]["datasets"]["things"]
    ds["in_domains"] = "tok_rgb-meg_mask"
    ds["out_domains"] = "tok_rgb"
    errors = validate_data_config(cfg, _REPO)
    assert any("meg_mask" in e and "presence" in e for e in errors)


def test_neural_in_smoke_validates() -> None:
    cfg = load_yaml(CONFIG_DIR / "4m_smoke_things_neural_in_data.yaml")
    assert validate_data_config(cfg, _REPO) == []


def test_neural_modality_in_out_domains_rejected() -> None:
    cfg = load_yaml(CONFIG_DIR / "4m_smoke_things_data.yaml")
    ds = cfg["train"]["datasets"]["things"]
    ds["out_domains"] = "tok_rgb-tok_meg"  # tok_meg is input-only
    ds["target_alphas"] = "1.0-1.0"
    errors = validate_data_config(cfg, _REPO)
    assert any("tok_meg" in e and "input-only" in e for e in errors)


def test_cc12m_smoke_alphas_match_in_out_domain_counts() -> None:
    cfg = load_yaml(CONFIG_DIR / "4m_smoke_cc12m_data.yaml")
    ds = cfg["train"]["datasets"]["cc12m"]
    n_in = len(ds["in_domains"].split("-"))
    n_out = len(ds["out_domains"].split("-"))
    assert len(ds["input_alphas"].split("-")) == n_in
    assert len(ds["target_alphas"].split("-")) == n_out


def test_tok_rgb_registered_after_neural_import() -> None:
    from fourm.data.modality_info import MODALITY_INFO

    for dom in ("tok_rgb", "tok_depth", "tok_meg", "tok_eeg"):
        assert dom in MODALITY_INFO, dom
