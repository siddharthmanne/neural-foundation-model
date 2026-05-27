"""Loaded from ``/opt/repo/4m_training/_modal_load.py`` inside Modal containers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_modal_image() -> ModuleType:
    if "modal_image" in sys.modules:
        return sys.modules["modal_image"]
    for path in (
        Path("/opt/repo/4m_training/modal_image.py"),
        Path(__file__).resolve().parent / "modal_image.py",
    ):
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("modal_image", path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules["modal_image"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError(
        "modal_image.py not found under /opt/repo/4m_training (repo mount missing?)"
    )
