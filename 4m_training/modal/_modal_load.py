"""Bridge between Modal scripts and modal_image.py.

Why this exists:
  - On your laptop, modal_image.py sits next to this file.
  - In the Modal container, the repo is mounted at /opt/repo/4m_training/.
  This loader tries both locations so every modal_*.py can share one image file.

You usually don't need to edit this — copy it verbatim into new training folders.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_modal_image() -> ModuleType:
    """Import modal_image.py and return it as a module."""
    # Reuse if already loaded (e.g. multiple modal scripts in one process).
    if "modal_image" in sys.modules:
        return sys.modules["modal_image"]

    # Try container path first, then local path next to this file.
    for path in (
        Path("/opt/repo/4m_training/modal/modal_image.py"),
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
