"""The one file to edit if your paths differ from the defaults.

Everything (training, validation, the Modal wrappers, the tokenizer) reads its
paths from here, so editing the two constants below is permanent across every
terminal and every `modal run` — no environment variables, no shell setup.
"""

from __future__ import annotations

import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
#  EDIT HERE — the only two settings most users ever change
# ═══════════════════════════════════════════════════════════════════════════

# 1. Where your 4M checkout lives. Leave as None to use the bundled
#    external/ml-4m submodule; otherwise give an absolute path, e.g.
#    ML4M_DIR_OVERRIDE = "/Users/you/code/ml-4m"
ML4M_DIR_OVERRIDE: str | None = None

# 2. The name of your Modal data + checkpoint volume (mounted at /project).
PROJECT_VOLUME_NAME: str = "project"

# ═══════════════════════════════════════════════════════════════════════════
#  Derived paths — you should not need to touch anything below.
# ═══════════════════════════════════════════════════════════════════════════

# This file is at <repo>/4m_training/lib/repo_paths.py
LIB_DIR = Path(__file__).resolve().parent      # 4m_training/lib (library modules)
TRAINING_DIR = LIB_DIR.parent                  # 4m_training
REPO_ROOT = TRAINING_DIR.parent                # repo root


def _resolve(p: str) -> Path:
    q = Path(p)
    return (q if q.is_absolute() else (REPO_ROOT / q)).resolve()


# Resolution order for the 4M checkout:
#   1) FOURM_ML4M_DIR env var  — used internally by the Modal wrappers to point at
#      the in-container mount; also a power-user override.
#   2) ML4M_DIR_OVERRIDE above — the normal way a user relocates 4M.
#   3) the bundled external/ml-4m submodule.
_env = os.environ.get("FOURM_ML4M_DIR")
if _env:
    ML4M_DIR = _resolve(_env)
elif ML4M_DIR_OVERRIDE:
    ML4M_DIR = _resolve(ML4M_DIR_OVERRIDE)
else:
    ML4M_DIR = (REPO_ROOT / "external" / "ml-4m").resolve()

# The 4M text tokenizer ships inside the 4M checkout.
TEXT_TOKENIZER = ML4M_DIR / "fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json"
