"""MEG tokenizer subpackage.

Public surface — one Tokenizer-protocol class per phase, all share the same
data / splits / config utilities.
"""

from .meg_config import (
    BRAINOMNI_DEFAULT,
    BRAINOMNI_EVAL_DEFAULTS,
    EVAL_DEFAULTS,
    LEARNABLE_SPLIT_DEFAULTS,
    MEG_BANDS,
    MEG_DATA,
    MU_SPLIT_DEFAULTS,
    MU_TRANSFORM_DEFAULT,
    BrainOmniConfig,
    EvalDefaults,
    MEGDataSpec,
    MuTransformConfig,
    SplitDefaults,
)
from .brainomni import BrainOmniTokenizer, load_braintokenizer
from .mu_transform import MuCalibration, MuTransformTokenizer, fit_calibration
from .splits import Splits, split_by_image

__all__ = [
    "BRAINOMNI_DEFAULT",
    "BRAINOMNI_EVAL_DEFAULTS",
    "BrainOmniConfig",
    "BrainOmniTokenizer",
    "EVAL_DEFAULTS",
    "EvalDefaults",
    "LEARNABLE_SPLIT_DEFAULTS",
    "MEG_BANDS",
    "MEG_DATA",
    "MEGDataSpec",
    "MU_SPLIT_DEFAULTS",
    "MU_TRANSFORM_DEFAULT",
    "MuCalibration",
    "MuTransformConfig",
    "MuTransformTokenizer",
    "SplitDefaults",
    "Splits",
    "fit_calibration",
    "load_braintokenizer",
    "split_by_image",
]
