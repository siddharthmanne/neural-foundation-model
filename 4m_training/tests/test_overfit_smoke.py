"""The minimum bar for a DL system: it can overfit one batch.

Runs a short overfit (CPU, tiny model, one fixed batch with all modalities present)
and asserts every predicted modality's loss drops. This is the regression guard for
"the forward/backward path actually learns" — and, because neural is SYMMETRIC, it is
the local proof that the MEG (4 RVQ) and EEG decoding heads are wired and receiving
gradient: their losses must descend too. Slower than the rest of the suite (~40s).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural_constants import EEG_MODALITY, MEG_RVQ_MODALITIES
from overfit_smoke import _OUT_DOMAINS, assert_decreasing, run_overfit


def test_overfit_one_batch_descends():
    hist = run_overfit(steps=60, lr=2e-3, log_every=999, device="cpu")
    # Every predicted modality must drop well below its init (~ln(vocab)).
    assert_decreasing(hist, min_drop=1.0)
    for mod in _OUT_DOMAINS:
        assert hist[mod][0] > hist[mod][-1]


def test_all_neural_heads_descend():
    """Explicit: each of the 4 MEG RVQ heads and the EEG head learns on one batch."""
    hist = run_overfit(steps=60, lr=2e-3, log_every=999, device="cpu")
    for mod in (*MEG_RVQ_MODALITIES, EEG_MODALITY):
        assert mod in hist, mod
        assert min(hist[mod][-3:]) < min(hist[mod][:3]) - 1.0, (
            f"{mod}: neural head loss did not descend "
            f"({min(hist[mod][:3]):.3f} -> {min(hist[mod][-3:]):.3f})"
        )


def test_starts_near_ln_vocab():
    """At init, cross-entropy ≈ ln(vocab): a wiring sanity check."""
    import math

    from neural_constants import TOK_DEPTH_VOCAB_SIZE, TOK_RGB_VOCAB_SIZE

    hist = run_overfit(steps=1, lr=2e-3, log_every=999, device="cpu")
    assert abs(hist["tok_rgb"][0] - math.log(TOK_RGB_VOCAB_SIZE)) < 1.5
    assert abs(hist["tok_depth"][0] - math.log(TOK_DEPTH_VOCAB_SIZE)) < 1.5
