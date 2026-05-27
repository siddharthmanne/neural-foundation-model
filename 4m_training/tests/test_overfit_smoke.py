"""The minimum bar for a DL system: it can overfit one batch.

Runs a short overfit (CPU, tiny model, one fixed batch with all modalities present)
and asserts every predicted modality's loss drops. This is the regression guard
for "the forward/backward path actually learns" — including that MEG/EEG flow
through the encoder without error. Slower than the rest of the suite (~30s).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from overfit_smoke import _OUT_DOMAINS, assert_decreasing, run_overfit


def test_overfit_one_batch_descends():
    hist = run_overfit(steps=35, lr=2e-3, log_every=999, device="cpu")
    # Each predicted (vision) modality must drop well below its init (~ln(vocab)).
    assert_decreasing(hist, min_drop=1.5)
    for mod in _OUT_DOMAINS:
        assert hist[mod][0] > hist[mod][-1]


def test_starts_near_ln_vocab():
    """At init, cross-entropy ≈ ln(vocab): a wiring sanity check."""
    import math

    from neural_constants import TOK_DEPTH_VOCAB_SIZE, TOK_RGB_VOCAB_SIZE

    hist = run_overfit(steps=1, lr=2e-3, log_every=999, device="cpu")
    assert abs(hist["tok_rgb"][0] - math.log(TOK_RGB_VOCAB_SIZE)) < 1.5
    assert abs(hist["tok_depth"][0] - math.log(TOK_DEPTH_VOCAB_SIZE)) < 1.5
