"""One-off inspection: what's inside epochs.event_id for THINGS-MEG?

We need to know if MNE's event_id dict already encodes concept names
(in which case we can derive image_id → concept mapping locally), or if
it just holds integer-string codes (in which case we need the OpenNeuro
events.tsv metadata).

Cheap to run (~$0.10 — CPU container, no GPU, ~3 min).

Invoke from the inner repo root:
    modal run neural_tokenizers/meg/modal/modal_meg_inspect_labels.py::inspect
"""

from __future__ import annotations

from pathlib import Path
import sys

try:
    import neural_tokenizers  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import modal  # noqa: E402


app = modal.App("neural-fm")
project_volume = modal.Volume.from_name("project")

inspect_image = (
    modal.Image.debian_slim(python_version="3.11")
    # torch is pulled in transitively because neural_tokenizers/meg/__init__.py
    # re-exports mu_transform; cheaper to add it than to refactor for lazy imports.
    .pip_install("mne", "numpy", "torch")
    .add_local_python_source("neural_tokenizers")
)


@app.function(
    image=inspect_image,
    volumes={"/project": project_volume},
    cpu=2.0,
    memory=8 * 1024,
    timeout=60 * 10,
)
def inspect_labels_remote() -> dict:
    """Dump event_id keys + a sample for one subject. Returns a small summary."""
    import mne

    mne.set_log_level("ERROR")

    # Import the constant directly (not through meg/__init__.py) so we
    # don't pull torch through the mu_transform re-export chain.
    from neural_tokenizers.meg.meg_config import MEG_DATA

    primary = f"{MEG_DATA.data_dir}/preprocessed_P1-epo.fif"
    epochs = mne.read_epochs(primary, preload=False, verbose="ERROR")
    event_id = dict(epochs.event_id)
    keys = list(event_id.keys())

    summary = {
        "n_event_id_keys": len(event_id),
        "first_20_keys": keys[:20],
        "last_5_keys": keys[-5:],
        "key_value_examples": [(k, event_id[k]) for k in keys[:5]],
        "trigger_code_range": [int(min(event_id.values())), int(max(event_id.values()))],
    }
    print("[inspect] summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


@app.local_entrypoint()
def inspect():
    out = inspect_labels_remote.remote()
    print("\n[inspect] returned (local view):")
    for k, v in out.items():
        print(f"  {k}: {v}")
