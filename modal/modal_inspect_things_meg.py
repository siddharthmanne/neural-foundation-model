"""One-off inspection of preprocessed THINGS-MEG .fif files on the `project`
Modal Volume. Run this whenever the data layout assumptions in
[`../neural_tokenizers/meg/CLAUDE.md`](../neural_tokenizers/meg/CLAUDE.md)
need re-verification.

Run:
    modal run modal_inspect_things_meg.py::inspect

Prints, for each subject's first .fif split: sampling rate, channel count
(broken down by ch_type), epoch count, epoch duration, timepoints per epoch,
and the trigger/event label range. These are the numbers the MEG tokenizer
plan is grounded in — re-run after any preprocessing pipeline change.
"""

import os
from glob import glob

import modal

from modal_app import app, image

project_volume = modal.Volume.from_name("project")

inspect_image = (
    image
    .pip_install("mne", "numpy")
    .add_local_python_source("modal_app")
)

DATA_DIR = "/project/data/things-meg/preprocessed"


@app.function(
    image=inspect_image,
    volumes={"/project": project_volume},
    cpu=2.0,
    memory=8 * 1024,
    timeout=60 * 15,
)
def inspect():
    import mne
    import numpy as np

    mne.set_log_level("WARNING")

    all_files = sorted(glob(os.path.join(DATA_DIR, "*-epo.fif")))
    print(f"[INFO] Found {len(all_files)} primary -epo.fif files in {DATA_DIR}")
    for f in all_files:
        print(f"  - {os.path.basename(f)}  ({os.path.getsize(f) / 1e9:.2f} GB)")

    for f in all_files:
        print("\n" + "=" * 72)
        print(f"FILE: {f}")
        print("=" * 72)
        epochs = mne.read_epochs(f, preload=False, verbose="ERROR")
        info = epochs.info

        ch_types = [info.get_channel_types()[i] for i in range(len(info["ch_names"]))]
        type_counts = {t: ch_types.count(t) for t in sorted(set(ch_types))}

        print(f"  sfreq:        {info['sfreq']} Hz")
        print(f"  n_channels:   {len(info['ch_names'])}  -> {type_counts}")
        print(f"  n_epochs:     {len(epochs)}")
        print(f"  tmin, tmax:   {epochs.tmin:.4f} s, {epochs.tmax:.4f} s")
        print(f"  timepoints:   {len(epochs.times)}")

        ev = epochs.events
        print(f"  events shape: {ev.shape}")
        print(f"  event_id keys (first 5): {list(epochs.event_id.keys())[:5]}")
        print(f"  unique label codes: {len(np.unique(ev[:, 2]))}")
        print(f"  data dtype on disk: {epochs._data.dtype if epochs.preload else 'lazy'}")


@app.local_entrypoint()
def run():
    inspect.remote()
