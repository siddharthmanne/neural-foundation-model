"""Download THINGS-MEG preprocessed epoch files (.fif, ~33 GB) into
the existing `project` Modal Volume at /project/data/things-meg/preprocessed/.

Why this path / volume:
  The team's `project` Volume already holds liubr's cc12m image data at
  /project/data/train/cc12m/. We sit alongside under
      /project/data/things-meg/preprocessed/
  outside the train/val split because the MEG train/val split happens
  by held-out image ID (not by trial), so directory-level partitioning
  doesn't work and we'll express splits via a sibling
  /project/data/things-meg/splits/<policy>.json manifest later.

Why not the shared `neural-fm-data` Volume:
  modal_app.py imports the team's default `neural-fm-data` Volume, but
  the rest of the project is already using `project`. We override locally
  here instead of editing the shared modal_app.py.

Why this source (OpenNeuro S3, not figshare):
  Figshare's ndownloader is behind AWS WAF and serves a JS challenge
  page to anonymous clients (empty HTTP 202). The same preprocessed
  data lives at
      s3://openneuro.org/ds004212/derivatives/preprocessed/
  alongside .mat (CoSMo) and .csv (eye-tracking) variants. We pull only
  the .fif (MNE-Python epoched MEG, 33 GB) — directly loadable via
  mne.read_epochs() into a PyTorch DataLoader. We skip .mat (60 GB,
  MATLAB duplicate of the same data) and .csv eye-tracking (33 GB,
  not needed for a first-pass MEG FM).

Run:
    modal run --detach modal_download_things_meg.py::download
"""

import os
import subprocess

import modal

from modal_app import app, image

# Override the volume locally. `from_name` without `create_if_missing`
# means we error loudly if `project` doesn't exist, instead of silently
# creating a new empty volume of that name.
project_volume = modal.Volume.from_name("project")

# awscli for `aws s3 sync` against the public OpenNeuro bucket.
# add_local_python_source("modal_app") ships modal_app.py into the
# container so the `from modal_app import ...` line above resolves at
# remote-runtime.
download_image = (
    image
    .pip_install("awscli")
    .add_local_python_source("modal_app")
)

S3_SOURCE = "s3://openneuro.org/ds004212/derivatives/preprocessed/"
DEST_DIR = "/project/data/things-meg/preprocessed"

# Post-condition floor. 16 .fif files totaling ~33 GB on S3; anything
# under 25 GB after a successful sync means files are missing.
MIN_EXPECTED_BYTES = 25 * 1024 * 1024 * 1024


@app.function(
    image=download_image,
    # Mount `project` volume at /project (NOT /data) so the path inside
    # the container mirrors the volume's internal layout: liubr's stuff
    # at /project/data/train/cc12m/, ours at /project/data/things-meg/...
    volumes={"/project": project_volume},
    cpu=4.0,
    memory=8 * 1024,
    timeout=60 * 60 * 3,
)
def download():
    """aws s3 sync, .fif only, into /project/data/things-meg/preprocessed/."""
    os.makedirs(DEST_DIR, exist_ok=True)

    # --exclude "*" then --include "*.fif" is aws-cli's idiom for "only
    # files matching this pattern." Order matters (left-to-right).
    subprocess.run(
        [
            "aws", "s3", "sync",
            "--no-sign-request",
            "--no-progress",
            "--exclude", "*",
            "--include", "*.fif",
            S3_SOURCE,
            DEST_DIR,
        ],
        check=True,
    )

    # Post-condition: total size in the expected range. Catches silent
    # partial syncs (zero exit code but missing files).
    files = [f for f in os.listdir(DEST_DIR)
             if os.path.isfile(os.path.join(DEST_DIR, f))]
    total = sum(os.path.getsize(os.path.join(DEST_DIR, f)) for f in files)
    print(f"Synced {len(files)} files, total {total / 1e9:.2f} GB")
    if total < MIN_EXPECTED_BYTES:
        raise RuntimeError(
            f"Synced data is suspiciously small ({total / 1e9:.2f} GB). "
            f"Expected >{MIN_EXPECTED_BYTES / 1e9:.0f} GB. "
            f"Check the S3 prefix and include filter."
        )

    subprocess.run(["du", "-sh", DEST_DIR], check=True)
    subprocess.run(["ls", "-la", DEST_DIR], check=True)
    project_volume.commit()
