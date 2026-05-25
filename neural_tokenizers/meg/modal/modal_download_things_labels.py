"""Download + parse THINGS-MEG events.tsv files to build the
image_id → concept_id mapping the §5.3 linear probe needs.

Why this script exists (and why the mapping isn't on the Volume already):
  - /project/data/things-meg/preprocessed/*.fif holds the MEG data + integer
    trigger codes (22,449 unique). Those codes are THINGS *image* numbers,
    not concept numbers.
  - Each trial's THINGS concept number lives only in OpenNeuro's per-run
    `events.tsv` files (under `sub-BIGMEG{1..4}/ses-*/meg/*_events.tsv`),
    which we did NOT pull when we downloaded the preprocessed derivatives.
  - The mapping is one-to-many in trials but functional in image→concept:
    each image_id maps to exactly one things_category_nr ∈ [1, 1854].

Verified events.tsv schema (from
https://raw.githubusercontent.com/OpenNeuroDatasets/ds004212/main/task-main_events.json):
    value:               THINGS image number          (= trigger code)
    things_category_nr:  THINGS category number       (= concept_id, 1..1854)
    trial_type:          {exp, catch, test}
        - exp:   experimental THINGS trial
        - test:  repeat THINGS trial (200 special images per session)
        - catch: artificially-generated oddball (NOT a real THINGS image)

We keep `exp` + `test` rows (these match the preprocessed .fif trials) and
drop `catch`.

Output (written to the Volume and ALSO returned via .remote() so it lands
in git):
    /project/data/things-meg/labels/image_id_to_concept.json
    neural_tokenizers/meg/data/image_id_to_concept.json   (local copy)

Cost: ~$0.10 (CPU container, ~5 MB of tsv downloads + parsing).

Run from inner repo root:
    modal run neural_tokenizers/meg/modal/modal_download_things_labels.py::download
"""

from __future__ import annotations

from pathlib import Path

# Intentionally NO sys.path tweak / neural_tokenizers import here: this
# script doesn't depend on the project's tokenizer code. It's a pure
# OpenNeuro-fetch + parse job and the only project deps are pandas + awscli.

import modal


app = modal.App("neural-fm")
project_volume = modal.Volume.from_name("project")
PROJECT_MOUNT = "/project"
OUTPUT_DIR_REMOTE = "/project/data/things-meg/labels"
OUTPUT_FILE = "image_id_to_concept.json"

OPENNEURO_S3 = "s3://openneuro.org/ds004212"
EVENTS_SUBJECTS = ("sub-BIGMEG1", "sub-BIGMEG2", "sub-BIGMEG3", "sub-BIGMEG4")

labels_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("awscli", "pandas")
)


@app.function(
    image=labels_image,
    volumes={PROJECT_MOUNT: project_volume},
    cpu=2.0,
    memory=4 * 1024,
    timeout=60 * 20,
)
def download_remote() -> dict:
    """Sync events.tsv files, parse, build the mapping, write JSON. Returns
    the same JSON payload so the local entrypoint can write a git-trackable
    copy.
    """
    import json
    import os
    import subprocess
    import tempfile
    from collections import Counter
    from glob import glob

    import pandas as pd

    os.makedirs(OUTPUT_DIR_REMOTE, exist_ok=True)
    tmpdir = tempfile.mkdtemp()

    # Pull events.tsv from each subject's session/run hierarchy. We restrict
    # the s3 sync to /sub-BIGMEG*/ to avoid touching the 33 GB of MEG data
    # again. The include filter only keeps `*_events.tsv`.
    for subj in EVENTS_SUBJECTS:
        print(f"[labels] syncing events.tsv for {subj} ...")
        subprocess.run(
            [
                "aws", "s3", "sync",
                "--no-sign-request",
                "--no-progress",
                "--exclude", "*",
                "--include", "*_events.tsv",
                f"{OPENNEURO_S3}/{subj}/",
                f"{tmpdir}/{subj}/",
            ],
            check=True,
        )

    tsv_files = sorted(glob(f"{tmpdir}/**/*_events.tsv", recursive=True))
    print(f"[labels] downloaded {len(tsv_files)} events.tsv files")

    # Parse: build image_id → concept_id, also count trials per concept.
    image_to_concept: dict[int, int] = {}
    trial_count_per_concept: Counter[int] = Counter()
    kept_trial_types = ("exp", "test")
    rows_total = rows_kept = 0
    inconsistencies: list[tuple[int, int, int]] = []  # (image_id, concept_old, concept_new)

    for f in tsv_files:
        df = pd.read_csv(f, sep="\t")
        rows_total += len(df)
        for _, row in df.iterrows():
            tt = row.get("trial_type")
            if tt not in kept_trial_types:
                continue
            v = row.get("value")
            c = row.get("things_category_nr")
            if pd.isna(v) or pd.isna(c):
                continue
            image_id = int(v)
            concept_id = int(c)
            rows_kept += 1
            trial_count_per_concept[concept_id] += 1
            prior = image_to_concept.get(image_id)
            if prior is None:
                image_to_concept[image_id] = concept_id
            elif prior != concept_id:
                inconsistencies.append((image_id, prior, concept_id))

    if inconsistencies:
        raise RuntimeError(
            f"image→concept mapping not functional in {len(inconsistencies)} cases; "
            f"first 5: {inconsistencies[:5]}"
        )

    n_concepts = len({c for c in image_to_concept.values()})
    print(f"[labels] parsed {rows_total} total rows, {rows_kept} kept "
          f"(trial_type in {kept_trial_types})")
    print(f"[labels] unique image_ids: {len(image_to_concept)}")
    print(f"[labels] unique concept_ids: {n_concepts} (expected ~1854)")
    print(f"[labels] top-5 concepts by trial count: "
          f"{trial_count_per_concept.most_common(5)}")

    payload = {
        "image_id_to_concept_id": {str(k): v for k, v in image_to_concept.items()},
        "n_image_ids": len(image_to_concept),
        "n_concepts": n_concepts,
        "trial_type_filter": list(kept_trial_types),
        "source": f"{OPENNEURO_S3}/sub-BIGMEG{{1..4}}/ses-*/meg/*_events.tsv",
        "events_schema_source": "task-main_events.json (THINGS-MEG OpenNeuro v3.0.0)",
    }

    out_path = f"{OUTPUT_DIR_REMOTE}/{OUTPUT_FILE}"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[labels] wrote {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")

    project_volume.commit()

    # Strip the big dict from the return so .remote() doesn't ship 2 MB
    # over RPC; the local entrypoint will re-load the file via volume.get.
    summary = {k: v for k, v in payload.items() if k != "image_id_to_concept_id"}
    summary["mapping_size_bytes"] = os.path.getsize(out_path)
    summary["top_5_concepts_by_count"] = trial_count_per_concept.most_common(5)
    summary["payload"] = payload  # ship the full thing — only ~2 MB JSON
    return summary


@app.local_entrypoint()
def download(
    output: str = "neural_tokenizers/meg/data/image_id_to_concept.json",
):
    """Run the download remotely, write a git-trackable local copy.

    The file is small enough (<2 MB) to keep in git per the same logic as
    calibration.json: it's versioned tokenizer-eval state, and you don't
    want eval results to depend on whether the Volume was re-synced.
    """
    import json

    summary = download_remote.remote()
    payload = summary.pop("payload")
    print("\n[labels] summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[labels] wrote local copy to {out_path}")
