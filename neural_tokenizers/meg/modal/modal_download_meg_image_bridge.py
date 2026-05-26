"""Build the MEG-trigger-code → THINGS-universal-image-id bridge.

Why this script exists (and why we don't reuse events.tsv):
  events.tsv (downloaded by modal_download_things_labels.py) only has the
  per-trial integer trigger code (`value`) and the THINGS concept number.
  It has no image filename. To pair an MEG trial with an RGB shard we need
  to resolve trigger_code → THINGS_filename → universal 9-digit image_id
  in things_catalog.json.

  The OpenNeuro source files `sourcedata/sample_attributes_P{1..4}.csv`
  carry exactly that extra column (`image_path = images_meg/<concept>/<file>`).
  This script pulls those four CSVs, takes the basename of `image_path`,
  joins on things_catalog, and writes the resulting bridge JSON.

Cross-subject consistency:
  The THINGS image_nr is a canonical THINGS identifier, so the same trigger
  must map to the same filename across all 4 subjects. We verify this and
  abort if not — silent inconsistency would silently misalign MEG↔RGB
  pairing for the rest of the project.

Catalog coverage:
  Every MEG trigger code MUST resolve to a filename that's in
  things_catalog.json. Per the user's "abort on gap" policy, we surface
  any unresolved triggers as a hard error.

Cost: ~$0.10 (CPU, ~5 MB downloads, < 1 min).

Run from inner repo root:
    modal run neural_tokenizers/meg/modal/modal_download_meg_image_bridge.py::build
"""

from __future__ import annotations

from pathlib import Path

import modal


app = modal.App("neural-fm")
project_volume = modal.Volume.from_name("project")
PROJECT_MOUNT = "/project"

OPENNEURO_S3 = "s3://openneuro.org/ds004212"
SAMPLE_ATTR_SUBJECTS = ("P1", "P2", "P3", "P4")

CATALOG_PATH_REMOTE = "/project/data/things_catalog.json"
OUTPUT_DIR_REMOTE = "/project/data/things-meg/labels"
OUTPUT_FILE = "meg_trigger_to_image_id.json"
DEFAULT_LOCAL_OUT = "neural_tokenizers/meg/data/meg_trigger_to_image_id.json"
DEFAULT_LOCAL_MEG_COVERAGE = "neural_tokenizers/meg/data/meg_coverage.json"


def _build_meg_coverage_payload(trigger_to_image_id: dict[str, str]) -> dict:
    image_ids = sorted(set(trigger_to_image_id.values()))
    return {
        "version": "1",
        "modality": "meg",
        "n_image_ids": len(image_ids),
        "id_format": (
            "9-digit zero-padded alphabetical rank of the THINGS image filename"
        ),
        "source": "things-meg/labels/meg_trigger_to_image_id.json",
        "image_ids": image_ids,
    }

bridge_image = (
    modal.Image.debian_slim(python_version="3.11").pip_install("awscli", "pandas")
)


@app.function(
    image=bridge_image,
    volumes={PROJECT_MOUNT: project_volume},
    cpu=2.0,
    memory=4 * 1024,
    timeout=60 * 15,
)
def build_bridge_remote() -> dict:
    """Pull sample_attributes_P*.csv, join with things_catalog, write bridge JSON."""
    import json
    import os
    import subprocess
    import tempfile
    from collections import defaultdict

    import pandas as pd

    if not os.path.exists(CATALOG_PATH_REMOTE):
        raise FileNotFoundError(
            f"things_catalog.json missing at {CATALOG_PATH_REMOTE}. "
            f"Run modal/modal_things_repack.py::plan first."
        )
    catalog = json.loads(Path(CATALOG_PATH_REMOTE).read_text())
    # Invert catalog for filename → image_id lookup.
    filename_to_image_id: dict[str, str] = {}
    for image_id, filename in catalog["image_id_to_filename"].items():
        if filename in filename_to_image_id:
            raise ValueError(
                f"things_catalog has duplicate filename {filename!r}; "
                f"can't invert."
            )
        filename_to_image_id[filename] = image_id
    print(f"[bridge] catalog: {len(filename_to_image_id)} unique filenames")

    # Pull sample_attributes_P*.csv from S3. Each is ~3 MB.
    tmpdir = tempfile.mkdtemp()
    csv_paths: dict[str, str] = {}
    for subj in SAMPLE_ATTR_SUBJECTS:
        dst = f"{tmpdir}/sample_attributes_{subj}.csv"
        print(f"[bridge] downloading sample_attributes_{subj}.csv ...")
        subprocess.run(
            [
                "aws", "s3", "cp",
                "--no-sign-request", "--no-progress",
                f"{OPENNEURO_S3}/sourcedata/sample_attributes_{subj}.csv",
                dst,
            ],
            check=True,
        )
        csv_paths[subj] = dst

    # Build trigger → filename per subject, then merge with consistency check.
    per_subject: dict[str, dict[int, str]] = {}
    for subj, p in csv_paths.items():
        df = pd.read_csv(p)
        if "things_image_nr" not in df.columns or "image_path" not in df.columns:
            raise RuntimeError(
                f"{p} missing expected columns; got {list(df.columns)}"
            )
        # Keep exp + test rows (the rows that produce actual MEG trials). Catch
        # rows have things_image_nr but the image isn't a THINGS image we care
        # about; trial_type=='catch' rows have image_path pointing outside images_meg/.
        mask = df["trial_type"].isin(("exp", "test"))
        df = df[mask].dropna(subset=["things_image_nr", "image_path"])
        m: dict[int, str] = {}
        for trig, path in zip(df["things_image_nr"].astype(int), df["image_path"]):
            basename = os.path.basename(str(path))
            prior = m.get(trig)
            if prior is not None and prior != basename:
                raise RuntimeError(
                    f"{subj}: trigger {trig} has conflicting filenames within "
                    f"this subject: {prior!r} vs {basename!r}"
                )
            m[trig] = basename
        per_subject[subj] = m
        print(f"[bridge] {subj}: {len(m)} unique (trigger, filename) pairs")

    # Cross-subject consistency: same trigger must map to same filename in all subjects.
    all_triggers: set[int] = set()
    for m in per_subject.values():
        all_triggers.update(m.keys())

    conflicts: list[tuple[int, dict[str, str]]] = []
    merged: dict[int, str] = {}
    for trig in sorted(all_triggers):
        seen: dict[str, str] = {}
        for subj, m in per_subject.items():
            if trig in m:
                seen[subj] = m[trig]
        unique_filenames = set(seen.values())
        if len(unique_filenames) > 1:
            conflicts.append((trig, seen))
        else:
            merged[trig] = next(iter(unique_filenames))
    if conflicts:
        raise RuntimeError(
            f"{len(conflicts)} triggers disagree across subjects. "
            f"First 5: {conflicts[:5]}"
        )
    print(f"[bridge] cross-subject consistent: {len(merged)} unique triggers")

    # Resolve filename → universal image_id via catalog.
    trigger_to_image_id: dict[int, str] = {}
    unresolved: list[tuple[int, str]] = []
    for trig in sorted(merged):
        fn = merged[trig]
        image_id = filename_to_image_id.get(fn)
        if image_id is None:
            unresolved.append((trig, fn))
        else:
            trigger_to_image_id[trig] = image_id

    if unresolved:
        raise RuntimeError(
            f"{len(unresolved)} MEG triggers have no image in things_catalog. "
            f"First 10: {unresolved[:10]}. "
            f"Aborting (user policy: surface unresolved gaps, do not silently drop)."
        )

    print(f"[bridge] resolved {len(trigger_to_image_id)} triggers → image_ids")

    # Distribution sanity: count exp vs test triggers.
    per_trial_type: dict[str, int] = defaultdict(int)
    for p in csv_paths.values():
        df = pd.read_csv(p)
        df = df[df["trial_type"].isin(("exp", "test"))]
        for tt in df["trial_type"]:
            per_trial_type[tt] += 1
    print(f"[bridge] total trial rows across subjects: {dict(per_trial_type)}")

    os.makedirs(OUTPUT_DIR_REMOTE, exist_ok=True)
    payload = {
        "version": "1",
        "n_triggers": len(trigger_to_image_id),
        "n_filenames_in_catalog": len(filename_to_image_id),
        "source": f"{OPENNEURO_S3}/sourcedata/sample_attributes_P{{1..4}}.csv",
        "trial_types_kept": ["exp", "test"],
        "trigger_to_filename": {str(k): merged[k] for k in sorted(merged)},
        "trigger_to_image_id": {
            str(k): trigger_to_image_id[k] for k in sorted(trigger_to_image_id)
        },
    }
    out_path = f"{OUTPUT_DIR_REMOTE}/{OUTPUT_FILE}"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[bridge] wrote {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")

    meg_coverage_path = f"{OUTPUT_DIR_REMOTE}/meg_coverage.json"
    meg_coverage = _build_meg_coverage_payload(payload["trigger_to_image_id"])
    with open(meg_coverage_path, "w") as fh:
        json.dump(meg_coverage, fh, indent=2)
    print(
        f"[bridge] wrote {meg_coverage_path} "
        f"({os.path.getsize(meg_coverage_path) / 1024:.1f} KB)"
    )

    project_volume.commit()

    # Ship full payload back so the local entrypoint can git-track it.
    return {
        "n_triggers": payload["n_triggers"],
        "n_filenames_in_catalog": payload["n_filenames_in_catalog"],
        "trial_type_totals": dict(per_trial_type),
        "payload": payload,
    }


@app.local_entrypoint()
def build(output: str = DEFAULT_LOCAL_OUT, meg_coverage: str = DEFAULT_LOCAL_MEG_COVERAGE):
    """Run the bridge build remotely, write git-trackable local copies."""
    import json

    summary = build_bridge_remote.remote()
    payload = summary.pop("payload")
    print("\n[bridge] summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    local_path = Path(output)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(payload, indent=2))
    print(f"[bridge] wrote local copy to {local_path}")

    meg_cov = _build_meg_coverage_payload(payload["trigger_to_image_id"])
    meg_path = Path(meg_coverage)
    meg_path.parent.mkdir(parents=True, exist_ok=True)
    meg_path.write_text(json.dumps(meg_cov, indent=2))
    print(f"[bridge] wrote meg_coverage to {meg_path}")
