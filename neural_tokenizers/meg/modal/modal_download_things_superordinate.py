"""Download + parse the THINGS concept_id → 27-superordinate mapping.

Why this exists, and why it's a separate script from `modal_download_things_labels.py`:
  - `modal_download_things_labels.py` extracts `image_id → concept_id` from
    THINGS-MEG OpenNeuro events.tsv (modality-specific provenance).
  - The 27 "high-level" / superordinate category assignments are NOT in
    OpenNeuro — they live in canonical THINGS metadata (`category_mat_manual.tsv`)
    which is hosted in the THINGS-data repo. It's modality-independent: the
    same `concept_id → superordinate` mapping is the right label space for
    MEG, EEG, fMRI, and intracortical data.
  - Therefore one download produces one artifact shared across modalities.

The canonical source is the THINGS database on OSF (project `jum2f`),
specifically `concepts-metadata_things.tsv` (1854 rows × 25 columns; row N
== concept_id N+1 from THINGS events.tsv). The relevant column is
**"Bottom-up Category (Human Raters)"** — the empirically-determined
high-level category from crowdsourcing (Hebart 2019, §"high-level
categories").

Field semantics:
  - empty cell  → no high-level category assignment (concept does not
                  belong to any of the 27 most populated categories; 928
                  concepts).
  - single name → single-category concept (e.g. "animal"); 892 concepts.
  - "A, B"      → multi-category concept (e.g. "food, fruit"); 34 concepts,
                  excluded from the probe label space to keep classification
                  K-way rather than multi-label.

The 27 unique single-category labels are stable across releases; we sort
them alphabetically when assigning indices 0..26 so the mapping is
reproducible.

Output (Volume + git):
    /project/data/things/labels/concept_id_to_superordinate.json
    neural_tokenizers/meg/data/concept_id_to_superordinate.json

Cost: ~$0.05 (CPU, ~500 KB TSV download).

Run from inner repo root:
    modal run neural_tokenizers/meg/modal/modal_download_things_superordinate.py::download
"""

from __future__ import annotations

from pathlib import Path

import modal


app = modal.App("neural-fm")
project_volume = modal.Volume.from_name("project")
PROJECT_MOUNT = "/project"
OUTPUT_DIR_REMOTE = "/project/data/things/labels"
OUTPUT_FILE = "concept_id_to_superordinate.json"

DEFAULT_SOURCE_URL = "https://osf.io/download/um6a9/"
CATEGORY_COLUMN = "Bottom-up Category (Human Raters)"
N_EXPECTED_CONCEPTS = 1854
N_EXPECTED_CATEGORIES = 27

labels_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "requests")
)


def _parse_concepts_metadata(
    tsv_text: str,
) -> tuple[dict[int, int], list[str], dict[int, list[str]], dict[int, str]]:
    """Parse the THINGS concepts metadata TSV.

    Returns:
        concept_to_super: {concept_id (1..1854): superordinate_index (0..26)}
                          for concepts with exactly one bottom-up category.
        category_names:   alphabetically-sorted list of 27 high-level
                          category names — index ↔ label.
        multi_membership: {concept_id: [names...]} for the 34 concepts
                          assigned to two categories (e.g. "food, fruit").
                          Excluded from the probe label space; recorded
                          here for inspection.
        unique_ids:       {concept_id: uniqueID string} — for cross-referencing
                          back to the THINGSplus uniqueID system.
    """
    import io
    import pandas as pd

    df = pd.read_csv(io.StringIO(tsv_text), sep="\t")
    if len(df) != N_EXPECTED_CONCEPTS:
        raise RuntimeError(
            f"expected {N_EXPECTED_CONCEPTS} concept rows, got {len(df)}"
        )
    if CATEGORY_COLUMN not in df.columns:
        raise RuntimeError(
            f"missing column {CATEGORY_COLUMN!r}; columns present: "
            f"{list(df.columns)}"
        )

    raw = df[CATEGORY_COLUMN].fillna("")
    # "A, B" → ["A", "B"]; empty string → []
    memberships_per_concept: list[list[str]] = []
    for cell in raw:
        cell = str(cell).strip()
        if not cell:
            memberships_per_concept.append([])
        else:
            memberships_per_concept.append([s.strip() for s in cell.split(",") if s.strip()])

    single_labels = sorted({m[0] for m in memberships_per_concept if len(m) == 1})
    if len(single_labels) != N_EXPECTED_CATEGORIES:
        raise RuntimeError(
            f"expected {N_EXPECTED_CATEGORIES} distinct single-category "
            f"labels, got {len(single_labels)}: {single_labels}"
        )
    label_to_index = {name: i for i, name in enumerate(single_labels)}

    concept_to_super: dict[int, int] = {}
    multi_membership: dict[int, list[str]] = {}
    unique_ids: dict[int, str] = {}
    for row_idx, (memberships, uid) in enumerate(
        zip(memberships_per_concept, df["uniqueID"].astype(str))
    ):
        # concept_id == row index + 1 (events.tsv uses 1-indexed `things_category_nr`
        # which is the row number in this metadata file).
        concept_id = row_idx + 1
        unique_ids[concept_id] = uid
        if len(memberships) == 1:
            concept_to_super[concept_id] = label_to_index[memberships[0]]
        elif len(memberships) >= 2:
            multi_membership[concept_id] = memberships
        # else: no membership → omit (downstream encode() marks invalid)

    return concept_to_super, single_labels, multi_membership, unique_ids


@app.function(
    image=labels_image,
    volumes={PROJECT_MOUNT: project_volume},
    cpu=1.0,
    memory=2 * 1024,
    timeout=60 * 5,
)
def download_remote(source_url: str = DEFAULT_SOURCE_URL) -> dict:
    """Fetch the category matrix, parse, and write JSON. Returns full
    payload so the local entrypoint can mirror it to git."""
    import json
    import os
    from collections import Counter

    import requests

    print(f"[super] GET {source_url}")
    resp = requests.get(source_url, timeout=60)
    resp.raise_for_status()
    tsv_text = resp.text
    print(f"[super] downloaded {len(tsv_text)} bytes")

    concept_to_super, category_names, multi, unique_ids = _parse_concepts_metadata(tsv_text)
    print(f"[super] {len(concept_to_super)} concepts with single-category membership")
    print(f"[super] {len(multi)} concepts with multi-category membership (excluded)")
    print(f"[super] {N_EXPECTED_CATEGORIES} category names: {category_names}")

    # Per-category concept count distribution — useful sanity check.
    dist = Counter(concept_to_super.values())
    by_name = [(category_names[i], dist[i]) for i in range(len(category_names))]
    print(f"[super] concepts per category:")
    for name, n in sorted(by_name, key=lambda x: -x[1]):
        print(f"    {name:<32s} {n}")

    payload = {
        "concept_id_to_superordinate_index": {
            str(c): int(s) for c, s in concept_to_super.items()
        },
        "category_names": list(category_names),
        "n_categories": len(category_names),
        "n_concepts_with_label": len(concept_to_super),
        "n_concepts_multi_membership": len(multi),
        "multi_membership": {str(c): names for c, names in multi.items()},
        "concept_id_to_unique_id": {str(c): uid for c, uid in unique_ids.items()},
        "source_url": source_url,
    }

    os.makedirs(OUTPUT_DIR_REMOTE, exist_ok=True)
    out_path = f"{OUTPUT_DIR_REMOTE}/{OUTPUT_FILE}"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[super] wrote {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")

    project_volume.commit()
    return payload


@app.local_entrypoint()
def download(
    output: str = "neural_tokenizers/meg/data/concept_id_to_superordinate.json",
    source_url: str = DEFAULT_SOURCE_URL,
):
    """Run remotely, write a git-trackable local copy.

    Same logic as `modal_download_things_labels.py`: the JSON is small
    versioned eval-state, so it lives in git and eval reproducibility
    doesn't depend on the Volume.
    """
    import json

    payload = download_remote.remote(source_url=source_url)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[super] wrote local copy to {out_path}")
    print(f"[super] n_concepts_with_label: {payload['n_concepts_with_label']}")
    print(f"[super] n_categories:          {payload['n_categories']}")
