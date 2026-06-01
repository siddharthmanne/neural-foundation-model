"""Build THINGS split JSON artifacts without touching legacy manifests or shards.

Writes only:
  /project/data/things-meg/labels/meg_coverage.json
  /project/data/things_split.json

Does NOT write to train/things_manifest.json, val/things_manifest.json,
or any shard tars. Existing 85/15 layout stays intact for later comparison.

Run from modal/:
    modal run modal_build_things_split.py::build
    modal run modal_build_things_split.py::build --force

Local dry-run (when Volume paths are mirrored locally):
    python modal_build_things_split.py \\
        --catalog /path/to/things_catalog.json \\
        --bridge neural_tokenizers/meg/data/meg_trigger_to_image_id.json \\
        --eeg-coverage /path/to/eeg_coverage.json \\
        --out-dir /tmp/things_split_out
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import modal

import things_manifest
from modal_app import app, image


project_volume = modal.Volume.from_name("project")

build_image = image.add_local_python_source("things_manifest", "modal_app")

PROJECT_MOUNT = "/project"
CATALOG_PATH = "/project/data/things_catalog.json"
BRIDGE_PATH = "/project/data/things-meg/labels/meg_trigger_to_image_id.json"
EEG_COVERAGE_PATH = "/project/data/eeg_coverage.json"
EEG_COVERAGE_PATH_ALT = "/project/data/things-eeg/labels/eeg_coverage.json"
MEG_COVERAGE_PATH = "/project/data/things-meg/labels/meg_coverage.json"
THINGS_SPLIT_PATH = "/project/data/things_split.json"

DEFAULT_LOCAL_MEG_COVERAGE = (
    "neural_tokenizers/meg/data/meg_coverage.json"
)
DEFAULT_LOCAL_THINGS_SPLIT = "modal/data/things_split.json"

VAL_FRAC = 0.20
SEED = 0


def _write_json(path: str | Path, payload: dict, force: bool) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists — pass --force to overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"[build] wrote {path} ({path.stat().st_size / 1024:.1f} KB)")


def build_things_split_artifacts(
    catalog_path: str | Path,
    bridge_path: str | Path,
    eeg_coverage_path: str | Path,
    *,
    meg_coverage_out: str | Path,
    things_split_out: str | Path,
    val_frac: float = VAL_FRAC,
    seed: int = SEED,
    force: bool = False,
) -> dict:
    """Pure build logic usable locally or on the Volume."""
    catalog = json.loads(Path(catalog_path).read_text())
    bridge = json.loads(Path(bridge_path).read_text())
    eeg_cov = json.loads(Path(eeg_coverage_path).read_text())

    catalog_ids = list(catalog["image_id_to_filename"].keys())
    meg_coverage = things_manifest.build_meg_coverage_payload(
        bridge["trigger_to_image_id"]
    )
    meg_ids = meg_coverage["image_ids"]
    eeg_parsed = things_manifest.parse_eeg_coverage(eeg_cov)

    split_payload = things_manifest.build_things_split_payload(
        catalog_ids,
        meg_ids,
        eeg_parsed,
        val_frac=val_frac,
        seed=seed,
    )

    _write_json(meg_coverage_out, meg_coverage, force)
    _write_json(things_split_out, split_payload, force)

    summary = {
        "n_catalog": split_payload["n_catalog"],
        "n_meg": split_payload["n_meg"],
        "n_eeg1": split_payload["eeg"]["n_eeg1"],
        "n_eeg2": split_payload["eeg"]["n_eeg2"],
        "n_eeg_intersection": split_payload["eeg"]["n_intersection"],
        "n_eeg_union": split_payload["eeg"]["n_union"],
        "n_intersection": split_payload["n_intersection"],
        "n_train": split_payload["n_train"],
        "n_val": split_payload["n_val"],
        "meg_coverage_path": str(meg_coverage_out),
        "things_split_path": str(things_split_out),
    }
    return summary


def _resolve_eeg_coverage_path() -> str:
    if os.path.exists(EEG_COVERAGE_PATH):
        return EEG_COVERAGE_PATH
    if os.path.exists(EEG_COVERAGE_PATH_ALT):
        return EEG_COVERAGE_PATH_ALT
    raise FileNotFoundError(
        f"EEG coverage missing at {EEG_COVERAGE_PATH} "
        f"and {EEG_COVERAGE_PATH_ALT}"
    )


@app.function(
    image=build_image,
    volumes={PROJECT_MOUNT: project_volume},
    cpu=2.0,
    memory=4 * 1024,
    timeout=60 * 15,
)
def build_remote(force: bool = False) -> dict:
    """Build meg_coverage.json + things_split.json on the project Volume."""
    eeg_path = _resolve_eeg_coverage_path()
    for p in (CATALOG_PATH, BRIDGE_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required input: {p}")

    summary = build_things_split_artifacts(
        CATALOG_PATH,
        BRIDGE_PATH,
        eeg_path,
        meg_coverage_out=MEG_COVERAGE_PATH,
        things_split_out=THINGS_SPLIT_PATH,
        val_frac=VAL_FRAC,
        seed=SEED,
        force=force,
    )
    project_volume.commit()
    return summary


@app.local_entrypoint()
def build(
    force: bool = False,
    local_meg_coverage: str = DEFAULT_LOCAL_MEG_COVERAGE,
    local_things_split: str = DEFAULT_LOCAL_THINGS_SPLIT,
):
    """Run on Modal Volume, then mirror JSONs into the git repo."""
    summary = build_remote.remote(force=force)
    print("\n[build] summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Pull Volume artifacts back for git tracking. Re-read from summary paths
    # isn't available remotely; rebuild locally from git-tracked bridge if
    # Volume files aren't mounted. Instead, fetch via a second remote read.
    payloads = read_artifacts.remote()
    repo_root = Path(__file__).resolve().parent.parent
    meg_out = repo_root / local_meg_coverage
    split_out = repo_root / local_things_split
    meg_out.parent.mkdir(parents=True, exist_ok=True)
    split_out.parent.mkdir(parents=True, exist_ok=True)
    meg_out.write_text(json.dumps(payloads["meg_coverage"], indent=2))
    split_out.write_text(json.dumps(payloads["things_split"], indent=2))
    print(f"[build] git mirror: {meg_out}")
    print(f"[build] git mirror: {split_out}")


@app.function(
    image=build_image,
    volumes={PROJECT_MOUNT: project_volume},
    cpu=1.0,
    memory=2 * 1024,
    timeout=60 * 5,
)
def read_artifacts() -> dict:
    """Read back the JSON artifacts written by build_remote."""
    return {
        "meg_coverage": json.loads(Path(MEG_COVERAGE_PATH).read_text()),
        "things_split": json.loads(Path(THINGS_SPLIT_PATH).read_text()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build THINGS split JSONs locally (no Modal)."
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--eeg-coverage", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    summary = build_things_split_artifacts(
        args.catalog,
        args.bridge,
        args.eeg_coverage,
        meg_coverage_out=out_dir / "meg_coverage.json",
        things_split_out=out_dir / "things_split.json",
        val_frac=args.val_frac,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
