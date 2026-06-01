"""Comprehensive end-to-end audit of the MEG tokenization pipeline.

Reads everything on the project Volume and runs 15 cross-layer checks:
  Layer 1 (structural):   does everything exist where expected?
  Layer 2 (consistency):  do the JSONs / caches / shards agree with each other?
  Layer 3 (data integrity): are the bytes correct?
  Layer 4 (protocol):     does the corpus match the THINGS-MEG protocol?

Exits 0 if every check passes. If any check FAILs, prints the offending
details and exits with a non-zero summary so it's safe to script.

Cost: ~$0.10 (CPU only, ~5 min).

Run from inner repo root (uses modal/ working dir for the image):
    cd modal
    modal run modal_meg_pipeline_audit.py::audit
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

import modal

import meg_token_shard
import things_manifest
from modal_app import app, image


project_volume = modal.Volume.from_name("project")
PROJECT_MOUNT = "/project"

CATALOG_PATH = "/project/data/things_catalog.json"
BRIDGE_PATH = "/project/data/things-meg/labels/meg_trigger_to_image_id.json"
CACHE_DIR = "/project/data/things-meg/tokens/brainomni/V512_rvq4_win512_sf256_3b"
TRAIN_RGB_MANIFEST = "/project/data/train/things_manifest.json"
VAL_RGB_MANIFEST = "/project/data/val/things_manifest.json"
TRAIN_MEG_MANIFEST = "/project/data/train/things_meg_manifest.json"
VAL_MEG_MANIFEST = "/project/data/val/things_meg_manifest.json"
TRAIN_MEG_DIR = "/project/data/train/things/tok_meg"
VAL_MEG_DIR = "/project/data/val/things/tok_meg"
SUBJECTS = ("P1", "P2", "P3", "P4")
EXPECTED_CODEBOOK_SIZE = 512
EXPECTED_NUM_QUANTIZERS = 4
EXPECTED_TOKEN_SHAPE = (16, 8, 4)
EXPECTED_TOTAL_TRIALS = 98_592   # 88,992 exp + 9,600 test from bridge totals
EXPECTED_PER_SUBJECT_TRIALS = 24_648
SAMPLE_ROUNDTRIP_N = 200

audit_image = (
    image
    .pip_install("numpy")
    .add_local_python_source("meg_token_shard", "things_manifest", "modal_app")
)


# ---------- result harness -----------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str


def _ok(name: str, msg: str = "") -> CheckResult:
    print(f"  [PASS] {name}  {msg}")
    return CheckResult(name, True, msg)


def _fail(name: str, msg: str) -> CheckResult:
    print(f"  [FAIL] {name}  {msg}")
    return CheckResult(name, False, msg)


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------- helpers (read from volume) ------------------------------------

def _load_catalog() -> dict:
    return json.loads(open(CATALOG_PATH).read())


def _load_bridge() -> dict:
    return json.loads(open(BRIDGE_PATH).read())


def _load_manifests() -> tuple[dict, dict, dict, dict]:
    return (
        json.loads(open(TRAIN_RGB_MANIFEST).read()),
        json.loads(open(VAL_RGB_MANIFEST).read()),
        json.loads(open(TRAIN_MEG_MANIFEST).read()),
        json.loads(open(VAL_MEG_MANIFEST).read()),
    )


def _load_cache_arrays(subject: str):
    import numpy as np

    return np.load(os.path.join(CACHE_DIR, f"{subject}.npz"))


def _enumerate_meg_shard(tar_path: str):
    """Yield (filename, np.ndarray) for every entry in a MEG shard tar."""
    import numpy as np

    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.name.endswith(".meg.npy"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
            yield member.name, arr


# ---------- the actual checks --------------------------------------------

def layer1_structural() -> list[CheckResult]:
    _section("Layer 1 — Structural")
    results: list[CheckResult] = []

    # L1.1: file presence
    expected_files = [
        CATALOG_PATH, BRIDGE_PATH,
        TRAIN_RGB_MANIFEST, VAL_RGB_MANIFEST,
        TRAIN_MEG_MANIFEST, VAL_MEG_MANIFEST,
        os.path.join(CACHE_DIR, "config.json"),
    ] + [os.path.join(CACHE_DIR, f"{s}.npz") for s in SUBJECTS]
    missing = [p for p in expected_files if not os.path.exists(p)]
    if missing:
        results.append(_fail("L1.1 file presence", f"missing: {missing}"))
    else:
        results.append(_ok("L1.1 file presence", f"{len(expected_files)} files present"))

    # L1.2: schema sanity on each JSON
    for path, required_keys in (
        (CATALOG_PATH, {"version", "n_images", "image_id_to_filename"}),
        (BRIDGE_PATH, {"trigger_to_image_id", "n_triggers"}),
        (TRAIN_RGB_MANIFEST, {"split", "n_shards", "n_images", "shards"}),
        (VAL_RGB_MANIFEST, {"split", "n_shards", "n_images", "shards"}),
        (TRAIN_MEG_MANIFEST, {"split", "n_shards", "n_entries", "shards", "shards_subpath", "tokenizer_cfg"}),
        (VAL_MEG_MANIFEST, {"split", "n_shards", "n_entries", "shards", "shards_subpath", "tokenizer_cfg"}),
    ):
        try:
            payload = json.loads(open(path).read())
            missing_keys = required_keys - set(payload.keys())
            if missing_keys:
                results.append(_fail(f"L1.2 schema {os.path.basename(path)}", f"missing keys {missing_keys}"))
            else:
                results.append(_ok(f"L1.2 schema {os.path.basename(path)}", "keys ok"))
        except json.JSONDecodeError as e:
            results.append(_fail(f"L1.2 schema {os.path.basename(path)}", f"malformed JSON: {e}"))

    # L1.3: shard count on disk == manifest count per split
    for split_name, manifest_path, shard_dir in (
        ("train MEG", TRAIN_MEG_MANIFEST, TRAIN_MEG_DIR),
        ("val MEG", VAL_MEG_MANIFEST, VAL_MEG_DIR),
    ):
        m = json.loads(open(manifest_path).read())
        on_disk = sorted(f for f in os.listdir(shard_dir) if f.endswith(".tar"))
        expected = sorted(f"{s}.tar" for s in m["shards"])
        if on_disk == expected:
            results.append(_ok(f"L1.3 shard count {split_name}", f"{len(on_disk)} shards"))
        else:
            results.append(_fail(
                f"L1.3 shard count {split_name}",
                f"on_disk={len(on_disk)} vs manifest={len(expected)}; "
                f"missing={sorted(set(expected) - set(on_disk))[:5]}"
            ))

    return results


def layer2_consistency(
    catalog: dict, bridge: dict, train_rgb: dict, val_rgb: dict,
    train_meg: dict, val_meg: dict, cache_summaries: dict,
) -> list[CheckResult]:
    _section("Layer 2 — Cross-artifact consistency")
    results: list[CheckResult] = []

    catalog_filenames = set(catalog["image_id_to_filename"].values())
    catalog_image_ids = set(catalog["image_id_to_filename"].keys())
    bridge_triggers = {int(k) for k in bridge["trigger_to_image_id"]}
    bridge_filenames = set(bridge["trigger_to_filename"].values())
    bridge_image_ids = set(bridge["trigger_to_image_id"].values())

    # L2.1: every bridge filename exists in catalog
    missing_in_catalog = bridge_filenames - catalog_filenames
    if missing_in_catalog:
        results.append(_fail(
            "L2.1 bridge⊆catalog filenames",
            f"{len(missing_in_catalog)} unresolved; first 5: {sorted(missing_in_catalog)[:5]}",
        ))
    else:
        results.append(_ok(
            "L2.1 bridge⊆catalog filenames",
            f"{len(bridge_filenames)} bridge filenames all in catalog",
        ))

    # L2.2: every cache trigger exists in bridge
    cache_triggers: set[int] = set()
    for s in SUBJECTS:
        cache_triggers.update(cache_summaries[s]["triggers"])
    not_in_bridge = cache_triggers - bridge_triggers
    if not_in_bridge:
        results.append(_fail(
            "L2.2 cache⊆bridge triggers",
            f"{len(not_in_bridge)} cache triggers not in bridge; first 5: {sorted(not_in_bridge)[:5]}",
        ))
    else:
        results.append(_ok(
            "L2.2 cache⊆bridge triggers",
            f"all {len(cache_triggers)} unique cache triggers in bridge",
        ))

    # L2.3: every MEG shard image_id is in the matching RGB shard
    bad_pairings: list[str] = []
    for split_name, meg_manifest, rgb_manifest, shard_dir in (
        ("train", train_meg, train_rgb, TRAIN_MEG_DIR),
        ("val", val_meg, val_rgb, VAL_MEG_DIR),
    ):
        for shard_id in sorted(meg_manifest["shards"]):
            rgb_image_ids = set(rgb_manifest["shards"][shard_id]["image_ids"])
            meg_image_ids: set[str] = set()
            tar_path = os.path.join(shard_dir, f"{shard_id}.tar")
            for name, _arr in _enumerate_meg_shard(tar_path):
                parsed = meg_token_shard.parse_meg_filename(name)
                if parsed is None:
                    continue
                meg_image_ids.add(parsed[0])
            unexpected = meg_image_ids - rgb_image_ids
            if unexpected:
                bad_pairings.append(
                    f"{split_name}/{shard_id}: {len(unexpected)} MEG ids not in RGB; "
                    f"first: {sorted(unexpected)[:2]}"
                )
    if bad_pairings:
        results.append(_fail("L2.3 MEG⊆RGB per shard", "; ".join(bad_pairings[:3])))
    else:
        results.append(_ok("L2.3 MEG⊆RGB per shard", "all 27 shards pair cleanly"))

    # L2.4: train/val image_id disjointness (both RGB and MEG)
    rgb_train_ids = {iid for s in train_rgb["shards"].values() for iid in s["image_ids"]}
    rgb_val_ids = {iid for s in val_rgb["shards"].values() for iid in s["image_ids"]}
    if rgb_train_ids.isdisjoint(rgb_val_ids):
        results.append(_ok("L2.4a RGB train ∩ val = ∅", f"|train|={len(rgb_train_ids)} |val|={len(rgb_val_ids)}"))
    else:
        leak = rgb_train_ids & rgb_val_ids
        results.append(_fail("L2.4a RGB train ∩ val", f"{len(leak)} leaked; first: {sorted(leak)[:3]}"))

    # MEG image_id sets derived from cache entries (no need to re-enumerate tars):
    meg_image_ids_per_subject: dict[str, set[str]] = {}
    for s in SUBJECTS:
        meg_image_ids_per_subject[s] = {
            bridge["trigger_to_image_id"][str(t)] for t in cache_summaries[s]["triggers"]
        }
    meg_all_image_ids = set().union(*meg_image_ids_per_subject.values())
    meg_train_ids = meg_all_image_ids & rgb_train_ids
    meg_val_ids = meg_all_image_ids & rgb_val_ids
    if meg_train_ids.isdisjoint(meg_val_ids):
        results.append(_ok("L2.4b MEG train ∩ val = ∅", f"|train|={len(meg_train_ids)} |val|={len(meg_val_ids)}"))
    else:
        results.append(_fail("L2.4b MEG train ∩ val", f"{len(meg_train_ids & meg_val_ids)} leaked"))

    # L2.5: cache total entries == shard total entries (no loss in pack)
    cache_total = sum(cache_summaries[s]["n"] for s in SUBJECTS)
    shard_total = train_meg["n_entries"] + val_meg["n_entries"]
    if cache_total == shard_total:
        results.append(_ok("L2.5 cache total == shard total", f"{cache_total} entries"))
    else:
        results.append(_fail(
            "L2.5 cache total == shard total",
            f"cache={cache_total}, shards={shard_total}, diff={cache_total - shard_total}"
        ))

    return results


def layer3_integrity(train_meg: dict, val_meg: dict, cache_summaries: dict, bridge: dict) -> list[CheckResult]:
    """Full enumeration over all 27 MEG shards.
    Also collects metadata that Layer 4 uses (vocab histograms, trial counts)."""
    import numpy as np
    _section("Layer 3 — Data integrity (full enumeration)")
    results: list[CheckResult] = []

    bad_filenames: list[tuple[str, str]] = []
    bad_shapes: list[tuple[str, str, tuple]] = []
    bad_dtypes: list[tuple[str, str, str]] = []
    out_of_range: list[tuple[str, str, int, int]] = []
    keys_seen: set[tuple[str, str, str, int]] = set()  # (shard, image, subject, trial_idx)
    duplicates: list[tuple] = []

    # Stash all loaded tokens keyed by (image_id, subject, trial_idx) so L3.5
    # can do a random round-trip sample without re-reading tars.
    shard_tokens_lookup: dict[tuple[str, str, int], np.ndarray] = {}

    # Per-quantizer code histogram (4 layers × 512 codes).
    code_seen = np.zeros((EXPECTED_TOKEN_SHAPE[2], EXPECTED_CODEBOOK_SIZE), dtype=bool)

    n_total_entries = 0
    for split_name, meg_manifest, shard_dir in (
        ("train", train_meg, TRAIN_MEG_DIR),
        ("val", val_meg, VAL_MEG_DIR),
    ):
        for shard_id in sorted(meg_manifest["shards"]):
            tar_path = os.path.join(shard_dir, f"{shard_id}.tar")
            for name, arr in _enumerate_meg_shard(tar_path):
                n_total_entries += 1
                parsed = meg_token_shard.parse_meg_filename(name)
                if parsed is None:
                    bad_filenames.append((f"{split_name}/{shard_id}", name))
                    continue
                image_id, subject, trial_idx = parsed
                if arr.shape != EXPECTED_TOKEN_SHAPE:
                    bad_shapes.append((shard_id, name, arr.shape))
                    continue
                if arr.dtype != np.int16:
                    bad_dtypes.append((shard_id, name, str(arr.dtype)))
                    continue
                a_min, a_max = int(arr.min()), int(arr.max())
                if a_min < 0 or a_max >= EXPECTED_CODEBOOK_SIZE:
                    out_of_range.append((shard_id, name, a_min, a_max))
                key = (shard_id, image_id, subject, trial_idx)
                if key in keys_seen:
                    duplicates.append(key)
                else:
                    keys_seen.add(key)
                shard_tokens_lookup[(image_id, subject, trial_idx)] = arr
                # Update per-quantizer codebook histogram.
                # arr shape: (16=C, 8=T, 4=Q). For each q, accumulate seen codes.
                for q in range(EXPECTED_TOKEN_SHAPE[2]):
                    code_seen[q, np.unique(arr[:, :, q])] = True

    # L3.1 / L3.2 / L3.3 collated:
    if bad_filenames:
        results.append(_fail("L3.1 filename parse", f"{len(bad_filenames)}; first: {bad_filenames[0]}"))
    else:
        results.append(_ok("L3.1 filename parse", f"{n_total_entries} entries parse ok"))

    if bad_shapes:
        results.append(_fail("L3.2 token shape", f"{len(bad_shapes)}; first: {bad_shapes[0]}"))
    else:
        results.append(_ok("L3.2 token shape", f"all (16,8,4)"))
    if bad_dtypes:
        results.append(_fail("L3.2 token dtype", f"{len(bad_dtypes)}; first: {bad_dtypes[0]}"))
    else:
        results.append(_ok("L3.2 token dtype", "all int16"))

    if out_of_range:
        results.append(_fail("L3.3 vocab range [0,512)", f"{len(out_of_range)}; first: {out_of_range[0]}"))
    else:
        results.append(_ok("L3.3 vocab range [0,512)", "all entries in range"))

    # L3.4: duplicate (image_id, subject, trial_idx) anywhere
    # We tracked per-shard duplicates; also check across shards.
    cross_shard_keys: dict[tuple[str, str, int], str] = {}
    cross_dups: list = []
    for shard_id, iid, subj, t in keys_seen:
        k2 = (iid, subj, t)
        if k2 in cross_shard_keys:
            cross_dups.append((k2, cross_shard_keys[k2], shard_id))
        else:
            cross_shard_keys[k2] = shard_id
    all_dups = duplicates + cross_dups
    if all_dups:
        results.append(_fail("L3.4 no duplicate (image, subject, trial)", f"{len(all_dups)}; first: {all_dups[0]}"))
    else:
        results.append(_ok("L3.4 no duplicate (image, subject, trial)", f"{len(cross_shard_keys)} unique keys"))

    # L3.5: cache↔shard round-trip on a random sample.
    import random
    rng = random.Random(0)
    keys_sample = rng.sample(sorted(cross_shard_keys.keys()), k=min(SAMPLE_ROUNDTRIP_N, len(cross_shard_keys)))
    mismatches: list = []
    bridge_image_id_to_trigger = {v: int(k) for k, v in bridge["trigger_to_image_id"].items()}
    for image_id, subject, trial_idx in keys_sample:
        trigger = bridge_image_id_to_trigger.get(image_id)
        if trigger is None:
            mismatches.append((image_id, subject, trial_idx, "no trigger"))
            continue
        cache = cache_summaries[subject]
        # Find the index in cache where (trigger, trial_idx) match.
        mask = (cache["triggers_arr"] == trigger) & (cache["trial_idx_arr"] == trial_idx)
        positions = np.flatnonzero(mask)
        if len(positions) != 1:
            mismatches.append((image_id, subject, trial_idx, f"{len(positions)} cache hits"))
            continue
        cache_tokens = cache["tokens_arr"][int(positions[0])]
        shard_tokens = shard_tokens_lookup[(image_id, subject, trial_idx)]
        if not np.array_equal(cache_tokens, shard_tokens):
            mismatches.append((image_id, subject, trial_idx, "tokens differ"))
    if mismatches:
        results.append(_fail("L3.5 cache↔shard round-trip", f"{len(mismatches)}/{len(keys_sample)}; first: {mismatches[0]}"))
    else:
        results.append(_ok("L3.5 cache↔shard round-trip", f"{len(keys_sample)} samples byte-identical"))

    # Stash the code histogram for L4.4.
    layer3_integrity._code_seen = code_seen  # type: ignore[attr-defined]
    layer3_integrity._n_entries = n_total_entries  # type: ignore[attr-defined]
    return results


def layer4_protocol(train_meg: dict, val_meg: dict, cache_summaries: dict, bridge: dict) -> list[CheckResult]:
    import numpy as np
    _section("Layer 4 — Domain-level (THINGS-MEG protocol)")
    results: list[CheckResult] = []

    # L4.1: total trials
    total = train_meg["n_entries"] + val_meg["n_entries"]
    if total == EXPECTED_TOTAL_TRIALS:
        results.append(_ok("L4.1 total trials", f"{total}"))
    else:
        results.append(_fail("L4.1 total trials", f"{total}, expected {EXPECTED_TOTAL_TRIALS}"))

    # L4.2: per-subject trials in cache
    for s in SUBJECTS:
        n = cache_summaries[s]["n"]
        if n == EXPECTED_PER_SUBJECT_TRIALS:
            results.append(_ok(f"L4.2 cache {s} count", f"{n}"))
        else:
            results.append(_fail(f"L4.2 cache {s} count", f"{n}, expected {EXPECTED_PER_SUBJECT_TRIALS}"))

    # L4.3: trial-count-per-(image, subject) distribution.
    # Iterate triggers_arr (raw per-trial), not triggers (set), so test
    # repeats produce >1 count per (trigger, subject).
    pair_counts: Counter = Counter()
    for s in SUBJECTS:
        for t in cache_summaries[s]["triggers_arr"]:
            pair_counts[(int(t), s)] += 1
    counts_only = list(pair_counts.values())
    dist = Counter(counts_only)
    n_singletons = dist.get(1, 0)
    # exp images: ~22,248 × 4 subj × 1 trial = 88,992 singletons
    # test images: ~200 × 4 subj × ~12 trials = many higher-count pairs
    n_high = sum(c for v, c in dist.items() if v >= 10)
    if n_singletons > 80_000 and n_high > 500:
        results.append(_ok("L4.3 trial-count distribution", f"singletons={n_singletons}, ≥10-trial pairs={n_high}"))
    else:
        results.append(_fail("L4.3 trial-count distribution", f"singletons={n_singletons}, ≥10-trial pairs={n_high}"))
    # Also print the top values for visibility.
    print(f"    (top trial-counts/pair: {sorted(dist.items())[:5]}...{sorted(dist.items())[-3:]})")

    # L4.4: codebook utilization per quantizer layer
    code_seen = getattr(layer3_integrity, "_code_seen", None)
    if code_seen is None:
        results.append(_fail("L4.4 codebook utilization", "code_seen not computed (Layer 3 didn't run?)"))
    else:
        per_layer_used = code_seen.sum(axis=1)
        unused_per_layer = EXPECTED_CODEBOOK_SIZE - per_layer_used
        if (unused_per_layer == 0).all():
            results.append(_ok("L4.4 codebook utilization", f"all 512 codes seen on all {EXPECTED_NUM_QUANTIZERS} layers"))
        else:
            results.append(_fail(
                "L4.4 codebook utilization",
                f"dead codes per layer: {unused_per_layer.tolist()}"
            ))

    return results


# ---------- master runner ------------------------------------------------

@app.function(
    image=audit_image,
    volumes={PROJECT_MOUNT: project_volume},
    cpu=4.0,
    memory=16 * 1024,
    timeout=60 * 30,
)
def audit_remote() -> dict:
    import numpy as np

    print("=" * 60)
    print("MEG TOKENIZATION PIPELINE — FULL AUDIT")
    print("=" * 60)

    catalog = _load_catalog()
    bridge = _load_bridge()
    train_rgb, val_rgb, train_meg, val_meg = _load_manifests()

    cache_summaries: dict = {}
    print("\n[load] reading per-subject token caches...")
    for s in SUBJECTS:
        arr = _load_cache_arrays(s)
        triggers = arr["meg_trigger_codes"]
        trial_idx = arr["trial_idx"]
        cache_summaries[s] = {
            "n": int(arr["tokens"].shape[0]),
            "triggers": set(int(t) for t in triggers),
            "triggers_arr": triggers,
            "trial_idx_arr": trial_idx,
            "tokens_arr": arr["tokens"],
        }
        print(f"  {s}: {cache_summaries[s]['n']} trials, {len(cache_summaries[s]['triggers'])} unique triggers")

    results: list[CheckResult] = []
    results.extend(layer1_structural())
    results.extend(layer2_consistency(
        catalog, bridge, train_rgb, val_rgb, train_meg, val_meg, cache_summaries
    ))
    results.extend(layer3_integrity(train_meg, val_meg, cache_summaries, bridge))
    results.extend(layer4_protocol(train_meg, val_meg, cache_summaries, bridge))

    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)
    failures = [r for r in results if not r.passed]

    print("\n" + "=" * 60)
    print(f"SUMMARY  {n_pass} pass  /  {n_fail} fail   ({len(results)} total checks)")
    print("=" * 60)
    if failures:
        print("\nFAILED CHECKS:")
        for r in failures:
            print(f"  {r.name}: {r.message}")
        print("\nRESULT: AUDIT FAILED — fix above before walking away from the pipeline.")
    else:
        print("\nRESULT: AUDIT PASSED — pipeline is safe to leave behind.")

    return {
        "n_pass": n_pass,
        "n_fail": n_fail,
        "results": [(r.name, r.passed, r.message) for r in results],
    }


@app.local_entrypoint()
def audit():
    summary = audit_remote.remote()
    print(f"\n[local] {summary['n_pass']} pass / {summary['n_fail']} fail")
    if summary["n_fail"] > 0:
        raise SystemExit(1)
