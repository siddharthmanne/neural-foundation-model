"""Diagnose legacy vs LaBraM cache key alignment."""
import io, json, os, re, tarfile, glob
import modal
app = modal.App("diag-eeg-keys")
vol = modal.Volume.from_name("project")
P = "/project"
CACHE = f"{P}/data/things-eeg/tokens/labram/V8192_d64_ch17_sr200_train-eeg1+2_e5"
RE = re.compile(r"^(?P<iid>\d{9})_(?P<ds>eeg[12])sub(?P<sub>\d+)_t(?P<t>\d+)\.eeg\.npy$")
img = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")

@app.function(image=img, volumes={P: vol}, timeout=120, memory=16*1024)
def diag():
    import numpy as np
    z = np.load(f"{CACHE}/eeg1_sub-01.npz")
    iid = "000000001"
    mask = z["image_id"] == iid
    print("=== cache rows for 000000001 eeg1 sub-01 ===")
    idxs = np.flatnonzero(mask)
    print("n_rows", len(idxs), "trial_idx values", sorted(set(int(x) for x in z["trial_idx"][mask])))

    # legacy file from train shard_000
    tar = tarfile.open(f"{P}/data/train/things/tok_eeg/shard_000.tar")
    legacy = {}
    for m in tar:
        if not m.name.startswith(iid + "_"):
            continue
        if m.name.endswith(".eeg.npy"):
            legacy[m.name] = np.load(io.BytesIO(tar.extractfile(m).read()), allow_pickle=False)
    print("\n=== legacy files for 000000001 ===")
    for name, arr in sorted(legacy.items()):
        print(name, arr.shape, arr[:3].tolist())

    # try match legacy eeg1sub01_t0 to cache by image+subject only
    m = RE.match("000000001_eeg1sub01_t0.eeg.npy")
    subj = f"sub-{int(m.group('sub')):02d}"
    rows = np.flatnonzero((z["image_id"]==iid) & (z["trial_idx"]==0))
    print("\n=== compare eeg1sub01_t0 to cache trial_idx=0 rows ===")
    print("n cache rows with trial_idx=0:", len(rows))
    if len(rows):
        ca = z["tokens"][int(rows[0])]
        le = legacy.get("000000001_eeg1sub01_t0.eeg.npy")
        print("equal?", le is not None and np.array_equal(le, ca))

    # mismatch example eeg2
    name = "000020102_eeg2sub03_t2.eeg.npy"
    z2 = np.load(f"{CACHE}/eeg2_sub-03.npz")
    m = RE.match(name)
    iid2 = m.group("iid")
    t = int(m.group("t"))
    with tarfile.open(f"{P}/data/train/things/tok_eeg/shard_020.tar") as tar:
        f = tar.extractfile(name)
        le = np.load(io.BytesIO(f.read()), allow_pickle=False) if f else None
    hits = np.flatnonzero((z2["image_id"]==iid2) & (z2["trial_idx"]==t))
    print(f"\n=== mismatch probe {name} ===")
    print("legacy", le[:5] if le is not None else None)
    print("cache hits for same trial_idx", len(hits))
    if len(hits):
        ca = z2["tokens"][int(hits[0])]
        print("cache", ca[:5], "equal?", np.array_equal(le, ca))
    # all rows for this image in cache
    mask2 = z2["image_id"] == iid2
    print("cache n_rows for image", mask2.sum(), "trial_idxs", sorted(set(int(x) for x in z2["trial_idx"][mask2])))

    # trial_idx semantics in cache globally
    print("\n=== cache trial_idx distribution (eeg1 sub-01 sample) ===")
    print("unique trial_idx count", len(set(int(x) for x in z["trial_idx"])))
    from collections import Counter
    c = Counter(int(x) for x in z["trial_idx"])
    print("top trial_idx", c.most_common(10))

@app.local_entrypoint()
def main():
    diag.remote()
