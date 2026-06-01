"""Identity check with correct legacy→cache key: (image_id, source, subject), trial_idx=0."""
import io, json, os, random, re, tarfile, glob
import modal
app = modal.App("verify-eeg-id2")
vol = modal.Volume.from_name("project")
P = "/project"
CACHE = f"{P}/data/things-eeg/tokens/labram/V8192_d64_ch17_sr200_train-eeg1+2_e5"
TRAIN = f"{P}/data/train/things/tok_eeg"
VAL = f"{P}/data/val/things/tok_eeg"
RE = re.compile(r"^(?P<iid>\d{9})_(?P<ds>eeg[12])sub(?P<sub>\d+)_t(?P<t>\d+)\.eeg\.npy$")
img = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")

@app.function(image=img, volumes={P: vol}, timeout=180, memory=16*1024)
def verify():
    import numpy as np
    cache = {}
    for p in glob.glob(f"{CACHE}/*.npz"):
        z = np.load(p)
        cache[(str(z["source"]), str(z["subject"]))] = z

    def cache_token(iid, ds, sub_num):
        subj = f"sub-{int(sub_num):02d}"
        z = cache[(ds, subj)]
        hits = np.flatnonzero((z["image_id"] == iid) & (z["trial_idx"] == 0))
        if len(hits) != 1:
            return None, len(hits)
        return z["tokens"][int(hits[0])], 1

    rng = random.Random(0)
    checked = match = mismatch = missing = 0

    # sample val loose
    val_names = [n for n in os.listdir(VAL) if n.endswith(".eeg.npy")]
    for name in rng.sample(val_names, 200):
        m = RE.match(name)
        with open(os.path.join(VAL, name), "rb") as f:
            le = np.load(io.BytesIO(f.read()), allow_pickle=False)
        ca, nh = cache_token(m.group("iid"), m.group("ds"), m.group("sub"))
        checked += 1
        if nh != 1:
            missing += 1
        elif np.array_equal(le, ca):
            match += 1
        else:
            mismatch += 1

    # exhaustive shard_000
    with tarfile.open(f"{TRAIN}/shard_000.tar") as tar:
        for member in tar:
            if not member.name.endswith(".eeg.npy"):
                continue
            m = RE.match(member.name)
            le = np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=False)
            ca, nh = cache_token(m.group("iid"), m.group("ds"), m.group("sub"))
            checked += 1
            if nh != 1:
                missing += 1
            elif np.array_equal(le, ca):
                match += 1
            else:
                mismatch += 1

    print(f"checked={checked} match={match} mismatch={mismatch} missing={missing}")
    ok = mismatch == 0 and missing == 0
    print("PASS" if ok else "FAIL")
    return {"ok": ok, "checked": checked, "match": match, "mismatch": mismatch, "missing": missing}

@app.local_entrypoint()
def main():
    r = verify.remote()
    print(r)
    if not r["ok"]:
        raise SystemExit(1)
