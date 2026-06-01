"""Peek LaBraM EEG cache npz keys (one-off)."""
import modal
app = modal.App("peek-eeg-npz")
vol = modal.Volume.from_name("project")
img = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
CACHE = "/project/data/things-eeg/tokens/labram/V8192_d64_ch17_sr200_train-eeg1+2_e5"

@app.function(image=img, volumes={"/project": vol}, timeout=120)
def peek():
    import numpy as np, os, glob
    paths = sorted(glob.glob(f"{CACHE}/eeg1_sub-01.npz"))
    if not paths:
        paths = sorted(glob.glob(f"{CACHE}/eeg1_sub-*.npz"))[:1]
    p = paths[0]
    z = np.load(p)
    print("file:", p)
    print("keys:", list(z.files))
    for k in z.files:
        v = z[k]
        print(f"  {k}: shape={getattr(v,'shape',None)} dtype={getattr(v,'dtype',type(v))}")

@app.local_entrypoint()
def main():
    peek.remote()
