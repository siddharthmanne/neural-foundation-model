"""
Shared Modal setup for the neural-fm project.

Defines the App, Image, and Volume that other files in this folder import
from. Run `modal run modal_app.py::hello` to sanity-check your setup.

Secrets are NOT defined here on purpose. Modal secrets are workspace-scoped,
so anything we define here is usable (and effectively readable) by every
member of the neural-fm workspace. Bad for personal credentials like HF
tokens. Each teammate creates their own personally-named secret:

    modal secret create hf-brandon HF_TOKEN=hf_xxxxx
    modal secret create hf-sid HF_TOKEN=hf_xxxxx
    modal secret create hf-liane HF_TOKEN=hf_xxxxx

Then reference it in your own Modal file via:

    import modal
    my_hf = modal.Secret.from_name("hf-yourname")

    @app.function(image=image, secrets=[my_hf], ...)
    def my_job(): ...

See MODAL_GUIDE.md for the pattern.
"""

import modal

# Our Modal App, all our runs group under this name in the dashboard.
app = modal.App("neural-fm")

# The container Image: a Linux box with Python and our dependencies. Please add dependencies
# as needed. Modal builds this once and caches it across runs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
)

# One persistent Volume for everything we want to keep across runs:
# datasets, pretrained weights, training outputs. Mounted at /data.
data_volume = modal.Volume.from_name("neural-fm-data", create_if_missing=True)

@app.function(image=image, volumes={"/data": data_volume})
def hello():
    """Sanity check. Run: modal run modal_app.py::hello"""
    import sys
    import os
    print(f"Hello from Modal! Python {sys.version.split()[0]}")
    files = os.listdir("/data") if os.path.exists("/data") else []
    print(f"Files in /data on the shared Volume: {sorted(files)}")


@app.local_entrypoint()
def main():
    hello.remote()
