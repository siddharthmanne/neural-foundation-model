"""
Modal wrapper that runs `demo_train.py` on a remote Modal container.

This is a TEMPLATE you can copy and modify for your own training jobs.

To make your own Modal training job:
  1. Write your training code as a plain Python file (see `demo_train.py`
     for the pattern — argparse, no Modal imports).
  2. Copy this file to `modal_<your_thing>.py`.
  3. Replace "demo_train" below with the name of your script.
  4. Update the subprocess.run command to pass the args your script wants.
  5. Adjust `gpu=` and `timeout=` for your workload (or remove `gpu=` if CPU is fine).
  6. Run it:
       modal run modal_<your_thing>.py            # for short jobs
       modal run --detach modal_<your_thing>.py   # for anything longer than ~10 min;
                                                  # the job keeps running on Modal's
                                                  # servers even if you close your
                                                  # terminal or your laptop sleeps.

What this wrapper does:
  - Adds your training script to the container's image (`add_local_python_source`).
  - Spins up a container with the right GPU and the shared /project Volume mounted.
  - Runs your script inside the container via subprocess.
  - Commits any /project writes back to the Volume so they persist.
"""

import subprocess

from modal_app import app, image, data_volume

# Ship the training script into the container's image.
# If you rename your script, change the name on this line.
training_image = image.add_local_python_source("demo_train")


@app.function(
    image=training_image,
    volumes={"/project": data_volume},
    # If you need an API token (HF, W&B, etc.), create your own personal Modal
    # secret first — see modal_app.py docstring — and reference it here:
    #     import modal
    #     my_hf = modal.Secret.from_name("hf-yourname")
    #     ...then add `secrets=[my_hf],` to this decorator.
    # gpu="A10",                # uncomment if you need a GPU (T4 / L40S / A10 / A100 / H100)
    timeout=60 * 5,             # 5 minutes — bump way up for real training
)
def run_demo():
    """Run demo_train.py inside the container.

    The subprocess.run call below is equivalent to typing this on a
    command line:

        python -u -m demo_train --output-dir /project/demo_output --steps 3

    except it's happening inside the Modal container instead of on
    your laptop. Each list item is one space-separated argument.
    """
    subprocess.run(
        [
            "python",                            
            "-u",                                # unbuffered output, so logs stream live
            "-m", "demo_train",                  # run the demo_train module
                                                 # (matches `add_local_python_source("demo_train")` above)
            "--output-dir", "/project/demo_output", # passed to demo_train.py's argparse
            "--steps", "3",                      # passed to demo_train.py's argparse
        ],
        check=True,  # raise an exception if the script returns a non-zero exit code
    )
    # Flush /project writes back to the Volume so they persist after the container ends.
    data_volume.commit()


@app.local_entrypoint()
def main():
    run_demo.remote()
