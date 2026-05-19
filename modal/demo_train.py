"""
Demo "training" script: pure Python, no Modal anywhere!

This file exists to demonstrate the pattern: write your real training
code as a plain Python script (with argparse, no Modal imports), then
have a small Modal wrapper file (`modal_demo.py`) that runs THIS script
on a remote container.

This is for demonstrating how you can write your training code as a plain Python 
script and then have a Modal wrapper file that runs this scfript on a remote container.

So:

- You can run this directly with:
    python demo_train.py --output-dir /tmp/demo_out --steps 3

- Or you can run it on Modal via the wrapper:
    modal run modal_demo.py

Either way the code is identical. That is to say that your training
script doesn't need to know it's running on Modal :)
"""

import argparse
import time
from pathlib import Path


def main(args):
    print(f"Starting demo 'training' with {args.steps} steps...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pretend to train. Replace this with your actual training loop.
    for step in range(1, args.steps + 1):
        print(f"  Step {step}/{args.steps}")
        time.sleep(0.5)

    # Write a fake "trained model" file to the output directory.
    result_file = output_dir / "result.txt"
    result_file.write_text(f"Trained for {args.steps} steps.\n")
    print(f"Done. Wrote {result_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True,
                   help="Directory to write 'trained' output to")
    p.add_argument("--steps", type=int, default=5,
                   help="Pretend to train for this many steps")
    main(p.parse_args())
