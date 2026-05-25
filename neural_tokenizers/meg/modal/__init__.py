"""Modal wrappers for MEG tokenizer phases.

Each script in this directory is a thin wrapper that:
  1. Spins up a Modal container with the `project` Volume mounted.
  2. Imports the tokenizer / calibration / harness code from
     `neural_tokenizers.meg.*` (pure Python, runnable on a laptop too).
  3. Executes the run-on-real-data version of what tests cover on synthetic.

Generic vs phase-specific:
  - modal_meg_eval.py        — GENERIC, --tokenizer <name>. Adds new phases
                                with a small dispatch entry, not a new file.
  - modal_meg_calibrate_mu.py — Phase-1 specific (calibration is per-phase prep).
"""
