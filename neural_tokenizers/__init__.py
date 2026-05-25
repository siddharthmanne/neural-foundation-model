"""neural_tokenizers — per-modality tokenizers for the mini-4M pipeline.

This file exists so `neural_tokenizers` is a *regular* package, not a
namespace package. Subtle implications:
  - `import neural_tokenizers` resolves to a single, deterministic path
    (namespace packages aggregate every matching directory on sys.path,
    which makes `add_local_python_source("neural_tokenizers")` ship a
    confusing union when the repo is checked out twice on the same machine).
  - Editable installs / Modal's source packing behave predictably.

We deliberately do NOT re-export anything from here. Each modality subpackage
(meg, stubs, …) and the eval harness expose their own public APIs via their
own __init__.py — that keeps import surfaces small and lets you delete a
subpackage without rippling through this file.
"""
