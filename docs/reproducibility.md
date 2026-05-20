# Reproducibility Guide

This project keeps runtime reproducibility lightweight and local-first. The default environment is CPU-only, with deep learning experiment execution kept in notebooks under `notebooks/*.ipynb`.

## Python Runtime

- Project declaration: Python `>=3.10`.
- Validated local runtime: Python `3.13.7`.
- Validated local platform: Windows 11.
- Validated deep learning backend: PyTorch CPU runtime. The local build reports `torch 2.10.0+cpu`, while the pinned installation manifest uses `torch==2.10.0`.
- CUDA/GPU is not required for default training, evaluation, API inference, tests, or notebook validation.

## Dependency Pinning

Install the exact pinned environment with:

```powershell
python -m pip install -r requirements.txt
```

`pyproject.toml` remains the package manifest for dependency groups, while `requirements.txt` records exact reproducibility pins for runtime, test, research, and notebook dependencies.

## Random Seeds

Use `traffic_prediction.config.set_global_determinism(seed)` at the start of scripts, tests, notebooks, and one-off experiments that need repeatable results.

The helper sets:

- `PYTHONHASHSEED`
- Python `random`
- NumPy random state
- PyTorch random state when PyTorch is installed
- PyTorch deterministic algorithms with warning-only mode

For notebook-based deep learning work, place the seed call in the first executable setup cell before data loading, model creation, or training.

## CPU-Only Assumptions

The default deployment target assumes:

- No GPU is available.
- Model artifacts are loaded on CPU.
- API inference should stay lightweight.
- Deep learning validation remains notebook-first, not a standalone `.py` runner.

These assumptions match the current project direction: a monolithic, CPU-friendly traffic prediction service with reproducible offline notebooks and local artifact storage.
