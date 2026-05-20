"""Helpers for reproducible local runs."""

from __future__ import annotations

import os
import platform
import random
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReproducibilitySummary:
    """Runtime summary after deterministic seed setup."""

    seed: int
    python_version: str
    platform: str
    numpy_version: str
    torch_version: str | None
    cuda_available: bool
    cpu_only: bool
    deterministic_torch: bool


def set_global_determinism(seed: int, deterministic_torch: bool = True) -> ReproducibilitySummary:
    """Set deterministic seeds for Python, NumPy, and optional PyTorch."""

    if seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_version: str | None = None
    cuda_available = False

    try:
        import torch

        torch_version = str(torch.__version__)
        torch.manual_seed(seed)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

    return ReproducibilitySummary(
        seed=seed,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        numpy_version=np.__version__,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cpu_only=not cuda_available,
        deterministic_torch=deterministic_torch,
    )


def cpu_runtime_assumptions() -> dict[str, Any]:
    """Return the default runtime assumptions for deployment documentation."""

    return {
        "runtime": "cpu-only",
        "requires_gpu": False,
        "deep_learning_execution": "notebook-first",
        "model_device": "cpu",
        "dependency_lockfile": "requirements.txt",
    }


__all__ = [
    "ReproducibilitySummary",
    "cpu_runtime_assumptions",
    "set_global_determinism",
]
