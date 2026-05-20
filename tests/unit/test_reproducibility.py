from __future__ import annotations

import os
import random

import numpy as np
import pytest

from traffic_prediction.config import cpu_runtime_assumptions, set_global_determinism


def test_set_global_determinism_repeats_python_and_numpy_random_values() -> None:
    first_summary = set_global_determinism(123)
    first_python = random.random()
    first_numpy = np.random.random()

    second_summary = set_global_determinism(123)
    second_python = random.random()
    second_numpy = np.random.random()

    assert first_python == second_python
    assert first_numpy == second_numpy
    assert first_summary.seed == second_summary.seed == 123
    assert os.environ["PYTHONHASHSEED"] == "123"
    assert isinstance(first_summary.cpu_only, bool)
    assert first_summary.python_version
    assert first_summary.numpy_version


def test_set_global_determinism_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_global_determinism(-1)


def test_cpu_runtime_assumptions_are_explicit() -> None:
    assumptions = cpu_runtime_assumptions()

    assert assumptions["runtime"] == "cpu-only"
    assert assumptions["requires_gpu"] is False
    assert assumptions["deep_learning_execution"] == "notebook-first"
    assert assumptions["dependency_lockfile"] == "requirements.txt"
