from __future__ import annotations

import re
import tomllib
from pathlib import Path


def test_pyproject_declares_required_dependency_manifest() -> None:
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    declared = _all_declared_dependencies(payload)

    required = {
        "apscheduler",
        "catboost",
        "fastapi",
        "httpx",
        "hypothesis",
        "lightgbm",
        "matplotlib",
        "numpy",
        "pandas",
        "pydantic",
        "pytest",
        "requests",
        "scikit-learn",
        "scipy",
        "seaborn",
        "statsmodels",
        "torch",
        "uvicorn",
        "xgboost",
    }

    assert required <= declared


def test_pyproject_uses_pytorch_as_the_deep_learning_framework() -> None:
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    declared = _all_declared_dependencies(payload)

    assert "torch" in declared
    assert "tensorflow" not in declared


def test_requirements_file_pins_declared_dependency_versions() -> None:
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    declared = _all_declared_dependencies(payload)
    pinned = _all_pinned_requirements(Path("requirements.txt"))

    assert declared <= set(pinned)
    assert all(specifier.startswith("==") for specifier in pinned.values())


def test_reproducibility_document_records_runtime_assumptions() -> None:
    docs = Path("docs/reproducibility.md").read_text(encoding="utf-8")

    assert "Python `3.13.7`" in docs
    assert "CPU-only" in docs
    assert "requirements.txt" in docs
    assert "set_global_determinism" in docs


def _all_declared_dependencies(payload: dict) -> set[str]:
    dependencies = list(payload["project"].get("dependencies", []))
    optional = payload["project"].get("optional-dependencies", {})
    for group in optional.values():
        dependencies.extend(group)
    return {_normalize_requirement_name(item) for item in dependencies}


def _normalize_requirement_name(requirement: str) -> str:
    name = re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def _all_pinned_requirements(path: Path) -> dict[str, str]:
    pinned: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, specifier = re.split(r"(?===)", line, maxsplit=1)
        pinned[_normalize_requirement_name(name)] = specifier
    return pinned
