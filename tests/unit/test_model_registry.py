from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from traffic_prediction.models.registry import ModelRegistry


def make_test_root() -> Path:
    root = Path("artifacts") / "test_runs" / "model_registry" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_artifact(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "feature_manifest.json").write_text("{}", encoding="utf-8")
    (path / "scaler_params.joblib").write_text("scaler", encoding="utf-8")
    return path


def test_registry_registers_and_activates_model() -> None:
    root = make_test_root()
    artifact = make_artifact(root / "models" / "offline-1")
    registry = ModelRegistry(root / "models" / "registry.json")

    entry = registry.register(
        artifact_path=artifact,
        model_version="v1",
        metrics={"mae": 1.2},
        config={"lookback": 12},
        activate=True,
    )

    assert entry.model_version == "v1"
    assert registry.get_active().model_version == "v1"
    assert registry.get("v1").metrics["mae"] == 1.2
    assert (root / "models" / "latest_model.txt").read_text(encoding="utf-8") == "offline-1"


def test_registry_generates_timestamped_version_when_missing() -> None:
    root = make_test_root()
    artifact = make_artifact(root / "models" / "candidate")
    registry = ModelRegistry(root / "models" / "registry.json")

    entry = registry.register(artifact_path=artifact, model_type="lstm")

    assert entry.model_version.startswith("lstm-")
    assert len(entry.model_version) >= len("lstm-20260519-130000")


def test_registry_prevents_duplicate_versions() -> None:
    root = make_test_root()
    artifact = make_artifact(root / "models" / "offline-1")
    registry = ModelRegistry(root / "models" / "registry.json")

    registry.register(artifact, model_version="v1")

    with pytest.raises(ValueError, match="already registered"):
        registry.register(artifact, model_version="v1")


def test_registry_discovers_and_activates_latest_offline_artifact() -> None:
    root = make_test_root()
    models_dir = root / "models"
    first = make_artifact(models_dir / "offline-1")
    second = make_artifact(models_dir / "offline-2")
    first_time = 1_700_000_000
    second_time = first_time + 10
    for path in [first, first / "feature_manifest.json", first / "scaler_params.joblib"]:
        os.utime(path, (first_time, first_time))
    for path in [second, second / "feature_manifest.json", second / "scaler_params.joblib"]:
        os.utime(path, (second_time, second_time))
    registry = ModelRegistry(models_dir / "registry.json")

    registered = registry.register_discovered_offline_artifacts(models_dir)

    assert [entry.model_version for entry in registered] == ["offline-1", "offline-2"]
    assert registry.get_active().model_version == "offline-2"
    payload = json.loads((models_dir / "registry.json").read_text(encoding="utf-8"))
    assert payload["active_model_version"] == "offline-2"
    assert (models_dir / "latest_model.txt").read_text(encoding="utf-8") == "offline-2"


def test_registry_rollback_reactivates_previous_model() -> None:
    root = make_test_root()
    models_dir = root / "models"
    first = make_artifact(models_dir / "offline-1")
    second = make_artifact(models_dir / "offline-2")
    registry = ModelRegistry(models_dir / "registry.json")
    registry.register(first, model_version="v1", activate=True)
    registry.register(second, model_version="v2", activate=True)

    active = registry.rollback("v1")

    assert active.model_version == "v1"
    assert registry.get_active().model_version == "v1"
    assert (models_dir / "latest_model.txt").read_text(encoding="utf-8") == "offline-1"


def test_registry_resolve_prefers_configured_version() -> None:
    root = make_test_root()
    models_dir = root / "models"
    first = make_artifact(models_dir / "offline-1")
    second = make_artifact(models_dir / "offline-2")
    registry = ModelRegistry(models_dir / "registry.json")
    registry.register(first, model_version="v1", activate=False)
    registry.register(second, model_version="v2", activate=True)

    assert registry.resolve("v1").model_version == "v1"
    assert registry.resolve().model_version == "v2"
