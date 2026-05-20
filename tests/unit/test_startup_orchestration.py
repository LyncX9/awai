from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from traffic_prediction.api.app import AppState
from traffic_prediction.config.settings import load_config
from traffic_prediction.models.registry import ModelRegistry
from traffic_prediction.orchestration.startup import build_startup_report


from unittest.mock import patch

def test_startup_report_requires_core_runtime_artifacts() -> None:
    report = build_startup_report(
        roads_loaded=True,
        model_loaded=True,
        buffer_available=True,
        tomtom_configured=False,
        scheduler_registered=True,
        scheduler_running=False,
        scheduler_enabled=False,
        model_version="model-v1",
    )

    assert report.ready is True
    assert report.by_name()["tomtom"].critical is False
    assert report.by_name()["tomtom"].ready is False
    assert report.by_name()["scheduler"].status == "disabled"
    assert report.details()["model"] == "active model loaded: model-v1"


def test_startup_report_blocks_readiness_when_model_is_missing() -> None:
    report = build_startup_report(
        roads_loaded=True,
        model_loaded=False,
        buffer_available=True,
        tomtom_configured=True,
        scheduler_registered=True,
        scheduler_running=False,
        scheduler_enabled=False,
    )

    assert report.ready is False
    assert report.by_name()["model"].status == "missing"


@patch("traffic_prediction.api.app.PyTorchModelRunner.load_from_artifact")
def test_app_state_registers_scheduler_without_autostart_by_default(mock_load) -> None:
    config = load_config(project_root=".", load_dotenv_file=False)
    models_dir = Path("artifacts/test_runs/startup_orchestration") / uuid4().hex / "models"
    _make_model_artifact(models_dir / "offline-mock")
    config = replace(config, paths=replace(config.paths, models_dir=models_dir.resolve()))
    state = AppState(config)

    state.load_static_resources()
    report = state.readiness_report()

    assert "tomtom_ingestion" in state.scheduler.jobs
    assert "buffer_persistence" in state.scheduler.jobs
    assert state.scheduler.running is False
    assert report.by_name()["scheduler"].status == "disabled"
    assert report.by_name()["restart_recovery"].status == "recovered"
    assert state.restart_recovery["buffer_source"] in {"persisted_snapshot", "history_seeded"}
    assert state.restart_recovery["model_source"] in {"registry", "latest_pointer"}
    assert state.restart_recovery["scheduler_job_count"] >= 6
    assert report.ready is True


def test_app_state_keeps_scheduler_stopped_when_critical_artifacts_are_missing() -> None:
    config = load_config(project_root=".", load_dotenv_file=False)
    missing_model_root = Path("artifacts/test_runs/startup_orchestration") / uuid4().hex / "models"
    missing_model_root.mkdir(parents=True, exist_ok=True)
    config = replace(
        config,
        paths=replace(config.paths, models_dir=missing_model_root.resolve()),
        runtime=replace(config.runtime, scheduler_enabled=True),
    )
    state = AppState(config)

    state.load_static_resources()
    report = state.readiness_report()

    assert state.model_loaded is False
    assert state.scheduler.running is False
    assert report.ready is False
    assert report.by_name()["scheduler"].status == "stopped"


@patch("traffic_prediction.api.app.PyTorchModelRunner.load_from_artifact")
def test_app_state_uses_configured_registry_model_version(mock_load) -> None:
    config = load_config(project_root=".", load_dotenv_file=False)
    models_dir = Path("artifacts/test_runs/startup_orchestration") / uuid4().hex / "models"
    first = _make_model_artifact(models_dir / "offline-1")
    second = _make_model_artifact(models_dir / "offline-2")
    registry = ModelRegistry(models_dir / "registry.json")
    registry.register(first, model_version="v1", activate=False)
    registry.register(second, model_version="v2", activate=True)
    config = replace(
        config,
        paths=replace(config.paths, models_dir=models_dir.resolve()),
        runtime=replace(config.runtime, active_model_version="v1"),
    )
    state = AppState(config)

    state.load_static_resources()

    assert state.model_version == "v1"
    assert state.model_artifact_path == first.resolve()
    assert state.restart_recovery["model_source"] == "registry"


@patch("traffic_prediction.api.app.PyTorchModelRunner.load_from_artifact")
def test_app_state_restart_recovery_falls_back_to_history_after_buffer_restore_failure(mock_load) -> None:
    config = load_config(project_root=".", load_dotenv_file=False)
    buffer_dir = Path("artifacts/test_runs/startup_orchestration") / uuid4().hex / "buffers"
    buffer_dir.mkdir(parents=True, exist_ok=True)
    (buffer_dir / "live_buffer.pkl").write_text("not a pickle", encoding="utf-8")
    
    models_dir = Path("artifacts/test_runs/startup_orchestration") / uuid4().hex / "models"
    _make_model_artifact(models_dir / "offline-mock")
    
    config = replace(config, paths=replace(config.paths, buffers_dir=buffer_dir.resolve(), models_dir=models_dir.resolve()))
    state = AppState(config)

    state.load_static_resources()
    report = state.readiness_report()

    assert state.buffer_restore_error is not None
    assert state.live_buffer_seeded_from_history is True
    assert state.restart_recovery["buffer_source"] == "history_seeded"
    assert state.restart_recovery["buffer_restore_error"] is not None
    assert report.by_name()["restart_recovery"].status == "recovered"
    assert "buffer=history_seeded" in report.details()["restart_recovery"]


def test_app_state_restart_recovery_allows_stale_buffer_when_tomtom_is_unavailable() -> None:
    config = load_config(project_root=".", load_dotenv_file=False)
    config = replace(config, tomtom=replace(config.tomtom, api_key=None, api_keys=()))
    state = AppState(config)

    state.load_static_resources()

    assert state.restart_recovery["tomtom_status"] == "missing_credentials"
    assert state.restart_recovery["stale_buffer_allowed"] is True


def _make_model_artifact(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "feature_manifest.json").write_text("{}", encoding="utf-8")
    (path / "scaler_params.joblib").write_text("scaler", encoding="utf-8")
    return path
