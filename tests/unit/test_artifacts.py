from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from traffic_prediction.artifacts import (
    ARTIFACT_METADATA_KEY,
    ArtifactLayout,
    load_buffer_snapshot,
    load_evaluation_report,
    load_feature_manifest,
    load_model_card,
    load_model_config,
    load_scaler_params,
    load_training_history,
    save_buffer_snapshot,
    save_evaluation_report,
    save_feature_manifest,
    save_model_card,
    save_model_config,
    save_scaler_params,
    save_training_history,
    timestamp_id,
    write_json_artifact,
)
from traffic_prediction.data.schemas import FeatureManifest


def test_artifact_layout_standardizes_paths() -> None:
    layout = _layout("paths")

    assert layout.model_dir("model-v1") == layout.models_dir / "model-v1"
    assert layout.model_checkpoint_path("model-v1").name == "model.pt"
    assert layout.feature_manifest_path("model-v1").name == "feature_manifest.json"
    assert layout.scaler_params_path("model-v1").name == "scaler_params.joblib"
    assert layout.model_config_path("model-v1").name == "model_config.json"
    assert layout.training_history_path("model-v1").name == "training_history.json"
    assert layout.evaluation_report_path("model-v1").name == "evaluation_report.json"
    assert layout.model_card_path("model-v1").name == "model_card.md"
    assert layout.report_path("summary", "20260519-050000").name == "summary_20260519-050000.json"
    assert layout.figure_dir("run-1") == layout.figures_dir / "run-1"
    assert layout.log_path("tomtom_ingestion", "20260519").name == "tomtom_ingestion-20260519.jsonl"
    assert layout.buffer_snapshot_path() == layout.buffers_dir / "live_buffer.pkl"


def test_latest_model_resolution_prefers_registry_then_pointer_then_mtime() -> None:
    registry_layout = _layout("latest_registry")
    first = _make_artifact(registry_layout.models_dir / "offline-1", offline=True)
    second = _make_artifact(registry_layout.models_dir / "lstm-2", offline=False)
    registry_model = _make_artifact(registry_layout.models_dir / "registered-model", offline=False)
    _set_mtime(first, 1_700_000_000)
    _set_mtime(second, 1_700_000_100)
    _set_mtime(registry_model, 1_700_000_050)
    registry_path = registry_layout.models_dir / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "active_model_version": "registered",
                "models": [
                    {
                        "model_version": "registered",
                        "artifact_path": str(registry_model),
                        "created_at": "2026-05-19T05:00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert registry_layout.resolve_latest_model() == registry_model.resolve()

    pointer_layout = _layout("latest_pointer")
    pointer_first = _make_artifact(pointer_layout.models_dir / "offline-1", offline=True)
    pointer_second = _make_artifact(pointer_layout.models_dir / "lstm-2", offline=False)
    _set_mtime(pointer_first, 1_700_000_000)
    _set_mtime(pointer_second, 1_700_000_100)
    pointer_layout.write_latest_model_pointer(pointer_first)
    assert pointer_layout.resolve_latest_model() == pointer_first.resolve()

    mtime_layout = _layout("latest_mtime")
    mtime_first = _make_artifact(mtime_layout.models_dir / "offline-1", offline=True)
    mtime_second = _make_artifact(mtime_layout.models_dir / "lstm-2", offline=False)
    _set_mtime(mtime_first, 1_700_000_000)
    _set_mtime(mtime_second, 1_700_000_100)
    assert mtime_layout.resolve_latest_model() == mtime_second.resolve()


def test_json_artifact_includes_version_and_timestamp_metadata() -> None:
    layout = _layout("metadata")
    target = layout.report_path("summary", "20260519-050000")

    write_json_artifact(
        target,
        {"status": "ok"},
        artifact_type="report",
        artifact_version="summary-20260519-050000",
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    metadata = payload[ARTIFACT_METADATA_KEY]
    assert payload["status"] == "ok"
    assert metadata["artifact_type"] == "report"
    assert metadata["artifact_version"] == "summary-20260519-050000"
    assert metadata["created_at"]
    assert metadata["schema_version"] == "1.0"


def test_timestamp_id_uses_expected_convention() -> None:
    assert timestamp_id("offline").startswith("offline-")
    assert len(timestamp_id("offline")) == len("offline-20260519-050000")


def test_artifact_persistence_helpers_round_trip_model_files() -> None:
    layout = _layout("persistence")
    model_version = "model-v1"
    manifest = FeatureManifest(["current_speed", "hour_of_day"], "current_speed", 12, 4)

    save_model_config(layout, model_version, {"model_type": "lstm", "created_at": "2026-05-19T00:00:00"})
    save_feature_manifest(layout, model_version, manifest)
    save_scaler_params(layout, model_version, {"mean": [1.0], "scale": [2.0]})
    save_training_history(layout, model_version, [{"epoch": 1, "validation_loss": 0.5}])
    save_evaluation_report(layout, model_version, {"mae": 3.1, "rmse": 4.7})
    save_model_card(layout, model_version, "# Model Card\n")

    assert load_model_config(layout, model_version)["model_type"] == "lstm"
    assert load_feature_manifest(layout, model_version) == manifest
    assert load_scaler_params(layout, model_version)["scale"] == [2.0]
    assert load_training_history(layout, model_version)["history"][0]["epoch"] == 1
    assert load_evaluation_report(layout, model_version)["rmse"] == 4.7
    assert "Model Card" in load_model_card(layout, model_version)


def test_buffer_snapshot_helper_round_trips_pickle_payload() -> None:
    layout = _layout("buffer_snapshot")

    save_buffer_snapshot(layout, {"roads": ["R1"], "count": 1}, name="unit_buffer")

    assert load_buffer_snapshot(layout, name="unit_buffer") == {"roads": ["R1"], "count": 1}


def _layout(name: str) -> ArtifactLayout:
    root = Path("artifacts/test_runs/artifact_layout") / name / uuid4().hex
    layout = ArtifactLayout(
        artifact_dir=root,
        reports_dir=root / "reports",
        models_dir=root / "models",
        buffers_dir=root / "buffers",
        figures_dir=root / "figures",
        logs_dir=root / "logs",
    )
    layout.ensure_directories()
    return layout


def _make_artifact(path: Path, *, offline: bool) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if offline:
        (path / "feature_manifest.json").write_text("{}", encoding="utf-8")
        (path / "scaler_params.joblib").write_text("scaler", encoding="utf-8")
    else:
        (path / "model.pt").write_text("checkpoint", encoding="utf-8")
    return path


def _set_mtime(path: Path, value: int) -> None:
    for item in [path, *path.iterdir()]:
        os.utime(item, (value, value))
