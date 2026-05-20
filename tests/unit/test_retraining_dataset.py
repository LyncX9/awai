from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd

from traffic_prediction.config.settings import load_config
from traffic_prediction.data.retraining import RetrainingDatasetConfig, RetrainingDatasetManager
from traffic_prediction.data.schemas import LiveTrafficRecord


def test_retraining_candidate_preserves_history_and_limits_live_fraction() -> None:
    root = _scratch_dir("balance")
    history_path = root / "history.csv"
    _write_history(history_path, road_count=2, periods=2)
    live_records = [
        LiveTrafficRecord(
            road_id=f"R{i % 2:02d}",
            current_speed=25.0 + i,
            confidence=0.9,
            timestamp=datetime(2026, 1, 3) + timedelta(minutes=15 * i),
        )
        for i in range(10)
    ]
    manager = RetrainingDatasetManager(
        RetrainingDatasetConfig(max_live_fraction=0.50, keep_versions=2)
    )

    manifest = manager.build_candidate(history_path, live_records, root / "outputs", now=datetime(2026, 1, 4))
    dataset = pd.read_csv(manifest.dataset_path)

    assert manifest.historical_records == 4
    assert manifest.live_records == 4
    assert set(dataset["source"]) == {"historical", "live"}
    assert dataset["split"].value_counts().sum() == len(dataset)
    assert pd.read_csv(history_path).shape[0] == 4


def test_retraining_diversity_validates_road_time_and_congestion_coverage() -> None:
    root = _scratch_dir("diversity")
    history_path = root / "history.csv"
    history = _write_history(history_path, road_count=10, periods=24 * 7)
    roads = pd.DataFrame({"road_id": [f"R{i:02d}" for i in range(10)]})
    manager = RetrainingDatasetManager()

    report = manager.validate_diversity(history.assign(source="historical"), roads)

    assert report.is_valid is True
    assert report.road_coverage == 1.0
    assert report.covered_hours == list(range(24))
    assert report.covered_days_of_week == list(range(7))
    assert set(report.congestion_levels) == {"free_flow", "moderate", "congested"}


def test_retraining_status_and_archive_pruning() -> None:
    root = _scratch_dir("status")
    history_path = root / "history.csv"
    _write_history(history_path, road_count=1, periods=4)
    manager = RetrainingDatasetManager(RetrainingDatasetConfig(keep_versions=2))

    for day in range(4):
        manager.build_candidate(history_path, [], root / "outputs", now=datetime(2026, 1, 10 + day))

    status = manager.status(history_path, now=datetime(2026, 1, 10))
    versions = manager.list_versions(root / "outputs")

    assert status["historical_records"] == 4
    assert status["live_records"] == 0
    assert len(versions) == 2


def test_runtime_retraining_job_reports_dataset_status() -> None:
    from traffic_prediction.api.app import AppState

    config = load_config(project_root=".", load_dotenv_file=False)
    buffer_dir = _scratch_dir("app-buffer")
    paths = replace(config.paths, buffers_dir=buffer_dir.resolve())
    paths.buffers_dir.mkdir(parents=True, exist_ok=True)
    state = AppState(replace(config, paths=paths))

    state.load_static_resources()
    response = state.trigger_job("retraining_check")

    assert response.data_quality is not None
    assert "retraining_dataset" in response.data_quality
    assert response.data_quality["retraining_dataset"]["historical_records"] > 0


def _scratch_dir(name: str) -> Path:
    path = Path("artifacts/test_runs/retraining_dataset") / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _write_history(path: Path, road_count: int, periods: int) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01 00:00:00")
    rows = []
    speeds = [58.0, 38.0, 20.0]
    for road_index in range(road_count):
        for step in range(periods):
            rows.append(
                {
                    "id": road_index * periods + step,
                    "road_id": f"R{road_index:02d}",
                    "road_name": f"Road {road_index}",
                    "city": "SUKABUMI",
                    "road_weight": 0.5,
                    "current_speed": speeds[(road_index + step) % len(speeds)],
                    "free_flow_speed": 60.0,
                    "current_travel_time": 100,
                    "free_flow_travel_time": 80,
                    "confidence": 0.95,
                    "road_closure": 0,
                    "frc": "FRC2",
                    "sample_lat": -6.9,
                    "sample_lon": 106.9,
                    "collected_at_wib": start + pd.Timedelta(hours=step),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame
