from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from traffic_prediction.data.schemas import LiveTrafficRecord
from traffic_prediction.monitoring.drift import DriftMonitor


def test_drift_monitor_detects_feature_and_pattern_drift() -> None:
    monitor = DriftMonitor(
        speed_delta_threshold_kmh=10.0,
        pattern_change_min_roads=2,
        pattern_change_fraction=0.50,
    )
    now = datetime(2026, 5, 19, 8, 0)
    records = [
        LiveTrafficRecord("R1", 42.0, 0.9, now),
        LiveTrafficRecord("R2", 12.0, 0.9, now),
        LiveTrafficRecord("R3", 31.0, 0.9, now),
    ]

    report = monitor.evaluate(
        records=records,
        historical_mean_speed={"R1": 25.0, "R2": 28.0, "R3": 30.0},
        now=now,
    )

    assert report.status == "degraded"
    assert report.drift_road_count == 2
    assert report.drift_roads == ["R1", "R2"]
    assert report.traffic_pattern_changed is True
    assert report.indicators["feature_drift_roads"] == 2


def test_drift_monitor_tracks_prediction_error_degradation_and_writes_log() -> None:
    monitor = DriftMonitor(prediction_mae_threshold_kmh=5.0)
    now = datetime(2026, 5, 19, 8, 0)
    report = monitor.evaluate(
        records=[LiveTrafficRecord("R1", 25.0, 0.9, now)],
        historical_mean_speed={"R1": 24.0},
        now=now,
        prediction_errors=[7.0, -8.0, 6.0],
    )
    scratch = Path("artifacts/test_runs/drift_monitor") / uuid4().hex

    log_path = monitor.write_log(report, scratch)
    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

    assert report.status == "degraded"
    assert report.model_degradation is True
    assert report.prediction_error_mae == 7.0
    assert report.prediction_error_count == 3
    assert log_path.name == "drift_monitor-20260519.jsonl"
    assert payload["event_name"] == "drift_check"
    assert payload["report"]["model_degradation"] is True
