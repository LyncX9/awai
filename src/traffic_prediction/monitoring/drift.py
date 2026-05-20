from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from traffic_prediction.data.schemas import LiveTrafficRecord


@dataclass(frozen=True)
class DriftReport:
    timestamp: datetime
    status: str
    road_count_checked: int
    drift_road_count: int
    drift_roads: list[str]
    max_abs_delta_kmh: float
    mean_abs_delta_kmh: float
    threshold_kmh: float
    traffic_pattern_changed: bool
    prediction_error_mae: float | None = None
    prediction_error_count: int = 0
    model_degradation: bool = False
    indicators: dict[str, int | float | bool | str | None] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


class DriftMonitor:
    """Lightweight drift monitor for live-vs-historical traffic patterns."""

    def __init__(
        self,
        *,
        speed_delta_threshold_kmh: float = 15.0,
        pattern_change_min_roads: int = 5,
        pattern_change_fraction: float = 0.10,
        prediction_mae_threshold_kmh: float = 6.0,
    ) -> None:
        self.speed_delta_threshold_kmh = speed_delta_threshold_kmh
        self.pattern_change_min_roads = pattern_change_min_roads
        self.pattern_change_fraction = pattern_change_fraction
        self.prediction_mae_threshold_kmh = prediction_mae_threshold_kmh

    def evaluate(
        self,
        *,
        records: Sequence[LiveTrafficRecord],
        historical_mean_speed: Mapping[str, float],
        now: datetime,
        prediction_errors: Sequence[float] | None = None,
    ) -> DriftReport:
        deltas = self._speed_deltas(records, historical_mean_speed)
        drift_roads = sorted(
            road_id
            for road_id, delta in deltas.items()
            if abs(delta) > self.speed_delta_threshold_kmh
        )
        abs_deltas = [abs(delta) for delta in deltas.values()]
        road_count_checked = len(deltas)
        traffic_pattern_changed = self._traffic_pattern_changed(len(drift_roads), road_count_checked)
        prediction_error_mae = self._prediction_error_mae(prediction_errors)
        model_degradation = (
            prediction_error_mae is not None
            and prediction_error_mae > self.prediction_mae_threshold_kmh
        )
        status = self._status(traffic_pattern_changed, model_degradation, len(drift_roads))
        return DriftReport(
            timestamp=now,
            status=status,
            road_count_checked=road_count_checked,
            drift_road_count=len(drift_roads),
            drift_roads=drift_roads[:20],
            max_abs_delta_kmh=round(max(abs_deltas, default=0.0), 6),
            mean_abs_delta_kmh=round(float(np.mean(abs_deltas)) if abs_deltas else 0.0, 6),
            threshold_kmh=self.speed_delta_threshold_kmh,
            traffic_pattern_changed=traffic_pattern_changed,
            prediction_error_mae=(
                round(prediction_error_mae, 6) if prediction_error_mae is not None else None
            ),
            prediction_error_count=len(prediction_errors or []),
            model_degradation=model_degradation,
            indicators={
                "feature_drift_roads": len(drift_roads),
                "traffic_pattern_changed": traffic_pattern_changed,
                "model_degradation": model_degradation,
                "prediction_mae_threshold_kmh": self.prediction_mae_threshold_kmh,
            },
        )

    def write_log(self, report: DriftReport, log_dir: str | Path) -> Path:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        log_path = path / f"drift_monitor-{report.timestamp:%Y%m%d}.jsonl"
        event = {
            "event_name": "drift_check",
            "logged_at": datetime.now().astimezone().isoformat(),
            "report": _json_safe(report.to_dict()),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        return log_path

    def _speed_deltas(
        self,
        records: Sequence[LiveTrafficRecord],
        historical_mean_speed: Mapping[str, float],
    ) -> dict[str, float]:
        deltas: dict[str, float] = {}
        for record in records:
            if record.road_id not in historical_mean_speed:
                continue
            deltas[record.road_id] = float(record.current_speed - historical_mean_speed[record.road_id])
        return deltas

    def _traffic_pattern_changed(self, drift_road_count: int, road_count_checked: int) -> bool:
        if road_count_checked <= 0:
            return False
        return (
            drift_road_count >= self.pattern_change_min_roads
            or drift_road_count / road_count_checked >= self.pattern_change_fraction
        )

    @staticmethod
    def _prediction_error_mae(prediction_errors: Sequence[float] | None) -> float | None:
        if not prediction_errors:
            return None
        return float(np.mean([abs(float(error)) for error in prediction_errors]))

    @staticmethod
    def _status(traffic_pattern_changed: bool, model_degradation: bool, drift_road_count: int) -> str:
        if traffic_pattern_changed or model_degradation:
            return "degraded"
        if drift_road_count:
            return "warning"
        return "healthy"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value
