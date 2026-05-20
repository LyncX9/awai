from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from traffic_prediction.data.schemas import LiveTrafficRecord


@dataclass(frozen=True)
class FallbackPrediction:
    predicted_speed: float
    method: str
    lookup_quality: str
    uncertainty_margin: float
    confidence_score: float
    degraded: bool
    reason: str


class FallbackPredictor:
    """Provides degraded predictions when online model inference is unavailable."""

    def __init__(
        self,
        historical_lookup: Any | None = None,
        road_mean_speed: Any | None = None,
        global_default_speed: float = 30.0,
    ) -> None:
        self.historical_lookup = historical_lookup
        self.road_mean_speed = road_mean_speed
        self.global_default_speed = global_default_speed

    def predict(
        self,
        road_id: str,
        target_time: datetime,
        horizon_minutes: int,
        latest_live_record: LiveTrafficRecord | None = None,
        prefer_persistence: bool = False,
    ) -> FallbackPrediction:
        if prefer_persistence and latest_live_record is not None:
            return self._from_persistence(latest_live_record, horizon_minutes)
        predicted_speed, lookup_quality = self._historical_prediction(road_id, target_time)
        return FallbackPrediction(
            predicted_speed=predicted_speed,
            method="historical_average_fallback",
            lookup_quality=lookup_quality,
            uncertainty_margin=self.uncertainty_margin(horizon_minutes, lookup_quality),
            confidence_score=self.confidence_score(lookup_quality),
            degraded=True,
            reason="live_buffer_unavailable",
        )

    def _from_persistence(self, record: LiveTrafficRecord, horizon_minutes: int) -> FallbackPrediction:
        lookup_quality = "persistence_latest_observation"
        return FallbackPrediction(
            predicted_speed=float(record.current_speed),
            method="persistence_fallback",
            lookup_quality=lookup_quality,
            uncertainty_margin=self.uncertainty_margin(horizon_minutes, lookup_quality),
            confidence_score=min(float(record.confidence), self.confidence_score(lookup_quality)),
            degraded=True,
            reason="model_inference_unavailable",
        )

    def _historical_prediction(self, road_id: str, target_time: datetime) -> tuple[float, str]:
        if self.historical_lookup is not None:
            key = (road_id, int(target_time.hour), int(target_time.weekday()))
            if key in self.historical_lookup.index:
                return float(self.historical_lookup.loc[key]), "road_hour_day_average"
        if self.road_mean_speed is not None and road_id in self.road_mean_speed.index:
            return float(self.road_mean_speed.loc[road_id]), "road_average"
        return self.global_default_speed, "global_default"

    def uncertainty_margin(self, horizon_minutes: int, lookup_quality: str) -> float:
        base_by_horizon = {
            15: 2.69,
            30: 2.69,
            45: 2.69,
            60: 2.68,
        }
        multiplier = {
            "persistence_latest_observation": 1.15,
            "road_hour_day_average": 1.0,
            "road_average": 1.35,
            "global_default": 2.0,
        }.get(lookup_quality, 1.5)
        return base_by_horizon.get(horizon_minutes, 3.0) * multiplier

    def confidence_score(self, lookup_quality: str) -> float:
        return {
            "persistence_latest_observation": 0.68,
            "road_hour_day_average": 0.72,
            "road_average": 0.55,
            "global_default": 0.35,
        }.get(lookup_quality, 0.40)
