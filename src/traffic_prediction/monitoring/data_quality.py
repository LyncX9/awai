from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

import numpy as np

from traffic_prediction.data.schemas import DataQualityReport, LiveTrafficRecord


class DataQualityMonitor:
    """Evaluates live data completeness, freshness, confidence, and outliers."""

    def __init__(
        self,
        stale_after: timedelta = timedelta(minutes=30),
        delayed_after: timedelta = timedelta(minutes=5),
        low_confidence_threshold: float = 0.3,
        outlier_speed_delta: float = 40.0,
        api_event_window: int = 100,
    ) -> None:
        self.stale_after = stale_after
        self.delayed_after = delayed_after
        self.low_confidence_threshold = low_confidence_threshold
        self.outlier_speed_delta = outlier_speed_delta
        self.previous_speed_by_road: dict[str, float] = {}
        self.api_events: deque[bool] = deque(maxlen=api_event_window)

    def evaluate(
        self,
        records: list[LiveTrafficRecord],
        expected_road_ids: set[str],
        now: datetime,
        api_success: bool | None = None,
    ) -> DataQualityReport:
        if api_success is not None:
            self.record_api_result(api_success)

        by_road = {record.road_id: record for record in records}
        missing_roads = sorted(expected_road_ids - set(by_road))
        stale_roads = sorted(
            road_id
            for road_id, record in by_road.items()
            if now - record.timestamp > self.stale_after
        )
        delayed_roads = sorted(
            record.road_id
            for record in records
            if self._record_delay(record, now) > self.delayed_after
        )
        low_confidence_roads = sorted(
            record.road_id for record in records if record.confidence < self.low_confidence_threshold
        )
        outlier_roads = self._detect_outliers(records)

        fresh_roads = expected_road_ids - set(missing_roads) - set(stale_roads)
        completeness = len(fresh_roads) / max(len(expected_road_ids), 1)
        average_confidence = float(np.mean([record.confidence for record in records])) if records else 0.0
        api_uptime = self.api_uptime()
        quality_issues = {
            "missing_roads": len(missing_roads),
            "stale_roads": len(stale_roads),
            "delayed_roads": len(delayed_roads),
            "low_confidence_roads": len(low_confidence_roads),
            "outlier_roads": len(outlier_roads),
            "api_failures": len(self.api_events) - sum(1 for event in self.api_events if event),
        }
        status = self._derive_status(
            completeness,
            stale_roads,
            delayed_roads,
            low_confidence_roads,
            outlier_roads,
            api_uptime,
        )
        fallback_recommendation = self._fallback_recommendation(status, completeness, api_uptime)

        for record in records:
            self.previous_speed_by_road[record.road_id] = record.current_speed

        return DataQualityReport(
            timestamp=now,
            completeness=completeness,
            average_confidence=average_confidence,
            stale_roads=stale_roads,
            missing_roads=missing_roads,
            delayed_roads=delayed_roads,
            low_confidence_roads=low_confidence_roads,
            outlier_roads=outlier_roads,
            api_uptime=api_uptime,
            fallback_recommendation=fallback_recommendation,
            quality_issues=quality_issues,
            status=status,
        )

    def record_api_result(self, success: bool) -> None:
        self.api_events.append(bool(success))

    def api_uptime(self) -> float:
        if not self.api_events:
            return 1.0
        return sum(1 for event in self.api_events if event) / len(self.api_events)

    def _record_delay(self, record: LiveTrafficRecord, now: datetime) -> timedelta:
        if record.freshness_indicator is not None:
            return record.freshness_indicator
        return now - record.timestamp

    def _detect_outliers(self, records: list[LiveTrafficRecord]) -> list[str]:
        outlier_roads = []
        for record in records:
            previous = self.previous_speed_by_road.get(record.road_id)
            if previous is not None and abs(record.current_speed - previous) > self.outlier_speed_delta:
                outlier_roads.append(record.road_id)
        return sorted(outlier_roads)

    def _derive_status(
        self,
        completeness: float,
        stale_roads: list[str],
        delayed_roads: list[str],
        low_confidence_roads: list[str],
        outlier_roads: list[str],
        api_uptime: float,
    ) -> str:
        if completeness < 0.50 or api_uptime < 0.50:
            return "unavailable"
        if completeness < 0.90 or stale_roads or delayed_roads or low_confidence_roads or outlier_roads:
            return "degraded"
        return "healthy"

    def _fallback_recommendation(self, status: str, completeness: float, api_uptime: float) -> str:
        if status == "healthy":
            return "use_live_lstm"
        if status == "unavailable" or completeness < 0.50 or api_uptime < 0.50:
            return "use_historical_average_fallback"
        return "use_live_prediction_with_quality_penalty"
