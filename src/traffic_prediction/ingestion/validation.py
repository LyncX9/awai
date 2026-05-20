from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from traffic_prediction.data.schemas import LiveTrafficRecord
from traffic_prediction.ingestion.buffer import LiveBufferManager


@dataclass(frozen=True)
class LiveRecordValidationConfig:
    timezone: str = "Asia/Jakarta"
    min_speed: float = 0.0
    max_speed: float = 120.0
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    max_record_age: timedelta = timedelta(minutes=30)
    future_tolerance: timedelta = timedelta(minutes=2)


class LiveRecordValidator:
    """Validates live observations before they enter the rolling buffer."""

    def __init__(
        self,
        expected_road_ids: set[str],
        buffer_manager: LiveBufferManager | None = None,
        config: LiveRecordValidationConfig | None = None,
    ) -> None:
        self.expected_road_ids = {str(road_id) for road_id in expected_road_ids}
        self.buffer_manager = buffer_manager
        self.config = config or LiveRecordValidationConfig()
        self.timezone = ZoneInfo(self.config.timezone)

    def validate(self, record: LiveTrafficRecord, now: datetime) -> LiveTrafficRecord:
        normalized_now = self._normalize_datetime(now)
        normalized = self.normalize(record, normalized_now)

        if normalized.road_id not in self.expected_road_ids:
            raise ValueError(f"Unknown road mapping for {normalized.road_id}")
        if not self.config.min_speed <= normalized.current_speed <= self.config.max_speed:
            raise ValueError(f"Invalid speed for {normalized.road_id}: {normalized.current_speed}")
        if not self.config.min_confidence <= normalized.confidence <= self.config.max_confidence:
            raise ValueError(f"Invalid confidence for {normalized.road_id}: {normalized.confidence}")
        if normalized.timestamp > normalized_now + self.config.future_tolerance:
            raise ValueError(f"Future TomTom record for {normalized.road_id}: {normalized.timestamp.isoformat()}")
        if normalized_now - normalized.timestamp > self.config.max_record_age:
            raise ValueError(f"Stale TomTom record for {normalized.road_id}: {normalized.timestamp.isoformat()}")
        if self._is_duplicate(normalized):
            raise ValueError(f"Duplicate TomTom record for {normalized.road_id}: {normalized.timestamp.isoformat()}")

        return normalized

    def normalize(self, record: LiveTrafficRecord, now: datetime) -> LiveTrafficRecord:
        normalized_now = self._normalize_datetime(now)
        timestamp = self._normalize_datetime(record.timestamp)
        return replace(record, timestamp=timestamp, freshness_indicator=normalized_now - timestamp)

    def _is_duplicate(self, record: LiveTrafficRecord) -> bool:
        if self.buffer_manager is None:
            return False
        return any(existing.timestamp == record.timestamp for existing in self.buffer_manager.get_latest(record.road_id))

    def _normalize_datetime(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=self.timezone)
        return value.astimezone(self.timezone)
