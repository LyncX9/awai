from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol

UTC = timezone.utc
from zoneinfo import ZoneInfo

from traffic_prediction.data.schemas import DataQualityReport, LiveTrafficRecord
from traffic_prediction.ingestion.buffer import LiveBufferManager
from traffic_prediction.ingestion.tomtom_client import TomTomFetchResult, TomTomSegmentQuery, TomTomTrafficObservation
from traffic_prediction.ingestion.tomtom_mapping import TomTomMappingError, TomTomRoadMapper
from traffic_prediction.ingestion.validation import LiveRecordValidationConfig, LiveRecordValidator
from traffic_prediction.monitoring.data_quality import DataQualityMonitor


CacheInvalidator = Callable[[], None]
BufferPersister = Callable[[], object]


class TomTomObservationClient(Protocol):
    def fetch_flow_segments(self, queries: list[TomTomSegmentQuery]) -> TomTomFetchResult:
        ...


class IngestionEventWriter(Protocol):
    def write(self, summary: "TomTomIngestionSummary", event_name: str = "tomtom_ingestion") -> object:
        ...


@dataclass(frozen=True)
class TomTomIngestionSummary:
    fetched_count: int
    accepted_count: int
    rejected_count: int
    fetch_error_count: int
    cache_invalidated: bool
    ingested_at: datetime
    response_time_seconds: float
    errors: dict[str, str] = field(default_factory=dict)
    buffer_stats: dict | None = None
    data_quality: DataQualityReport | None = None
    event_log_path: str | None = None
    buffer_persisted: bool = False
    buffer_persist_path: str | None = None


class TomTomIngestor:
    """Fetch, map, validate, and append TomTom observations into the live buffer."""

    def __init__(
        self,
        client: TomTomObservationClient,
        mapper: TomTomRoadMapper,
        buffer_manager: LiveBufferManager,
        expected_road_ids: set[str],
        timezone: str = "Asia/Jakarta",
        min_speed: float = 0.0,
        max_speed: float = 120.0,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        max_record_age: timedelta = timedelta(minutes=30),
        quality_monitor: DataQualityMonitor | None = None,
        cache_invalidator: CacheInvalidator | None = None,
        buffer_persister: BufferPersister | None = None,
        event_logger: IngestionEventWriter | None = None,
    ) -> None:
        self.client = client
        self.mapper = mapper
        self.buffer_manager = buffer_manager
        self.expected_road_ids = expected_road_ids
        self.timezone = ZoneInfo(timezone)
        self.validator = LiveRecordValidator(
            expected_road_ids=expected_road_ids,
            buffer_manager=buffer_manager,
            config=LiveRecordValidationConfig(
                timezone=timezone,
                min_speed=min_speed,
                max_speed=max_speed,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
                max_record_age=max_record_age,
            ),
        )
        self.quality_monitor = quality_monitor or DataQualityMonitor()
        self.cache_invalidator = cache_invalidator
        self.buffer_persister = buffer_persister
        self.event_logger = event_logger

    def ingest_once(self, now: datetime | None = None) -> TomTomIngestionSummary:
        ingested_at = self._normalize_now(now)
        result = self.client.fetch_flow_segments(self.mapper.to_queries())
        accepted: list[LiveTrafficRecord] = []
        errors = dict(result.errors)

        for observation in result.observations:
            try:
                record = self.mapper.to_live_record(observation, received_at=ingested_at.astimezone(UTC))
                record = self.validator.validate(record, ingested_at)
                accepted.append(record)
            except (TomTomMappingError, ValueError) as exc:
                errors[observation.tomtom_segment_id] = str(exc)

        self.buffer_manager.append_many(accepted)
        cache_invalidated = False
        if accepted and self.cache_invalidator is not None:
            self.cache_invalidator()
            cache_invalidated = True

        buffer_persisted = False
        buffer_persist_path = None
        if accepted and self.buffer_persister is not None:
            persisted = self.buffer_persister()
            buffer_persisted = True
            buffer_persist_path = str(persisted) if persisted is not None else None

        latest_records = [
            records[-1]
            for road_id in sorted(self.buffer_manager.buffers)
            if (records := self.buffer_manager.get_latest(road_id))
        ]
        data_quality = self.quality_monitor.evaluate(
            latest_records,
            expected_road_ids=self.expected_road_ids,
            now=ingested_at,
            api_success=len(result.errors) == 0,
        )

        summary = TomTomIngestionSummary(
            fetched_count=len(result.observations),
            accepted_count=len(accepted),
            rejected_count=len(errors),
            fetch_error_count=len(result.errors),
            cache_invalidated=cache_invalidated,
            ingested_at=ingested_at,
            response_time_seconds=result.response_time_seconds,
            errors=errors,
            buffer_stats=self.buffer_manager.stats(expected_road_ids=self.expected_road_ids, now=ingested_at),
            data_quality=data_quality,
            buffer_persisted=buffer_persisted,
            buffer_persist_path=buffer_persist_path,
        )
        if self.event_logger is not None:
            event_path = self.event_logger.write(summary)
            summary = replace(summary, event_log_path=str(event_path))
        return summary

    def _normalize_now(self, value: datetime | None) -> datetime:
        if value is None:
            return datetime.now(self.timezone)
        if value.tzinfo is None:
            return value.replace(tzinfo=self.timezone)
        return value.astimezone(self.timezone)
