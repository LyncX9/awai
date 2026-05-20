from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
from pathlib import Path
from uuid import uuid4

from traffic_prediction.ingestion.buffer import LiveBufferManager
from traffic_prediction.ingestion.events import IngestionEventLogger
from traffic_prediction.ingestion.ingestor import TomTomIngestor
from traffic_prediction.ingestion.tomtom_client import TomTomFetchResult, TomTomTrafficObservation
from traffic_prediction.ingestion.tomtom_mapping import TomTomRoadMapper, TomTomRoadMapping


@dataclass
class FakeTomTomClient:
    result: TomTomFetchResult

    def fetch_flow_segments(self, queries):
        return self.result


def test_tomtom_ingestor_appends_valid_records_and_invalidates_cache() -> None:
    now_utc = datetime(2026, 5, 18, 0, 0, tzinfo=UTC)
    invalidations: list[bool] = []
    persisted_paths: list[str] = []
    buffer = LiveBufferManager(min_timesteps=1, max_timesteps=3)
    ingestor = TomTomIngestor(
        client=FakeTomTomClient(
            TomTomFetchResult(
                observations=[
                    TomTomTrafficObservation("tt-001", 24.0, 0.9, now_utc - timedelta(minutes=10)),
                    TomTomTrafficObservation("tt-002", 28.0, 0.8, now_utc - timedelta(minutes=5)),
                ],
                response_time_seconds=0.12,
            )
        ),
        mapper=_mapper(),
        buffer_manager=buffer,
        expected_road_ids={"R1", "R2"},
        cache_invalidator=lambda: invalidations.append(True),
        buffer_persister=lambda: persisted_paths.append("live-buffer.pkl") or "live-buffer.pkl",
    )

    summary = ingestor.ingest_once(now=now_utc)

    assert summary.fetched_count == 2
    assert summary.accepted_count == 2
    assert summary.rejected_count == 0
    assert summary.cache_invalidated is True
    assert invalidations == [True]
    assert summary.buffer_persisted is True
    assert summary.buffer_persist_path == "live-buffer.pkl"
    assert persisted_paths == ["live-buffer.pkl"]
    assert [record.road_id for record in buffer.get_latest("R1")] == ["R1"]
    assert buffer.has_minimum_history("R2")
    assert summary.data_quality is not None
    assert summary.data_quality.status == "degraded"
    assert summary.data_quality.delayed_roads == ["R1"]
    assert summary.data_quality.api_uptime == 1.0


def test_tomtom_ingestor_writes_event_log() -> None:
    now_utc = datetime(2026, 5, 19, 0, 0, tzinfo=UTC)
    scratch = Path("artifacts/test_runs/tomtom_ingestor_events") / uuid4().hex
    ingestor = TomTomIngestor(
        client=FakeTomTomClient(
            TomTomFetchResult(
                observations=[TomTomTrafficObservation("tt-001", 24.0, 0.9, now_utc)],
                response_time_seconds=0.12,
            )
        ),
        mapper=_mapper(),
        buffer_manager=LiveBufferManager(min_timesteps=1, max_timesteps=3),
        expected_road_ids={"R1"},
        event_logger=IngestionEventLogger(scratch),
    )

    summary = ingestor.ingest_once(now=now_utc)

    assert summary.event_log_path is not None
    assert "tomtom_ingestion-20260519.jsonl" in summary.event_log_path


def test_tomtom_ingestor_rejects_stale_invalid_and_duplicate_records() -> None:
    now_utc = datetime(2026, 5, 18, 0, 0, tzinfo=UTC)
    buffer = LiveBufferManager(min_timesteps=1, max_timesteps=3)
    persisted_paths: list[str] = []
    buffer.append(_mapper().to_live_record(TomTomTrafficObservation("tt-001", 20.0, 0.9, now_utc), now_utc))
    ingestor = TomTomIngestor(
        client=FakeTomTomClient(
            TomTomFetchResult(
                observations=[
                    TomTomTrafficObservation("tt-001", 21.0, 0.9, now_utc),
                    TomTomTrafficObservation("tt-002", 130.0, 0.8, now_utc - timedelta(minutes=1)),
                    TomTomTrafficObservation("tt-003", 25.0, 1.2, now_utc - timedelta(minutes=1)),
                    TomTomTrafficObservation("tt-004", 25.0, 0.8, now_utc - timedelta(hours=1)),
                ]
            )
        ),
        mapper=_mapper(),
        buffer_manager=buffer,
        expected_road_ids={"R1", "R2", "R3", "R4"},
        cache_invalidator=lambda: None,
        buffer_persister=lambda: persisted_paths.append("should-not-run"),
    )

    summary = ingestor.ingest_once(now=now_utc)

    assert summary.accepted_count == 0
    assert summary.rejected_count == 4
    assert "Duplicate TomTom record" in summary.errors["tt-001"]
    assert "Invalid speed" in summary.errors["tt-002"]
    assert "Invalid confidence" in summary.errors["tt-003"]
    assert "Stale TomTom record" in summary.errors["tt-004"]
    assert summary.cache_invalidated is False
    assert summary.buffer_persisted is False
    assert summary.buffer_persist_path is None
    assert persisted_paths == []


def test_tomtom_ingestor_preserves_fetch_and_mapping_errors() -> None:
    now_utc = datetime(2026, 5, 18, 0, 0, tzinfo=UTC)
    ingestor = TomTomIngestor(
        client=FakeTomTomClient(
            TomTomFetchResult(
                observations=[TomTomTrafficObservation("unknown", 25.0, 0.8, now_utc)],
                errors={"tt-timeout": "TomTom request failed"},
            )
        ),
        mapper=_mapper(),
        buffer_manager=LiveBufferManager(),
        expected_road_ids={"R1", "R2"},
    )

    summary = ingestor.ingest_once(now=now_utc)

    assert summary.fetch_error_count == 1
    assert summary.accepted_count == 0
    assert summary.rejected_count == 2
    assert summary.errors["tt-timeout"] == "TomTom request failed"
    assert "Missing road mapping" in summary.errors["unknown"]


def _mapper() -> TomTomRoadMapper:
    return TomTomRoadMapper(
        [
            TomTomRoadMapping("tt-001", "R1", -6.90, 106.90),
            TomTomRoadMapping("tt-002", "R2", -6.91, 106.91),
            TomTomRoadMapping("tt-003", "R3", -6.92, 106.92),
            TomTomRoadMapping("tt-004", "R4", -6.93, 106.93),
        ]
    )
