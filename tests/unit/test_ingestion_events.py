from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from traffic_prediction.ingestion.events import IngestionEventLogger
from traffic_prediction.ingestion.ingestor import TomTomIngestionSummary


def test_ingestion_event_logger_writes_jsonl_summary() -> None:
    scratch = Path("artifacts/test_runs/ingestion_events") / uuid4().hex
    logger = IngestionEventLogger(scratch)
    summary = TomTomIngestionSummary(
        fetched_count=2,
        accepted_count=1,
        rejected_count=1,
        fetch_error_count=0,
        cache_invalidated=True,
        ingested_at=datetime(2026, 5, 19, 8, 30),
        response_time_seconds=0.25,
        errors={"tt-002": "Invalid speed"},
    )

    path = logger.write(summary)

    assert path.name == "tomtom_ingestion-20260519.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["event_name"] == "tomtom_ingestion"
    assert payload["summary"]["accepted_count"] == 1
    assert payload["summary"]["errors"] == {"tt-002": "Invalid speed"}
