from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from traffic_prediction.monitoring.runtime_logging import (
    LOG_CATEGORIES,
    RuntimeEventLogger,
    configure_structured_logging,
)
from traffic_prediction.orchestration.scheduler import InProcessScheduler


def test_runtime_event_logger_writes_redacted_category_jsonl() -> None:
    scratch = _scratch_dir("event_write")
    logger = RuntimeEventLogger(scratch)

    path = logger.write(
        "prediction",
        "prediction_request",
        {
            "road_id": "SBM_BHY_01",
            "api_key": "secret",
            "nested": {"authorization": "Bearer secret"},
        },
        occurred_at=datetime(2026, 5, 19, 17, 30),
    )

    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert path.name == "prediction-20260519.jsonl"
    assert payload["category"] == "prediction"
    assert payload["event_name"] == "prediction_request"
    assert payload["payload"]["road_id"] == "SBM_BHY_01"
    assert payload["payload"]["api_key"] == "[REDACTED]"
    assert payload["payload"]["nested"]["authorization"] == "[REDACTED]"


def test_runtime_event_logger_removes_expired_daily_files() -> None:
    scratch = _scratch_dir("retention")
    old_path = scratch / "api-20260401.jsonl"
    old_path.write_text("{}\n", encoding="utf-8")
    removed: list[Path] = []

    class RecordingRetentionLogger(RuntimeEventLogger):
        def _remove_expired_file(self, path: Path) -> None:
            removed.append(path)

    RecordingRetentionLogger(scratch, retention_days=30).write(
        "api",
        "request",
        {"status_code": 200},
        occurred_at=datetime(2026, 5, 19, 17, 30),
    )

    assert removed == [old_path]


def test_configure_structured_logging_uses_daily_rotating_json_handler() -> None:
    scratch = _scratch_dir("python_logging")
    logger_name = f"traffic_prediction.test.{uuid4().hex}"
    logger = configure_structured_logging(scratch, level="DEBUG", logger_name=logger_name)

    logger.info(
        "runtime ready",
        extra={"category": "startup", "event_name": "startup_recovery", "status": "completed"},
    )
    for handler in logger.handlers:
        handler.flush()

    payload = json.loads((scratch / "application.log").read_text(encoding="utf-8").splitlines()[0])
    assert logger.level == logging.DEBUG
    assert payload["message"] == "runtime ready"
    assert payload["category"] == "startup"
    assert any(getattr(handler, "backupCount", None) == 30 for handler in logger.handlers)


def test_scheduler_writes_structured_job_events() -> None:
    scratch = _scratch_dir("scheduler")
    event_logger = RuntimeEventLogger(scratch)
    scheduler = InProcessScheduler(event_logger=event_logger)
    scheduler.add_interval_job("demo_job", 60, lambda: {"ok": True})

    result = scheduler.trigger("demo_job")

    from datetime import timezone
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    payload = json.loads((scratch / f"scheduler-{today_str}.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert result.status == "completed"
    assert payload["event_name"] == "job_run"
    assert payload["payload"]["job_name"] == "demo_job"
    assert payload["payload"]["status"] == "completed"


def test_runtime_logging_categories_cover_required_domains() -> None:
    assert {
        "api",
        "data_quality",
        "drift",
        "errors",
        "ingestion",
        "prediction",
        "scheduler",
        "training",
    } <= LOG_CATEGORIES


def _scratch_dir(name: str) -> Path:
    path = Path("artifacts/test_runs/runtime_logging") / name / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()
