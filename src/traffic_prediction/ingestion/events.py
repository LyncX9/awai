from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


class IngestionEventLogger:
    """Writes append-only JSONL events for live ingestion runs."""

    def __init__(self, log_dir: str | Path, file_prefix: str = "tomtom_ingestion") -> None:
        self.log_dir = Path(log_dir)
        self.file_prefix = file_prefix

    def write(self, summary: Any, event_name: str = "tomtom_ingestion") -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ingested_at = getattr(summary, "ingested_at", datetime.now())
        log_path = self.log_dir / f"{self.file_prefix}-{ingested_at:%Y%m%d}.jsonl"
        payload = {
            "event_name": event_name,
            "logged_at": datetime.now().astimezone().isoformat(),
            "summary": _json_safe(summary),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
        return log_path


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Path):
        return str(value)
    return value
