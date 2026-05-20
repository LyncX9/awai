from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import sys
from typing import Any


LOG_CATEGORIES = {
    "api",
    "data_quality",
    "drift",
    "errors",
    "ingestion",
    "prediction",
    "scheduler",
    "startup",
    "training",
}

SENSITIVE_KEY_PARTS = (
    "api_key",
    "api_keys",
    "authorization",
    "password",
    "secret",
    "token",
    "x-api-key",
)


@dataclass(frozen=True)
class RuntimeLogEvent:
    category: str
    event_name: str
    status: str
    level: str
    logged_at: str
    payload: dict[str, Any]


class JsonLogFormatter(logging.Formatter):
    """Format standard Python logging records as compact JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "logged_at": datetime.fromtimestamp(record.created).astimezone().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        for key in ("category", "event_name", "status"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(_redact(_json_safe(payload)), sort_keys=True)


class RuntimeEventLogger:
    """Append structured runtime events to category-specific daily JSONL files."""

    def __init__(self, log_dir: str | Path, retention_days: int = 30) -> None:
        if retention_days <= 0:
            raise ValueError("retention_days must be positive")
        self.log_dir = Path(log_dir)
        self.retention_days = retention_days

    def write(
        self,
        category: str,
        event_name: str,
        payload: dict[str, Any] | None = None,
        *,
        status: str = "completed",
        level: str = "INFO",
        occurred_at: datetime | None = None,
    ) -> Path:
        normalized_category = _normalize_category(category)
        timestamp = occurred_at or datetime.now().astimezone()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._purge_expired_files(normalized_category, timestamp)
        path = self.log_dir / f"{normalized_category}-{timestamp:%Y%m%d}.jsonl"
        event = RuntimeLogEvent(
            category=normalized_category,
            event_name=event_name,
            status=status,
            level=level.upper(),
            logged_at=timestamp.isoformat(),
            payload=_redact(_json_safe(payload or {})),
        )
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        return path

    def _purge_expired_files(self, category: str, now: datetime) -> None:
        cutoff = now.timestamp() - self.retention_days * 24 * 60 * 60
        cutoff_date = (now - timedelta(days=self.retention_days)).date()
        for path in self.log_dir.glob(f"{category}-*.jsonl"):
            try:
                file_date = _date_from_daily_log_name(category, path)
                if (file_date is not None and file_date < cutoff_date) or path.stat().st_mtime < cutoff:
                    self._remove_expired_file(path)
            except OSError:
                continue

    def _remove_expired_file(self, path: Path) -> None:
        path.unlink()


def configure_structured_logging(
    log_dir: str | Path,
    *,
    level: str = "INFO",
    retention_days: int = 30,
    logger_name: str = "traffic_prediction",
) -> logging.Logger:
    """Configure package logging with daily rotation and 30-day default retention."""

    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(_logging_level(level))
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_awai_runtime_handler", False):
            logger.removeHandler(handler)
            handler.close()

    handler = TimedRotatingFileHandler(
        filename=path / "application.log",
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
    )
    handler.setFormatter(JsonLogFormatter())
    handler.setLevel(_logging_level(level))
    handler._awai_runtime_handler = True
    logger.addHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JsonLogFormatter())
    stream_handler.setLevel(_logging_level(level))
    stream_handler._awai_runtime_handler = True
    logger.addHandler(stream_handler)

    return logger


def _normalize_category(category: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", category.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("category must not be empty")
    return normalized


def _logging_level(level: str) -> int:
    return int(getattr(logging, level.upper(), logging.INFO))


def _date_from_daily_log_name(category: str, path: Path) -> date | None:
    match = re.fullmatch(rf"{re.escape(category)}-(\d{{8}})\.jsonl", path.name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


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


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).lower().replace("_", "-")
            if any(part.replace("_", "-") in normalized_key for part in SENSITIVE_KEY_PARTS):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


__all__ = [
    "LOG_CATEGORIES",
    "JsonLogFormatter",
    "RuntimeEventLogger",
    "RuntimeLogEvent",
    "configure_structured_logging",
]
