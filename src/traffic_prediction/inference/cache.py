from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass
class CacheEntry:
    value: Any
    expires_at: datetime


class PredictionCache:
    """Small in-memory TTL cache for repeated prediction requests."""

    def __init__(self, ttl_seconds: int = 900) -> None:
        self.ttl = timedelta(seconds=ttl_seconds)
        self._entries: dict[str, CacheEntry] = {}

    def make_key(self, model_version: str, road_id: str, horizon_minutes: int, timestamp_bucket: datetime) -> str:
        return f"{model_version}:{road_id}:{horizon_minutes}:{timestamp_bucket.isoformat()}"

    def get(self, key: str, now: datetime | None = None):
        now = now or datetime.now()
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            self._entries.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value, now: datetime | None = None) -> None:
        now = now or datetime.now()
        self._entries[key] = CacheEntry(value=value, expires_at=now + self.ttl)

    def invalidate(self, prefix: str | None = None) -> None:
        if prefix is None:
            self._entries.clear()
            return
        for key in list(self._entries):
            if key.startswith(prefix):
                self._entries.pop(key, None)

