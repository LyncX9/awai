from __future__ import annotations

import pickle
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

from traffic_prediction.data.schemas import LiveTrafficRecord


class LiveBufferManager:
    """Rolling per-road live observation buffer."""

    def __init__(self, min_timesteps: int = 24, max_timesteps: int = 48) -> None:
        self.min_timesteps = min_timesteps
        self.max_timesteps = max_timesteps
        self.buffers: dict[str, deque[LiveTrafficRecord]] = defaultdict(lambda: deque(maxlen=max_timesteps))
        self.freshness: dict[str, datetime] = {}

    def append(self, record: LiveTrafficRecord) -> None:
        buffer = self.buffers[record.road_id]
        if buffer and buffer[-1].timestamp == record.timestamp:
            return
        buffer.append(record)
        self.freshness[record.road_id] = record.timestamp

    def append_many(self, records: list[LiveTrafficRecord]) -> None:
        for record in sorted(records, key=lambda item: (item.road_id, item.timestamp)):
            self.append(record)

    def get_latest(self, road_id: str, n: int | None = None) -> list[LiveTrafficRecord]:
        buffer = self.buffers.get(road_id)
        if not buffer:
            return []
        if n is None:
            return list(buffer)
        return list(buffer)[-n:]

    def is_stale(self, road_id: str, now: datetime, stale_after: timedelta = timedelta(minutes=30)) -> bool:
        last_seen = self.freshness.get(road_id)
        if last_seen is None:
            return True
        return now - last_seen > stale_after

    def has_minimum_history(self, road_id: str) -> bool:
        return len(self.buffers.get(road_id, [])) >= self.min_timesteps

    def stats(self, expected_road_ids: set[str] | None = None, now: datetime | None = None) -> dict:
        road_ids = expected_road_ids or set(self.buffers)
        now = now or datetime.now()
        fill_rates = {
            road_id: min(len(self.buffers.get(road_id, [])) / self.max_timesteps, 1.0)
            for road_id in sorted(road_ids)
        }
        stale_roads = [road_id for road_id in sorted(road_ids) if self.is_stale(road_id, now)]
        return {
            "total_roads": len(road_ids),
            "fresh_roads": len(road_ids) - len(stale_roads),
            "stale_roads": len(stale_roads),
            "fill_rate": fill_rates,
            "average_fill_rate": sum(fill_rates.values()) / len(fill_rates) if fill_rates else 0.0,
        }

    def persist_to_disk(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "min_timesteps": self.min_timesteps,
            "max_timesteps": self.max_timesteps,
            "buffers": {road_id: list(buffer) for road_id, buffer in self.buffers.items()},
            "freshness": self.freshness,
            "saved_at": datetime.now(),
        }
        with path.open("wb") as stream:
            pickle.dump(state, stream)

    @classmethod
    def restore_from_disk(cls, path: str | Path) -> "LiveBufferManager":
        path = Path(path)
        with path.open("rb") as stream:
            state = pickle.load(stream)
        manager = cls(
            min_timesteps=state.get("min_timesteps", 24),
            max_timesteps=state.get("max_timesteps", 48),
        )
        for road_id, records in state.get("buffers", {}).items():
            manager.buffers[road_id] = deque(records, maxlen=manager.max_timesteps)
        manager.freshness = state.get("freshness", {})
        return manager

