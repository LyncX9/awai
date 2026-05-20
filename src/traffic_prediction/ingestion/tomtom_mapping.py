from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from traffic_prediction.data.schemas import LiveTrafficRecord
from traffic_prediction.ingestion.tomtom_client import TomTomSegmentQuery, TomTomTrafficObservation


class TomTomMappingError(ValueError):
    """Raised when a TomTom segment cannot be mapped to an internal road id."""


@dataclass(frozen=True)
class TomTomRoadMapping:
    tomtom_segment_id: str
    road_id: str
    latitude: float
    longitude: float
    version: str = "mapping-v1"


class TomTomRoadMapper:
    REQUIRED_COLUMNS = {"tomtom_segment_id", "road_id", "latitude", "longitude"}

    def __init__(self, mappings: list[TomTomRoadMapping], timezone: str = "Asia/Jakarta") -> None:
        if not mappings:
            raise TomTomMappingError("TomTom mapping is empty")
        self.mappings = mappings
        self.timezone = ZoneInfo(timezone)
        self._by_segment_id: dict[str, TomTomRoadMapping] = {}
        for mapping in mappings:
            if mapping.tomtom_segment_id in self._by_segment_id:
                raise TomTomMappingError(f"Duplicate TomTom segment id: {mapping.tomtom_segment_id}")
            self._by_segment_id[mapping.tomtom_segment_id] = mapping

    @classmethod
    def from_csv(cls, path: str | Path, timezone: str = "Asia/Jakarta") -> "TomTomRoadMapper":
        frame = pd.read_csv(path)
        missing = cls.REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise TomTomMappingError(f"TomTom mapping file is missing columns: {sorted(missing)}")
        mappings = [
            TomTomRoadMapping(
                tomtom_segment_id=str(row["tomtom_segment_id"]),
                road_id=str(row["road_id"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                version=str(row.get("version", "mapping-v1")),
            )
            for _, row in frame.iterrows()
        ]
        return cls(mappings, timezone=timezone)

    @classmethod
    def from_roads_master(
        cls,
        roads: pd.DataFrame,
        timezone: str = "Asia/Jakarta",
        version: str = "roads-midpoint-placeholder-v1",
    ) -> "TomTomRoadMapper":
        required = {"road_id", "mid_lat", "mid_lon"}
        missing = required.difference(roads.columns)
        if missing:
            raise TomTomMappingError(f"Road master is missing columns for fallback mapping: {sorted(missing)}")
        mappings = [
            TomTomRoadMapping(
                tomtom_segment_id=str(row["road_id"]),
                road_id=str(row["road_id"]),
                latitude=float(row["mid_lat"]),
                longitude=float(row["mid_lon"]),
                version=version,
            )
            for _, row in roads.dropna(subset=["road_id", "mid_lat", "mid_lon"]).iterrows()
        ]
        return cls(mappings, timezone=timezone)

    def to_queries(self) -> list[TomTomSegmentQuery]:
        return [
            TomTomSegmentQuery(
                tomtom_segment_id=mapping.tomtom_segment_id,
                latitude=mapping.latitude,
                longitude=mapping.longitude,
            )
            for mapping in self.mappings
        ]

    def road_id_for_segment(self, tomtom_segment_id: str) -> str:
        mapping = self._by_segment_id.get(tomtom_segment_id)
        if mapping is None:
            raise TomTomMappingError(f"Missing road mapping for TomTom segment id: {tomtom_segment_id}")
        return mapping.road_id

    def to_live_record(
        self,
        observation: TomTomTrafficObservation,
        received_at: datetime | None = None,
    ) -> LiveTrafficRecord:
        road_id = self.road_id_for_segment(observation.tomtom_segment_id)
        timestamp = observation.timestamp_utc.astimezone(self.timezone)
        freshness = None
        if received_at is not None:
            freshness = received_at.astimezone(self.timezone) - timestamp
        return LiveTrafficRecord(
            road_id=road_id,
            current_speed=observation.current_speed,
            confidence=observation.confidence,
            timestamp=timestamp,
            freshness_indicator=freshness,
        )

    def to_live_records(
        self,
        observations: list[TomTomTrafficObservation],
        received_at: datetime | None = None,
    ) -> list[LiveTrafficRecord]:
        return [self.to_live_record(observation, received_at=received_at) for observation in observations]
