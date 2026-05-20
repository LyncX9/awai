from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from traffic_prediction.api.schemas import PredictionResponse
from traffic_prediction.data.schemas import LiveTrafficRecord
from traffic_prediction.models.registry import ModelRegistryEntry


@dataclass(frozen=True)
class StoredPrediction:
    prediction_id: str
    model_version: str
    road_id: str
    requested_at_wib: str
    horizon_minutes: int
    predicted_speed: float
    lower_bound: float | None
    upper_bound: float | None
    confidence_score: float | None
    quality_status: str
    metadata: dict[str, Any]


class SQLitePersistence:
    """Optional SQLite persistence layer.

    The application remains file-backed by default. This adapter is intended for
    deployments that need queryable local history without introducing a service
    dependency.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        self._memory_connection: sqlite3.Connection | None = None
        if self.database_path == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:")
            self._memory_connection.row_factory = sqlite3.Row
        else:
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS live_traffic_records (
                    road_id TEXT NOT NULL,
                    timestamp_wib TEXT NOT NULL,
                    current_speed REAL NOT NULL,
                    confidence REAL NOT NULL,
                    freshness_seconds INTEGER,
                    source TEXT NOT NULL DEFAULT 'tomtom',
                    PRIMARY KEY (road_id, timestamp_wib)
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    model_version TEXT NOT NULL,
                    road_id TEXT NOT NULL,
                    requested_at_wib TEXT NOT NULL,
                    horizon_minutes INTEGER NOT NULL,
                    predicted_speed REAL NOT NULL,
                    lower_bound REAL,
                    upper_bound REAL,
                    confidence_score REAL,
                    quality_status TEXT NOT NULL,
                    metadata_json TEXT
                );

                CREATE TABLE IF NOT EXISTS model_registry (
                    model_version TEXT PRIMARY KEY,
                    created_at_wib TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    model_type TEXT NOT NULL DEFAULT 'lstm',
                    framework TEXT NOT NULL DEFAULT 'pytorch',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    is_active INTEGER NOT NULL DEFAULT 0
                );
                """
            )

    def upsert_live_record(self, record: LiveTrafficRecord, source: str = "tomtom") -> None:
        self.upsert_live_records([record], source=source)

    def upsert_live_records(self, records: list[LiveTrafficRecord], source: str = "tomtom") -> int:
        if not records:
            return 0
        self.initialize()
        rows = [
            (
                record.road_id,
                record.timestamp.isoformat(),
                float(record.current_speed),
                float(record.confidence),
                self._freshness_seconds(record.freshness_indicator),
                source,
            )
            for record in records
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO live_traffic_records (
                    road_id, timestamp_wib, current_speed, confidence, freshness_seconds, source
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(road_id, timestamp_wib) DO UPDATE SET
                    current_speed = excluded.current_speed,
                    confidence = excluded.confidence,
                    freshness_seconds = excluded.freshness_seconds,
                    source = excluded.source
                """,
                rows,
            )
        return len(rows)

    def latest_live_records(self, road_id: str | None = None, limit: int = 100) -> list[LiveTrafficRecord]:
        self.initialize()
        where_clause = "WHERE road_id = ?" if road_id is not None else ""
        params: tuple[Any, ...] = (road_id, limit) if road_id is not None else (limit,)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT road_id, timestamp_wib, current_speed, confidence, freshness_seconds
                FROM live_traffic_records
                {where_clause}
                ORDER BY timestamp_wib DESC, road_id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [
            LiveTrafficRecord(
                road_id=str(row["road_id"]),
                current_speed=float(row["current_speed"]),
                confidence=float(row["confidence"]),
                timestamp=datetime.fromisoformat(str(row["timestamp_wib"])),
                freshness_indicator=(
                    timedelta(seconds=int(row["freshness_seconds"]))
                    if row["freshness_seconds"] is not None
                    else None
                ),
            )
            for row in rows
        ]

    def insert_prediction(
        self,
        prediction: PredictionResponse,
        *,
        requested_at: datetime,
        prediction_id: str | None = None,
    ) -> str:
        self.initialize()
        stored_id = prediction_id or uuid4().hex
        quality_status = str(prediction.data_quality.get("status", "unknown"))
        metadata = {
            "degraded": prediction.degraded,
            "prediction_method": prediction.prediction_method,
            "congestion_level": prediction.congestion_level,
            "data_quality": prediction.data_quality,
            "metadata": prediction.metadata,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    prediction_id, model_version, road_id, requested_at_wib, horizon_minutes,
                    predicted_speed, lower_bound, upper_bound, confidence_score, quality_status, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stored_id,
                    prediction.model_version or "unversioned",
                    prediction.road_id,
                    requested_at.isoformat(),
                    int(prediction.horizon_minutes),
                    float(prediction.predicted_speed),
                    float(prediction.uncertainty_lower),
                    float(prediction.uncertainty_upper),
                    float(prediction.confidence_score),
                    quality_status,
                    json.dumps(metadata),
                ),
            )
        return stored_id

    def latest_predictions(self, road_id: str | None = None, limit: int = 100) -> list[StoredPrediction]:
        self.initialize()
        where_clause = "WHERE road_id = ?" if road_id is not None else ""
        params: tuple[Any, ...] = (road_id, limit) if road_id is not None else (limit,)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT prediction_id, model_version, road_id, requested_at_wib, horizon_minutes,
                       predicted_speed, lower_bound, upper_bound, confidence_score, quality_status, metadata_json
                FROM predictions
                {where_clause}
                ORDER BY requested_at_wib DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [
            StoredPrediction(
                prediction_id=str(row["prediction_id"]),
                model_version=str(row["model_version"]),
                road_id=str(row["road_id"]),
                requested_at_wib=str(row["requested_at_wib"]),
                horizon_minutes=int(row["horizon_minutes"]),
                predicted_speed=float(row["predicted_speed"]),
                lower_bound=float(row["lower_bound"]) if row["lower_bound"] is not None else None,
                upper_bound=float(row["upper_bound"]) if row["upper_bound"] is not None else None,
                confidence_score=float(row["confidence_score"]) if row["confidence_score"] is not None else None,
                quality_status=str(row["quality_status"]),
                metadata=json.loads(row["metadata_json"] or "{}"),
            )
            for row in rows
        ]

    def upsert_model_entry(self, entry: ModelRegistryEntry) -> None:
        self.initialize()
        with self._connect() as connection:
            if entry.is_active:
                connection.execute("UPDATE model_registry SET is_active = 0")
            connection.execute(
                """
                INSERT INTO model_registry (
                    model_version, created_at_wib, artifact_path, metrics_json, config_json,
                    model_type, framework, tags_json, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_version) DO UPDATE SET
                    created_at_wib = excluded.created_at_wib,
                    artifact_path = excluded.artifact_path,
                    metrics_json = excluded.metrics_json,
                    config_json = excluded.config_json,
                    model_type = excluded.model_type,
                    framework = excluded.framework,
                    tags_json = excluded.tags_json,
                    is_active = excluded.is_active
                """,
                (
                    entry.model_version,
                    entry.created_at,
                    entry.artifact_path,
                    json.dumps(entry.metrics),
                    json.dumps(entry.config),
                    entry.model_type,
                    entry.framework,
                    json.dumps(entry.tags),
                    1 if entry.is_active else 0,
                ),
            )

    def list_model_entries(self) -> list[ModelRegistryEntry]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT model_version, created_at_wib, artifact_path, metrics_json, config_json,
                       model_type, framework, tags_json, is_active
                FROM model_registry
                ORDER BY created_at_wib ASC
                """
            ).fetchall()
        return [
            ModelRegistryEntry(
                model_version=str(row["model_version"]),
                artifact_path=str(row["artifact_path"]),
                created_at=str(row["created_at_wib"]),
                model_type=str(row["model_type"]),
                framework=str(row["framework"]),
                metrics=json.loads(row["metrics_json"] or "{}"),
                config=json.loads(row["config_json"] or "{}"),
                tags=json.loads(row["tags_json"] or "[]"),
                is_active=bool(row["is_active"]),
            )
            for row in rows
        ]

    def active_model_entry(self) -> ModelRegistryEntry | None:
        return next((entry for entry in self.list_model_entries() if entry.is_active), None)

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return self._memory_connection
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _freshness_seconds(value: timedelta | None) -> int | None:
        if value is None:
            return None
        return int(value.total_seconds())
