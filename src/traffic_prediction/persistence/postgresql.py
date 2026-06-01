from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Generator
from uuid import uuid4

import psycopg2
from psycopg2.extras import DictCursor

from traffic_prediction.api.schemas import PredictionResponse
from traffic_prediction.data.schemas import LiveTrafficRecord
from traffic_prediction.models.registry import ModelRegistryEntry
from traffic_prediction.persistence.sqlite import StoredPrediction

logger = logging.getLogger(__name__)


class PostgreSQLPersistence:
    """PostgreSQL/Supabase persistence layer.

    Synchronizes live traffic buffers, forecast predictions, and model registry
    entries to a managed PostgreSQL database. Helps maintain state and seed buffers
    in containerized, stateless deployments like Render.
    """

    def __init__(self, connection_uri: str) -> None:
        self.connection_uri = connection_uri

    @contextmanager
    def _connect(self) -> Generator[psycopg2.extensions.connection, None, None]:
        connection = psycopg2.connect(self.connection_uri)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS live_traffic_records (
                        road_id VARCHAR(50) NOT NULL,
                        timestamp_wib VARCHAR(50) NOT NULL,
                        current_speed DOUBLE PRECISION NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        freshness_seconds INTEGER,
                        source VARCHAR(50) NOT NULL DEFAULT 'tomtom',
                        PRIMARY KEY (road_id, timestamp_wib)
                    );

                    CREATE TABLE IF NOT EXISTS predictions (
                        prediction_id VARCHAR(50) PRIMARY KEY,
                        model_version VARCHAR(100) NOT NULL,
                        road_id VARCHAR(50) NOT NULL,
                        requested_at_wib VARCHAR(50) NOT NULL,
                        horizon_minutes INTEGER NOT NULL,
                        predicted_speed DOUBLE PRECISION NOT NULL,
                        lower_bound DOUBLE PRECISION,
                        upper_bound DOUBLE PRECISION,
                        confidence_score DOUBLE PRECISION,
                        quality_status VARCHAR(50) NOT NULL,
                        metadata_json TEXT
                    );

                    CREATE TABLE IF NOT EXISTS model_registry (
                        model_version VARCHAR(100) PRIMARY KEY,
                        created_at_wib VARCHAR(50) NOT NULL,
                        artifact_path VARCHAR(255) NOT NULL,
                        metrics_json TEXT NOT NULL,
                        config_json TEXT NOT NULL,
                        model_type VARCHAR(50) NOT NULL DEFAULT 'lstm',
                        framework VARCHAR(50) NOT NULL DEFAULT 'pytorch',
                        tags_json TEXT NOT NULL DEFAULT '[]',
                        is_active BOOLEAN NOT NULL DEFAULT FALSE
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
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO live_traffic_records (
                        road_id, timestamp_wib, current_speed, confidence, freshness_seconds, source
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT(road_id, timestamp_wib) DO UPDATE SET
                        current_speed = EXCLUDED.current_speed,
                        confidence = EXCLUDED.confidence,
                        freshness_seconds = EXCLUDED.freshness_seconds,
                        source = EXCLUDED.source
                    """,
                    rows,
                )
        return len(rows)

    def latest_live_records(self, road_id: str | None = None, limit: int = 100) -> list[LiveTrafficRecord]:
        self.initialize()
        where_clause = "WHERE road_id = %s" if road_id is not None else ""
        params: tuple[Any, ...] = (road_id, limit) if road_id is not None else (limit,)
        with self._connect() as connection:
            with connection.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(
                    f"""
                    SELECT road_id, timestamp_wib, current_speed, confidence, freshness_seconds
                    FROM live_traffic_records
                    {where_clause}
                    ORDER BY timestamp_wib DESC, road_id ASC
                    LIMIT %s
                    """,
                    params,
                )
                rows = cursor.fetchall()
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

    def clear_live_records(self) -> None:
        self.initialize()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM live_traffic_records")

    def prune_live_records(self, retain_hours: int = 12) -> int:
        """Delete live traffic records older than ``retain_hours`` hours.

        The LSTM model needs at most 24 timesteps (= 6 hours at 15-min intervals)
        to make predictions.  Keeping 12 hours gives 2x safety margin while
        drastically cutting row count and therefore database storage.

        Returns the number of rows deleted.
        """
        self.initialize()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=retain_hours)
        cutoff_str = cutoff.isoformat()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM live_traffic_records
                    WHERE timestamp_wib < %s
                    """,
                    (cutoff_str,),
                )
                deleted = cursor.rowcount
        logger.info("pruned_live_records", extra={"deleted": deleted, "retain_hours": retain_hours})
        return deleted

    def prune_old_predictions(self, retain_days: int = 3) -> int:
        """Delete prediction log rows older than ``retain_days`` days.

        Prediction history older than a few days is not needed for real-time
        operation.  Keeping 3 days of logs is sufficient for drift monitoring
        while keeping the table small.

        Returns the number of rows deleted.
        """
        self.initialize()
        cutoff = datetime.now(timezone.utc) - timedelta(days=retain_days)
        cutoff_str = cutoff.isoformat()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM predictions
                    WHERE requested_at_wib < %s
                    """,
                    (cutoff_str,),
                )
                deleted = cursor.rowcount
        logger.info("pruned_old_predictions", extra={"deleted": deleted, "retain_days": retain_days})
        return deleted

    def get_db_size_stats(self) -> dict[str, Any]:
        """Return row counts and estimated table sizes for monitoring.

        Uses ``pg_total_relation_size`` to estimate disk bytes per table.
        Values are approximations; actual Supabase billing uses a slightly
        different measurement.
        """
        self.initialize()
        stats: dict[str, Any] = {}
        tables = ("live_traffic_records", "predictions", "model_registry")
        with self._connect() as connection:
            with connection.cursor() as cursor:
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                    cursor.execute(
                        "SELECT pg_total_relation_size(%s)",
                        (table,),
                    )
                    size_bytes = cursor.fetchone()[0]
                    stats[table] = {
                        "row_count": row_count,
                        "size_bytes": size_bytes,
                        "size_mb": round(size_bytes / (1024 * 1024), 3),
                    }
        total_bytes = sum(t["size_bytes"] for t in stats.values())
        stats["total"] = {
            "size_bytes": total_bytes,
            "size_mb": round(total_bytes / (1024 * 1024), 3),
        }
        return stats

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
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO predictions (
                        prediction_id, model_version, road_id, requested_at_wib, horizon_minutes,
                        predicted_speed, lower_bound, upper_bound, confidence_score, quality_status, metadata_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        stored_id,
                        prediction.model_version or "unversioned",
                        prediction.road_id,
                        requested_at.isoformat(),
                        int(prediction.horizon_minutes),
                        float(prediction.predicted_speed),
                        float(prediction.uncertainty_lower) if prediction.uncertainty_lower is not None else None,
                        float(prediction.uncertainty_upper) if prediction.uncertainty_upper is not None else None,
                        float(prediction.confidence_score) if prediction.confidence_score is not None else None,
                        quality_status,
                        json.dumps(metadata),
                    ),
                )
        return stored_id

    def latest_predictions(self, road_id: str | None = None, limit: int = 100) -> list[StoredPrediction]:
        self.initialize()
        where_clause = "WHERE road_id = %s" if road_id is not None else ""
        params: tuple[Any, ...] = (road_id, limit) if road_id is not None else (limit,)
        with self._connect() as connection:
            with connection.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(
                    f"""
                    SELECT prediction_id, model_version, road_id, requested_at_wib, horizon_minutes,
                           predicted_speed, lower_bound, upper_bound, confidence_score, quality_status, metadata_json
                    FROM predictions
                    {where_clause}
                    ORDER BY requested_at_wib DESC
                    LIMIT %s
                    """,
                    params,
                )
                rows = cursor.fetchall()
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
            with connection.cursor() as cursor:
                if entry.is_active:
                    cursor.execute("UPDATE model_registry SET is_active = FALSE")
                cursor.execute(
                    """
                    INSERT INTO model_registry (
                        model_version, created_at_wib, artifact_path, metrics_json, config_json,
                        model_type, framework, tags_json, is_active
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(model_version) DO UPDATE SET
                        created_at_wib = EXCLUDED.created_at_wib,
                        artifact_path = EXCLUDED.artifact_path,
                        metrics_json = EXCLUDED.metrics_json,
                        config_json = EXCLUDED.config_json,
                        model_type = EXCLUDED.model_type,
                        framework = EXCLUDED.framework,
                        tags_json = EXCLUDED.tags_json,
                        is_active = EXCLUDED.is_active
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
                        entry.is_active,
                    ),
                )

    def list_model_entries(self) -> list[ModelRegistryEntry]:
        self.initialize()
        with self._connect() as connection:
            with connection.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT model_version, created_at_wib, artifact_path, metrics_json, config_json,
                           model_type, framework, tags_json, is_active
                    FROM model_registry
                    ORDER BY created_at_wib ASC
                    """
                )
                rows = cursor.fetchall()
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

    @staticmethod
    def _freshness_seconds(value: timedelta | None) -> int | None:
        if value is None:
            return None
        return int(value.total_seconds())
