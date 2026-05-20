from __future__ import annotations

from datetime import datetime, timedelta
from traffic_prediction.api.schemas import PredictionResponse
from traffic_prediction.data.schemas import LiveTrafficRecord
from traffic_prediction.models.registry import ModelRegistryEntry
from traffic_prediction.persistence.sqlite import SQLitePersistence


def test_sqlite_persistence_initializes_expected_tables() -> None:
    store = SQLitePersistence(_db_path("schema"))

    store.initialize()

    with store._connect() as connection:
        table_names = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert {"live_traffic_records", "predictions", "model_registry"}.issubset(table_names)


def test_sqlite_persistence_round_trips_live_records() -> None:
    store = SQLitePersistence(_db_path("live"))
    timestamp = datetime(2026, 5, 19, 8, 15)
    record = LiveTrafficRecord(
        road_id="R1",
        current_speed=24.5,
        confidence=0.91,
        timestamp=timestamp,
        freshness_indicator=timedelta(seconds=30),
    )

    count = store.upsert_live_records([record], source="manual")
    store.upsert_live_record(
        LiveTrafficRecord("R1", 25.5, 0.92, timestamp, timedelta(seconds=10)),
        source="manual",
    )
    latest = store.latest_live_records("R1")

    assert count == 1
    assert len(latest) == 1
    assert latest[0].current_speed == 25.5
    assert latest[0].freshness_indicator == timedelta(seconds=10)


def test_sqlite_persistence_stores_predictions() -> None:
    store = SQLitePersistence(_db_path("predictions"))
    prediction = PredictionResponse(
        road_id="R1",
        horizon_minutes=60,
        predicted_speed=28.0,
        congestion_level="moderate",
        uncertainty_lower=23.0,
        uncertainty_upper=33.0,
        confidence_score=0.81,
        model_version="lstm-v1",
        prediction_method="historical_average_fallback",
        degraded=True,
        data_quality={"status": "degraded", "completeness": 0.7},
        metadata={"cache_hit": False},
    )

    prediction_id = store.insert_prediction(
        prediction,
        requested_at=datetime(2026, 5, 19, 8, 0),
        prediction_id="prediction-1",
    )
    latest = store.latest_predictions("R1")

    assert prediction_id == "prediction-1"
    assert latest[0].model_version == "lstm-v1"
    assert latest[0].quality_status == "degraded"
    assert latest[0].metadata["prediction_method"] == "historical_average_fallback"


def test_sqlite_persistence_mirrors_model_registry_entries() -> None:
    store = SQLitePersistence(_db_path("registry"))
    first = ModelRegistryEntry(
        model_version="v1",
        artifact_path="artifacts/models/v1",
        created_at="2026-05-19T08:00:00",
        metrics={"rmse": 4.0},
        config={"hidden": 32},
        tags=["candidate"],
        is_active=True,
    )
    second = ModelRegistryEntry(
        model_version="v2",
        artifact_path="artifacts/models/v2",
        created_at="2026-05-19T09:00:00",
        metrics={"rmse": 3.5},
        config={"hidden": 64},
        tags=["candidate"],
        is_active=True,
    )

    store.upsert_model_entry(first)
    store.upsert_model_entry(second)
    entries = store.list_model_entries()
    active = store.active_model_entry()

    assert [entry.model_version for entry in entries] == ["v1", "v2"]
    assert active is not None
    assert active.model_version == "v2"
    assert active.metrics["rmse"] == 3.5


def _db_path(name: str) -> str:
    return ":memory:"
