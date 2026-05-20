from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from traffic_prediction.api.schemas import PredictionResponse
from traffic_prediction.data.schemas import LiveTrafficRecord
from traffic_prediction.models.registry import ModelRegistryEntry
from traffic_prediction.persistence.postgresql import PostgreSQLPersistence


@pytest.fixture
def mock_db():
    with patch("psycopg2.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        persistence = PostgreSQLPersistence("postgresql://user:pass@host:5432/dbname")
        yield persistence, mock_connect, mock_conn, mock_cursor


def test_postgresql_persistence_initializes_expected_tables(mock_db) -> None:
    persistence, mock_connect, mock_conn, mock_cursor = mock_db

    persistence.initialize()

    mock_connect.assert_called_once_with("postgresql://user:pass@host:5432/dbname")
    mock_cursor.execute.assert_called_once()
    sql_arg = mock_cursor.execute.call_args[0][0]
    
    assert "CREATE TABLE IF NOT EXISTS live_traffic_records" in sql_arg
    assert "CREATE TABLE IF NOT EXISTS predictions" in sql_arg
    assert "CREATE TABLE IF NOT EXISTS model_registry" in sql_arg
    mock_conn.commit.assert_called_once()


def test_postgresql_persistence_upserts_live_records(mock_db) -> None:
    persistence, _, _, mock_cursor = mock_db
    timestamp = datetime(2026, 5, 20, 8, 15)
    record = LiveTrafficRecord(
        road_id="R1",
        current_speed=35.5,
        confidence=0.95,
        timestamp=timestamp,
        freshness_indicator=timedelta(seconds=45),
    )

    count = persistence.upsert_live_records([record], source="manual")

    assert count == 1
    assert mock_cursor.executemany.call_count == 1
    
    args = mock_cursor.executemany.call_args[0]
    sql, params = args[0], args[1]
    
    assert "INSERT INTO live_traffic_records" in sql
    assert "ON CONFLICT(road_id, timestamp_wib) DO UPDATE" in sql
    assert params[0] == ("R1", timestamp.isoformat(), 35.5, 0.95, 45, "manual")


def test_postgresql_persistence_retrieves_latest_live_records(mock_db) -> None:
    persistence, _, _, mock_cursor = mock_db
    timestamp = datetime(2026, 5, 20, 8, 15)
    
    mock_cursor.fetchall.return_value = [
        {
            "road_id": "R1",
            "timestamp_wib": timestamp.isoformat(),
            "current_speed": 35.5,
            "confidence": 0.95,
            "freshness_seconds": 45,
        }
    ]

    records = persistence.latest_live_records(road_id="R1", limit=50)

    assert len(records) == 1
    assert records[0].road_id == "R1"
    assert records[0].current_speed == 35.5
    assert records[0].confidence == 0.95
    assert records[0].timestamp == timestamp
    assert records[0].freshness_indicator == timedelta(seconds=45)

    mock_cursor.execute.assert_called()
    sql, params = mock_cursor.execute.call_args[0]
    assert "SELECT road_id, timestamp_wib" in sql
    assert "WHERE road_id = %s" in sql
    assert params == ("R1", 50)


def test_postgresql_persistence_stores_prediction(mock_db) -> None:
    persistence, _, _, mock_cursor = mock_db
    prediction = PredictionResponse(
        road_id="R1",
        horizon_minutes=30,
        predicted_speed=42.0,
        congestion_level="free_flow",
        uncertainty_lower=38.0,
        uncertainty_upper=46.0,
        confidence_score=0.90,
        model_version="lstm-v2",
        prediction_method="lstm_real_model",
        degraded=False,
        data_quality={"status": "optimal", "completeness": 1.0},
        metadata={"cache_hit": False},
    )

    pred_id = persistence.insert_prediction(
        prediction,
        requested_at=datetime(2026, 5, 20, 8, 0),
        prediction_id="pred-123",
    )

    assert pred_id == "pred-123"
    mock_cursor.execute.assert_called()
    sql, params = mock_cursor.execute.call_args_list[-1][0]
    
    assert "INSERT INTO predictions" in sql
    assert params[0] == "pred-123"
    assert params[1] == "lstm-v2"
    assert params[2] == "R1"
    assert params[3] == datetime(2026, 5, 20, 8, 0).isoformat()
    assert params[4] == 30
    assert params[5] == 42.0
    assert params[6] == 38.0
    assert params[7] == 46.0
    assert params[8] == 0.90
    assert params[9] == "optimal"
    
    meta = json.loads(params[10])
    assert meta["prediction_method"] == "lstm_real_model"
    assert meta["degraded"] is False


def test_postgresql_persistence_retrieves_latest_predictions(mock_db) -> None:
    persistence, _, _, mock_cursor = mock_db
    
    mock_cursor.fetchall.return_value = [
        {
            "prediction_id": "pred-123",
            "model_version": "lstm-v2",
            "road_id": "R1",
            "requested_at_wib": datetime(2026, 5, 20, 8, 0).isoformat(),
            "horizon_minutes": 30,
            "predicted_speed": 42.0,
            "lower_bound": 38.0,
            "upper_bound": 46.0,
            "confidence_score": 0.90,
            "quality_status": "optimal",
            "metadata_json": json.dumps({
                "degraded": False,
                "prediction_method": "lstm_real_model",
                "congestion_level": "free_flow",
                "data_quality": {"status": "optimal"},
                "metadata": {},
            }),
        }
    ]

    preds = persistence.latest_predictions(road_id="R1", limit=10)

    assert len(preds) == 1
    assert preds[0].prediction_id == "pred-123"
    assert preds[0].model_version == "lstm-v2"
    assert preds[0].road_id == "R1"
    assert preds[0].predicted_speed == 42.0
    assert preds[0].lower_bound == 38.0
    assert preds[0].upper_bound == 46.0
    assert preds[0].confidence_score == 0.90
    assert preds[0].quality_status == "optimal"
    assert preds[0].metadata["prediction_method"] == "lstm_real_model"


def test_postgresql_persistence_mirrors_model_registry(mock_db) -> None:
    persistence, _, _, mock_cursor = mock_db
    entry = ModelRegistryEntry(
        model_version="lstm-v2",
        artifact_path="artifacts/models/lstm-v2",
        created_at="2026-05-20T08:00:00",
        model_type="lstm",
        framework="pytorch",
        metrics={"rmse": 3.2},
        config={"hidden_size": 64},
        tags=["production"],
        is_active=True,
    )

    persistence.upsert_model_entry(entry)
    
    # Assert DDL update and insert execution
    assert mock_cursor.execute.call_count >= 2
    
    # Find the registry insert execute call
    insert_call = None
    for call in mock_cursor.execute.call_args_list:
        sql = call[0][0]
        if "INSERT INTO model_registry" in sql:
            insert_call = call
            break
            
    assert insert_call is not None
    sql, params = insert_call[0]
    
    assert params[0] == "lstm-v2"
    assert params[1] == "2026-05-20T08:00:00"
    assert params[2] == "artifacts/models/lstm-v2"
    assert json.loads(params[3]) == {"rmse": 3.2}
    assert json.loads(params[4]) == {"hidden_size": 64}
    assert params[5] == "lstm"
    assert params[6] == "pytorch"
    assert json.loads(params[7]) == ["production"]
    assert params[8] is True
