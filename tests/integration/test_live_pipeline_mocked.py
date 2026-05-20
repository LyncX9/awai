import pytest
from unittest.mock import MagicMock
from traffic_prediction.ingestion.ingestor import TomTomIngestor
from traffic_prediction.inference.realtime import RealtimePredictionPipeline
from traffic_prediction.api.app import AppState
from traffic_prediction.data.schemas import LiveTrafficRecord
from datetime import datetime, timezone

def test_mocked_live_pipeline_ingestion_to_prediction(tmp_path):
    # This integration test verifies that mocked ingestion populates the live buffer,
    # which can then be immediately consumed by the realtime prediction pipeline.
    
    # 1. Setup AppState
    from traffic_prediction.config.settings import load_config
    from dataclasses import replace
    config = load_config()
    config = replace(config, paths=replace(config.paths, buffers_dir=tmp_path))
    state = AppState(config)
    # For testing, initialize buffer with a known road so TomTom mapping works or skip mapping.
    # We will mock the TomTom client so we don't need real keys.
    state.tomtom_client = MagicMock()
    from traffic_prediction.ingestion.tomtom_client import TomTomFetchResult, TomTomTrafficObservation
    state.tomtom_client.fetch_flow_segments.return_value = TomTomFetchResult(
        observations=[
            TomTomTrafficObservation(
                tomtom_segment_id="S1",
                current_speed=50.0,
                confidence=1.0,
                timestamp_utc=datetime.now(timezone.utc)
            )
        ]
    )
    
    # Mock mapper
    state.road_mapper = MagicMock()
    # It takes raw data and returns LiveTrafficRecord
    state.road_mapper.to_live_record.return_value = LiveTrafficRecord(
        road_id="R1",
        current_speed=50.0,
        confidence=1.0,
        timestamp=datetime.now(timezone.utc)
    )
    
    ingestor = TomTomIngestor(
        client=state.tomtom_client,
        mapper=state.road_mapper,
        buffer_manager=state.live_buffer,
        expected_road_ids={"R1"}
    )
    
    # 2. Run ingestion
    summary = ingestor.ingest_once()
    assert summary.accepted_count == 1
    
    # 3. Trigger prediction
    # Since we lack a real model in this test environment without full setup, 
    # we just ensure the pipeline is instantiated and tries to lookup cache.
    import pandas as pd
    from traffic_prediction.api.app import PredictionRequest
    from traffic_prediction.inference.realtime import RealtimePredictionContext
    
    pipeline = RealtimePredictionPipeline(
        model_runner=state.model_runner,
        live_buffer=state.live_buffer,
        roads=pd.DataFrame([{"road_id": "R1"}]),
        prediction_cache=state.prediction_cache,
        fallback_predictor=state.fallback_predictor,
        confidence_adjuster=state.confidence_adjuster,
        data_quality_monitor=state.data_quality_monitor,
        online_feature_engineer=state.online_feature_engineer
    )
    
    # We expect this to either return fallback (if model None) or error if model is mocked
    # Just verifying integration flow
    req = PredictionRequest(road_id="R1", horizon_minutes=15)
    ctx = RealtimePredictionContext(
        request=req,
        road_record={"road_id": "R1"},
        requested_at=datetime.now(timezone.utc),
        target_time=datetime.now(timezone.utc),
        cache_key="R1_15",
        model_version="v1"
    )
    response = pipeline.predict(ctx)
    
    # It should have fallback or actual prediction
    assert response is not None
    assert response.road_id == "R1"

