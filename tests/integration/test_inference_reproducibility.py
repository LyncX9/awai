from traffic_prediction.inference.realtime import RealtimePredictionPipeline, RealtimePredictionContext
from traffic_prediction.inference.fallback import FallbackPredictor
from traffic_prediction.data.schemas import LiveTrafficRecord
from traffic_prediction.inference.cache import PredictionCache
from traffic_prediction.inference.confidence import ConfidenceAdjuster
from traffic_prediction.monitoring.data_quality import DataQualityMonitor
from traffic_prediction.features.online import OnlineFeatureResult
from traffic_prediction.api.app import PredictionRequest
from datetime import datetime, timezone
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

def test_inference_pipeline_is_deterministic_across_repeated_calls():
    # The pipeline should return identical predictions given the same state
    runner = MagicMock()
    runner.predict_kmh.return_value = np.array([45.0, 45.0, 45.0, 45.0])
    
    buffer = MagicMock()
    buffer.has_sufficient_data.return_value = True
    
    engineer = MagicMock()
    feature_result = OnlineFeatureResult(
        road_id="R1",
        feature_frame=pd.DataFrame(),
        sequence=np.zeros((1, 10, 5)),
        quality=MagicMock()
    )
    engineer.build_online_sequence.return_value = (feature_result, None)
    
    roads = pd.DataFrame([{"road_id": "R1"}])
    
    pipeline = RealtimePredictionPipeline(
        model_runner=runner,
        live_buffer=buffer,
        roads=roads,
        prediction_cache=PredictionCache(60),
        fallback_predictor=FallbackPredictor(),
        confidence_adjuster=ConfidenceAdjuster(),
        data_quality_monitor=DataQualityMonitor(buffer, roads),
        online_feature_engineer=engineer
    )
    
    req = PredictionRequest(road_id="R1", horizon_minutes=15)
    ctx = RealtimePredictionContext(
        request=req,
        road_record={"road_id": "R1"},
        requested_at=datetime.now(timezone.utc),
        target_time=datetime.now(timezone.utc),
        cache_key="R1_15",
        model_version="v1"
    )
    
    pred1 = pipeline.predict(ctx)
    pred2 = pipeline.predict(ctx)
    
    assert pred1.predicted_speed == pred2.predicted_speed
    assert pred1.congestion_level == pred2.congestion_level
    assert pred1.confidence_score == pred2.confidence_score
