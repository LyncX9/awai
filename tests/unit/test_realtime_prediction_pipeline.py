from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from traffic_prediction.api.schemas import PredictionRequest
from traffic_prediction.data.schemas import FeatureManifest, LiveTrafficRecord
from traffic_prediction.features.online import OnlineFeatureEngineer
from traffic_prediction.inference.cache import PredictionCache
from traffic_prediction.inference.confidence import ConfidenceAdjuster
from traffic_prediction.inference.fallback import FallbackPredictor
from traffic_prediction.inference.realtime import RealtimePredictionContext, RealtimePredictionPipeline
from traffic_prediction.ingestion.buffer import LiveBufferManager
from traffic_prediction.monitoring.data_quality import DataQualityMonitor


class StaticModelRunner:
    def predict_kmh(self, sequence: np.ndarray) -> np.ndarray:
        assert sequence.shape == (1, 3, 5)
        return np.array([30.0, 31.0, 32.0, 33.0], dtype=float)


def test_realtime_prediction_pipeline_returns_live_model_response_and_caches_it() -> None:
    pipeline, context = _pipeline(model_runner=StaticModelRunner())

    response = pipeline.predict(context)
    cached = pipeline.predict(context)

    assert response.prediction_method == "live_lstm_runtime"
    assert response.predicted_speed == 33.0
    assert response.degraded is False
    assert response.metadata["online_features_built"] is True
    assert response.metadata["model_runner_available"] is True
    assert response.metadata["cache_hit"] is False
    assert cached.metadata["cache_hit"] is True
    assert cached.predicted_speed == response.predicted_speed


def test_realtime_prediction_pipeline_falls_back_without_runtime_model_runner() -> None:
    pipeline, context = _pipeline(model_runner=None)

    response = pipeline.predict(context)

    assert response.prediction_method == "historical_average_fallback"
    assert response.degraded is True
    assert response.metadata["realtime_pipeline"] is True
    assert response.metadata["online_features_built"] is True
    assert response.metadata["model_runner_available"] is False
    assert response.metadata["feature_quality"]["status"] == "healthy"


def _pipeline(model_runner) -> tuple[RealtimePredictionPipeline, RealtimePredictionContext]:
    start = datetime(2026, 5, 19, 7, 0)
    roads = pd.DataFrame(
        {
            "road_id": ["R1", "R2"],
            "road_weight": [0.4, 0.6],
            "free_flow_speed": [45.0, 50.0],
            "mid_lat": [-6.9, -6.91],
            "mid_lon": [106.9, 106.91],
        }
    )
    buffer = LiveBufferManager(min_timesteps=3, max_timesteps=8)
    for index in range(4):
        timestamp = start + timedelta(minutes=15 * index)
        buffer.append(LiveTrafficRecord("R1", 20.0 + index, 0.95, timestamp))
        buffer.append(LiveTrafficRecord("R2", 30.0 + index, 0.95, timestamp))

    manifest = FeatureManifest(
        feature_columns=["current_speed", "confidence", "lag_1", "rolling_mean_3", "neighbor_speed_mean"],
        target_column="current_speed",
        lookback=3,
        horizon=4,
    )
    engineer = OnlineFeatureEngineer(
        manifest=manifest,
        buffer_manager=buffer,
        roads=roads,
        neighbor_mapping={"R1": ["R2"], "R2": ["R1"]},
    )
    pipeline = RealtimePredictionPipeline(
        live_buffer=buffer,
        roads=roads,
        prediction_cache=PredictionCache(ttl_seconds=900),
        fallback_predictor=FallbackPredictor(global_default_speed=29.0),
        confidence_adjuster=ConfidenceAdjuster(),
        data_quality_monitor=DataQualityMonitor(),
        online_feature_engineer=engineer,
        model_runner=model_runner,
    )
    requested_at = start + timedelta(minutes=45)
    request = PredictionRequest(road_id="R1", horizon_minutes=60, requested_at=requested_at)
    context = RealtimePredictionContext(
        request=request,
        road_record=roads.iloc[0].to_dict(),
        requested_at=requested_at,
        target_time=requested_at + timedelta(minutes=60),
        cache_key="model:R1:60:2026-05-19T07:45:00",
        model_version="model",
    )
    return pipeline, context
