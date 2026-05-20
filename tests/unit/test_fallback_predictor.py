from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from traffic_prediction.data.schemas import LiveTrafficRecord
from traffic_prediction.inference.fallback import FallbackPredictor


def test_fallback_predictor_uses_road_hour_day_average() -> None:
    lookup = pd.Series(
        [24.5],
        index=pd.MultiIndex.from_tuples(
            [("R1", 8, 1)],
            names=["road_id", "hour", "day_of_week"],
        ),
    )
    predictor = FallbackPredictor(historical_lookup=lookup)

    prediction = predictor.predict("R1", datetime(2026, 5, 19, 8, 0), horizon_minutes=60)

    assert prediction.predicted_speed == pytest.approx(24.5)
    assert prediction.method == "historical_average_fallback"
    assert prediction.lookup_quality == "road_hour_day_average"
    assert prediction.confidence_score == pytest.approx(0.72)
    assert prediction.degraded is True


def test_fallback_predictor_uses_road_average_then_global_default() -> None:
    road_mean = pd.Series([31.0], index=pd.Index(["R1"], name="road_id"))
    predictor = FallbackPredictor(road_mean_speed=road_mean, global_default_speed=29.0)

    road_average = predictor.predict("R1", datetime(2026, 5, 19, 8, 0), horizon_minutes=30)
    global_default = predictor.predict("missing", datetime(2026, 5, 19, 8, 0), horizon_minutes=30)

    assert road_average.predicted_speed == pytest.approx(31.0)
    assert road_average.lookup_quality == "road_average"
    assert global_default.predicted_speed == pytest.approx(29.0)
    assert global_default.lookup_quality == "global_default"


def test_fallback_predictor_supports_persistence_fallback() -> None:
    record = LiveTrafficRecord(
        road_id="R1",
        current_speed=18.0,
        confidence=0.61,
        timestamp=datetime(2026, 5, 19, 8, 0),
    )
    predictor = FallbackPredictor()

    prediction = predictor.predict(
        "R1",
        datetime(2026, 5, 19, 8, 15),
        horizon_minutes=15,
        latest_live_record=record,
        prefer_persistence=True,
    )

    assert prediction.predicted_speed == pytest.approx(18.0)
    assert prediction.method == "persistence_fallback"
    assert prediction.lookup_quality == "persistence_latest_observation"
    assert prediction.reason == "model_inference_unavailable"
    assert prediction.confidence_score == pytest.approx(0.61)
