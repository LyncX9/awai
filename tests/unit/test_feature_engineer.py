from __future__ import annotations

import pandas as pd

from traffic_prediction.features.offline import FeatureEngineer


def test_feature_engineer_adds_temporal_lag_rolling_and_spatial_features() -> None:
    timestamps = pd.date_range("2026-04-01", periods=10, freq="15min", tz="Asia/Jakarta")
    frame = pd.DataFrame(
        {
            "road_id": ["R1"] * 10 + ["R2"] * 10,
            "collected_at_wib": list(timestamps) + list(timestamps),
            "current_speed": list(range(10, 20)) + list(range(20, 30)),
            "free_flow_speed": [40.0] * 20,
            "confidence": [1.0] * 20,
            "speed_ratio": [0.5] * 20,
        }
    )

    engineer = FeatureEngineer(neighbor_mapping={"R1": ["R2"], "R2": ["R1"]})
    featured = engineer.extract_features(frame)

    expected = {
        "hour_sin",
        "hour_cos",
        "lag_1",
        "lag_8",
        "rolling_mean_3",
        "speed_delta",
        "speed_volatility",
        "neighbor_speed_mean",
    }
    assert expected.issubset(featured.columns)
    assert featured[list(expected)].isna().sum().sum() == 0
    r1_first = featured[featured["road_id"] == "R1"].iloc[0]
    assert r1_first["neighbor_speed_mean"] == 20.0

