from __future__ import annotations

import pandas as pd
import pytest

from traffic_prediction.data.exceptions import DataLeakageError
from traffic_prediction.features.leakage import LeakageValidator
from traffic_prediction.features.offline import FeatureEngineer


def test_leakage_validator_rejects_split_overlap_and_non_train_scaler_fit() -> None:
    validator = LeakageValidator()
    train = _base_frame().iloc[:2].copy()
    validation = _base_frame().iloc[1:3].copy()
    test = _base_frame().iloc[3:].copy()

    with pytest.raises(DataLeakageError, match="share row keys"):
        validator.validate_split_isolation(train, validation, test)

    with pytest.raises(DataLeakageError, match="training split only"):
        validator.validate_scaler_fit_source("validation")


def test_leakage_validator_accepts_causal_lag_rolling_and_spatial_features() -> None:
    frame = _two_road_frame()
    featured = FeatureEngineer(
        neighbor_mapping={"R1": ["R2"], "R2": ["R1"]},
        lag_steps=(1, 2),
        rolling_windows=(3,),
    ).extract_features(frame)
    validator = LeakageValidator()

    validator.validate_lag_causality(featured, lag_steps=(1, 2))
    validator.validate_rolling_causality(featured, rolling_windows=(3,))
    validator.validate_spatial_same_timestamp(featured, neighbor_mapping={"R1": ["R2"], "R2": ["R1"]})


def test_leakage_validator_rejects_future_lag_and_rolling_values() -> None:
    frame = _two_road_frame()
    featured = FeatureEngineer(lag_steps=(1,), rolling_windows=(3,)).extract_features(frame)
    validator = LeakageValidator()

    future_lag = featured.copy()
    future_lag["lag_1"] = future_lag.groupby("road_id")["current_speed"].shift(-1).fillna(future_lag["current_speed"])
    with pytest.raises(DataLeakageError, match="lag_1"):
        validator.validate_lag_causality(future_lag, lag_steps=(1,))

    future_roll = featured.copy()
    future_roll["rolling_mean_3"] = future_roll["current_speed"]
    with pytest.raises(DataLeakageError, match="rolling_mean_3"):
        validator.validate_rolling_causality(future_roll, rolling_windows=(3,))


def test_leakage_validator_rejects_spatial_future_timestamp_and_eval_augmentation() -> None:
    featured = FeatureEngineer(neighbor_mapping={"R1": ["R2"], "R2": ["R1"]}).extract_features(_two_road_frame())
    validator = LeakageValidator()
    corrupted = featured.copy()
    r1_mask = corrupted["road_id"] == "R1"
    corrupted.loc[r1_mask, "neighbor_speed_mean"] = (
        corrupted[corrupted["road_id"] == "R2"]["current_speed"].shift(-1).bfill().to_numpy()
    )

    with pytest.raises(DataLeakageError, match="same-timestamp"):
        validator.validate_spatial_same_timestamp(corrupted, neighbor_mapping={"R1": ["R2"], "R2": ["R1"]})

    with pytest.raises(DataLeakageError, match="only allowed on the training split"):
        validator.validate_augmentation_boundary("test")


def _base_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-04-01 00:00", periods=4, freq="15min", tz="Asia/Jakarta")
    return pd.DataFrame(
        {
            "road_id": ["R1", "R1", "R1", "R1"],
            "collected_at_wib": timestamps,
            "current_speed": [10.0, 11.0, 12.0, 13.0],
        }
    )


def _two_road_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-04-01 00:00", periods=8, freq="15min", tz="Asia/Jakarta")
    return pd.DataFrame(
        {
            "road_id": ["R1"] * 8 + ["R2"] * 8,
            "collected_at_wib": list(timestamps) + list(timestamps),
            "current_speed": list(range(10, 18)) + list(range(20, 28)),
            "free_flow_speed": [40.0] * 16,
            "confidence": [0.9] * 16,
            "speed_ratio": [0.5] * 16,
        }
    )
