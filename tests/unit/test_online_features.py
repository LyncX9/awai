from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from traffic_prediction.data.schemas import FeatureManifest, LiveTrafficRecord
from traffic_prediction.features.online import OnlineFeatureEngineer
from traffic_prediction.ingestion.buffer import LiveBufferManager


def test_online_feature_engineer_builds_manifest_ordered_sequence() -> None:
    buffer = LiveBufferManager(min_timesteps=3, max_timesteps=8)
    start = datetime(2026, 5, 19, 7, 0)
    for index in range(4):
        timestamp = start + timedelta(minutes=15 * index)
        buffer.append(LiveTrafficRecord("R1", 20.0 + index, 0.9, timestamp))
        buffer.append(LiveTrafficRecord("R2", 30.0 + index, 0.8, timestamp))

    manifest = FeatureManifest(
        feature_columns=[
            "road_weight",
            "current_speed",
            "confidence",
            "hour_of_day",
            "lag_1",
            "rolling_mean_3",
            "neighbor_speed_mean",
        ],
        target_column="current_speed",
        lookback=3,
        horizon=4,
    )
    engineer = OnlineFeatureEngineer(
        manifest=manifest,
        buffer_manager=buffer,
        roads=_roads(),
        neighbor_mapping={"R1": ["R2"], "R2": ["R1"]},
    )

    result = engineer.build_for_road("R1")

    assert result.sequence.shape == (1, 3, len(manifest.feature_columns))
    assert list(result.feature_frame.columns) == manifest.feature_columns
    assert result.quality.status == "healthy"
    assert result.quality.has_minimum_history is True
    assert result.quality.padded_timesteps == 0
    assert result.feature_frame["neighbor_speed_mean"].iloc[-1] == 33.0


def test_online_feature_engineer_reports_incomplete_history_and_missing_columns() -> None:
    buffer = LiveBufferManager(min_timesteps=3, max_timesteps=8)
    buffer.append(LiveTrafficRecord("R1", 20.0, 0.9, datetime(2026, 5, 19, 7, 0)))
    manifest = FeatureManifest(
        feature_columns=["current_speed", "lag_1", "future_external_context"],
        target_column="current_speed",
        lookback=3,
        horizon=4,
    )
    engineer = OnlineFeatureEngineer(manifest=manifest, buffer_manager=buffer, roads=_roads())

    result = engineer.build_for_road("R1")

    assert result.sequence.shape == (1, 3, 3)
    assert result.quality.status == "degraded"
    assert result.quality.has_minimum_history is False
    assert result.quality.padded_timesteps == 2
    assert result.quality.missing_feature_columns == ["future_external_context"]


def test_online_feature_engineer_rejects_missing_road_buffer() -> None:
    engineer = OnlineFeatureEngineer(
        manifest=FeatureManifest(["current_speed"], "current_speed", lookback=3, horizon=4),
        buffer_manager=LiveBufferManager(),
        roads=_roads(),
    )

    try:
        engineer.build_for_road("R1")
    except ValueError as exc:
        assert "No live buffer records" in str(exc)
    else:
        raise AssertionError("Expected missing live buffer records to raise ValueError")


def _roads() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "road_id": ["R1", "R2"],
            "road_weight": [0.4, 0.6],
            "mid_lat": [-6.9, -6.91],
            "mid_lon": [106.9, 106.91],
        }
    )
