from __future__ import annotations

import pandas as pd

from traffic_prediction.config.settings import DataConfig, FeatureConfig
from traffic_prediction.data.processor import DataProcessor


def test_clean_sorts_deduplicates_and_imputes_missing_speed() -> None:
    raw = pd.DataFrame(
        [
            {
                "road_id": "R1",
                "current_speed": 20.0,
                "free_flow_speed": 40.0,
                "confidence": 0.8,
                "collected_at_wib": "2026-04-01 00:30:00",
            },
            {
                "road_id": "R1",
                "current_speed": 10.0,
                "free_flow_speed": 40.0,
                "confidence": 0.9,
                "collected_at_wib": "2026-04-01 00:00:00",
            },
            {
                "road_id": "R1",
                "current_speed": 12.0,
                "free_flow_speed": 40.0,
                "confidence": 0.7,
                "collected_at_wib": "2026-04-01 00:00:00",
            },
        ]
    )

    processor = DataProcessor(DataConfig(), FeatureConfig())
    cleaned = processor.clean(raw)

    assert len(cleaned) == 3
    assert cleaned["current_speed"].isna().sum() == 0
    assert cleaned.iloc[0]["current_speed"] == 10.0
    assert cleaned["collected_at_wib"].dt.tz is not None


def test_chronological_split_uses_expected_day_windows() -> None:
    timestamps = pd.date_range("2026-04-01", periods=30, freq="D", tz="Asia/Jakarta")
    frame = pd.DataFrame(
        {
            "road_id": ["R1"] * 30,
            "collected_at_wib": timestamps,
            "current_speed": [1.0] * 30,
        }
    )
    processor = DataProcessor(DataConfig(), FeatureConfig())
    train, validation, test, stats = processor.chronological_split(frame)

    assert len(train) == 18
    assert len(validation) == 6
    assert len(test) == 6
    assert stats.train_rows == 18
