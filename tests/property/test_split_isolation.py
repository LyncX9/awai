from __future__ import annotations

import pandas as pd
from hypothesis import given, strategies as st

from traffic_prediction.config.settings import DataConfig, FeatureConfig
from traffic_prediction.data.processor import DataProcessor


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=30),  # day of april
            st.integers(min_value=0, max_value=23),  # hour
            st.sampled_from(["R1", "R2"])            # road_id
        ),
        min_size=50,
        max_size=200
    )
)
def test_chronological_split_preserves_strict_isolation(records) -> None:
    # Build unique timestamps to simulate records
    data = []
    for day, hour, road in records:
        ts = f"2026-04-{day:02d} {hour:02d}:00:00"
        data.append({"road_id": road, "collected_at_wib": ts, "current_speed": 40.0, "confidence": 0.9, "free_flow_speed": 50.0})
    
    df = pd.DataFrame(data)
    
    # Needs to be sorted and have timestamp parsed for processor
    processor = DataProcessor(DataConfig(train_days=15, validation_days=5, frequency="1h"), FeatureConfig())
    df["collected_at_wib"] = pd.to_datetime(df["collected_at_wib"]).dt.tz_localize("Asia/Jakarta")
    
    # We may not have enough data spanning the full 20 days required by the custom config
    # The processor doesn't strictly fail if validation or test is empty, but LeakageValidator might
    # So we'll catch ValueError which is raised by LeakageValidator if splits are empty or overlap.
    from traffic_prediction.data.exceptions import DataLeakageError
    try:
        train, val, test, stats = processor.chronological_split(df)
    except DataLeakageError:
        # Expected if generated dates are too sparse to form all 3 splits.
        return
        
    if train.empty or val.empty or test.empty:
        return
        
    train_max = train["collected_at_wib"].max()
    val_min = val["collected_at_wib"].min()
    val_max = val["collected_at_wib"].max()
    test_min = test["collected_at_wib"].min()

    assert train_max < val_min, "Train leaks into Validation"
    assert val_max < test_min, "Validation leaks into Test"
