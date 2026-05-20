from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from traffic_prediction.data.scalers import ScalerStore

# Generate datasets with standard scaling candidates
@given(
    data_frames(
        columns=[
            column("current_speed", elements=st.floats(min_value=0, max_value=120, allow_nan=False, allow_infinity=False)),
            column("lag_1", elements=st.floats(min_value=0, max_value=120, allow_nan=False, allow_infinity=False)),
            column("hour_of_day", elements=st.floats(min_value=0, max_value=23, allow_nan=False, allow_infinity=False)),
            column("neighbor_speed_mean", elements=st.floats(min_value=0, max_value=120, allow_nan=False, allow_infinity=False)),
        ],
        index=range_indexes(min_size=20, max_size=100)
    )
)
def test_scaler_reversibility_is_symmetric(df: pd.DataFrame) -> None:
    # Scikit-learn scalers fail or warn if std=0 (all constant).
    # We guarantee variance by forcing two distinct rows.
    df = df.copy()
    df.loc[0] = [10.0, 10.0, 0.0, 10.0]
    df.loc[1] = [100.0, 100.0, 23.0, 100.0]

    store = ScalerStore(
        speed_columns=["current_speed", "lag_1"],
        minmax_columns=["hour_of_day"],
        standard_columns=["neighbor_speed_mean"]
    )

    transformed = store.fit_transform(df)

    # Inverse transform speeds
    # We must match the shape of speed_columns
    scaled_speeds = transformed[["current_speed", "lag_1"]].to_numpy()
    inversed = store.inverse_transform_speed(scaled_speeds)

    original_speeds = df[["current_speed", "lag_1"]].to_numpy()

    np.testing.assert_allclose(inversed, original_speeds, rtol=1e-5, atol=1e-5)

