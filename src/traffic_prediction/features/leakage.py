from __future__ import annotations

import numpy as np
import pandas as pd

from traffic_prediction.data.exceptions import DataLeakageError


class LeakageValidator:
    """Small, explicit checks that protect the time-series training contract."""

    def validate_temporal_split(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        timestamp_column: str = "collected_at_wib",
    ) -> None:
        if train.empty or validation.empty or test.empty:
            raise DataLeakageError("Train, validation, and test splits must all be non-empty")

        train_end = train[timestamp_column].max()
        validation_start = validation[timestamp_column].min()
        validation_end = validation[timestamp_column].max()
        test_start = test[timestamp_column].min()

        if not train_end < validation_start:
            raise DataLeakageError("Training data overlaps validation data")
        if not validation_end < test_start:
            raise DataLeakageError("Validation data overlaps test data")

    def validate_split_isolation(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        key_columns: tuple[str, ...] = ("road_id", "collected_at_wib"),
    ) -> None:
        self._validate_key_columns(train, key_columns)
        self._validate_key_columns(validation, key_columns)
        self._validate_key_columns(test, key_columns)
        train_keys = self._key_set(train, key_columns)
        validation_keys = self._key_set(validation, key_columns)
        test_keys = self._key_set(test, key_columns)
        if train_keys & validation_keys:
            raise DataLeakageError("Training and validation splits share row keys")
        if train_keys & test_keys:
            raise DataLeakageError("Training and test splits share row keys")
        if validation_keys & test_keys:
            raise DataLeakageError("Validation and test splits share row keys")

    def validate_scaler_fit_source(self, split_name: str) -> None:
        if split_name.lower() != "train":
            raise DataLeakageError(f"Scalers must be fit on the training split only, not {split_name!r}")

    def validate_feature_columns(self, df: pd.DataFrame, feature_columns: list[str]) -> None:
        missing = sorted(set(feature_columns) - set(df.columns))
        if missing:
            raise DataLeakageError(f"Feature columns missing from dataframe: {missing}")
        if df[feature_columns].isna().any().any():
            raise DataLeakageError("Feature dataframe contains NaN values after preprocessing")
        numeric = df[feature_columns].select_dtypes(include="number")
        if len(numeric.columns) != len(feature_columns):
            bad = sorted(set(feature_columns) - set(numeric.columns))
            raise DataLeakageError(f"Feature columns must be numeric: {bad}")

    def validate_feature_monotonicity(
        self,
        df: pd.DataFrame,
        timestamp_column: str = "collected_at_wib",
    ) -> None:
        for road_id, road_df in df.groupby("road_id", sort=False):
            if not road_df[timestamp_column].is_monotonic_increasing:
                raise DataLeakageError(f"Feature data is not chronological for road_id={road_id}")

    def validate_lag_causality(
        self,
        df: pd.DataFrame,
        lag_steps: tuple[int, ...],
        speed_column: str = "current_speed",
    ) -> None:
        self.validate_feature_monotonicity(df)
        grouped = df.groupby("road_id", sort=False)[speed_column]
        for lag in lag_steps:
            column = f"lag_{lag}"
            if column not in df.columns:
                continue
            expected = grouped.shift(lag)
            self._assert_series_matches_expected(df[column], expected, column)

    def validate_rolling_causality(
        self,
        df: pd.DataFrame,
        rolling_windows: tuple[int, ...],
        speed_column: str = "current_speed",
    ) -> None:
        self.validate_feature_monotonicity(df)
        grouped = df.groupby("road_id", sort=False)[speed_column]
        shifted = grouped.shift(1)
        for window in rolling_windows:
            rolling = shifted.groupby(df["road_id"], sort=False).rolling(window=window, min_periods=1)
            expected_by_suffix = {
                "mean": rolling.mean().reset_index(level=0, drop=True),
                "std": rolling.std().reset_index(level=0, drop=True),
                "min": rolling.min().reset_index(level=0, drop=True),
                "max": rolling.max().reset_index(level=0, drop=True),
            }
            for suffix, expected in expected_by_suffix.items():
                column = f"rolling_{suffix}_{window}"
                if column in df.columns:
                    self._assert_series_matches_expected(df[column], expected, column)

    def validate_spatial_same_timestamp(
        self,
        df: pd.DataFrame,
        neighbor_mapping: dict[str, list[str]],
        timestamp_column: str = "collected_at_wib",
        speed_column: str = "current_speed",
    ) -> None:
        if not neighbor_mapping or "neighbor_speed_mean" not in df.columns:
            return
        speed_lookup = {
            (str(row.road_id), row_timestamp): float(row_speed)
            for row in df[["road_id", timestamp_column, speed_column]].itertuples(index=False)
            for row_timestamp, row_speed in [(getattr(row, timestamp_column), getattr(row, speed_column))]
        }
        for row in df[["road_id", timestamp_column, "neighbor_speed_mean"]].itertuples(index=False):
            road_id = str(row.road_id)
            timestamp = getattr(row, timestamp_column)
            neighbor_speeds = [
                speed_lookup[(neighbor_id, timestamp)]
                for neighbor_id in neighbor_mapping.get(road_id, [])
                if (neighbor_id, timestamp) in speed_lookup
            ]
            if not neighbor_speeds:
                continue
            expected = float(np.mean(neighbor_speeds))
            actual = float(row.neighbor_speed_mean)
            if not np.isclose(actual, expected, rtol=1e-6, atol=1e-6):
                raise DataLeakageError(
                    f"Spatial feature neighbor_speed_mean for road_id={road_id} uses values outside same-timestamp neighbors"
                )

    def validate_augmentation_boundary(self, split_name: str) -> None:
        if split_name.lower() != "train":
            raise DataLeakageError(f"Data augmentation is only allowed on the training split, not {split_name!r}")

    def _validate_key_columns(self, df: pd.DataFrame, key_columns: tuple[str, ...]) -> None:
        missing = sorted(set(key_columns) - set(df.columns))
        if missing:
            raise DataLeakageError(f"Split isolation key columns missing from dataframe: {missing}")

    def _key_set(self, df: pd.DataFrame, key_columns: tuple[str, ...]) -> set[tuple]:
        return set(map(tuple, df[list(key_columns)].to_numpy()))

    def _assert_series_matches_expected(self, actual: pd.Series, expected: pd.Series, column: str) -> None:
        comparable = expected.notna()
        if not comparable.any():
            return
        actual_values = actual[comparable].to_numpy(dtype=float)
        expected_values = expected[comparable].to_numpy(dtype=float)
        if not np.allclose(actual_values, expected_values, rtol=1e-6, atol=1e-6, equal_nan=True):
            raise DataLeakageError(f"Feature column {column} violates past-only causality")
