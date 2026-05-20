from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from traffic_prediction.data.exceptions import DataLeakageError
from traffic_prediction.features.leakage import LeakageValidator


IDENTIFIER_COLUMNS = {
    "id",
    "road_id",
    "road_name",
    "city",
    "frc",
    "collected_at_wib",
}


@dataclass
class ScalerStore:
    leakage_validator: LeakageValidator = field(default_factory=LeakageValidator)
    speed_scaler: StandardScaler | None = None
    standard_scaler: StandardScaler | None = None
    minmax_scaler: MinMaxScaler | None = None
    standard_columns: list[str] = field(default_factory=list)
    minmax_columns: list[str] = field(default_factory=list)
    fitted: bool = False

    def infer_feature_columns(self, df: pd.DataFrame) -> list[str]:
        numeric_columns = [
            col for col in df.columns
            if col not in IDENTIFIER_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
        ]
        return numeric_columns

    def fit(self, train_df: pd.DataFrame, split_name: str = "train") -> None:
        self.leakage_validator.validate_scaler_fit_source(split_name)
        feature_columns = self.infer_feature_columns(train_df)
        self.minmax_columns = [
            col for col in feature_columns
            if col.startswith(("hour_", "day_", "is_", "time_since_midnight"))
            or col in {"road_weight", "road_closure"}
        ]
        self.standard_columns = [
            col for col in feature_columns
            if col not in self.minmax_columns
        ]

        self.speed_scaler = StandardScaler()
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

        self.speed_scaler.fit(train_df[["current_speed"]])
        if self.standard_columns:
            self.standard_scaler.fit(train_df[self.standard_columns])
        if self.minmax_columns:
            self.minmax_scaler.fit(train_df[self.minmax_columns])
        self.fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or self.standard_scaler is None or self.minmax_scaler is None:
            raise DataLeakageError("Scalers must be fit on training data before transform.")

        result = df.copy()
        if self.standard_columns:
            result[self.standard_columns] = self.standard_scaler.transform(result[self.standard_columns])
        if self.minmax_columns:
            result[self.minmax_columns] = self.minmax_scaler.transform(result[self.minmax_columns])
        self._assert_finite(result[self.standard_columns + self.minmax_columns])
        return result

    def inverse_speed(self, values: np.ndarray) -> np.ndarray:
        if self.speed_scaler is None:
            raise DataLeakageError("Speed scaler is not fitted.")
        flat = values.reshape(-1, 1)
        restored = self.speed_scaler.inverse_transform(flat)
        return restored.reshape(values.shape)

    def save(self, path: str | Path) -> None:
        payload: dict[str, Any] = {
            "speed_scaler": self.speed_scaler,
            "standard_scaler": self.standard_scaler,
            "minmax_scaler": self.minmax_scaler,
            "standard_columns": self.standard_columns,
            "minmax_columns": self.minmax_columns,
            "fitted": self.fitted,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "ScalerStore":
        payload = joblib.load(path)
        store = cls()
        store.speed_scaler = payload["speed_scaler"]
        store.standard_scaler = payload["standard_scaler"]
        store.minmax_scaler = payload["minmax_scaler"]
        store.standard_columns = payload["standard_columns"]
        store.minmax_columns = payload["minmax_columns"]
        store.fitted = payload["fitted"]
        return store

    def _assert_finite(self, df: pd.DataFrame) -> None:
        if not np.isfinite(df.to_numpy(dtype=float)).all():
            raise ValueError("Scaled features contain NaN or infinite values.")

