from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from traffic_prediction.data.exceptions import DataLeakageError


class ScalerStore:
    """Owns train-fitted scalers and prevents refitting during evaluation/inference."""

    def __init__(
        self,
        speed_columns: list[str] | None = None,
        minmax_columns: list[str] | None = None,
        standard_columns: list[str] | None = None,
    ) -> None:
        self.speed_columns = speed_columns or ["current_speed"]
        self.minmax_columns = minmax_columns or []
        self.standard_columns = standard_columns or []
        self.speed_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "ScalerStore":
        self.speed_columns = [column for column in self.speed_columns if column in df.columns]
        self.minmax_columns = [column for column in self.minmax_columns if column in df.columns]
        self.standard_columns = [column for column in self.standard_columns if column in df.columns]

        if self.speed_columns:
            self.speed_scaler.fit(df[self.speed_columns])
        if self.minmax_columns:
            self.minmax_scaler.fit(df[self.minmax_columns])
        if self.standard_columns:
            self.standard_scaler.fit(df[self.standard_columns])
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise DataLeakageError("ScalerStore.transform called before fitting on training data")

        transformed = df.copy()
        if self.speed_columns:
            transformed[self.speed_columns] = self.speed_scaler.transform(transformed[self.speed_columns])
        if self.minmax_columns:
            transformed[self.minmax_columns] = self.minmax_scaler.transform(transformed[self.minmax_columns])
        if self.standard_columns:
            transformed[self.standard_columns] = self.standard_scaler.transform(transformed[self.standard_columns])
        return transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def inverse_transform_speed(self, values):
        if not self.is_fitted:
            raise DataLeakageError("Cannot inverse-transform with unfitted scalers")
        return self.speed_scaler.inverse_transform(values)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "ScalerStore":
        return joblib.load(path)

