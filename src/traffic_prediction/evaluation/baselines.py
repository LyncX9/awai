from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from traffic_prediction.evaluation.metrics import RegressionMetrics, calculate_regression_metrics


class BaselineModel(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> "BaselineModel":
        ...

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        ...


@dataclass(frozen=True)
class BaselineResult:
    name: str
    predictions: np.ndarray
    metrics: RegressionMetrics
    status: str = "ok"
    detail: str = ""


@dataclass(frozen=True)
class BaselineComparison:
    results: list[BaselineResult]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "model": result.name,
                "status": result.status,
                "mae": result.metrics.mae,
                "rmse": result.metrics.rmse,
                "mape": result.metrics.mape,
                "r2": result.metrics.r2,
                "sample_count": result.metrics.sample_count,
                "detail": result.detail,
            }
            for result in self.results
        ]
        return pd.DataFrame(rows).sort_values(["status", "rmse", "mae"]).reset_index(drop=True)

    @property
    def available_results(self) -> list[BaselineResult]:
        return [result for result in self.results if result.status == "ok"]

    def best_by_rmse(self) -> BaselineResult | None:
        available = self.available_results
        if not available:
            return None
        return min(available, key=lambda result: result.metrics.rmse)


class PersistenceBaseline:
    name = "naive_persistence"

    def fit(self, train: pd.DataFrame) -> "PersistenceBaseline":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        _require_columns(test, {"lag_1", "actual_speed"})
        return test["lag_1"].fillna(test["actual_speed"]).to_numpy(dtype=float)


class HistoricalAverageBaseline:
    name = "historical_average"

    def __init__(self) -> None:
        self.global_mean_: float | None = None
        self.lookup_: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> "HistoricalAverageBaseline":
        _require_columns(train, {"road_id", "timestamp", "actual_speed"})
        work = train.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"])
        work["hour_of_day"] = work["timestamp"].dt.hour
        work["day_of_week"] = work["timestamp"].dt.dayofweek
        self.global_mean_ = float(work["actual_speed"].mean())
        self.lookup_ = work.groupby(["road_id", "hour_of_day", "day_of_week"])["actual_speed"].mean()
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if self.lookup_ is None or self.global_mean_ is None:
            raise ValueError("HistoricalAverageBaseline must be fit before predict")
        _require_columns(test, {"road_id", "timestamp"})
        work = test.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"])
        predictions: list[float] = []
        for row in work[["road_id", "timestamp"]].itertuples(index=False):
            key = (row.road_id, row.timestamp.hour, row.timestamp.dayofweek)
            predictions.append(float(self.lookup_.get(key, self.global_mean_)))
        return np.asarray(predictions, dtype=float)


class LinearRegressionBaseline:
    name = "linear_regression"

    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self.feature_columns = feature_columns
        self.model = LinearRegression()
        self.fitted_columns_: list[str] | None = None

    def fit(self, train: pd.DataFrame) -> "LinearRegressionBaseline":
        _require_columns(train, {"actual_speed"})
        features = self._resolve_feature_columns(train)
        if not features:
            raise ValueError("LinearRegressionBaseline requires at least one numeric feature")
        self.fitted_columns_ = features
        self.model.fit(train[features].to_numpy(dtype=float), train["actual_speed"].to_numpy(dtype=float))
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if self.fitted_columns_ is None:
            raise ValueError("LinearRegressionBaseline must be fit before predict")
        return self.model.predict(test[self.fitted_columns_].to_numpy(dtype=float))

    def _resolve_feature_columns(self, train: pd.DataFrame) -> list[str]:
        if self.feature_columns is not None:
            _require_columns(train, set(self.feature_columns))
            return self.feature_columns
        excluded = {"actual_speed", "predicted_speed", "timestamp", "road_id", "frc", "road_class"}
        return [
            column
            for column in train.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(train[column])
        ]


class OptionalDependencyBaseline:
    """Placeholder for baselines whose third-party package is not installed."""

    def __init__(self, name: str, package_name: str) -> None:
        self.name = name
        self.package_name = package_name
        self.available = self._is_available(package_name)

    def fit(self, train: pd.DataFrame) -> "OptionalDependencyBaseline":
        if not self.available:
            raise ImportError(f"{self.name} requires optional package '{self.package_name}'")
        raise NotImplementedError(
            f"{self.name} dependency is available, but concrete adapter is not implemented in this lightweight stage"
        )

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        raise RuntimeError(f"{self.name} cannot predict before a concrete fitted adapter is available")

    @staticmethod
    def _is_available(package_name: str) -> bool:
        try:
            __import__(package_name)
        except Exception:
            return False
        return True


class BaselineEvaluator:
    """Fits and evaluates baseline models against the same test split."""

    def __init__(self, baselines: list[BaselineModel] | None = None) -> None:
        self.baselines = baselines or default_baselines()

    def evaluate(self, train: pd.DataFrame, test: pd.DataFrame) -> BaselineComparison:
        _require_columns(test, {"actual_speed"})
        results: list[BaselineResult] = []
        for baseline in self.baselines:
            try:
                fitted = baseline.fit(train)
                predictions = fitted.predict(test)
                metrics = calculate_regression_metrics(test["actual_speed"].to_numpy(dtype=float), predictions)
                results.append(
                    BaselineResult(
                        name=baseline.name,
                        predictions=np.asarray(predictions, dtype=float),
                        metrics=metrics,
                    )
                )
            except Exception as exc:
                results.append(
                    BaselineResult(
                        name=baseline.name,
                        predictions=np.asarray([], dtype=float),
                        metrics=_empty_metrics(),
                        status="unavailable",
                        detail=str(exc),
                    )
                )
        return BaselineComparison(results=results)


def default_baselines() -> list[BaselineModel]:
    return [
        PersistenceBaseline(),
        HistoricalAverageBaseline(),
        LinearRegressionBaseline(),
        OptionalDependencyBaseline("arima", "statsmodels"),
        OptionalDependencyBaseline("xgboost", "xgboost"),
        OptionalDependencyBaseline("lightgbm", "lightgbm"),
        OptionalDependencyBaseline("catboost", "catboost"),
    ]


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _empty_metrics() -> RegressionMetrics:
    return RegressionMetrics(mae=float("nan"), rmse=float("nan"), mape=float("nan"), r2=float("nan"), sample_count=0)

