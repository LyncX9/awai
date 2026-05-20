from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from traffic_prediction.inference.congestion import classify_congestion


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    mape: float
    r2: float
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    overall: RegressionMetrics
    by_horizon: dict[int, RegressionMetrics]
    by_congestion_level: dict[str, RegressionMetrics]
    by_peak_period: dict[str, RegressionMetrics]
    by_road_class: dict[str, RegressionMetrics]
    by_road_id: dict[str, RegressionMetrics]
    by_weekend: dict[str, RegressionMetrics]
    worst_roads_by_rmse: pd.DataFrame
    worst_time_periods_by_rmse: pd.DataFrame


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _prepare_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _prepare_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(true - pred))))


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    true, pred = _prepare_arrays(y_true, y_pred)
    denominator = np.maximum(np.abs(true), epsilon)
    return float(np.mean(np.abs((true - pred) / denominator)) * 100.0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _prepare_arrays(y_true, y_pred)
    residual_sum = float(np.sum(np.square(true - pred)))
    total_sum = float(np.sum(np.square(true - np.mean(true))))
    if total_sum == 0.0:
        return 1.0 if residual_sum == 0.0 else 0.0
    return float(1.0 - residual_sum / total_sum)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    true, pred = _prepare_arrays(y_true, y_pred)
    return RegressionMetrics(
        mae=mean_absolute_error(true, pred),
        rmse=root_mean_squared_error(true, pred),
        mape=mean_absolute_percentage_error(true, pred),
        r2=r2_score(true, pred),
        sample_count=int(true.size),
    )


def horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_minutes: Iterable[int] | None = None,
) -> dict[int, RegressionMetrics]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred shape mismatch: {true.shape} != {pred.shape}")
    if true.ndim < 2:
        raise ValueError("horizon_metrics expects at least 2D arrays with horizon dimension at axis=1")

    horizon_count = int(true.shape[1])
    horizons = list(horizon_minutes) if horizon_minutes is not None else [(index + 1) * 15 for index in range(horizon_count)]
    if len(horizons) != horizon_count:
        raise ValueError("horizon_minutes length must match prediction horizon")

    return {
        int(horizon): calculate_regression_metrics(true[:, index, ...], pred[:, index, ...])
        for index, horizon in enumerate(horizons)
    }


def classify_peak_period(timestamp: pd.Timestamp) -> str:
    hour = int(timestamp.hour)
    if 7 <= hour < 9:
        return "morning_rush"
    if 17 <= hour < 19:
        return "evening_rush"
    return "off_peak"


class ModelEvaluator:
    """Computes publication-ready core regression metrics and grouped reports."""

    REQUIRED_COLUMNS = {"actual_speed", "predicted_speed"}

    def evaluate_dataframe(self, df: pd.DataFrame) -> EvaluationReport:
        self._validate_dataframe(df)
        enriched = self._enrich_metadata(df)
        return EvaluationReport(
            overall=self._metrics_for_frame(enriched),
            by_horizon=self._grouped_metrics(enriched, "horizon_minutes", int),
            by_congestion_level=self._grouped_metrics(enriched, "congestion_level", str),
            by_peak_period=self._grouped_metrics(enriched, "peak_period", str),
            by_road_class=self._grouped_metrics(enriched, "road_class", str),
            by_road_id=self._grouped_metrics(enriched, "road_id", str),
            by_weekend=self._grouped_metrics(enriched, "weekend_label", str),
            worst_roads_by_rmse=self.worst_groups_by_rmse(enriched, "road_id", top_n=10),
            worst_time_periods_by_rmse=self.worst_groups_by_rmse(enriched, "time_period", top_n=10),
        )

    def evaluate_arrays(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon_minutes: Iterable[int] | None = None,
    ) -> tuple[RegressionMetrics, dict[int, RegressionMetrics]]:
        return calculate_regression_metrics(y_true, y_pred), horizon_metrics(y_true, y_pred, horizon_minutes)

    def worst_groups_by_rmse(self, df: pd.DataFrame, group_column: str, top_n: int = 10) -> pd.DataFrame:
        if group_column not in df.columns:
            return pd.DataFrame(columns=[group_column, "rmse", "mae", "mape", "r2", "sample_count"])

        rows = []
        for group_value, group in df.groupby(group_column, dropna=False):
            metrics = self._metrics_for_frame(group)
            rows.append(
                {
                    group_column: group_value,
                    "rmse": metrics.rmse,
                    "mae": metrics.mae,
                    "mape": metrics.mape,
                    "r2": metrics.r2,
                    "sample_count": metrics.sample_count,
                }
            )
        return pd.DataFrame(rows).sort_values("rmse", ascending=False).head(top_n).reset_index(drop=True)

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Evaluation dataframe missing required columns: {sorted(missing)}")
        if df.empty:
            raise ValueError("Evaluation dataframe must not be empty")

    def _enrich_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = df.copy()
        if "free_flow_speed" in enriched.columns and "congestion_level" not in enriched.columns:
            enriched["congestion_level"] = [
                classify_congestion(speed, free_flow)
                for speed, free_flow in zip(enriched["actual_speed"], enriched["free_flow_speed"])
            ]
        elif "congestion_level" not in enriched.columns:
            enriched["congestion_level"] = "unknown"

        if "timestamp" in enriched.columns:
            timestamps = pd.to_datetime(enriched["timestamp"])
            enriched["peak_period"] = [classify_peak_period(timestamp) for timestamp in timestamps]
            enriched["weekend_label"] = np.where(timestamps.dt.dayofweek >= 5, "weekend", "weekday")
            enriched["time_period"] = timestamps.dt.strftime("%Y-%m-%d %H:%M")
        else:
            enriched["peak_period"] = "unknown"
            enriched["weekend_label"] = "unknown"
            enriched["time_period"] = "unknown"

        if "frc" in enriched.columns and "road_class" not in enriched.columns:
            enriched["road_class"] = enriched["frc"].astype(str)
        elif "road_class" not in enriched.columns:
            enriched["road_class"] = "unknown"

        if "road_id" not in enriched.columns:
            enriched["road_id"] = "unknown"
        if "horizon_minutes" not in enriched.columns:
            enriched["horizon_minutes"] = 0
        return enriched

    def _grouped_metrics(self, df: pd.DataFrame, group_column: str, key_type) -> dict:
        if group_column not in df.columns:
            return {}
        output = {}
        for group_value, group in df.groupby(group_column, dropna=False):
            output[key_type(group_value)] = self._metrics_for_frame(group)
        return output

    def _metrics_for_frame(self, df: pd.DataFrame) -> RegressionMetrics:
        return calculate_regression_metrics(
            df["actual_speed"].to_numpy(dtype=float),
            df["predicted_speed"].to_numpy(dtype=float),
        )


def _prepare_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred shape mismatch: {true.shape} != {pred.shape}")
    if true.size == 0:
        raise ValueError("Metric arrays must not be empty")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metric arrays must contain only finite values")
    return true, pred
