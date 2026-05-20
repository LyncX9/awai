from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float
    confidence_level: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class StatisticalTestResult:
    comparison_name: str
    metric_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: ConfidenceInterval
    significant: bool
    alpha: float
    test_name: str
    sample_count: int

    def to_dict(self) -> dict[str, float | str | bool | int | dict[str, float]]:
        payload = asdict(self)
        payload["confidence_interval"] = self.confidence_interval.to_dict()
        return payload


@dataclass(frozen=True)
class ForecastComparisonReport:
    paired_t_test: StatisticalTestResult
    diebold_mariano_test: StatisticalTestResult

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                self.paired_t_test.to_dict(),
                self.diebold_mariano_test.to_dict(),
            ]
        )


def absolute_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true, pred = _prepare_arrays(y_true, y_pred)
    return np.abs(true - pred)


def squared_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true, pred = _prepare_arrays(y_true, y_pred)
    return np.square(true - pred)


def confidence_interval_mean(
    values: np.ndarray,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    data = _prepare_values(values)
    mean = float(np.mean(data))
    if data.size == 1:
        return ConfidenceInterval(mean=mean, lower=mean, upper=mean, confidence_level=confidence_level)

    standard_error = float(stats.sem(data))
    critical = float(stats.t.ppf((1.0 + confidence_level) / 2.0, df=data.size - 1))
    margin = critical * standard_error
    return ConfidenceInterval(
        mean=mean,
        lower=mean - margin,
        upper=mean + margin,
        confidence_level=confidence_level,
    )


def cohens_d_paired(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a, b = _prepare_pair(values_a, values_b)
    diff = a - b
    std = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    if std == 0.0:
        return 0.0
    return float(np.mean(diff) / std)


def paired_t_test_errors(
    model_errors: np.ndarray,
    baseline_errors: np.ndarray,
    comparison_name: str = "model_vs_baseline",
    metric_name: str = "absolute_error",
    alpha: float = 0.05,
    confidence_level: float = 0.95,
) -> StatisticalTestResult:
    model, baseline = _prepare_pair(model_errors, baseline_errors)
    diff = model - baseline
    if diff.size == 1:
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.ttest_rel(model, baseline, nan_policy="raise")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    return StatisticalTestResult(
        comparison_name=comparison_name,
        metric_name=metric_name,
        statistic=statistic,
        p_value=p_value,
        effect_size=cohens_d_paired(model, baseline),
        confidence_interval=confidence_interval_mean(diff, confidence_level=confidence_level),
        significant=bool(p_value < alpha),
        alpha=alpha,
        test_name="paired_t_test",
        sample_count=int(diff.size),
    )


def diebold_mariano_test(
    y_true: np.ndarray,
    model_pred: np.ndarray,
    baseline_pred: np.ndarray,
    comparison_name: str = "model_vs_baseline",
    loss: str = "squared",
    horizon: int = 1,
    alpha: float = 0.05,
    confidence_level: float = 0.95,
) -> StatisticalTestResult:
    true, model = _prepare_arrays(y_true, model_pred)
    _, baseline = _prepare_arrays(y_true, baseline_pred)
    if model.shape != baseline.shape:
        raise ValueError("model_pred and baseline_pred must have the same shape")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    if loss == "squared":
        model_loss = np.square(true - model)
        baseline_loss = np.square(true - baseline)
    elif loss == "absolute":
        model_loss = np.abs(true - model)
        baseline_loss = np.abs(true - baseline)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    loss_diff = model_loss - baseline_loss
    statistic, p_value = _dm_statistic(loss_diff, horizon=horizon)
    return StatisticalTestResult(
        comparison_name=comparison_name,
        metric_name=f"{loss}_loss_difference",
        statistic=statistic,
        p_value=p_value,
        effect_size=_standardized_mean(loss_diff),
        confidence_interval=confidence_interval_mean(loss_diff, confidence_level=confidence_level),
        significant=bool(p_value < alpha),
        alpha=alpha,
        test_name="diebold_mariano",
        sample_count=int(loss_diff.size),
    )


class StatisticalTester:
    """Builds publication-ready statistical comparison reports."""

    def __init__(self, alpha: float = 0.05, confidence_level: float = 0.95) -> None:
        self.alpha = alpha
        self.confidence_level = confidence_level

    def compare_forecasts(
        self,
        y_true: np.ndarray,
        model_pred: np.ndarray,
        baseline_pred: np.ndarray,
        comparison_name: str = "model_vs_baseline",
        horizon: int = 1,
    ) -> ForecastComparisonReport:
        model_abs = absolute_errors(y_true, model_pred)
        baseline_abs = absolute_errors(y_true, baseline_pred)
        return ForecastComparisonReport(
            paired_t_test=paired_t_test_errors(
                model_abs,
                baseline_abs,
                comparison_name=comparison_name,
                metric_name="absolute_error",
                alpha=self.alpha,
                confidence_level=self.confidence_level,
            ),
            diebold_mariano_test=diebold_mariano_test(
                y_true,
                model_pred,
                baseline_pred,
                comparison_name=comparison_name,
                loss="squared",
                horizon=horizon,
                alpha=self.alpha,
                confidence_level=self.confidence_level,
            ),
        )

    def compare_against_many(
        self,
        y_true: np.ndarray,
        model_pred: np.ndarray,
        baseline_predictions: dict[str, np.ndarray],
        horizon: int = 1,
    ) -> pd.DataFrame:
        frames = []
        for name, baseline_pred in baseline_predictions.items():
            report = self.compare_forecasts(
                y_true,
                model_pred,
                baseline_pred,
                comparison_name=f"model_vs_{name}",
                horizon=horizon,
            )
            frames.append(report.to_dataframe())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _dm_statistic(loss_diff: np.ndarray, horizon: int) -> tuple[float, float]:
    diff = _prepare_values(loss_diff)
    n = diff.size
    if n < 2:
        return 0.0, 1.0

    mean_diff = float(np.mean(diff))
    variance = _newey_west_variance(diff, max_lag=max(horizon - 1, 0))
    if variance <= 0.0:
        statistic = 0.0 if mean_diff == 0.0 else math.copysign(float("inf"), mean_diff)
    else:
        statistic = mean_diff / math.sqrt(variance / n)

    if math.isinf(statistic):
        p_value = 0.0
    else:
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(statistic))))
    return float(statistic), p_value


def _newey_west_variance(values: np.ndarray, max_lag: int) -> float:
    centered = values - np.mean(values)
    n = centered.size
    gamma0 = float(np.dot(centered, centered) / n)
    variance = gamma0
    for lag in range(1, min(max_lag, n - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        variance += 2.0 * weight * gamma
    return max(variance, 0.0)


def _standardized_mean(values: np.ndarray) -> float:
    data = _prepare_values(values)
    std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
    if std == 0.0:
        return 0.0
    return float(np.mean(data) / std)


def _prepare_pair(values_a: np.ndarray, values_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = _prepare_values(values_a)
    b = _prepare_values(values_b)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays shape mismatch: {a.shape} != {b.shape}")
    return a, b


def _prepare_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = _prepare_values(y_true)
    pred = _prepare_values(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred shape mismatch: {true.shape} != {pred.shape}")
    return true, pred


def _prepare_values(values: np.ndarray | Iterable[float]) -> np.ndarray:
    data = np.asarray(values, dtype=float).reshape(-1)
    if data.size == 0:
        raise ValueError("statistical test arrays must not be empty")
    if not np.isfinite(data).all():
        raise ValueError("statistical test arrays must contain only finite values")
    return data

