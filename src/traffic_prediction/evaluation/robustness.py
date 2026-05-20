from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

from traffic_prediction.evaluation.metrics import RegressionMetrics, calculate_regression_metrics


PredictionFn = Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class RobustnessScenario:
    name: str
    description: str
    transform: Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class RobustnessResult:
    scenario_name: str
    description: str
    metrics: RegressionMetrics
    delta_mae: float
    delta_rmse: float
    degradation_mae_pct: float
    degradation_rmse_pct: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["metrics"] = self.metrics.to_dict()
        return payload


@dataclass(frozen=True)
class RobustnessReport:
    clean_metrics: RegressionMetrics
    scenario_results: list[RobustnessResult]
    baseline_results: dict[str, list[RobustnessResult]]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for result in self.scenario_results:
            rows.append({"model": "model", **_flatten_result(result)})
        for baseline_name, results in self.baseline_results.items():
            for result in results:
                rows.append({"model": baseline_name, **_flatten_result(result)})
        return pd.DataFrame(rows)

    def worst_scenarios(self, top_n: int = 5) -> pd.DataFrame:
        table = self.to_dataframe()
        if table.empty:
            return table
        return table.sort_values("delta_rmse", ascending=False).head(top_n).reset_index(drop=True)


class RobustnessScenarioFactory:
    """Creates adverse-condition transforms used for robustness evaluation."""

    def __init__(self, random_seed: int = 42) -> None:
        self.rng = np.random.default_rng(random_seed)

    def noisy_speed(self, speed_column: str = "actual_speed", std_kmh: float = 2.0) -> RobustnessScenario:
        def transform(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            noise = self.rng.normal(0.0, std_kmh, size=len(out))
            out[speed_column] = (out[speed_column].to_numpy(dtype=float) + noise).clip(0.0, 120.0)
            return out

        return RobustnessScenario("noisy_input", f"Gaussian speed noise std={std_kmh} km/h", transform)

    def missing_intervals(self, drop_fraction: float = 0.10) -> RobustnessScenario:
        if not 0.0 <= drop_fraction < 1.0:
            raise ValueError("drop_fraction must be in [0.0, 1.0)")

        def transform(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df.copy()
            keep_count = max(1, int(round(len(df) * (1.0 - drop_fraction))))
            keep_indices = np.sort(self.rng.choice(df.index.to_numpy(), size=keep_count, replace=False))
            return df.loc[keep_indices].copy().reset_index(drop=True)

        return RobustnessScenario("missing_intervals", f"Randomly drop {drop_fraction:.0%} of intervals", transform)

    def degraded_confidence(self, confidence_column: str = "confidence", factor: float = 0.8) -> RobustnessScenario:
        def transform(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if confidence_column in out.columns:
                out[confidence_column] = (out[confidence_column].to_numpy(dtype=float) * factor).clip(0.0, 1.0)
            return out

        return RobustnessScenario("degraded_confidence", f"Reduce confidence by {(1.0 - factor):.0%}", transform)

    def delayed_updates(self, timestamp_column: str = "timestamp", delay_minutes: int = 15) -> RobustnessScenario:
        def transform(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if timestamp_column in out.columns:
                out[timestamp_column] = pd.to_datetime(out[timestamp_column]) + pd.Timedelta(minutes=delay_minutes)
            return out

        return RobustnessScenario("delayed_updates", f"Delay timestamps by {delay_minutes} minutes", transform)

    def low_confidence(self, confidence_column: str = "confidence", value: float = 0.3) -> RobustnessScenario:
        def transform(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if confidence_column in out.columns:
                out[confidence_column] = value
            return out

        return RobustnessScenario("low_confidence", f"Set confidence to {value}", transform)

    def congestion_spike(self, speed_column: str = "actual_speed", drop_kmh: float = 30.0) -> RobustnessScenario:
        def transform(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if len(out) == 0:
                return out
            spike_start = len(out) // 2
            out.loc[out.index[spike_start:], speed_column] = (
                out.loc[out.index[spike_start:], speed_column].to_numpy(dtype=float) - drop_kmh
            ).clip(0.0, 120.0)
            return out

        return RobustnessScenario("congestion_spike", f"Inject speed drop of {drop_kmh} km/h", transform)

    def api_outage(self, outage_fraction: float = 0.25) -> RobustnessScenario:
        if not 0.0 <= outage_fraction < 1.0:
            raise ValueError("outage_fraction must be in [0.0, 1.0)")

        def transform(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if out.empty:
                return out
            outage_count = max(1, int(round(len(out) * outage_fraction)))
            outage_start = max(0, (len(out) - outage_count) // 2)
            return out.drop(out.index[outage_start : outage_start + outage_count]).reset_index(drop=True)

        return RobustnessScenario("api_outage", f"Remove contiguous outage block covering {outage_fraction:.0%}", transform)

    def default_scenarios(self) -> list[RobustnessScenario]:
        return [
            self.noisy_speed(),
            self.missing_intervals(),
            self.degraded_confidence(),
            self.delayed_updates(),
            self.low_confidence(),
            self.congestion_spike(),
            self.api_outage(),
        ]


class RobustnessEvaluator:
    """Evaluates metric degradation under adverse data conditions."""

    def __init__(self, actual_column: str = "actual_speed") -> None:
        self.actual_column = actual_column

    def evaluate(
        self,
        clean_data: pd.DataFrame,
        predict_fn: PredictionFn,
        scenarios: list[RobustnessScenario],
        baseline_predictors: dict[str, PredictionFn] | None = None,
    ) -> RobustnessReport:
        self._validate(clean_data)
        clean_predictions = predict_fn(clean_data.copy())
        clean_metrics = self._metrics(clean_data, clean_predictions)

        scenario_results = [
            self._evaluate_one(clean_data, clean_metrics, scenario, predict_fn)
            for scenario in scenarios
        ]

        baseline_results: dict[str, list[RobustnessResult]] = {}
        for baseline_name, baseline_fn in (baseline_predictors or {}).items():
            baseline_clean_metrics = self._metrics(clean_data, baseline_fn(clean_data.copy()))
            baseline_results[baseline_name] = [
                self._evaluate_one(clean_data, baseline_clean_metrics, scenario, baseline_fn)
                for scenario in scenarios
            ]

        return RobustnessReport(
            clean_metrics=clean_metrics,
            scenario_results=scenario_results,
            baseline_results=baseline_results,
        )

    def _evaluate_one(
        self,
        clean_data: pd.DataFrame,
        clean_metrics: RegressionMetrics,
        scenario: RobustnessScenario,
        predict_fn: PredictionFn,
    ) -> RobustnessResult:
        scenario_data = scenario.transform(clean_data.copy())
        self._validate(scenario_data)
        predictions = predict_fn(scenario_data.copy())
        metrics = self._metrics(scenario_data, predictions)
        return RobustnessResult(
            scenario_name=scenario.name,
            description=scenario.description,
            metrics=metrics,
            delta_mae=metrics.mae - clean_metrics.mae,
            delta_rmse=metrics.rmse - clean_metrics.rmse,
            degradation_mae_pct=_percent_delta(metrics.mae, clean_metrics.mae),
            degradation_rmse_pct=_percent_delta(metrics.rmse, clean_metrics.rmse),
        )

    def _metrics(self, df: pd.DataFrame, predictions: np.ndarray) -> RegressionMetrics:
        actual = df[self.actual_column].to_numpy(dtype=float)
        predictions = np.asarray(predictions, dtype=float).reshape(-1)
        if len(actual) != len(predictions):
            raise ValueError(
                f"Prediction length mismatch: actual={len(actual)}, predicted={len(predictions)}"
            )
        return calculate_regression_metrics(actual, predictions)

    def _validate(self, df: pd.DataFrame) -> None:
        if self.actual_column not in df.columns:
            raise ValueError(f"Missing actual column: {self.actual_column}")
        if df.empty:
            raise ValueError("Robustness evaluation data must not be empty")


def _percent_delta(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return float(((value - baseline) / baseline) * 100.0)


def _flatten_result(result: RobustnessResult) -> dict:
    return {
        "scenario": result.scenario_name,
        "description": result.description,
        "mae": result.metrics.mae,
        "rmse": result.metrics.rmse,
        "mape": result.metrics.mape,
        "r2": result.metrics.r2,
        "sample_count": result.metrics.sample_count,
        "delta_mae": result.delta_mae,
        "delta_rmse": result.delta_rmse,
        "degradation_mae_pct": result.degradation_mae_pct,
        "degradation_rmse_pct": result.degradation_rmse_pct,
    }

