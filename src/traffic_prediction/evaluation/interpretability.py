from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import Callable, Literal

import numpy as np
import pandas as pd

from traffic_prediction.evaluation.metrics import (
    RegressionMetrics,
    calculate_regression_metrics,
    mean_absolute_error,
    root_mean_squared_error,
)


PredictionFn = Callable[[pd.DataFrame], np.ndarray]
MetricName = Literal["mae", "rmse"]


@dataclass(frozen=True)
class FeatureImportanceResult:
    feature: str
    baseline_score: float
    permuted_score: float
    importance: float
    metric: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class AblationResult:
    feature: str
    strategy: str
    baseline_metrics: RegressionMetrics
    ablated_metrics: RegressionMetrics
    delta_mae: float
    delta_rmse: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["baseline_metrics"] = self.baseline_metrics.to_dict()
        payload["ablated_metrics"] = self.ablated_metrics.to_dict()
        return payload


@dataclass(frozen=True)
class RedundancyPair:
    feature_a: str
    feature_b: str
    correlation: float
    absolute_correlation: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureConsistencyResult:
    feature: str
    mean_importance: float
    std_importance: float
    coefficient_of_variation: float
    observation_count: int

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class ShapAvailability:
    available: bool
    reason: str


@dataclass(frozen=True)
class InterpretabilityReport:
    permutation_importance: list[FeatureImportanceResult]
    ablation_results: list[AblationResult]
    redundancy_pairs: list[RedundancyPair]
    consistency_results: list[FeatureConsistencyResult]
    shap_availability: ShapAvailability

    def permutation_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([result.to_dict() for result in self.permutation_importance])

    def ablation_dataframe(self) -> pd.DataFrame:
        rows = []
        for result in self.ablation_results:
            rows.append(
                {
                    "feature": result.feature,
                    "strategy": result.strategy,
                    "baseline_mae": result.baseline_metrics.mae,
                    "baseline_rmse": result.baseline_metrics.rmse,
                    "ablated_mae": result.ablated_metrics.mae,
                    "ablated_rmse": result.ablated_metrics.rmse,
                    "delta_mae": result.delta_mae,
                    "delta_rmse": result.delta_rmse,
                    "sample_count": result.ablated_metrics.sample_count,
                }
            )
        return pd.DataFrame(rows)

    def redundancy_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([pair.to_dict() for pair in self.redundancy_pairs])

    def consistency_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([result.to_dict() for result in self.consistency_results])


class PermutationImportanceAnalyzer:
    """Computes model-agnostic permutation importance on tabular evaluation data."""

    def __init__(
        self,
        actual_column: str = "actual_speed",
        metric: MetricName = "rmse",
        random_seed: int = 42,
    ) -> None:
        self.actual_column = actual_column
        self.metric = metric
        self.rng = np.random.default_rng(random_seed)

    def analyze(
        self,
        data: pd.DataFrame,
        predict_fn: PredictionFn,
        feature_columns: list[str] | None = None,
    ) -> list[FeatureImportanceResult]:
        self._validate(data)
        features = _resolve_features(data, self.actual_column, feature_columns)
        baseline_predictions = predict_fn(data.copy())
        baseline_score = self._score(data[self.actual_column].to_numpy(dtype=float), baseline_predictions)

        results = []
        for feature in features:
            permuted = data.copy()
            permuted[feature] = self.rng.permutation(permuted[feature].to_numpy())
            predictions = predict_fn(permuted)
            permuted_score = self._score(permuted[self.actual_column].to_numpy(dtype=float), predictions)
            results.append(
                FeatureImportanceResult(
                    feature=feature,
                    baseline_score=baseline_score,
                    permuted_score=permuted_score,
                    importance=permuted_score - baseline_score,
                    metric=self.metric,
                )
            )
        return sorted(results, key=lambda result: result.importance, reverse=True)

    def to_dataframe(self, results: list[FeatureImportanceResult]) -> pd.DataFrame:
        return pd.DataFrame([result.to_dict() for result in results])

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.metric == "mae":
            return mean_absolute_error(y_true, y_pred)
        return root_mean_squared_error(y_true, y_pred)

    def _validate(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("Interpretability data must not be empty")
        if self.actual_column not in data.columns:
            raise ValueError(f"Missing actual column: {self.actual_column}")


class AblationAnalyzer:
    """Measures metric degradation after replacing selected features with neutral values."""

    def __init__(self, actual_column: str = "actual_speed") -> None:
        self.actual_column = actual_column

    def analyze(
        self,
        data: pd.DataFrame,
        predict_fn: PredictionFn,
        feature_columns: list[str] | None = None,
        strategy: Literal["mean", "median", "zero"] = "mean",
        reference_data: pd.DataFrame | None = None,
    ) -> list[AblationResult]:
        self._validate(data)
        features = _resolve_features(data, self.actual_column, feature_columns)
        reference = data if reference_data is None else reference_data
        baseline_metrics = self._metrics(data, predict_fn(data.copy()))

        results = []
        for feature in features:
            ablated = data.copy()
            ablated[feature] = self._replacement_value(reference, feature, strategy)
            ablated_metrics = self._metrics(ablated, predict_fn(ablated))
            results.append(
                AblationResult(
                    feature=feature,
                    strategy=strategy,
                    baseline_metrics=baseline_metrics,
                    ablated_metrics=ablated_metrics,
                    delta_mae=ablated_metrics.mae - baseline_metrics.mae,
                    delta_rmse=ablated_metrics.rmse - baseline_metrics.rmse,
                )
            )
        return sorted(results, key=lambda result: result.delta_rmse, reverse=True)

    def to_dataframe(self, results: list[AblationResult]) -> pd.DataFrame:
        return InterpretabilityReport([], results, [], [], ShapAvailability(False, "")).ablation_dataframe()

    def _metrics(self, data: pd.DataFrame, predictions: np.ndarray) -> RegressionMetrics:
        actual = data[self.actual_column].to_numpy(dtype=float)
        predicted = np.asarray(predictions, dtype=float).reshape(-1)
        if len(actual) != len(predicted):
            raise ValueError(f"Prediction length mismatch: actual={len(actual)}, predicted={len(predicted)}")
        return calculate_regression_metrics(actual, predicted)

    def _replacement_value(
        self,
        reference_data: pd.DataFrame,
        feature: str,
        strategy: Literal["mean", "median", "zero"],
    ) -> float:
        if strategy == "zero":
            return 0.0
        if feature not in reference_data.columns:
            raise ValueError(f"Reference data missing feature: {feature}")
        values = reference_data[feature].to_numpy(dtype=float)
        if strategy == "median":
            return float(np.nanmedian(values))
        return float(np.nanmean(values))

    def _validate(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("Ablation data must not be empty")
        if self.actual_column not in data.columns:
            raise ValueError(f"Missing actual column: {self.actual_column}")


class FeatureRedundancyAnalyzer:
    """Finds highly correlated numeric feature pairs."""

    def __init__(self, threshold: float = 0.95) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold

    def analyze(self, data: pd.DataFrame, feature_columns: list[str] | None = None) -> list[RedundancyPair]:
        features = feature_columns or data.select_dtypes(include=[np.number]).columns.tolist()
        features = [feature for feature in features if feature in data.columns]
        if len(features) < 2:
            return []

        corr = data[features].corr(numeric_only=True)
        pairs: list[RedundancyPair] = []
        for left_index, feature_a in enumerate(features):
            for feature_b in features[left_index + 1 :]:
                value = corr.loc[feature_a, feature_b]
                if pd.isna(value):
                    continue
                abs_value = float(abs(value))
                if abs_value >= self.threshold:
                    pairs.append(
                        RedundancyPair(
                            feature_a=feature_a,
                            feature_b=feature_b,
                            correlation=float(value),
                            absolute_correlation=abs_value,
                        )
                    )
        return sorted(pairs, key=lambda pair: pair.absolute_correlation, reverse=True)

    def to_dataframe(self, pairs: list[RedundancyPair]) -> pd.DataFrame:
        return pd.DataFrame([pair.to_dict() for pair in pairs])


class FeatureImportanceConsistencyAnalyzer:
    """Summarizes how stable feature importance is across folds, roads, or scenarios."""

    def analyze(
        self,
        importance_tables: list[pd.DataFrame],
        feature_column: str = "feature",
        importance_column: str = "importance",
    ) -> list[FeatureConsistencyResult]:
        if not importance_tables:
            return []
        combined = pd.concat(importance_tables, ignore_index=True)
        missing = {feature_column, importance_column} - set(combined.columns)
        if missing:
            raise ValueError(f"Importance table missing required columns: {sorted(missing)}")

        results = []
        for feature, group in combined.groupby(feature_column):
            values = group[importance_column].to_numpy(dtype=float)
            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            denominator = max(abs(mean_value), 1e-12)
            results.append(
                FeatureConsistencyResult(
                    feature=str(feature),
                    mean_importance=mean_value,
                    std_importance=std_value,
                    coefficient_of_variation=float(std_value / denominator),
                    observation_count=int(len(values)),
                )
            )
        return sorted(results, key=lambda result: result.coefficient_of_variation)

    def to_dataframe(self, results: list[FeatureConsistencyResult]) -> pd.DataFrame:
        return pd.DataFrame([result.to_dict() for result in results])


class ShapExplainer:
    """Thin optional SHAP adapter for tree-based baselines."""

    def availability(self) -> ShapAvailability:
        if find_spec("shap") is None:
            return ShapAvailability(
                available=False,
                reason="Package 'shap' is not installed in the current environment.",
            )
        return ShapAvailability(available=True, reason="Package 'shap' is available.")

    def tree_explainer(self, model):
        availability = self.availability()
        if not availability.available:
            raise ImportError(availability.reason)
        import shap  # type: ignore[import-not-found]

        return shap.TreeExplainer(model)

    def explain_tree_model(self, model, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        explainer = self.tree_explainer(model)
        return np.asarray(explainer.shap_values(features))


def _resolve_features(
    data: pd.DataFrame,
    actual_column: str,
    feature_columns: list[str] | None,
) -> list[str]:
    if feature_columns is None:
        features = data.select_dtypes(include=[np.number]).columns.tolist()
        features = [feature for feature in features if feature != actual_column]
    else:
        missing = sorted(set(feature_columns) - set(data.columns))
        if missing:
            raise ValueError(f"Feature columns missing from data: {missing}")
        features = list(feature_columns)
    if not features:
        raise ValueError("At least one feature column is required")
    return features
