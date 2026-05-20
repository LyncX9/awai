from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.evaluation.interpretability import (
    AblationAnalyzer,
    FeatureImportanceConsistencyAnalyzer,
    FeatureRedundancyAnalyzer,
    PermutationImportanceAnalyzer,
    ShapExplainer,
)


def make_data() -> pd.DataFrame:
    feature_a = np.arange(1.0, 13.0)
    feature_b = feature_a * 2.0
    feature_c = np.array([2.0, 4.0, 2.0, 5.0, 1.0, 3.0, 7.0, 6.0, 8.0, 5.0, 9.0, 4.0])
    actual = 10.0 + (feature_a * 3.0) + (feature_c * 0.25)
    return pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "feature_c": feature_c,
            "actual_speed": actual,
        }
    )


def model_predict(df: pd.DataFrame) -> np.ndarray:
    return 10.0 + (df["feature_a"].to_numpy(dtype=float) * 3.0) + (df["feature_c"].to_numpy(dtype=float) * 0.25)


def test_permutation_importance_ranks_influential_features() -> None:
    data = make_data()
    analyzer = PermutationImportanceAnalyzer(random_seed=7)

    results = analyzer.analyze(
        data,
        model_predict,
        feature_columns=["feature_a", "feature_c"],
    )
    table = analyzer.to_dataframe(results)

    assert results[0].feature == "feature_a"
    assert results[0].importance > results[1].importance
    assert results[0].baseline_score == pytest.approx(0.0)
    assert {"feature", "importance", "metric"}.issubset(table.columns)


def test_ablation_reports_metric_degradation() -> None:
    data = make_data()
    analyzer = AblationAnalyzer()

    results = analyzer.analyze(
        data,
        model_predict,
        feature_columns=["feature_a", "feature_c"],
        strategy="mean",
    )
    table = analyzer.to_dataframe(results)

    assert results[0].feature == "feature_a"
    assert results[0].baseline_metrics.rmse == pytest.approx(0.0)
    assert results[0].delta_rmse > results[1].delta_rmse
    assert {"baseline_rmse", "ablated_rmse", "delta_rmse"}.issubset(table.columns)


def test_redundancy_analyzer_detects_high_correlation_pairs() -> None:
    data = make_data()
    analyzer = FeatureRedundancyAnalyzer(threshold=0.99)

    pairs = analyzer.analyze(data, feature_columns=["feature_a", "feature_b", "feature_c"])

    assert len(pairs) == 1
    assert pairs[0].feature_a == "feature_a"
    assert pairs[0].feature_b == "feature_b"
    assert pairs[0].absolute_correlation == pytest.approx(1.0)


def test_feature_importance_consistency_summarizes_multiple_tables() -> None:
    fold_1 = pd.DataFrame({"feature": ["feature_a", "feature_c"], "importance": [8.0, 1.0]})
    fold_2 = pd.DataFrame({"feature": ["feature_a", "feature_c"], "importance": [10.0, 1.2]})
    analyzer = FeatureImportanceConsistencyAnalyzer()

    results = analyzer.analyze([fold_1, fold_2])
    table = analyzer.to_dataframe(results)

    by_feature = {result.feature: result for result in results}
    assert by_feature["feature_a"].mean_importance == pytest.approx(9.0)
    assert by_feature["feature_a"].observation_count == 2
    assert by_feature["feature_c"].coefficient_of_variation < by_feature["feature_a"].coefficient_of_variation
    assert {"feature", "mean_importance", "coefficient_of_variation"}.issubset(table.columns)


def test_shap_explainer_reports_optional_dependency_status() -> None:
    explainer = ShapExplainer()
    availability = explainer.availability()

    if availability.available:
        assert availability.reason == "Package 'shap' is available."
    else:
        assert "not installed" in availability.reason
        with pytest.raises(ImportError, match="shap"):
            explainer.tree_explainer(model=object())
