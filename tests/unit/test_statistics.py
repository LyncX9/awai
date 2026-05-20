from __future__ import annotations

import numpy as np
import pytest

from traffic_prediction.evaluation.statistics import (
    StatisticalTester,
    absolute_errors,
    cohens_d_paired,
    confidence_interval_mean,
    diebold_mariano_test,
    paired_t_test_errors,
    squared_errors,
)


def test_error_helpers_and_confidence_interval() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 18.0, 33.0])

    assert absolute_errors(y_true, y_pred).tolist() == [1.0, 2.0, 3.0]
    assert squared_errors(y_true, y_pred).tolist() == [1.0, 4.0, 9.0]

    interval = confidence_interval_mean(np.array([1.0, 2.0, 3.0]), confidence_level=0.95)
    assert interval.mean == pytest.approx(2.0)
    assert interval.lower < interval.mean < interval.upper


def test_paired_t_test_errors_detects_better_model() -> None:
    model_errors = np.array([1.0, 1.0, 2.0, 1.0, 2.0])
    baseline_errors = np.array([3.0, 2.0, 4.0, 3.0, 4.0])

    result = paired_t_test_errors(model_errors, baseline_errors, comparison_name="lstm_vs_baseline")

    assert result.test_name == "paired_t_test"
    assert result.comparison_name == "lstm_vs_baseline"
    assert result.sample_count == 5
    assert result.statistic < 0
    assert result.p_value < 0.05
    assert result.significant is True
    assert result.effect_size < 0
    assert result.confidence_interval.mean < 0


def test_cohens_d_paired_returns_zero_for_identical_differences() -> None:
    assert cohens_d_paired(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)


def test_diebold_mariano_test_compares_forecast_losses() -> None:
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    model_pred = np.array([10.5, 20.5, 29.5, 40.5, 49.5, 60.5])
    baseline_pred = np.array([13.0, 17.0, 33.0, 37.0, 53.0, 57.0])

    result = diebold_mariano_test(y_true, model_pred, baseline_pred, horizon=1)

    assert result.test_name == "diebold_mariano"
    assert result.sample_count == 6
    assert result.statistic < 0
    assert result.p_value < 0.05
    assert result.significant is True


def test_statistical_tester_builds_publication_table_for_many_baselines() -> None:
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    model_pred = np.array([10.5, 19.5, 30.5, 39.5])
    baselines = {
        "persistence": np.array([12.0, 18.5, 32.5, 37.5]),
        "historical_average": np.array([13.0, 17.5, 33.5, 36.5]),
    }

    table = StatisticalTester(alpha=0.05).compare_against_many(y_true, model_pred, baselines)

    assert len(table) == 4
    assert set(table["test_name"]) == {"paired_t_test", "diebold_mariano"}
    assert set(table["comparison_name"]) == {
        "model_vs_persistence",
        "model_vs_historical_average",
    }
    assert {"statistic", "p_value", "effect_size", "significant", "sample_count"}.issubset(table.columns)


def test_statistical_tests_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        paired_t_test_errors(np.array([1.0]), np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="shape mismatch"):
        diebold_mariano_test(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0]),
        )
