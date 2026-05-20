from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.evaluation.metrics import (
    ModelEvaluator,
    calculate_regression_metrics,
    classify_congestion,
    classify_peak_period,
    horizon_metrics,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)


def test_regression_metric_formulas() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    assert mean_absolute_error(y_true, y_pred) == pytest.approx(7.0 / 3.0)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(17.0 / 3.0))
    assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(((0.2 + 0.1 + 0.1) / 3.0) * 100.0)
    assert r2_score(y_true, y_pred) == pytest.approx(1.0 - 17.0 / 200.0)

    metrics = calculate_regression_metrics(y_true, y_pred)
    assert metrics.mae == pytest.approx(7.0 / 3.0)
    assert metrics.sample_count == 3


def test_horizon_metrics_uses_prediction_horizon_axis() -> None:
    y_true = np.array(
        [
            [[10.0], [20.0]],
            [[30.0], [40.0]],
        ]
    )
    y_pred = np.array(
        [
            [[11.0], [18.0]],
            [[29.0], [44.0]],
        ]
    )

    metrics = horizon_metrics(y_true, y_pred, horizon_minutes=[15, 30])

    assert set(metrics) == {15, 30}
    assert metrics[15].mae == pytest.approx(1.0)
    assert metrics[30].mae == pytest.approx(3.0)


def test_classification_helpers() -> None:
    assert classify_congestion(90.0, 100.0) == "free_flow"
    assert classify_congestion(70.0, 100.0) == "moderate"
    assert classify_congestion(45.0, 100.0) == "congested"
    assert classify_congestion(30.0, 100.0) == "severe"
    assert classify_peak_period(pd.Timestamp("2026-01-01 07:30:00")) == "morning_rush"
    assert classify_peak_period(pd.Timestamp("2026-01-01 17:30:00")) == "evening_rush"
    assert classify_peak_period(pd.Timestamp("2026-01-01 12:00:00")) == "off_peak"


def test_model_evaluator_dataframe_report_groups_conditions() -> None:
    df = pd.DataFrame(
        {
            "actual_speed": [80.0, 60.0, 40.0, 20.0, 50.0, 45.0],
            "predicted_speed": [78.0, 65.0, 35.0, 30.0, 48.0, 40.0],
            "free_flow_speed": [100.0] * 6,
            "horizon_minutes": [15, 15, 30, 30, 60, 60],
            "road_id": ["R1", "R1", "R2", "R2", "R3", "R3"],
            "frc": ["FRC1", "FRC1", "FRC2", "FRC2", "FRC3", "FRC3"],
            "timestamp": [
                "2026-01-05 07:30:00",
                "2026-01-05 08:00:00",
                "2026-01-05 17:30:00",
                "2026-01-05 18:00:00",
                "2026-01-10 12:00:00",
                "2026-01-10 12:15:00",
            ],
        }
    )

    report = ModelEvaluator().evaluate_dataframe(df)

    assert report.overall.sample_count == 6
    assert set(report.by_horizon) == {15, 30, 60}
    assert set(report.by_peak_period) == {"morning_rush", "evening_rush", "off_peak"}
    assert set(report.by_congestion_level) == {"free_flow", "moderate", "congested", "severe"}
    assert set(report.by_road_class) == {"FRC1", "FRC2", "FRC3"}
    assert set(report.by_weekend) == {"weekday", "weekend"}
    assert len(report.worst_roads_by_rmse) == 3
    assert report.worst_roads_by_rmse.iloc[0]["rmse"] >= report.worst_roads_by_rmse.iloc[-1]["rmse"]
    assert len(report.worst_time_periods_by_rmse) == 6


def test_metrics_reject_empty_or_mismatched_arrays() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_regression_metrics(np.array([]), np.array([]))

    with pytest.raises(ValueError, match="shape mismatch"):
        calculate_regression_metrics(np.array([1.0]), np.array([1.0, 2.0]))

