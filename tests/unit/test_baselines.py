from __future__ import annotations

import numpy as np
import pandas as pd

from traffic_prediction.evaluation.baselines import (
    BaselineEvaluator,
    HistoricalAverageBaseline,
    LinearRegressionBaseline,
    OptionalDependencyBaseline,
    PersistenceBaseline,
    default_baselines,
)


def make_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-05 07:00:00",
                    "2026-01-05 07:15:00",
                    "2026-01-06 07:00:00",
                    "2026-01-06 07:15:00",
                ]
            ),
            "road_id": ["R1", "R1", "R2", "R2"],
            "actual_speed": [40.0, 44.0, 30.0, 34.0],
            "lag_1": [39.0, 40.0, 29.0, 30.0],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0],
        }
    )
    test = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-12 07:00:00",
                    "2026-01-12 07:15:00",
                    "2026-01-13 07:00:00",
                    "2026-01-13 07:15:00",
                ]
            ),
            "road_id": ["R1", "R1", "R2", "R2"],
            "actual_speed": [42.0, 46.0, 32.0, 36.0],
            "lag_1": [41.0, 42.0, 31.0, 32.0],
            "feature_a": [1.5, 2.5, 3.5, 4.5],
            "feature_b": [15.0, 25.0, 35.0, 45.0],
        }
    )
    return train, test


def test_persistence_baseline_predicts_lag_1() -> None:
    _, test = make_train_test()
    baseline = PersistenceBaseline().fit(pd.DataFrame())

    predictions = baseline.predict(test)

    assert predictions.tolist() == [41.0, 42.0, 31.0, 32.0]


def test_historical_average_baseline_uses_road_hour_day_lookup() -> None:
    train, test = make_train_test()
    baseline = HistoricalAverageBaseline().fit(train)

    predictions = baseline.predict(test)

    assert predictions.tolist() == [42.0, 42.0, 32.0, 32.0]


def test_linear_regression_baseline_predicts_with_numeric_features() -> None:
    train, test = make_train_test()
    baseline = LinearRegressionBaseline(feature_columns=["feature_a", "feature_b"]).fit(train)

    predictions = baseline.predict(test)

    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()


def test_baseline_evaluator_compares_available_and_optional_models() -> None:
    train, test = make_train_test()
    evaluator = BaselineEvaluator(
        baselines=[
            PersistenceBaseline(),
            HistoricalAverageBaseline(),
            LinearRegressionBaseline(feature_columns=["feature_a", "feature_b"]),
            OptionalDependencyBaseline("missing_optional", "definitely_missing_package_for_test"),
        ]
    )

    comparison = evaluator.evaluate(train, test)
    table = comparison.to_dataframe()

    assert len(comparison.results) == 4
    assert len(comparison.available_results) == 3
    assert comparison.best_by_rmse() is not None
    assert set(table["model"]) == {
        "naive_persistence",
        "historical_average",
        "linear_regression",
        "missing_optional",
    }
    assert table.loc[table["model"] == "missing_optional", "status"].iloc[0] == "unavailable"


def test_default_baselines_include_required_model_names() -> None:
    names = [baseline.name for baseline in default_baselines()]

    assert names == [
        "naive_persistence",
        "historical_average",
        "linear_regression",
        "arima",
        "xgboost",
        "lightgbm",
        "catboost",
    ]
