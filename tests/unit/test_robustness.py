from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.evaluation.robustness import (
    RobustnessEvaluator,
    RobustnessScenarioFactory,
)


def make_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 07:00:00", periods=12, freq="15min"),
            "road_id": ["R1"] * 6 + ["R2"] * 6,
            "actual_speed": [50.0, 52.0, 53.0, 51.0, 49.0, 48.0, 40.0, 42.0, 41.0, 39.0, 38.0, 37.0],
            "confidence": [1.0] * 12,
        }
    )


def model_predict(df: pd.DataFrame) -> np.ndarray:
    return df["actual_speed"].to_numpy(dtype=float) - 1.0


def baseline_predict(df: pd.DataFrame) -> np.ndarray:
    return df["actual_speed"].to_numpy(dtype=float) - 3.0


def test_default_scenarios_are_created() -> None:
    factory = RobustnessScenarioFactory(random_seed=7)
    scenarios = factory.default_scenarios()

    assert [scenario.name for scenario in scenarios] == [
        "noisy_input",
        "missing_intervals",
        "degraded_confidence",
        "delayed_updates",
        "low_confidence",
        "congestion_spike",
        "api_outage",
    ]


def test_scenario_transforms_modify_expected_columns() -> None:
    data = make_data()
    factory = RobustnessScenarioFactory(random_seed=7)

    noisy = factory.noisy_speed(std_kmh=2.0).transform(data)
    missing = factory.missing_intervals(drop_fraction=0.25).transform(data)
    degraded = factory.degraded_confidence(factor=0.5).transform(data)
    delayed = factory.delayed_updates(delay_minutes=15).transform(data)
    low_confidence = factory.low_confidence(value=0.2).transform(data)
    spike = factory.congestion_spike(drop_kmh=30.0).transform(data)
    outage = factory.api_outage(outage_fraction=0.25).transform(data)

    assert not noisy["actual_speed"].equals(data["actual_speed"])
    assert len(missing) < len(data)
    assert degraded["confidence"].max() == pytest.approx(0.5)
    assert delayed["timestamp"].iloc[0] == data["timestamp"].iloc[0] + pd.Timedelta(minutes=15)
    assert low_confidence["confidence"].unique().tolist() == [0.2]
    assert spike["actual_speed"].iloc[-1] == pytest.approx(7.0)
    assert len(outage) < len(data)


def test_robustness_evaluator_reports_degradation_and_baselines() -> None:
    data = make_data()
    factory = RobustnessScenarioFactory(random_seed=7)
    scenarios = [
        factory.noisy_speed(std_kmh=2.0),
        factory.congestion_spike(drop_kmh=10.0),
    ]

    report = RobustnessEvaluator().evaluate(
        clean_data=data,
        predict_fn=model_predict,
        scenarios=scenarios,
        baseline_predictors={"persistence": baseline_predict},
    )
    table = report.to_dataframe()
    worst = report.worst_scenarios(top_n=2)

    assert report.clean_metrics.mae == pytest.approx(1.0)
    assert len(report.scenario_results) == 2
    assert "persistence" in report.baseline_results
    assert len(report.baseline_results["persistence"]) == 2
    assert set(table["model"]) == {"model", "persistence"}
    assert len(worst) == 2
    assert {"delta_mae", "delta_rmse", "degradation_rmse_pct"}.issubset(table.columns)


def test_robustness_evaluator_rejects_prediction_length_mismatch() -> None:
    data = make_data()
    scenario = RobustnessScenarioFactory(random_seed=7).noisy_speed()

    with pytest.raises(ValueError, match="Prediction length mismatch"):
        RobustnessEvaluator().evaluate(
            clean_data=data,
            predict_fn=lambda df: np.array([1.0]),
            scenarios=[scenario],
        )


def test_robustness_evaluator_requires_actual_column() -> None:
    data = make_data().drop(columns=["actual_speed"])

    with pytest.raises(ValueError, match="Missing actual column"):
        RobustnessEvaluator().evaluate(
            clean_data=data,
            predict_fn=lambda df: np.array([]),
            scenarios=[],
        )

