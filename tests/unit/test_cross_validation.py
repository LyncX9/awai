from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.models.lstm import LSTMModelConfig
from traffic_prediction.training.cross_validation import (
    CrossValidationTrainer,
    FoldFeaturePreparer,
    ReducedTimeSeriesCVConfig,
    ReducedTimeSeriesSplit,
)
from traffic_prediction.training.trainer import TrainingLoopConfig


def make_test_root() -> Path:
    root = Path("artifacts") / "test_runs" / "cross_validation" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_dataset(samples: int = 24, lookback: int = 4, features: int = 3, horizon: int = 2):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(samples, lookback, features)).astype(np.float32)
    y = x[:, -horizon:, :1].copy()
    return x, y


def test_reduced_time_series_split_expands_train_and_preserves_order() -> None:
    splitter = ReducedTimeSeriesSplit(
        ReducedTimeSeriesCVConfig(n_splits=3, validation_size=4, min_train_size=8)
    )

    folds = splitter.split_arrays(np.zeros((24, 2, 1), dtype=np.float32))

    assert len(folds) == 3
    assert [fold.train_size for fold in folds] == [8, 12, 16]
    assert [fold.validation_size for fold in folds] == [4, 4, 4]
    for fold in folds:
        assert np.intersect1d(fold.train_indices, fold.validation_indices).size == 0
        assert fold.train_indices.max() < fold.validation_indices.min()


def test_reduced_time_series_split_dataframe_uses_timestamp_blocks() -> None:
    timestamps = pd.date_range("2026-01-01", periods=10, freq="15min", tz="Asia/Jakarta")
    df = pd.DataFrame(
        {
            "collected_at_wib": list(timestamps) * 2,
            "road_id": ["A"] * 10 + ["B"] * 10,
            "value": range(20),
        }
    )
    splitter = ReducedTimeSeriesSplit(
        ReducedTimeSeriesCVConfig(n_splits=2, validation_size=2, min_train_size=4)
    )

    folds = splitter.split_dataframe(df)

    assert len(folds) == 2
    first_train, first_validation, first_fold = folds[0]
    assert first_train["collected_at_wib"].nunique() == 4
    assert first_validation["collected_at_wib"].nunique() == 2
    assert first_fold.train_size == 8
    assert first_fold.validation_size == 4
    assert first_train["collected_at_wib"].max() < first_validation["collected_at_wib"].min()


def test_reduced_time_series_split_rejects_too_few_samples() -> None:
    splitter = ReducedTimeSeriesSplit(
        ReducedTimeSeriesCVConfig(n_splits=3, validation_size=4, min_train_size=8)
    )

    with pytest.raises(ValueError, match="Not enough samples"):
        splitter.split_arrays(np.zeros((10, 2, 1), dtype=np.float32))


def test_fold_feature_preparer_selects_and_scales_features_per_fold() -> None:
    timestamps = pd.date_range("2026-01-01", periods=12, freq="15min", tz="Asia/Jakarta")
    df = pd.DataFrame(
        {
            "collected_at_wib": timestamps,
            "road_id": ["A"] * 12,
            "feature_a": np.arange(12, dtype=float),
            "feature_b": np.arange(100, 112, dtype=float),
            "label": np.arange(200, 212, dtype=float),
            "text": ["ignored"] * 12,
        }
    )
    splitter = ReducedTimeSeriesSplit(
        ReducedTimeSeriesCVConfig(n_splits=2, validation_size=2, min_train_size=4)
    )
    preparer = FoldFeaturePreparer(splitter)

    prepared = preparer.prepare(df, excluded_columns={"collected_at_wib", "road_id", "label", "text"})

    assert len(prepared) == 2
    assert prepared[0].feature_columns == ["feature_a", "feature_b"]
    assert prepared[1].feature_columns == ["feature_a", "feature_b"]
    assert prepared[0].train["feature_a"].mean() == pytest.approx(0.0)
    assert prepared[0].train["feature_b"].mean() == pytest.approx(0.0)
    assert prepared[0].validation["label"].tolist() == [204.0, 205.0]


def test_cross_validation_trainer_writes_summary() -> None:
    root = make_test_root()
    X, y = make_dataset()
    splitter = ReducedTimeSeriesSplit(
        ReducedTimeSeriesCVConfig(n_splits=2, validation_size=4, min_train_size=8)
    )
    trainer = CrossValidationTrainer(
        model_config_factory=lambda: LSTMModelConfig(
            input_size=3,
            prediction_horizon=2,
            hidden_sizes=(6, 3),
            dense_units=4,
        ),
        training_config=TrainingLoopConfig(
            max_epochs=1,
            batch_size=4,
            warmup_epochs=0,
            early_stopping_patience=2,
        ),
        splitter=splitter,
    )

    summary = trainer.run(X, y, artifact_root=root / "models", model_version_prefix="cv-test")

    assert summary.n_splits == 2
    assert len(summary.folds) == 2
    assert summary.mean_validation_rmse >= 0
    summary_path = root / "models" / "cv-test_cross_validation_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["n_splits"] == 2
