from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pytest

from traffic_prediction.training.cross_validation import ReducedTimeSeriesCVConfig
from traffic_prediction.training.trainer import TrainingLoopConfig
from traffic_prediction.training.tuning import (
    HyperparameterCandidate,
    HyperparameterSearchSpace,
    HyperparameterTuner,
)


def make_test_root() -> Path:
    root = Path("artifacts") / "test_runs" / "hyperparameter_tuning" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_candidate(name: str, learning_rate: float, dropout: float = 0.3, lookback: int = 4) -> HyperparameterCandidate:
    return HyperparameterCandidate(
        name=name,
        hidden_sizes=(8, 4),
        dense_units=4,
        dropout=dropout,
        recurrent_dropout=0.1,
        learning_rate=learning_rate,
        batch_size=4,
        weight_decay=0.0001,
        lookback=lookback,
    )


def test_search_space_generates_bounded_candidates() -> None:
    space = HyperparameterSearchSpace(
        hidden_sizes=((64, 32), (32, 16)),
        dense_units=(16,),
        dropout=(0.2, 0.3),
        recurrent_dropout=(0.2,),
        learning_rate=(0.001,),
        batch_size=(32, 64),
        weight_decay=(0.0001,),
        lookback=(12,),
    )

    candidates = space.generate_candidates(max_trials=3)

    assert len(candidates) == 3
    assert candidates[0].name == "candidate-001"
    assert candidates[0].hidden_sizes == (64, 32)
    assert candidates[0].lookback == 12


def test_hyperparameter_tuner_selects_best_and_writes_files() -> None:
    root = make_test_root()
    candidates = [
        make_candidate("slow", learning_rate=0.0005),
        make_candidate("best", learning_rate=0.001),
        make_candidate("too-fast", learning_rate=0.01),
    ]
    scores = {"slow": 0.9, "best": 0.5, "too-fast": 1.4}
    tuner = HyperparameterTuner(rank_metric="mean_validation_rmse", minimize=True)

    result = tuner.run(
        artifact_root=root,
        candidates=candidates,
        objective_fn=lambda candidate: ({"mean_validation_rmse": scores[candidate.name]}, None),
    )

    assert result.best_candidate.name == "best"
    assert result.best_score == pytest.approx(0.5)
    assert (root / "tuning_results.json").exists()
    assert (root / "best_config.json").exists()
    payload = json.loads((root / "best_config.json").read_text(encoding="utf-8"))
    assert payload["best_candidate"]["name"] == "best"


def test_hyperparameter_tuner_requires_rank_metric() -> None:
    root = make_test_root()
    tuner = HyperparameterTuner(rank_metric="mean_validation_rmse")

    with pytest.raises(ValueError, match="missing rank metric"):
        tuner.run(
            artifact_root=root,
            candidates=[make_candidate("bad", learning_rate=0.001)],
            objective_fn=lambda candidate: ({"mae": 1.0}, None),
        )


def test_run_lstm_cv_rejects_precomputed_array_with_wrong_lookback() -> None:
    root = make_test_root()
    X = np.zeros((8, 4, 3), dtype=np.float32)
    y = np.zeros((8, 2, 1), dtype=np.float32)
    tuner = HyperparameterTuner(rank_metric="mean_validation_rmse")

    with pytest.raises(ValueError, match="lookback does not match"):
        tuner.run_lstm_cv(
            X=X,
            y=y,
            artifact_root=root,
            input_size=3,
            prediction_horizon=2,
            base_training_config=TrainingLoopConfig(max_epochs=1, batch_size=2, warmup_epochs=0),
            cv_config=ReducedTimeSeriesCVConfig(n_splits=1, validation_size=2, min_train_size=4),
            candidates=[make_candidate("wrong-lookback", learning_rate=0.001, lookback=12)],
        )

