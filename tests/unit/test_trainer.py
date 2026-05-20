from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pytest

from traffic_prediction.models.lstm import LSTMModelConfig
from traffic_prediction.models.registry import ModelRegistry
from traffic_prediction.training.trainer import LSTMTrainer, TrainingLoopConfig


def make_test_root() -> Path:
    root = Path("artifacts") / "test_runs" / "trainer" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_dataset(samples: int = 16, lookback: int = 6, features: int = 5, horizon: int = 2):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(samples, lookback, features)).astype(np.float32)
    y_base = x[:, -1:, :1].repeat(horizon, axis=1)
    y = (y_base * 0.5).astype(np.float32)
    return x, y


def test_lstm_trainer_saves_checkpoint_and_registers_model() -> None:
    root = make_test_root()
    X_train, y_train = make_dataset(samples=20)
    X_validation, y_validation = make_dataset(samples=8)
    registry = ModelRegistry(root / "registry.json")
    trainer = LSTMTrainer(
        model_config=LSTMModelConfig(input_size=5, prediction_horizon=2, hidden_sizes=(8, 4), dense_units=4),
        training_config=TrainingLoopConfig(max_epochs=3, batch_size=4, early_stopping_patience=2),
    )

    result = trainer.train(
        X_train,
        y_train,
        X_validation,
        y_validation,
        artifact_root=root / "models",
        registry=registry,
        model_version="test-model",
    )

    artifact_path = Path(result.artifact_path)
    assert (artifact_path / "model.pt").exists()
    assert (artifact_path / "training_history.json").exists()
    assert (artifact_path / "model_config.json").exists()
    assert (artifact_path / "metrics.json").exists()
    assert registry.get_active().model_version == "test-model"
    assert result.validation_rmse >= 0
    history = json.loads((artifact_path / "training_history.json").read_text(encoding="utf-8"))
    assert len(history["history"]) >= 1


def test_lstm_trainer_validates_feature_count() -> None:
    root = make_test_root()
    X_train, y_train = make_dataset(features=5)
    X_validation, y_validation = make_dataset(samples=8, features=5)
    trainer = LSTMTrainer(
        model_config=LSTMModelConfig(input_size=6, prediction_horizon=2, hidden_sizes=(8, 4), dense_units=4),
        training_config=TrainingLoopConfig(max_epochs=1, batch_size=4),
    )

    with pytest.raises(ValueError, match="feature count"):
        trainer.train(X_train, y_train, X_validation, y_validation, artifact_root=root / "models")


def test_lstm_trainer_records_warmup_learning_rates() -> None:
    root = make_test_root()
    X_train, y_train = make_dataset(samples=12)
    X_validation, y_validation = make_dataset(samples=8)
    trainer = LSTMTrainer(
        model_config=LSTMModelConfig(input_size=5, prediction_horizon=2, hidden_sizes=(8, 4), dense_units=4),
        training_config=TrainingLoopConfig(
            max_epochs=3,
            batch_size=4,
            learning_rate=0.01,
            early_stopping_patience=5,
            warmup_epochs=3,
            warmup_start_factor=0.1,
        ),
    )

    result = trainer.train(
        X_train,
        y_train,
        X_validation,
        y_validation,
        artifact_root=root / "models",
        model_version="warmup-model",
    )

    history = json.loads((Path(result.artifact_path) / "training_history.json").read_text(encoding="utf-8"))
    learning_rates = [row["learning_rate"] for row in history["history"]]
    assert learning_rates == pytest.approx([0.004, 0.007, 0.01])
