from __future__ import annotations

import pytest
import torch

from traffic_prediction.models.lstm import build_lstm_model, count_trainable_parameters


def test_default_lstm_outputs_prediction_horizon_shape() -> None:
    model = build_lstm_model(input_size=41, prediction_horizon=4)
    model.eval()
    batch = torch.zeros((2, 12, 41), dtype=torch.float32)

    with torch.no_grad():
        output = model(batch)

    assert tuple(output.shape) == (2, 4, 1)
    assert count_trainable_parameters(model) > 0


def test_lightweight_lstm_uses_fewer_parameters_than_default() -> None:
    default_model = build_lstm_model(input_size=41, prediction_horizon=4)
    lightweight_model = build_lstm_model(input_size=41, prediction_horizon=4, lightweight=True)

    assert count_trainable_parameters(lightweight_model) < count_trainable_parameters(default_model)


def test_lstm_rejects_wrong_feature_count() -> None:
    model = build_lstm_model(input_size=41, prediction_horizon=4)
    bad_batch = torch.zeros((2, 12, 40), dtype=torch.float32)

    with pytest.raises(ValueError, match="Expected input_size"):
        model(bad_batch)


def test_lstm_supports_recurrent_dropout_during_training() -> None:
    model = build_lstm_model(input_size=5, prediction_horizon=2)
    model.train()
    batch = torch.zeros((2, 6, 5), dtype=torch.float32)

    output = model(batch)

    assert tuple(output.shape) == (2, 2, 1)
    assert model.config.recurrent_dropout == pytest.approx(0.2)
