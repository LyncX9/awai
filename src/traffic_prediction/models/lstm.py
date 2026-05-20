from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LSTMModelConfig:
    """Configuration for the lightweight small-dataset LSTM model."""

    input_size: int
    prediction_horizon: int = 4
    hidden_sizes: tuple[int, int] = (64, 32)
    dense_units: int = 16
    dropout: float = 0.3
    recurrent_dropout: float = 0.2
    bidirectional: bool = False

    @classmethod
    def lightweight(
        cls,
        input_size: int,
        prediction_horizon: int = 4,
        bidirectional: bool = False,
    ) -> "LSTMModelConfig":
        return cls(
            input_size=input_size,
            prediction_horizon=prediction_horizon,
            hidden_sizes=(32, 16),
            dense_units=16,
            dropout=0.3,
            recurrent_dropout=0.2,
            bidirectional=bidirectional,
        )


class TrafficLSTM(nn.Module):
    """
    Lightweight stacked LSTM for traffic speed prediction.

    Input shape:
        (batch_size, lookback, num_features)

    Output shape:
        (batch_size, prediction_horizon, 1)
    """

    def __init__(self, config: LSTMModelConfig) -> None:
        super().__init__()
        if len(config.hidden_sizes) != 2:
            raise ValueError("TrafficLSTM currently expects exactly two hidden sizes")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if not 0.0 <= config.recurrent_dropout < 1.0:
            raise ValueError("recurrent_dropout must be in [0.0, 1.0)")

        self.config = config
        direction_multiplier = 2 if config.bidirectional else 1
        first_hidden, second_hidden = config.hidden_sizes

        if config.bidirectional:
            self.lstm1 = nn.LSTM(
                input_size=config.input_size,
                hidden_size=first_hidden,
                batch_first=True,
                bidirectional=True,
            )
        else:
            self.lstm1 = nn.LSTMCell(
                input_size=config.input_size,
                hidden_size=first_hidden,
            )
        self.norm1 = nn.LayerNorm(first_hidden * direction_multiplier)
        self.dropout1 = nn.Dropout(config.dropout)

        if config.bidirectional:
            self.lstm2 = nn.LSTM(
                input_size=first_hidden * direction_multiplier,
                hidden_size=second_hidden,
                batch_first=True,
                bidirectional=True,
            )
        else:
            self.lstm2 = nn.LSTMCell(
                input_size=first_hidden,
                hidden_size=second_hidden,
            )
        self.norm2 = nn.LayerNorm(second_hidden * direction_multiplier)
        self.dropout2 = nn.Dropout(config.dropout)

        self.regressor = nn.Sequential(
            nn.Linear(second_hidden * direction_multiplier, config.dense_units),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dense_units, config.prediction_horizon),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("TrafficLSTM input must have shape (batch, lookback, features)")
        if inputs.shape[-1] != self.config.input_size:
            raise ValueError(
                f"Expected input_size={self.config.input_size}, got {inputs.shape[-1]}"
            )

        if self.config.bidirectional:
            x, _ = self.lstm1(inputs)
        else:
            x = self._run_lstm_cell_sequence(self.lstm1, inputs)
        x = self.norm1(x)
        x = self.dropout1(x)

        if self.config.bidirectional:
            x, _ = self.lstm2(x)
        else:
            x = self._run_lstm_cell_sequence(self.lstm2, x)
        x = self.norm2(x)
        last_timestep = self.dropout2(x[:, -1, :])

        prediction = self.regressor(last_timestep)
        return prediction.unsqueeze(-1)

    def _run_lstm_cell_sequence(self, cell: nn.LSTMCell, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        hidden_size = cell.hidden_size
        h = inputs.new_zeros((batch_size, hidden_size))
        c = inputs.new_zeros((batch_size, hidden_size))
        outputs: list[torch.Tensor] = []

        for index in range(sequence_length):
            recurrent_h = h
            if self.training and self.config.recurrent_dropout > 0:
                recurrent_h = F.dropout(
                    recurrent_h,
                    p=self.config.recurrent_dropout,
                    training=True,
                )
            h, c = cell(inputs[:, index, :], (recurrent_h, c))
            outputs.append(h)

        return torch.stack(outputs, dim=1)


def build_lstm_model(
    input_size: int,
    prediction_horizon: int = 4,
    lightweight: bool = False,
    bidirectional: bool = False,
) -> TrafficLSTM:
    """Build the default or alternative lightweight LSTM architecture."""

    if lightweight:
        config = LSTMModelConfig.lightweight(
            input_size=input_size,
            prediction_horizon=prediction_horizon,
            bidirectional=bidirectional,
        )
    else:
        config = LSTMModelConfig(
            input_size=input_size,
            prediction_horizon=prediction_horizon,
            bidirectional=bidirectional,
        )
    return TrafficLSTM(config)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
