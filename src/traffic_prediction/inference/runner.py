from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from traffic_prediction.data.scalers import ScalerStore
from traffic_prediction.inference.realtime import PredictionModelRunner
from traffic_prediction.models.lstm import TrafficLSTM, TrafficSeq2SeqLSTM, LSTMModelConfig
import torch.nn as nn

class PyTorchModelRunner(PredictionModelRunner):
    """
    Concrete implementation of PredictionModelRunner that executes
    a PyTorch LSTM model and handles inverse scaling of the outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        scaler: ScalerStore,
        device: torch.device,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.device = device
        self.model.eval()

    @classmethod
    def load_from_artifact(
        cls,
        artifact_path: str | Path,
        config: dict[str, Any] | None = None,
    ) -> "PyTorchModelRunner":
        """
        Instantiate the runner from an offline training artifact directory.
        """
        artifact_dir = Path(artifact_path)
        model_path = artifact_dir / "model.pt"
        scaler_path = artifact_dir / "scaler_params.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        scaler = ScalerStore.load(scaler_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve config: prefer explicit arg, fall back to model_config.json in artifact dir
        resolved_config = config or {}
        if not resolved_config or "input_size" not in resolved_config:
            config_json_path = artifact_dir / "model_config.json"
            if config_json_path.exists():
                import json
                raw = json.loads(config_json_path.read_text(encoding="utf-8"))
                # model_config.json may nest the config under "model_config" key
                resolved_config = raw.get("model_config", raw)

        if "input_size" not in resolved_config:
            raise ValueError(
                f"Model config requires 'input_size' but it was not found in the provided config "
                f"or in '{artifact_dir / 'model_config.json'}'"
            )

        model_config = LSTMModelConfig(
            input_size=resolved_config["input_size"],
            prediction_horizon=resolved_config.get("prediction_horizon", 4),
            hidden_sizes=tuple(resolved_config.get("hidden_sizes", (64, 32))),
            dense_units=resolved_config.get("dense_units", 16),
            dropout=resolved_config.get("dropout", 0.3),
            recurrent_dropout=resolved_config.get("recurrent_dropout", 0.2),
            bidirectional=resolved_config.get("bidirectional", False),
            seq2seq=resolved_config.get("seq2seq", False),
        )

        if getattr(model_config, "seq2seq", False):
            model = TrafficSeq2SeqLSTM(model_config)
        else:
            model = TrafficLSTM(model_config)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Handle both plain state_dict and full checkpoint dict
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.to(device)

        return cls(model=model, scaler=scaler, device=device)

    def predict_kmh(self, sequence: np.ndarray) -> np.ndarray:
        """
        Execute forward pass and unscale the result to original km/h.

        Args:
            sequence: (1, lookback, num_features) shaped numpy array from the online engineer.
        
        Returns:
            1D numpy array of shape (prediction_horizon,)
        """
        if sequence.ndim != 3 or sequence.shape[0] != 1:
            raise ValueError(f"Expected sequence shape (1, lookback, num_features), got {sequence.shape}")

        with torch.no_grad():
            tensor_seq = torch.from_numpy(sequence).to(self.device)
            # Output is (batch, horizon, 1) -> (1, horizon, 1)
            prediction = self.model(tensor_seq)

        # Convert to (horizon, 1)
        pred_scaled = prediction.detach().cpu().numpy()[0, :, :]
        
        # Inverse transform to get original km/h values
        pred_kmh = self.scaler.inverse_transform_speed(pred_scaled)
        
        # Flatten to (horizon,)
        return pred_kmh.flatten()
