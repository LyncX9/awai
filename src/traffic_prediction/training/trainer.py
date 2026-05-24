from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from traffic_prediction.models.lstm import LSTMModelConfig, TrafficLSTM, TrafficSeq2SeqLSTM, count_trainable_parameters
from traffic_prediction.models.registry import ModelRegistry, ModelRegistryEntry


@dataclass(frozen=True)
class TrainingLoopConfig:
    """Small-dataset training configuration for the PyTorch LSTM."""

    max_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    lr_plateau_patience: int = 5
    lr_plateau_factor: float = 0.5
    warmup_epochs: int = 10
    warmup_start_factor: float = 0.1
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.0001
    random_seed: int = 42
    device: str = "cpu"


@dataclass(frozen=True)
class TrainingResult:
    model_version: str
    artifact_path: str
    checkpoint_path: str
    best_epoch: int
    train_loss: float
    validation_loss: float
    validation_mae: float
    validation_rmse: float
    parameter_count: int
    registry_entry: ModelRegistryEntry | None = None


class LSTMTrainer:
    """Trains TrafficLSTM and writes a versioned local model artifact."""

    def __init__(
        self,
        model_config: LSTMModelConfig,
        training_config: TrainingLoopConfig | None = None,
    ) -> None:
        self.model_config = model_config
        self.training_config = training_config or TrainingLoopConfig()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
        artifact_root: str | Path,
        registry: ModelRegistry | None = None,
        model_version: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> TrainingResult:
        self._validate_arrays(X_train, y_train, X_validation, y_validation)
        self._set_seeds(self.training_config.random_seed)

        version = model_version or datetime.now().strftime("lstm-%Y%m%d-%H%M%S-%f")
        artifact_path = Path(artifact_root) / version
        artifact_path.mkdir(parents=True, exist_ok=True)

        device = torch.device(self.training_config.device)
        if getattr(self.model_config, "seq2seq", False):
            model = TrafficSeq2SeqLSTM(self.model_config).to(device)
        else:
            model = TrafficLSTM(self.model_config).to(device)
        # We use HuberLoss as it is more robust to traffic outliers than MSE
        criterion = nn.HuberLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.training_config.lr_plateau_factor,
            patience=self.training_config.lr_plateau_patience,
        )

        train_loader = self._build_loader(X_train, y_train, shuffle=False)
        validation_loader = self._build_loader(X_validation, y_validation, shuffle=False)

        history: list[dict[str, float | int]] = []
        best_validation_loss = math.inf
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, self.training_config.max_epochs + 1):
            self._apply_learning_rate_warmup(optimizer, epoch)
            train_loss = self._train_one_epoch(model, train_loader, criterion, optimizer, device)
            validation_loss, validation_mae, validation_rmse = self._evaluate(
                model,
                validation_loader,
                criterion,
                device,
            )
            if epoch > self.training_config.warmup_epochs:
                scheduler.step(validation_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "validation_mae": validation_mae,
                    "validation_rmse": validation_rmse,
                    "learning_rate": current_lr,
                }
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.training_config.early_stopping_patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        final_validation_loss, final_validation_mae, final_validation_rmse = self._evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )
        checkpoint_path = artifact_path / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(self.model_config),
                "training_config": asdict(self.training_config),
                "parameter_count": count_trainable_parameters(model),
            },
            checkpoint_path,
        )

        metadata = {
            "model_version": version,
            "created_at": datetime.now().isoformat(),
            "model_type": "lstm",
            "framework": "pytorch",
            "checkpoint_path": str(checkpoint_path),
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
            "parameter_count": count_trainable_parameters(model),
            "best_epoch": best_epoch,
            "metrics": {
                "train_loss": float(history[best_epoch - 1]["train_loss"]) if best_epoch else float("nan"),
                "validation_loss": final_validation_loss,
                "validation_mae": final_validation_mae,
                "validation_rmse": final_validation_rmse,
            },
            "extra_metadata": extra_metadata or {},
        }
        self._write_json(artifact_path / "training_history.json", {"history": history})
        self._write_json(artifact_path / "model_config.json", metadata)
        self._write_json(artifact_path / "metrics.json", metadata["metrics"])

        registry_entry = None
        if registry is not None:
            registry_entry = registry.register(
                artifact_path=artifact_path,
                model_version=version,
                model_type="lstm",
                framework="pytorch",
                metrics=metadata["metrics"],
                config={
                    "model_config": metadata["model_config"],
                    "training_config": metadata["training_config"],
                },
                tags=["trained-model"],
                activate=True,
            )

        return TrainingResult(
            model_version=version,
            artifact_path=str(artifact_path),
            checkpoint_path=str(checkpoint_path),
            best_epoch=best_epoch,
            train_loss=float(history[best_epoch - 1]["train_loss"]) if best_epoch else float("nan"),
            validation_loss=final_validation_loss,
            validation_mae=final_validation_mae,
            validation_rmse=final_validation_rmse,
            parameter_count=count_trainable_parameters(model),
            registry_entry=registry_entry,
        )

    def _train_one_epoch(
        self,
        model: TrafficLSTM,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> float:
        model.train()
        total_loss = 0.0
        total_samples = 0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_config.gradient_clip_norm)
            optimizer.step()
            batch_size = len(features)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
        return total_loss / max(total_samples, 1)

    def _evaluate(
        self,
        model: TrafficLSTM,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[float, float, float]:
        model.eval()
        total_loss = 0.0
        total_absolute_error = 0.0
        total_squared_error = 0.0
        total_values = 0
        total_samples = 0
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(device)
                targets = targets.to(device)
                predictions = model(features)
                loss = criterion(predictions, targets)
                errors = predictions - targets
                batch_size = len(features)
                total_loss += float(loss.item()) * batch_size
                total_absolute_error += float(torch.abs(errors).sum().item())
                total_squared_error += float(torch.square(errors).sum().item())
                total_values += int(targets.numel())
                total_samples += batch_size
        mse = total_squared_error / max(total_values, 1)
        return (
            total_loss / max(total_samples, 1),
            total_absolute_error / max(total_values, 1),
            math.sqrt(mse),
        )

    def _build_loader(self, features: np.ndarray, targets: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.as_tensor(features, dtype=torch.float32),
            torch.as_tensor(targets, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=self.training_config.batch_size, shuffle=shuffle)

    def _apply_learning_rate_warmup(self, optimizer: torch.optim.Optimizer, epoch: int) -> None:
        warmup_epochs = self.training_config.warmup_epochs
        if warmup_epochs <= 0 or epoch > warmup_epochs:
            return

        start_factor = self.training_config.warmup_start_factor
        if not 0.0 < start_factor <= 1.0:
            raise ValueError("warmup_start_factor must be in (0.0, 1.0]")

        progress = epoch / warmup_epochs
        factor = start_factor + (1.0 - start_factor) * progress
        warmup_lr = self.training_config.learning_rate * factor
        for group in optimizer.param_groups:
            group["lr"] = warmup_lr

    def _validate_arrays(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
    ) -> None:
        for name, value in {
            "X_train": X_train,
            "y_train": y_train,
            "X_validation": X_validation,
            "y_validation": y_validation,
        }.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a numpy array")
            if len(value) == 0:
                raise ValueError(f"{name} must not be empty")

        expected_x_shape = 3
        expected_y_shape = 3
        if X_train.ndim != expected_x_shape or X_validation.ndim != expected_x_shape:
            raise ValueError("X arrays must have shape (samples, lookback, features)")
        if y_train.ndim != expected_y_shape or y_validation.ndim != expected_y_shape:
            raise ValueError("y arrays must have shape (samples, horizon, 1)")
        if X_train.shape[-1] != self.model_config.input_size:
            raise ValueError("X_train feature count does not match model_config.input_size")
        if X_validation.shape[-1] != self.model_config.input_size:
            raise ValueError("X_validation feature count does not match model_config.input_size")
        if y_train.shape[1] != self.model_config.prediction_horizon:
            raise ValueError("y_train horizon does not match model_config.prediction_horizon")
        if y_validation.shape[1] != self.model_config.prediction_horizon:
            raise ValueError("y_validation horizon does not match model_config.prediction_horizon")

    @staticmethod
    def _set_seeds(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
