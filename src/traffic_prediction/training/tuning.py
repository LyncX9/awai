from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from traffic_prediction.models.lstm import LSTMModelConfig
from traffic_prediction.training.cross_validation import (
    CrossValidationSummary,
    CrossValidationTrainer,
    ReducedTimeSeriesCVConfig,
    ReducedTimeSeriesSplit,
)
from traffic_prediction.training.trainer import TrainingLoopConfig


@dataclass(frozen=True)
class HyperparameterCandidate:
    """One bounded hyperparameter candidate for small-dataset LSTM tuning."""

    name: str
    hidden_sizes: tuple[int, int]
    dense_units: int
    dropout: float
    recurrent_dropout: float
    learning_rate: float
    batch_size: int
    weight_decay: float
    lookback: int

    def to_model_config(self, input_size: int, prediction_horizon: int) -> LSTMModelConfig:
        return LSTMModelConfig(
            input_size=input_size,
            prediction_horizon=prediction_horizon,
            hidden_sizes=self.hidden_sizes,
            dense_units=self.dense_units,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
        )

    def to_training_config(self, base: TrainingLoopConfig | None = None) -> TrainingLoopConfig:
        base_config = base or TrainingLoopConfig()
        return TrainingLoopConfig(
            max_epochs=base_config.max_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            early_stopping_patience=base_config.early_stopping_patience,
            lr_plateau_patience=base_config.lr_plateau_patience,
            lr_plateau_factor=base_config.lr_plateau_factor,
            warmup_epochs=base_config.warmup_epochs,
            warmup_start_factor=base_config.warmup_start_factor,
            gradient_clip_norm=base_config.gradient_clip_norm,
            weight_decay=self.weight_decay,
            random_seed=base_config.random_seed,
            device=base_config.device,
        )


@dataclass(frozen=True)
class HyperparameterSearchSpace:
    """Conservative default search space for the limited traffic dataset."""

    hidden_sizes: tuple[tuple[int, int], ...] = ((64, 32), (32, 16))
    dense_units: tuple[int, ...] = (16,)
    dropout: tuple[float, ...] = (0.2, 0.3)
    recurrent_dropout: tuple[float, ...] = (0.1, 0.2)
    learning_rate: tuple[float, ...] = (0.001, 0.0005)
    batch_size: tuple[int, ...] = (32, 64)
    weight_decay: tuple[float, ...] = (0.0001,)
    lookback: tuple[int, ...] = (12,)

    def generate_candidates(self, max_trials: int | None = None) -> list[HyperparameterCandidate]:
        candidates: list[HyperparameterCandidate] = []
        grid = itertools.product(
            self.hidden_sizes,
            self.dense_units,
            self.dropout,
            self.recurrent_dropout,
            self.learning_rate,
            self.batch_size,
            self.weight_decay,
            self.lookback,
        )
        for index, values in enumerate(grid, start=1):
            hidden_sizes, dense_units, dropout, recurrent_dropout, learning_rate, batch_size, weight_decay, lookback = values
            candidates.append(
                HyperparameterCandidate(
                    name=f"candidate-{index:03d}",
                    hidden_sizes=hidden_sizes,
                    dense_units=dense_units,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    lookback=lookback,
                )
            )
            if max_trials is not None and len(candidates) >= max_trials:
                break
        return candidates


@dataclass(frozen=True)
class TuningTrialResult:
    candidate: HyperparameterCandidate
    metrics: dict[str, float]
    rank_metric: str
    score: float
    artifact_path: str | None = None


@dataclass(frozen=True)
class HyperparameterTuningResult:
    created_at: str
    rank_metric: str
    minimize: bool
    best_candidate: HyperparameterCandidate
    best_score: float
    trials: list[TuningTrialResult]


ObjectiveFn = Callable[[HyperparameterCandidate], tuple[dict[str, float], str | None]]


class HyperparameterTuner:
    """Runs bounded hyperparameter search and saves selected configuration."""

    def __init__(
        self,
        search_space: HyperparameterSearchSpace | None = None,
        rank_metric: str = "mean_validation_rmse",
        minimize: bool = True,
        max_trials: int | None = None,
    ) -> None:
        self.search_space = search_space or HyperparameterSearchSpace()
        self.rank_metric = rank_metric
        self.minimize = minimize
        self.max_trials = max_trials

    def run(
        self,
        artifact_root: str | Path,
        objective_fn: ObjectiveFn,
        candidates: Sequence[HyperparameterCandidate] | None = None,
    ) -> HyperparameterTuningResult:
        trial_candidates = list(candidates) if candidates is not None else self.search_space.generate_candidates(self.max_trials)
        if not trial_candidates:
            raise ValueError("No hyperparameter candidates were generated")

        artifact_root = Path(artifact_root)
        artifact_root.mkdir(parents=True, exist_ok=True)
        trials: list[TuningTrialResult] = []
        for candidate in trial_candidates:
            metrics, artifact_path = objective_fn(candidate)
            if self.rank_metric not in metrics:
                raise ValueError(f"Objective metrics missing rank metric: {self.rank_metric}")
            trials.append(
                TuningTrialResult(
                    candidate=candidate,
                    metrics=metrics,
                    rank_metric=self.rank_metric,
                    score=float(metrics[self.rank_metric]),
                    artifact_path=artifact_path,
                )
            )

        best = self._select_best(trials)
        result = HyperparameterTuningResult(
            created_at=datetime.now().isoformat(),
            rank_metric=self.rank_metric,
            minimize=self.minimize,
            best_candidate=best.candidate,
            best_score=best.score,
            trials=trials,
        )
        self._write_result_files(artifact_root, result)
        return result

    def run_lstm_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        artifact_root: str | Path,
        input_size: int,
        prediction_horizon: int,
        base_training_config: TrainingLoopConfig,
        cv_config: ReducedTimeSeriesCVConfig,
        candidates: Sequence[HyperparameterCandidate] | None = None,
    ) -> HyperparameterTuningResult:
        artifact_root = Path(artifact_root)

        def objective(candidate: HyperparameterCandidate) -> tuple[dict[str, float], str | None]:
            if X.shape[1] != candidate.lookback:
                raise ValueError(
                    "Candidate lookback does not match precomputed sequence array. "
                    "Regenerate sequences per lookback before using this candidate."
                )
            candidate_root = artifact_root / candidate.name
            cv_trainer = CrossValidationTrainer(
                model_config_factory=lambda: candidate.to_model_config(input_size, prediction_horizon),
                training_config=candidate.to_training_config(base_training_config),
                splitter=ReducedTimeSeriesSplit(cv_config),
            )
            summary = cv_trainer.run(
                X=X,
                y=y,
                artifact_root=candidate_root,
                registry=None,
                model_version_prefix=candidate.name,
            )
            metrics = self._metrics_from_cv_summary(summary)
            return metrics, str(candidate_root)

        return self.run(artifact_root=artifact_root, objective_fn=objective, candidates=candidates)

    def _select_best(self, trials: list[TuningTrialResult]) -> TuningTrialResult:
        key = lambda item: item.score
        return min(trials, key=key) if self.minimize else max(trials, key=key)

    @staticmethod
    def _metrics_from_cv_summary(summary: CrossValidationSummary) -> dict[str, float]:
        return {
            "mean_validation_loss": summary.mean_validation_loss,
            "mean_validation_mae": summary.mean_validation_mae,
            "mean_validation_rmse": summary.mean_validation_rmse,
        }

    def _write_result_files(self, artifact_root: Path, result: HyperparameterTuningResult) -> None:
        payload = asdict(result)
        (artifact_root / "tuning_results.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        best_payload = {
            "rank_metric": result.rank_metric,
            "minimize": result.minimize,
            "best_score": result.best_score,
            "best_candidate": asdict(result.best_candidate),
        }
        (artifact_root / "best_config.json").write_text(
            json.dumps(best_payload, indent=2, default=str),
            encoding="utf-8",
        )

