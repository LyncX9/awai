from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from traffic_prediction.models.lstm import LSTMModelConfig
from traffic_prediction.models.registry import ModelRegistry
from traffic_prediction.training.trainer import LSTMTrainer, TrainingLoopConfig, TrainingResult


@dataclass(frozen=True)
class ReducedTimeSeriesCVConfig:
    """Reduced expanding-window time-series cross-validation settings."""

    n_splits: int = 3
    validation_size: int | None = None
    min_train_size: int | None = None
    gap: int = 0


@dataclass(frozen=True)
class TimeSeriesFold:
    fold_index: int
    train_indices: np.ndarray
    validation_indices: np.ndarray

    @property
    def train_size(self) -> int:
        return int(len(self.train_indices))

    @property
    def validation_size(self) -> int:
        return int(len(self.validation_indices))


@dataclass(frozen=True)
class FoldTrainingSummary:
    fold_index: int
    train_size: int
    validation_size: int
    model_version: str
    artifact_path: str
    best_epoch: int
    train_loss: float
    validation_loss: float
    validation_mae: float
    validation_rmse: float


@dataclass(frozen=True)
class CrossValidationSummary:
    created_at: str
    n_splits: int
    folds: list[FoldTrainingSummary]
    mean_validation_loss: float
    mean_validation_mae: float
    mean_validation_rmse: float


@dataclass(frozen=True)
class PreparedFeatureFold:
    fold: TimeSeriesFold
    train: pd.DataFrame
    validation: pd.DataFrame
    feature_columns: list[str]


class ReducedTimeSeriesSplit:
    """
    Expanding-window chronological splitter.

    The caller should pass only the training/validation candidate data. Final test
    data must stay outside this splitter so it remains untouched for final
    evaluation.
    """

    def __init__(self, config: ReducedTimeSeriesCVConfig | None = None) -> None:
        self.config = config or ReducedTimeSeriesCVConfig()
        if self.config.n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        if self.config.gap < 0:
            raise ValueError("gap must be non-negative")

    def split_arrays(self, X: np.ndarray, y: np.ndarray | None = None) -> list[TimeSeriesFold]:
        if X.ndim == 0:
            raise ValueError("X must have at least one dimension")
        sample_count = int(len(X))
        if y is not None and len(y) != sample_count:
            raise ValueError("X and y must contain the same number of samples")
        return self._build_folds(sample_count)

    def split_dataframe(
        self,
        df: pd.DataFrame,
        timestamp_column: str = "collected_at_wib",
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, TimeSeriesFold]]:
        if timestamp_column not in df.columns:
            raise ValueError(f"timestamp column not found: {timestamp_column}")

        ordered = df.sort_values(timestamp_column).reset_index(drop=True)
        ordered["_cv_position"] = np.arange(len(ordered), dtype=int)
        unique_timestamps = ordered[timestamp_column].drop_duplicates().reset_index(drop=True)
        timestamp_folds = self._build_folds(len(unique_timestamps))
        output: list[tuple[pd.DataFrame, pd.DataFrame, TimeSeriesFold]] = []

        for fold in timestamp_folds:
            train_timestamps = set(unique_timestamps.iloc[fold.train_indices].tolist())
            validation_timestamps = set(unique_timestamps.iloc[fold.validation_indices].tolist())
            train = ordered[ordered[timestamp_column].isin(train_timestamps)].copy()
            validation = ordered[ordered[timestamp_column].isin(validation_timestamps)].copy()

            train_indices = train["_cv_position"].to_numpy(dtype=int)
            validation_indices = validation["_cv_position"].to_numpy(dtype=int)
            data_fold = TimeSeriesFold(
                fold_index=fold.fold_index,
                train_indices=train_indices,
                validation_indices=validation_indices,
            )
            self.validate_fold(data_fold)
            output.append(
                (
                    train.drop(columns=["_cv_position"]).reset_index(drop=True),
                    validation.drop(columns=["_cv_position"]).reset_index(drop=True),
                    data_fold,
                )
            )
        return output

    def validate_fold(self, fold: TimeSeriesFold) -> None:
        if fold.train_size == 0:
            raise ValueError(f"fold {fold.fold_index} has empty train indices")
        if fold.validation_size == 0:
            raise ValueError(f"fold {fold.fold_index} has empty validation indices")
        if np.intersect1d(fold.train_indices, fold.validation_indices).size:
            raise ValueError(f"fold {fold.fold_index} has overlapping train and validation indices")
        if int(fold.train_indices.max()) >= int(fold.validation_indices.min()):
            raise ValueError(f"fold {fold.fold_index} is not chronological")

    def _build_folds(self, sample_count: int) -> list[TimeSeriesFold]:
        validation_size = self.config.validation_size or sample_count // (self.config.n_splits + 1)
        if validation_size <= 0:
            raise ValueError("validation_size must be positive")

        min_train_size = self.config.min_train_size or validation_size
        required = min_train_size + self.config.gap + (validation_size * self.config.n_splits)
        if sample_count < required:
            raise ValueError(
                "Not enough samples for reduced time-series CV: "
                f"samples={sample_count}, required={required}"
            )

        folds: list[TimeSeriesFold] = []
        for fold_index in range(self.config.n_splits):
            train_end = min_train_size + (fold_index * validation_size)
            validation_start = train_end + self.config.gap
            validation_end = validation_start + validation_size
            fold = TimeSeriesFold(
                fold_index=fold_index + 1,
                train_indices=np.arange(0, train_end, dtype=int),
                validation_indices=np.arange(validation_start, validation_end, dtype=int),
            )
            self.validate_fold(fold)
            folds.append(fold)
        return folds


class FoldStandardScaler:
    """Per-fold feature scaler to avoid cross-validation leakage."""

    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = df.copy()
        transformed[self.feature_columns] = self.scaler.fit_transform(transformed[self.feature_columns])
        self.is_fitted = True
        return transformed

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("FoldStandardScaler.transform called before fit_transform")
        transformed = df.copy()
        transformed[self.feature_columns] = self.scaler.transform(transformed[self.feature_columns])
        return transformed


class FoldFeaturePreparer:
    """
    Prepares dataframe folds with train-fold-only feature selection and scaling.

    This keeps feature selection and scaler fitting inside each fold so
    validation data cannot influence preprocessing parameters.
    """

    DEFAULT_EXCLUDED_COLUMNS = {
        "id",
        "road_id",
        "road_name",
        "city",
        "frc",
        "collected_at_wib",
    }

    def __init__(self, splitter: ReducedTimeSeriesSplit | None = None) -> None:
        self.splitter = splitter or ReducedTimeSeriesSplit()

    def prepare(
        self,
        df: pd.DataFrame,
        timestamp_column: str = "collected_at_wib",
        excluded_columns: set[str] | None = None,
    ) -> list[PreparedFeatureFold]:
        prepared: list[PreparedFeatureFold] = []
        for train, validation, fold in self.splitter.split_dataframe(df, timestamp_column=timestamp_column):
            feature_columns = self.select_numeric_feature_columns(train, excluded_columns=excluded_columns)
            scaler = FoldStandardScaler(feature_columns)
            train_scaled = scaler.fit_transform(train)
            validation_scaled = scaler.transform(validation)
            prepared.append(
                PreparedFeatureFold(
                    fold=fold,
                    train=train_scaled,
                    validation=validation_scaled,
                    feature_columns=feature_columns,
                )
            )
        return prepared

    def select_numeric_feature_columns(
        self,
        train: pd.DataFrame,
        excluded_columns: set[str] | None = None,
    ) -> list[str]:
        excluded = excluded_columns or self.DEFAULT_EXCLUDED_COLUMNS
        return [
            column
            for column in train.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(train[column])
        ]


class CrossValidationTrainer:
    """Runs LSTM training over reduced chronological folds."""

    def __init__(
        self,
        model_config_factory: Callable[[], LSTMModelConfig],
        training_config: TrainingLoopConfig,
        splitter: ReducedTimeSeriesSplit | None = None,
    ) -> None:
        self.model_config_factory = model_config_factory
        self.training_config = training_config
        self.splitter = splitter or ReducedTimeSeriesSplit()

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        artifact_root: str | Path,
        registry: ModelRegistry | None = None,
        model_version_prefix: str = "cv-lstm",
    ) -> CrossValidationSummary:
        folds = self.splitter.split_arrays(X, y)
        artifact_root = Path(artifact_root)
        artifact_root.mkdir(parents=True, exist_ok=True)
        summaries: list[FoldTrainingSummary] = []

        for fold in folds:
            trainer = LSTMTrainer(
                model_config=self.model_config_factory(),
                training_config=self.training_config,
            )
            result = trainer.train(
                X_train=X[fold.train_indices],
                y_train=y[fold.train_indices],
                X_validation=X[fold.validation_indices],
                y_validation=y[fold.validation_indices],
                artifact_root=artifact_root,
                registry=registry,
                model_version=f"{model_version_prefix}-fold-{fold.fold_index}",
                extra_metadata={
                    "cv_fold_index": fold.fold_index,
                    "train_size": fold.train_size,
                    "validation_size": fold.validation_size,
                },
            )
            summaries.append(self._to_fold_summary(fold, result))

        summary = CrossValidationSummary(
            created_at=datetime.now().isoformat(),
            n_splits=len(summaries),
            folds=summaries,
            mean_validation_loss=self._mean(item.validation_loss for item in summaries),
            mean_validation_mae=self._mean(item.validation_mae for item in summaries),
            mean_validation_rmse=self._mean(item.validation_rmse for item in summaries),
        )
        self._write_summary(artifact_root / f"{model_version_prefix}_cross_validation_summary.json", summary)
        return summary

    @staticmethod
    def _to_fold_summary(fold: TimeSeriesFold, result: TrainingResult) -> FoldTrainingSummary:
        return FoldTrainingSummary(
            fold_index=fold.fold_index,
            train_size=fold.train_size,
            validation_size=fold.validation_size,
            model_version=result.model_version,
            artifact_path=result.artifact_path,
            best_epoch=result.best_epoch,
            train_loss=result.train_loss,
            validation_loss=result.validation_loss,
            validation_mae=result.validation_mae,
            validation_rmse=result.validation_rmse,
        )

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        values_list = list(values)
        return float(np.mean(values_list)) if values_list else float("nan")

    @staticmethod
    def _write_summary(path: Path, summary: CrossValidationSummary) -> None:
        payload = asdict(summary)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
