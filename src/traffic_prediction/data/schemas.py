from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from typing import Literal


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


class SerializableDataclassMixin:
    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True)
class TrafficRecord(SerializableDataclassMixin):
    road_id: str
    current_speed: float
    free_flow_speed: float
    confidence: float
    timestamp: datetime
    speed_ratio: float


@dataclass(frozen=True)
class LiveTrafficRecord(SerializableDataclassMixin):
    road_id: str
    current_speed: float
    confidence: float
    timestamp: datetime
    freshness_indicator: timedelta | None = None


@dataclass(frozen=True)
class SequenceMetadata(SerializableDataclassMixin):
    road_id: str
    input_start: datetime
    input_end: datetime
    target_start: datetime
    target_end: datetime


@dataclass(frozen=True)
class FeatureManifest(SerializableDataclassMixin):
    feature_columns: list[str]
    target_column: str
    lookback: int
    horizon: int
    feature_version: str = "features-v1"
    scaler_version: str = "scalers-v1"


@dataclass(frozen=True)
class ValidationReport(SerializableDataclassMixin):
    row_count: int
    road_count: int
    date_range_start: datetime | None
    date_range_end: datetime | None
    missing_values: dict[str, int]
    invalid_speed_count: int
    invalid_confidence_count: int
    duplicate_count: int
    is_chronological_per_road: bool


@dataclass(frozen=True)
class SplitStatistics(SerializableDataclassMixin):
    train_rows: int
    validation_rows: int
    test_rows: int
    train_start: datetime | None
    train_end: datetime | None
    validation_start: datetime | None
    validation_end: datetime | None
    test_start: datetime | None
    test_end: datetime | None
    train_road_count: int
    validation_road_count: int
    test_road_count: int


@dataclass(frozen=True)
class SplitReport(SerializableDataclassMixin):
    train_rows: int
    validation_rows: int
    test_rows: int
    train_range: tuple[datetime, datetime]
    validation_range: tuple[datetime, datetime]
    test_range: tuple[datetime, datetime]
    road_count_train: int
    road_count_validation: int
    road_count_test: int


@dataclass(frozen=True)
class DataQualityReport(SerializableDataclassMixin):
    timestamp: datetime
    completeness: float
    average_confidence: float
    stale_roads: list[str]
    missing_roads: list[str]
    delayed_roads: list[str]
    low_confidence_roads: list[str]
    outlier_roads: list[str]
    api_uptime: float
    fallback_recommendation: str
    quality_issues: dict[str, int]
    status: Literal["healthy", "degraded", "unavailable"]


@dataclass(frozen=True)
class DatasetBundle(SerializableDataclassMixin):
    X_train_shape: tuple[int, ...]
    y_train_shape: tuple[int, ...]
    X_validation_shape: tuple[int, ...]
    y_validation_shape: tuple[int, ...]
    X_test_shape: tuple[int, ...]
    y_test_shape: tuple[int, ...]
    feature_count: int
    train_samples: int
    validation_samples: int
    test_samples: int


@dataclass(frozen=True)
class PipelineSummary(SerializableDataclassMixin):
    traffic_validation: ValidationReport
    road_validation: ValidationReport
    split_report: SplitReport
    feature_manifest: FeatureManifest
    dataset_bundle: DatasetBundle
    leakage_status: Literal["passed"]


@dataclass(frozen=True)
class TrainingMetrics(SerializableDataclassMixin):
    train_loss: float
    validation_loss: float
    validation_mae: float
    validation_rmse: float
    best_epoch: int
    parameter_count: int


@dataclass(frozen=True)
class ModelArtifactReference(SerializableDataclassMixin):
    model_version: str
    artifact_path: str
    checkpoint_path: str | None
    framework: str
    created_at: datetime


@dataclass(frozen=True)
class TrainingResultSummary(SerializableDataclassMixin):
    model: ModelArtifactReference
    metrics: TrainingMetrics
    registry_active: bool = False
    extra_metadata: dict[str, Any] | None = None
