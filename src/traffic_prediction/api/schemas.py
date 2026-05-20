from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from traffic_prediction.data.schemas import FeatureManifest, TrainingResultSummary, ValidationReport


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = False
    model_version: str | None = None
    scheduler_running: bool = False
    uptime_seconds: float = 0.0


class ReadinessResponse(BaseModel):
    ready: bool
    model_loaded: bool
    roads_loaded: bool
    buffer_available: bool
    tomtom_configured: bool = False
    scheduler_registered: bool = False
    scheduler_running: bool = False
    resources: dict[str, dict[str, str | bool]] = Field(default_factory=dict)
    details: dict[str, str] = Field(default_factory=dict)


class DataQualityResponse(BaseModel):
    timestamp: datetime
    status: str
    completeness: float
    average_confidence: float
    stale_roads: list[str]
    missing_roads: list[str]
    delayed_roads: list[str]
    low_confidence_roads: list[str]
    outlier_roads: list[str]
    api_uptime: float = 1.0
    fallback_recommendation: str = "use_live_lstm"
    quality_issues: dict[str, int] = Field(default_factory=dict)
    buffer_available: bool
    buffer_stats: dict
    seeded_from_history: bool = False
    restore_error: str | None = None


class SchedulerJobStatusResponse(BaseModel):
    name: str
    enabled: bool
    interval_seconds: int
    run_count: int = 0
    failure_count: int = 0
    last_status: str | None = None
    next_run_at: datetime | None = None
    last_started_at: datetime | None = None
    last_finished_at: datetime | None = None
    last_error: str | None = None


class MetricsResponse(BaseModel):
    uptime_seconds: float
    model_loaded: bool
    model_version: str | None = None
    roads_total: int = 0
    buffer_available: bool = False
    buffer_fresh_roads: int = 0
    buffer_stale_roads: int = 0
    buffer_average_fill_rate: float = 0.0
    prediction_cache_size: int = 0
    scheduler_running: bool = False
    scheduler_job_count: int = 0
    scheduler_jobs: dict[str, SchedulerJobStatusResponse] = Field(default_factory=dict)
    data_quality_status: str | None = None
    tomtom_configured: bool = False


class RoadResponse(BaseModel):
    road_id: str
    road_name: str | None = None
    city: str | None = None
    road_weight: float | None = None
    start_lat: float | None = None
    start_lon: float | None = None
    end_lat: float | None = None
    end_lon: float | None = None
    mid_lat: float | None = None
    mid_lon: float | None = None



class ModelVersionResponse(BaseModel):
    model_version: str | None
    model_loaded: bool
    artifact_path: str | None = None


class ModelReloadResponse(BaseModel):
    model_version: str | None
    model_loaded: bool
    artifact_path: str | None = None
    cache_invalidated: bool
    reloaded_at: datetime


class JobTriggerResponse(BaseModel):
    job_name: str
    status: str
    triggered_at: datetime
    accepted_count: int = 0
    rejected_count: int = 0
    fetch_error_count: int = 0
    cache_invalidated: bool = False
    buffer_available: bool = False
    response_time_seconds: float = 0.0
    errors: dict[str, str] = Field(default_factory=dict)
    buffer_stats: dict | None = None
    data_quality: dict | None = None
    event_log_path: str | None = None
    buffer_persisted: bool = False
    buffer_persist_path: str | None = None


class ManualIngestRecord(BaseModel):
    road_id: str
    current_speed: float = Field(ge=0.0, le=120.0)
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime


class ManualIngestRequest(BaseModel):
    records: list[ManualIngestRecord] = Field(min_length=1, max_length=500)


class ManualIngestResponse(BaseModel):
    accepted_count: int
    rejected_count: int
    cache_invalidated: bool
    buffer_available: bool
    ingested_at: datetime
    buffer_stats: dict[str, float | int | dict[str, float]]


class PredictionRequest(BaseModel):
    road_id: str
    horizon_minutes: int = Field(default=60, ge=15, le=60)
    requested_at: datetime | None = None


class PredictionResponse(BaseModel):
    road_id: str
    horizon_minutes: int
    predicted_speed: float
    congestion_level: str
    uncertainty_lower: float
    uncertainty_upper: float
    confidence_score: float
    model_version: str | None
    prediction_method: str
    degraded: bool
    data_quality: dict[str, str | float | int | bool | dict]
    metadata: dict[str, str | float | int | bool | dict | None] = Field(default_factory=dict)


class PredictionBatchRequest(BaseModel):
    predictions: list[PredictionRequest] = Field(min_length=1, max_length=100)


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class PredictionBatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    requested_count: int
    successful_count: int
    failed_count: int = 0
    failures: list[ErrorResponse] = Field(default_factory=list)


class ValidationReportResponse(BaseModel):
    row_count: int
    road_count: int
    date_range_start: datetime | None
    date_range_end: datetime | None
    missing_values: dict[str, int]
    invalid_speed_count: int
    invalid_confidence_count: int
    duplicate_count: int
    is_chronological_per_road: bool

    @classmethod
    def from_report(cls, report: ValidationReport) -> "ValidationReportResponse":
        return cls(**report.to_dict())


class FeatureManifestResponse(BaseModel):
    feature_columns: list[str]
    target_column: str
    lookback: int
    horizon: int
    feature_version: str
    scaler_version: str

    @classmethod
    def from_manifest(cls, manifest: FeatureManifest) -> "FeatureManifestResponse":
        return cls(**manifest.to_dict())


class TrainingMetricsResponse(BaseModel):
    train_loss: float
    validation_loss: float
    validation_mae: float
    validation_rmse: float
    best_epoch: int
    parameter_count: int


class ModelArtifactResponse(BaseModel):
    model_version: str
    artifact_path: str
    checkpoint_path: str | None
    framework: str
    created_at: datetime


class TrainingResultResponse(BaseModel):
    model: ModelArtifactResponse
    metrics: TrainingMetricsResponse
    registry_active: bool = False
    extra_metadata: dict[str, Any] | None = None

    @classmethod
    def from_summary(cls, summary: TrainingResultSummary) -> "TrainingResultResponse":
        return cls(**summary.to_dict())
