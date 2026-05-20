from __future__ import annotations

from datetime import datetime, timedelta

from traffic_prediction.api.schemas import (
    ErrorResponse,
    FeatureManifestResponse,
    MetricsResponse,
    PredictionBatchResponse,
    PredictionResponse,
    SchedulerJobStatusResponse,
    TrainingResultResponse,
    ValidationReportResponse,
)
from traffic_prediction.data.schemas import (
    FeatureManifest,
    LiveTrafficRecord,
    ModelArtifactReference,
    TrafficRecord,
    TrainingMetrics,
    TrainingResultSummary,
    ValidationReport,
)


def test_runtime_dataclasses_export_json_ready_dicts() -> None:
    timestamp = datetime(2026, 5, 19, 7, 30)
    traffic = TrafficRecord("R1", 24.5, 50.0, 0.92, timestamp, 0.49)
    live = LiveTrafficRecord("R1", 24.5, 0.92, timestamp, timedelta(seconds=45))

    assert traffic.to_dict()["timestamp"] == "2026-05-19T07:30:00"
    assert live.to_dict()["freshness_indicator"] == 45.0


def test_validation_and_manifest_pydantic_responses_from_dataclasses() -> None:
    report = ValidationReport(
        row_count=10,
        road_count=2,
        date_range_start=datetime(2026, 5, 1),
        date_range_end=datetime(2026, 5, 2),
        missing_values={"current_speed": 0},
        invalid_speed_count=0,
        invalid_confidence_count=0,
        duplicate_count=0,
        is_chronological_per_road=True,
    )
    manifest = FeatureManifest(["current_speed", "hour_of_day"], "current_speed", 12, 4)

    report_response = ValidationReportResponse.from_report(report)
    manifest_response = FeatureManifestResponse.from_manifest(manifest)

    assert report_response.row_count == 10
    assert report_response.is_chronological_per_road is True
    assert manifest_response.feature_columns == ["current_speed", "hour_of_day"]
    assert manifest_response.horizon == 4


def test_training_result_schema_is_serializable() -> None:
    summary = TrainingResultSummary(
        model=ModelArtifactReference(
            model_version="lstm-test",
            artifact_path="artifacts/models/lstm-test",
            checkpoint_path="artifacts/models/lstm-test/model.pt",
            framework="pytorch",
            created_at=datetime(2026, 5, 19, 8, 0),
        ),
        metrics=TrainingMetrics(
            train_loss=0.1,
            validation_loss=0.2,
            validation_mae=3.0,
            validation_rmse=4.0,
            best_epoch=7,
            parameter_count=1234,
        ),
        registry_active=True,
        extra_metadata={"dataset_version": "candidate"},
    )

    response = TrainingResultResponse.from_summary(summary)
    payload = response.model_dump(mode="json")

    assert payload["model"]["model_version"] == "lstm-test"
    assert payload["metrics"]["validation_rmse"] == 4.0
    assert payload["registry_active"] is True
    assert payload["model"]["created_at"] == "2026-05-19T08:00:00"


def test_prediction_batch_and_error_response_shapes() -> None:
    prediction = PredictionResponse(
        road_id="R1",
        horizon_minutes=60,
        predicted_speed=25.0,
        congestion_level="moderate",
        uncertainty_lower=20.0,
        uncertainty_upper=30.0,
        confidence_score=0.8,
        model_version="lstm-test",
        prediction_method="historical_average_fallback",
        degraded=True,
        data_quality={"status": "degraded"},
    )
    error = ErrorResponse(error_code="unknown_road", message="Unknown road", details={"road_id": "R9"})
    batch = PredictionBatchResponse(
        predictions=[prediction],
        requested_count=2,
        successful_count=1,
        failed_count=1,
        failures=[error],
    )

    assert batch.failed_count == 1
    assert batch.predictions[0].road_id == "R1"
    assert batch.failures[0].error_code == "unknown_road"
    assert error.details["road_id"] == "R9"


def test_metrics_response_captures_runtime_scheduler_contract() -> None:
    job = SchedulerJobStatusResponse(
        name="buffer_persistence",
        enabled=True,
        interval_seconds=900,
        run_count=2,
        failure_count=0,
        last_status="completed",
        next_run_at=datetime(2026, 5, 19, 8, 15),
    )
    metrics = MetricsResponse(
        uptime_seconds=12.5,
        model_loaded=True,
        model_version="lstm-test",
        roads_total=50,
        buffer_available=True,
        buffer_fresh_roads=48,
        buffer_stale_roads=2,
        buffer_average_fill_rate=0.95,
        prediction_cache_size=4,
        scheduler_running=False,
        scheduler_job_count=1,
        scheduler_jobs={"buffer_persistence": job},
        data_quality_status="healthy",
        tomtom_configured=True,
    )
    payload = metrics.model_dump(mode="json")

    assert payload["roads_total"] == 50
    assert payload["scheduler_jobs"]["buffer_persistence"]["run_count"] == 2
    assert payload["scheduler_jobs"]["buffer_persistence"]["next_run_at"] == "2026-05-19T08:15:00"
