from __future__ import annotations

import asyncio
import json
import pickle
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncIterator

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from traffic_prediction.api.schemas import (
    DataQualityResponse,
    HealthResponse,
    JobTriggerResponse,
    ManualIngestRequest,
    ManualIngestResponse,
    ModelReloadResponse,
    ModelVersionResponse,
    ErrorResponse,
    MetricsResponse,
    PredictionBatchResponse,
    PredictionBatchRequest,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
    RoadResponse,
    SchedulerJobStatusResponse,
)
from traffic_prediction.artifacts import ArtifactLayout
from traffic_prediction.config.settings import AppConfig, load_config
from traffic_prediction.data.scalers import ScalerStore
from traffic_prediction.data.schemas import FeatureManifest
from traffic_prediction.data.retraining import RetrainingDatasetManager
from traffic_prediction.data.schemas import LiveTrafficRecord
from traffic_prediction.inference.congestion import classify_congestion
from traffic_prediction.inference.cache import PredictionCache
from traffic_prediction.inference.confidence import ConfidenceAdjuster
from traffic_prediction.inference.fallback import FallbackPredictor
from traffic_prediction.inference.realtime import RealtimePredictionContext, RealtimePredictionPipeline
from traffic_prediction.features.online import OnlineFeatureEngineer
from traffic_prediction.features.spatial import build_neighbor_mapping
from traffic_prediction.inference.runner import PyTorchModelRunner
from traffic_prediction.ingestion.buffer import LiveBufferManager
from traffic_prediction.ingestion.events import IngestionEventLogger
from traffic_prediction.ingestion.ingestor import TomTomIngestor
from traffic_prediction.ingestion.tomtom_client import TomTomTrafficClient
from traffic_prediction.ingestion.tomtom_mapping import TomTomMappingError, TomTomRoadMapper
from traffic_prediction.monitoring.data_quality import DataQualityMonitor
from traffic_prediction.monitoring.drift import DriftMonitor
from traffic_prediction.monitoring.runtime_logging import RuntimeEventLogger, configure_structured_logging
from traffic_prediction.models.registry import ModelRegistry, ModelRegistryEntry
from traffic_prediction.orchestration.scheduler import InProcessScheduler
from traffic_prediction.orchestration.startup import StartupReport, build_startup_report
from traffic_prediction.persistence.postgresql import PostgreSQLPersistence


class AppState:
    ALLOWED_HORIZONS = {15, 30, 45, 60}

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.db = PostgreSQLPersistence(config.api.database_url) if config.api.database_url else None
        self.started_at = time.monotonic()
        self.roads: pd.DataFrame | None = None
        self.model_version: str | None = None
        self.model_artifact_path: Path | None = None
        self.model_loaded = False
        self.scheduler_running = False
        self.buffer_available = False
        self.historical_lookup: pd.Series | None = None
        self.road_mean_speed: pd.Series | None = None
        self.latest_timestamp_by_road: pd.Series | None = None
        self.fallback_predictor = FallbackPredictor()
        self.confidence_adjuster = ConfidenceAdjuster()
        self.prediction_cache = PredictionCache(ttl_seconds=config.runtime.prediction_cache_ttl_seconds)
        self.runtime_event_logger = RuntimeEventLogger(
            config.paths.logs_dir,
            retention_days=config.runtime.log_retention_days,
        )
        self.app_logger = configure_structured_logging(
            config.paths.logs_dir,
            level=config.runtime.log_level,
            retention_days=config.runtime.log_retention_days,
        )
        self.online_feature_engineer: OnlineFeatureEngineer | None = None
        self.model_runner: PyTorchModelRunner | None = None
        self.artifact_layout = ArtifactLayout.from_paths(config.paths)
        self.live_buffer = LiveBufferManager()
        self.scheduler = InProcessScheduler(event_logger=self.runtime_event_logger)
        self.last_buffer_persisted_at: datetime | None = None
        self.buffer_restore_error: str | None = None
        self.buffer_recovery_source = "not_started"
        self.model_recovery_source = "not_started"
        self.tomtom_recovery_status = "not_checked"
        self.scheduler_recovery_status = "not_started"
        self.restart_recovery_status = "pending"
        self.restart_recovery_detail = "restart recovery has not run"
        self.restart_recovery: dict[str, str | bool | int | None] = {}
        self.live_buffer_seeded_from_history = False
        self.data_quality_monitor = DataQualityMonitor()
        self.drift_monitor = DriftMonitor()
        self.retraining_dataset_manager = RetrainingDatasetManager()
        self.startup_report: StartupReport | None = None

    def load_static_resources(self) -> None:
        self.restart_recovery_status = "running"
        if self.config.paths.roads_csv.exists():
            self.roads = pd.read_csv(self.config.paths.roads_csv)
        self.restore_live_buffer()
        if not self.live_buffer.buffers:
            self.seed_live_buffer_from_history()
        self._discover_active_model()
        self._configure_online_feature_engineer()
        self._load_historical_prediction_tables()
        self._verify_tomtom_recovery()
        self._configure_scheduler()
        self._finalize_restart_recovery()
        self._finalize_startup()
        self.runtime_event_logger.write(
            "startup",
            "startup_recovery",
            self.restart_recovery,
            status=self.restart_recovery_status,
        )

    def readiness_report(self) -> StartupReport:
        self._finalize_startup()
        if self.startup_report is None:
            raise RuntimeError("Startup report was not initialized")
        return self.startup_report

    def shutdown(self) -> None:
        self.scheduler.stop()
        if self.buffer_available:
            self.persist_live_buffer()

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        if self.roads is None or self.roads.empty:
            raise HTTPException(status_code=503, detail="Road master data is not loaded")
        if request.horizon_minutes not in self.ALLOWED_HORIZONS:
            raise HTTPException(status_code=422, detail="horizon_minutes must be one of 15, 30, 45, or 60")

        road = self._road_record(request.road_id)
        requested_at = request.requested_at or self._default_request_time(request.road_id)
        target_time = requested_at + timedelta(minutes=request.horizon_minutes)
        pipeline = RealtimePredictionPipeline(
            live_buffer=self.live_buffer,
            roads=self.roads,
            prediction_cache=self.prediction_cache,
            fallback_predictor=self.fallback_predictor,
            confidence_adjuster=self.confidence_adjuster,
            data_quality_monitor=self.data_quality_monitor,
            online_feature_engineer=self.online_feature_engineer,
            model_runner=self.model_runner,
        )
        response = pipeline.predict(
            RealtimePredictionContext(
                request=request,
                road_record=road,
                requested_at=requested_at,
                target_time=target_time,
                cache_key=self._cache_key(request.road_id, request.horizon_minutes, requested_at),
                model_version=self.model_version,
            )
        )
        response.metadata["active_model_loaded"] = self.model_loaded
        response.metadata["cache_ttl_seconds"] = self.config.runtime.prediction_cache_ttl_seconds
        if self.db is not None:
            try:
                self.db.insert_prediction(response, requested_at=requested_at)
            except Exception as e:
                self.app_logger.error("postgres_prediction_sync_failed", extra={"error": str(e)})
        self.runtime_event_logger.write(
            "prediction",
            "prediction_request",
            {
                "road_id": request.road_id,
                "horizon_minutes": request.horizon_minutes,
                "model_version": self.model_version,
                "prediction_source": response.metadata.get("prediction_source"),
                "cache_hit": response.metadata.get("cache_hit"),
                "confidence_score": response.confidence_score,
            },
        )
        return response

    def predict_batch(self, request: PredictionBatchRequest) -> PredictionBatchResponse:
        predictions: list[PredictionResponse] = []
        failures: list[ErrorResponse] = []
        for index, item in enumerate(request.predictions):
            try:
                predictions.append(self.predict(item))
            except HTTPException as exc:
                failures.append(
                    ErrorResponse(
                        error_code=self._prediction_error_code(exc.status_code),
                        message=str(exc.detail),
                        details={
                            "index": index,
                            "road_id": item.road_id,
                            "horizon_minutes": item.horizon_minutes,
                            "status_code": exc.status_code,
                        },
                    )
                )
        if not predictions and failures:
            first_failure = failures[0]
            raise HTTPException(
                status_code=int(first_failure.details.get("status_code", 400)),
                detail=first_failure.message,
            )
        return PredictionBatchResponse(
            predictions=predictions,
            requested_count=len(request.predictions),
            successful_count=len(predictions),
            failed_count=len(failures),
            failures=failures,
        )

    def invalidate_prediction_cache(self, model_version: str | None = None) -> None:
        if model_version is None:
            self.prediction_cache.invalidate()
            return
        self.prediction_cache.invalidate(prefix=f"{model_version}:")

    def reload_model(self) -> ModelReloadResponse:
        self.model_version = None
        self.model_artifact_path = None
        self.model_loaded = False
        self.model_runner = None
        self._discover_active_model()
        self._configure_online_feature_engineer()
        self.invalidate_prediction_cache()
        return ModelReloadResponse(
            model_version=self.model_version,
            model_loaded=self.model_loaded,
            artifact_path=str(self.model_artifact_path) if self.model_artifact_path else None,
            cache_invalidated=True,
            reloaded_at=datetime.now(),
        )

    def ingest_manual(self, request: ManualIngestRequest) -> ManualIngestResponse:
        if self.roads is None or self.roads.empty:
            raise HTTPException(status_code=503, detail="Road master data is not loaded")

        now = pd.Timestamp.now(tz=self.config.data.timezone).to_pydatetime()
        accepted: list[LiveTrafficRecord] = []
        for item in request.records:
            self._road_record(item.road_id)
            timestamp = self._normalize_live_timestamp(item.timestamp)
            accepted.append(
                LiveTrafficRecord(
                    road_id=item.road_id,
                    current_speed=item.current_speed,
                    confidence=item.confidence,
                    timestamp=timestamp,
                    freshness_indicator=now - timestamp,
                )
            )

        self.live_buffer.append_many(accepted)
        self.buffer_available = bool(self.live_buffer.buffers)
        if self.db is not None:
            try:
                self.db.upsert_live_records(accepted, source="manual")
            except Exception as e:
                self.app_logger.error("postgres_manual_ingest_sync_failed", extra={"error": str(e)})
        self.invalidate_prediction_cache()
        expected_roads = set(self.roads["road_id"].astype(str)) if self.roads is not None else set()
        response = ManualIngestResponse(
            accepted_count=len(accepted),
            rejected_count=0,
            cache_invalidated=True,
            buffer_available=self.buffer_available,
            ingested_at=now,
            buffer_stats=self.live_buffer.stats(expected_road_ids=expected_roads, now=now),
        )
        self.runtime_event_logger.write(
            "ingestion",
            "manual_ingest",
            {
                "accepted_count": response.accepted_count,
                "rejected_count": response.rejected_count,
                "cache_invalidated": response.cache_invalidated,
                "buffer_available": response.buffer_available,
            },
            occurred_at=now,
        )
        return response

    def trigger_job(self, job_name: str) -> JobTriggerResponse:
        normalized = job_name.replace("-", "_").lower()
        aliases = {
            "tomtom_ingestion": "tomtom_ingestion",
            "ingest_tomtom": "tomtom_ingestion",
            "buffer_persistence": "buffer_persistence",
            "persist_buffer": "buffer_persistence",
            "prediction_cache_refresh": "prediction_cache_refresh",
            "cache_refresh": "prediction_cache_refresh",
            "data_quality_summary": "data_quality_summary",
            "quality_summary": "data_quality_summary",
            "drift_check": "drift_check",
            "detect_drift": "drift_check",
            "retraining_candidate": "retraining_candidate",
            "retraining_check": "retraining_candidate",
        }
        scheduler_job_name = aliases.get(normalized)
        if scheduler_job_name is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_name: {job_name}")
        return self._run_scheduler_job(job_name=job_name, scheduler_job_name=scheduler_job_name)

    def _run_scheduler_job(self, job_name: str, scheduler_job_name: str) -> JobTriggerResponse:
        try:
            run_result = self.scheduler.trigger(scheduler_job_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job_name: {job_name}") from exc
        if run_result.status == "skipped_overlap":
            return JobTriggerResponse(
                job_name=job_name,
                status="skipped_overlap",
                triggered_at=run_result.started_at,
            )
        if run_result.error is not None:
            return JobTriggerResponse(
                job_name=job_name,
                status="failed",
                triggered_at=run_result.started_at,
                errors={"job": run_result.error},
            )
        response = run_result.result
        if isinstance(response, JobTriggerResponse):
            return response.model_copy(update={"job_name": job_name})
        raise HTTPException(status_code=500, detail=f"Job returned unsupported result: {scheduler_job_name}")

    def _trigger_tomtom_ingestion(self, job_name: str) -> JobTriggerResponse:
        if self.roads is None or self.roads.empty:
            raise HTTPException(status_code=503, detail="Road master data is not loaded")
        try:
            mapper = self._tomtom_mapper()
        except TomTomMappingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        expected_roads = set(self.roads["road_id"].astype(str))
        client = TomTomTrafficClient(
            api_key=self.config.tomtom.api_key,
            api_keys=self.config.tomtom.api_keys,
            base_url=self.config.tomtom.base_url,
            timeout_seconds=self.config.tomtom.timeout_seconds,
            max_retries=self.config.tomtom.max_retries,
            backoff_seconds=self.config.tomtom.backoff_seconds,
            key_cooldown_seconds=self.config.tomtom.key_cooldown_seconds,
        )
        ingestor = TomTomIngestor(
            client=client,
            mapper=mapper,
            buffer_manager=self.live_buffer,
            expected_road_ids=expected_roads,
            timezone=self.config.data.timezone,
            min_speed=self.config.data.min_speed,
            max_speed=self.config.data.max_speed,
            cache_invalidator=self.invalidate_prediction_cache,
            buffer_persister=self._persist_live_buffer_for_ingestion,
            event_logger=IngestionEventLogger(self.config.paths.logs_dir),
        )
        summary = ingestor.ingest_once()
        self.buffer_available = bool(self.live_buffer.buffers)
        status = "completed" if summary.rejected_count == 0 else "completed_with_errors"
        self.runtime_event_logger.write(
            "ingestion",
            "tomtom_ingestion",
            {
                "accepted_count": summary.accepted_count,
                "rejected_count": summary.rejected_count,
                "fetch_error_count": summary.fetch_error_count,
                "cache_invalidated": summary.cache_invalidated,
                "buffer_persisted": summary.buffer_persisted,
                "event_log_path": summary.event_log_path,
            },
            status=status,
            occurred_at=summary.ingested_at,
        )
        return JobTriggerResponse(
            job_name=job_name,
            status=status,
            triggered_at=summary.ingested_at,
            accepted_count=summary.accepted_count,
            rejected_count=summary.rejected_count,
            fetch_error_count=summary.fetch_error_count,
            cache_invalidated=summary.cache_invalidated,
            buffer_available=self.buffer_available,
            response_time_seconds=summary.response_time_seconds,
            errors=summary.errors,
            buffer_stats=summary.buffer_stats,
            data_quality=asdict(summary.data_quality) if summary.data_quality is not None else None,
            event_log_path=summary.event_log_path,
            buffer_persisted=summary.buffer_persisted,
            buffer_persist_path=summary.buffer_persist_path,
        )

    def persist_live_buffer(self) -> JobTriggerResponse:
        now = pd.Timestamp.now(tz=self.config.data.timezone).to_pydatetime()
        self.live_buffer.persist_to_disk(self._buffer_state_path())
        self.last_buffer_persisted_at = now
        if self.db is not None:
            try:
                records_to_sync = []
                for road_id, buffer in self.live_buffer.buffers.items():
                    records_to_sync.extend(list(buffer))
                if records_to_sync:
                    self.db.upsert_live_records(records_to_sync, source="tomtom")
            except Exception as e:
                self.app_logger.error("postgres_buffer_persist_sync_failed", extra={"error": str(e)})
        expected_roads = set(self.roads["road_id"].astype(str)) if self.roads is not None else set()
        return JobTriggerResponse(
            job_name="buffer_persistence",
            status="completed",
            triggered_at=now,
            buffer_available=bool(self.live_buffer.buffers),
            buffer_stats=self.live_buffer.stats(expected_road_ids=expected_roads, now=now),
            buffer_persisted=True,
            buffer_persist_path=str(self._buffer_state_path()),
        )

    def _persist_live_buffer_for_ingestion(self) -> Path:
        now = pd.Timestamp.now(tz=self.config.data.timezone).to_pydatetime()
        path = self._buffer_state_path()
        self.live_buffer.persist_to_disk(path)
        self.last_buffer_persisted_at = now
        if self.db is not None:
            try:
                records_to_sync = []
                for road_id, buffer in self.live_buffer.buffers.items():
                    records_to_sync.extend(list(buffer))
                if records_to_sync:
                    self.db.upsert_live_records(records_to_sync, source="tomtom")
            except Exception as e:
                self.app_logger.error("postgres_tomtom_ingest_sync_failed", extra={"error": str(e)})
        return path

    def refresh_prediction_cache_job(self) -> JobTriggerResponse:
        now = pd.Timestamp.now(tz=self.config.data.timezone).to_pydatetime()
        self.invalidate_prediction_cache()
        return JobTriggerResponse(
            job_name="prediction_cache_refresh",
            status="completed",
            triggered_at=now,
            cache_invalidated=True,
            buffer_available=self.buffer_available,
        )

    def data_quality_summary_job(self) -> JobTriggerResponse:
        response = self.data_quality()
        self.runtime_event_logger.write(
            "data_quality",
            "data_quality_summary",
            response.model_dump(mode="json"),
            status=response.status,
            occurred_at=response.timestamp,
        )
        return JobTriggerResponse(
            job_name="data_quality_summary",
            status="completed",
            triggered_at=response.timestamp,
            buffer_available=response.buffer_available,
            buffer_stats=response.buffer_stats,
            data_quality=response.model_dump(mode="json"),
        )

    def drift_check_job(self) -> JobTriggerResponse:
        now = pd.Timestamp.now(tz=self.config.data.timezone).to_pydatetime()
        if self.road_mean_speed is None:
            return JobTriggerResponse(
                job_name="drift_check",
                status="completed_with_errors",
                triggered_at=now,
                errors={"drift_check": "Historical road means are unavailable"},
            )
        latest_records = [
            records[-1]
            for road_id in sorted(self.live_buffer.buffers)
            if (records := self.live_buffer.get_latest(road_id))
        ]
        drift_report = self.drift_monitor.evaluate(
            records=latest_records,
            historical_mean_speed=self.road_mean_speed.to_dict(),
            now=now,
        )
        event_log_path = self.drift_monitor.write_log(drift_report, self.config.paths.logs_dir)
        status = "completed_with_warnings" if drift_report.status != "healthy" else "completed"
        self.runtime_event_logger.write(
            "drift",
            "drift_check",
            drift_report.to_dict(),
            status=status,
            occurred_at=now,
        )
        return JobTriggerResponse(
            job_name="drift_check",
            status=status,
            triggered_at=now,
            buffer_available=self.buffer_available,
            data_quality=drift_report.to_dict(),
            event_log_path=str(event_log_path),
        )

    def retraining_candidate_job(self) -> JobTriggerResponse:
        now = pd.Timestamp.now(tz=self.config.data.timezone).to_pydatetime()
        quality = self.data_quality()
        drift = self.drift_check_job()
        drift_count = int((drift.data_quality or {}).get("drift_road_count", 0))
        should_retrain = quality.status in {"degraded", "unavailable"} or drift_count >= 5
        live_records = [
            record
            for road_id in sorted(self.live_buffer.buffers)
            for record in self.live_buffer.get_latest(road_id)
        ]
        dataset_status = self.retraining_dataset_manager.status(
            self.config.paths.traffic_csv,
            live_records=live_records,
            roads=self.roads,
            now=now,
        )
        return JobTriggerResponse(
            job_name="retraining_candidate",
            status="candidate_found" if should_retrain else "completed",
            triggered_at=now,
            buffer_available=self.buffer_available,
            data_quality={
                "should_retrain": should_retrain,
                "quality_status": quality.status,
                "drift_road_count": drift_count,
                "retraining_dataset": dataset_status,
                "reason": "quality_or_drift_threshold_met" if should_retrain else "no_retraining_needed",
            },
        )

    def data_quality(self) -> DataQualityResponse:
        if self.roads is None or self.roads.empty:
            raise HTTPException(status_code=503, detail="Road master data is not loaded")
        now = pd.Timestamp.now(tz=self.config.data.timezone).to_pydatetime()
        expected_roads = set(self.roads["road_id"].astype(str))
        latest_records = [
            records[-1]
            for road_id in sorted(self.live_buffer.buffers)
            if (records := self.live_buffer.get_latest(road_id))
        ]
        report = self.data_quality_monitor.evaluate(latest_records, expected_road_ids=expected_roads, now=now)
        return DataQualityResponse(
            timestamp=report.timestamp,
            status=report.status,
            completeness=report.completeness,
            average_confidence=report.average_confidence,
            stale_roads=report.stale_roads,
            missing_roads=report.missing_roads,
            delayed_roads=report.delayed_roads,
            low_confidence_roads=report.low_confidence_roads,
            outlier_roads=report.outlier_roads,
            api_uptime=report.api_uptime,
            fallback_recommendation=report.fallback_recommendation,
            quality_issues=report.quality_issues,
            buffer_available=self.buffer_available,
            buffer_stats=self.live_buffer.stats(expected_road_ids=expected_roads, now=now),
            seeded_from_history=self.live_buffer_seeded_from_history,
            restore_error=self.buffer_restore_error,
        )

    def metrics(self) -> MetricsResponse:
        now = pd.Timestamp.now(tz=self.config.data.timezone).to_pydatetime()
        expected_roads = set(self.roads["road_id"].astype(str)) if self.roads is not None else set()
        buffer_stats = self.live_buffer.stats(expected_road_ids=expected_roads, now=now)
        scheduler_status = self.scheduler.status()
        try:
            quality_status = self.data_quality().status
        except HTTPException:
            quality_status = None
        return MetricsResponse(
            uptime_seconds=time.monotonic() - self.started_at,
            model_loaded=self.model_loaded,
            model_version=self.model_version,
            roads_total=len(expected_roads),
            buffer_available=self.buffer_available,
            buffer_fresh_roads=int(buffer_stats.get("fresh_roads", 0)),
            buffer_stale_roads=int(buffer_stats.get("stale_roads", 0)),
            buffer_average_fill_rate=float(buffer_stats.get("average_fill_rate", 0.0)),
            prediction_cache_size=len(self.prediction_cache._entries),
            scheduler_running=bool(scheduler_status["running"]),
            scheduler_job_count=int(scheduler_status["job_count"]),
            scheduler_jobs={
                name: SchedulerJobStatusResponse(
                    name=name,
                    enabled=bool(payload["enabled"]),
                    interval_seconds=int(payload["interval_seconds"]),
                    run_count=int(payload["run_count"]),
                    failure_count=int(payload["failure_count"]),
                    last_status=payload.get("last_status"),
                    next_run_at=self._parse_optional_datetime(payload.get("next_run_at")),
                    last_started_at=self._parse_optional_datetime(payload.get("last_started_at")),
                    last_finished_at=self._parse_optional_datetime(payload.get("last_finished_at")),
                    last_error=payload.get("last_error"),
                )
                for name, payload in scheduler_status["jobs"].items()
            },
            data_quality_status=quality_status,
            tomtom_configured=bool(self.config.tomtom.api_keys),
        )

    def _prediction_quality_report(self, now: datetime):
        if self.roads is None or self.roads.empty:
            return None
        now = self._normalize_live_timestamp(now)
        expected_roads = set(self.roads["road_id"].astype(str))
        latest_records = [
            records[-1]
            for road_id in sorted(self.live_buffer.buffers)
            if (records := self.live_buffer.get_latest(road_id))
        ]
        return self.data_quality_monitor.evaluate(latest_records, expected_road_ids=expected_roads, now=now)

    def restore_live_buffer(self) -> None:
        if self.db is not None:
            try:
                records = self.db.latest_live_records(limit=1000)
                if records:
                    self.live_buffer.append_many(records)
                    self.buffer_available = bool(self.live_buffer.buffers)
                    self.buffer_restore_error = None
                    self.buffer_recovery_source = "postgresql"
                    self.live_buffer_seeded_from_history = False
                    return
            except Exception as e:
                self.app_logger.error("postgres_buffer_restore_failed", extra={"error": str(e)})

        path = self._buffer_state_path()
        if not path.exists():
            self.buffer_recovery_source = "missing_snapshot"
            return
        try:
            self.live_buffer = LiveBufferManager.restore_from_disk(path)
            self.buffer_available = bool(self.live_buffer.buffers)
            self.buffer_restore_error = None
            self.buffer_recovery_source = "persisted_snapshot" if self.buffer_available else "empty_snapshot"
        except (EOFError, OSError, pickle.UnpicklingError, ValueError) as exc:
            self.live_buffer = LiveBufferManager()
            self.buffer_available = False
            self.buffer_restore_error = str(exc)
            self.buffer_recovery_source = "restore_failed"

    def seed_live_buffer_from_history(self) -> None:
        if not self.config.paths.traffic_csv.exists():
            return
        traffic = pd.read_csv(self.config.paths.traffic_csv)
        required = {"road_id", "current_speed", "confidence", "collected_at_wib"}
        if not required.issubset(traffic.columns):
            return
        traffic = traffic[list(required)].copy()
        traffic["collected_at_wib"] = pd.to_datetime(traffic["collected_at_wib"], errors="coerce")
        traffic = traffic.dropna(subset=["road_id", "current_speed", "confidence", "collected_at_wib"])
        traffic = traffic[
            traffic["current_speed"].between(self.config.data.min_speed, self.config.data.max_speed)
            & traffic["confidence"].between(0.0, 1.0)
        ].copy()
        if self.roads is not None and not self.roads.empty:
            valid_roads = set(self.roads["road_id"].astype(str))
            traffic = traffic[traffic["road_id"].astype(str).isin(valid_roads)].copy()
        if traffic.empty:
            return

        traffic = traffic.sort_values(["road_id", "collected_at_wib"])
        latest = traffic.groupby("road_id", group_keys=False).tail(self.live_buffer.max_timesteps)
        records: list[LiveTrafficRecord] = []
        for row in latest.itertuples(index=False):
            timestamp = self._normalize_live_timestamp(row.collected_at_wib)
            records.append(
                LiveTrafficRecord(
                    road_id=str(row.road_id),
                    current_speed=float(row.current_speed),
                    confidence=float(row.confidence),
                    timestamp=timestamp,
                )
            )
        self.live_buffer.append_many(records)
        self.buffer_available = bool(self.live_buffer.buffers)
        self.live_buffer_seeded_from_history = self.buffer_available
        if self.live_buffer_seeded_from_history:
            self.buffer_recovery_source = "history_seeded"

    def _discover_active_model(self) -> None:
        registry_path = self.config.paths.models_dir / "registry.json"
        if registry_path.exists():
            try:
                active = ModelRegistry(registry_path).resolve(self.config.runtime.active_model_version)
            except (KeyError, json.JSONDecodeError):
                active = None
            if active is not None:
                artifact = Path(active.artifact_path)
                self.model_version = active.model_version
                self.model_artifact_path = artifact
                self.model_loaded = artifact.exists()
                self._load_model_runner()
                self.model_recovery_source = "registry"
                if self.db is not None:
                    try:
                        self.db.upsert_model_entry(active)
                    except Exception as e:
                        self.app_logger.error("postgres_model_registry_sync_failed", extra={"error": str(e)})
                return
            if self.config.runtime.active_model_version:
                self.model_version = self.config.runtime.active_model_version
                self.model_artifact_path = None
                self.model_loaded = False
                self.model_recovery_source = "configured_missing"
                return

        latest = self.artifact_layout.resolve_latest_model(registry_path)
        if latest is not None:
            self.model_version = latest.name
            self.model_artifact_path = latest
            self.model_loaded = True
            self._load_model_runner()
            self.model_recovery_source = "latest_pointer"
            if self.db is not None:
                try:
                    entry = ModelRegistryEntry(
                        model_version=latest.name,
                        artifact_path=str(latest),
                        created_at=datetime.fromtimestamp(latest.stat().st_mtime).isoformat(),
                        is_active=True,
                    )
                    self.db.upsert_model_entry(entry)
                except Exception as e:
                    self.app_logger.error("postgres_model_registry_sync_failed", extra={"error": str(e)})
            return
        self.model_recovery_source = "missing"

    def _load_model_runner(self) -> None:
        self.model_runner = None
        if self.model_loaded and self.model_artifact_path is not None:
            try:
                self.model_runner = PyTorchModelRunner.load_from_artifact(self.model_artifact_path)
            except Exception as e:
                self.app_logger.error("failed_to_load_model_runner", extra={"error": str(e), "path": str(self.model_artifact_path)})
                self.model_loaded = False

    def _configure_online_feature_engineer(self) -> None:
        self.online_feature_engineer = None
        if self.roads is None or self.roads.empty:
            return
        manifest = self._load_active_feature_manifest()
        if manifest is None:
            return
        scaler_store = self._load_active_scaler_store()
        try:
            neighbor_mapping = build_neighbor_mapping(self.roads, self.config.features.spatial_neighbor_count)
        except ValueError:
            neighbor_mapping = {}
        self.online_feature_engineer = OnlineFeatureEngineer(
            manifest=manifest,
            buffer_manager=self.live_buffer,
            roads=self.roads,
            scaler_store=scaler_store,
            neighbor_mapping=neighbor_mapping,
        )

    def _load_active_feature_manifest(self) -> FeatureManifest | None:
        candidates = []
        if self.model_artifact_path is not None:
            candidates.append(self.model_artifact_path / "feature_manifest.json")
        candidates.append(self.config.paths.models_dir / "feature_manifest.json")
        for path in candidates:
            if path.exists():
                try:
                    return FeatureManifest(**json.loads(path.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, TypeError):
                    continue
        if self.model_artifact_path is None:
            return None
        config_path = self.model_artifact_path / "model_config.json"
        if not config_path.exists():
            return None
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        manifest_payload = (payload.get("extra_metadata") or {}).get("feature_manifest")
        if not manifest_payload:
            return None
        try:
            return FeatureManifest(**manifest_payload)
        except TypeError:
            return None

    def _load_active_scaler_store(self) -> ScalerStore | None:
        candidates = []
        if self.model_artifact_path is not None:
            candidates.append(self.model_artifact_path / "scaler_params.joblib")
        candidates.append(self.config.paths.models_dir / "scaler_params.joblib")
        for path in candidates:
            if path.exists():
                try:
                    return ScalerStore.load(path)
                except (OSError, ValueError, EOFError):
                    continue
        return None

    def _load_historical_prediction_tables(self) -> None:
        if not self.config.paths.traffic_csv.exists():
            return
        traffic = pd.read_csv(self.config.paths.traffic_csv)
        required = {"road_id", "collected_at_wib", "current_speed"}
        if not required.issubset(traffic.columns):
            return
        traffic["collected_at_wib"] = pd.to_datetime(traffic["collected_at_wib"], errors="coerce")
        traffic = traffic.dropna(subset=["collected_at_wib", "current_speed"]).copy()
        if "confidence" in traffic.columns:
            traffic = traffic[traffic["confidence"] >= self.config.data.min_confidence].copy()
        traffic["hour"] = traffic["collected_at_wib"].dt.hour
        traffic["day_of_week"] = traffic["collected_at_wib"].dt.dayofweek
        self.historical_lookup = traffic.groupby(["road_id", "hour", "day_of_week"])["current_speed"].mean()
        self.road_mean_speed = traffic.groupby("road_id")["current_speed"].mean()
        self.latest_timestamp_by_road = traffic.groupby("road_id")["collected_at_wib"].max()
        self.fallback_predictor = FallbackPredictor(
            historical_lookup=self.historical_lookup,
            road_mean_speed=self.road_mean_speed,
        )

    def _road_record(self, road_id: str) -> dict:
        matches = self.roads[self.roads["road_id"].astype(str) == str(road_id)] if self.roads is not None else pd.DataFrame()
        if matches.empty:
            raise HTTPException(status_code=404, detail=f"Unknown road_id: {road_id}")
        return matches.iloc[0].to_dict()

    def _default_request_time(self, road_id: str) -> datetime:
        if self.latest_timestamp_by_road is None or road_id not in self.latest_timestamp_by_road.index:
            return datetime.now()
        return pd.Timestamp(self.latest_timestamp_by_road.loc[road_id]).to_pydatetime()

    def _cache_key(self, road_id: str, horizon_minutes: int, requested_at: datetime) -> str:
        bucket = self._timestamp_bucket(requested_at)
        model_version = self.model_version or "unversioned"
        return self.prediction_cache.make_key(model_version, road_id, horizon_minutes, bucket)

    @staticmethod
    def _prediction_error_code(status_code: int) -> str:
        return {
            404: "unknown_road",
            422: "invalid_prediction_request",
            503: "prediction_unavailable",
        }.get(status_code, "prediction_failed")

    def _timestamp_bucket(self, value: datetime) -> datetime:
        timestamp = pd.Timestamp(value)
        return timestamp.floor(self.config.data.frequency).to_pydatetime()

    def _normalize_live_timestamp(self, value: datetime) -> datetime:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(self.config.data.timezone)
        else:
            timestamp = timestamp.tz_convert(self.config.data.timezone)
        return timestamp.to_pydatetime()

    @staticmethod
    def _parse_optional_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value)

    def _tomtom_mapper(self) -> TomTomRoadMapper:
        if self.config.paths.tomtom_mapping_csv.exists():
            return TomTomRoadMapper.from_csv(
                self.config.paths.tomtom_mapping_csv,
                timezone=self.config.data.timezone,
            )
        if self.roads is None or self.roads.empty:
            raise TomTomMappingError("Road master data is not loaded")
        return TomTomRoadMapper.from_roads_master(self.roads, timezone=self.config.data.timezone)

    def _configure_scheduler(self) -> None:
        scheduler_enabled = bool(self.config.runtime.scheduler_enabled)
        self.scheduler.add_interval_job(
            name="tomtom_ingestion",
            interval_seconds=self.config.runtime.ingestion_interval_seconds,
            action=lambda: self._trigger_tomtom_ingestion(job_name="tomtom_ingestion"),
            enabled=scheduler_enabled and bool(self.config.tomtom.api_keys),
        )
        self.scheduler.add_interval_job(
            name="buffer_persistence",
            interval_seconds=self.config.runtime.ingestion_interval_seconds,
            action=self.persist_live_buffer,
            enabled=scheduler_enabled,
        )
        self.scheduler.add_interval_job(
            name="prediction_cache_refresh",
            interval_seconds=self.config.runtime.prediction_cache_ttl_seconds,
            action=self.refresh_prediction_cache_job,
            enabled=scheduler_enabled,
        )
        self.scheduler.add_interval_job(
            name="data_quality_summary",
            interval_seconds=self.config.runtime.ingestion_interval_seconds,
            action=self.data_quality_summary_job,
            enabled=scheduler_enabled,
        )
        self.scheduler.add_interval_job(
            name="drift_check",
            interval_seconds=self.config.runtime.ingestion_interval_seconds,
            action=self.drift_check_job,
            enabled=scheduler_enabled,
        )
        self.scheduler.add_interval_job(
            name="retraining_candidate",
            interval_seconds=self.config.runtime.ingestion_interval_seconds,
            action=self.retraining_candidate_job,
            enabled=scheduler_enabled,
        )
        if scheduler_enabled and self._critical_startup_artifacts_available():
            self.scheduler.start()
        status = self.scheduler.status()
        if status["running"]:
            self.scheduler_recovery_status = "running"
        elif scheduler_enabled:
            self.scheduler_recovery_status = "registered_stopped"
        else:
            self.scheduler_recovery_status = "registered_manual"

    def _buffer_state_path(self) -> Path:
        return self.artifact_layout.buffer_snapshot_path()

    def _critical_startup_artifacts_available(self) -> bool:
        roads_loaded = self.roads is not None and not self.roads.empty
        return roads_loaded and self.model_loaded and self.buffer_available

    def _verify_tomtom_recovery(self) -> None:
        if not self.config.tomtom.api_keys:
            self.tomtom_recovery_status = "missing_credentials"
            return
        try:
            self._tomtom_mapper()
        except TomTomMappingError:
            self.tomtom_recovery_status = "mapping_unavailable"
            return
        self.tomtom_recovery_status = "configured"

    def _finalize_restart_recovery(self) -> None:
        roads_loaded = self.roads is not None and not self.roads.empty
        critical_ready = roads_loaded and self.model_loaded and self.buffer_available
        self.restart_recovery_status = "recovered" if critical_ready else "degraded"
        self.restart_recovery = {
            "roads_loaded": roads_loaded,
            "model_loaded": self.model_loaded,
            "model_source": self.model_recovery_source,
            "model_version": self.model_version,
            "buffer_available": self.buffer_available,
            "buffer_source": self.buffer_recovery_source,
            "buffer_restore_error": self.buffer_restore_error,
            "tomtom_status": self.tomtom_recovery_status,
            "scheduler_status": self.scheduler_recovery_status,
            "scheduler_job_count": len(self.scheduler.jobs),
            "stale_buffer_allowed": self.buffer_available and self.tomtom_recovery_status != "configured",
        }
        self.restart_recovery_detail = (
            f"buffer={self.buffer_recovery_source}; "
            f"model={self.model_recovery_source}; "
            f"tomtom={self.tomtom_recovery_status}; "
            f"scheduler={self.scheduler_recovery_status}"
        )

    def _finalize_startup(self) -> None:
        scheduler_status = self.scheduler.status()
        self.startup_report = build_startup_report(
            roads_loaded=self.roads is not None and not self.roads.empty,
            model_loaded=self.model_loaded,
            buffer_available=self.buffer_available,
            tomtom_configured=bool(self.config.tomtom.api_keys),
            scheduler_registered=bool(scheduler_status["job_count"]),
            scheduler_running=self.scheduler.running,
            scheduler_enabled=bool(self.config.runtime.scheduler_enabled),
            model_version=self.model_version,
            buffer_restore_error=self.buffer_restore_error,
            recovery_status=self.restart_recovery_status,
            recovery_detail=self.restart_recovery_detail,
        )


def build_app_state(config: AppConfig | None = None) -> AppState:
    state = AppState(config or load_config())
    state.load_static_resources()
    return state


def create_app(config: AppConfig | None = None) -> FastAPI:
    state = build_app_state(config)

    app = FastAPI(
        title="AWAI Traffic Prediction API",
        version="0.1.0",
        description="Lightweight traffic prediction service for Sukabumi road segments.",
        lifespan=api_lifespan,
    )
    app.state.app_state = state

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    register_security_middleware(app)
    register_request_logging_middleware(app)
    register_lifecycle_hooks(app)
    register_routes(app)
    return app


def register_security_middleware(app: FastAPI) -> None:
    state = get_app_state(app)
    config = state.config.api
    app.state.security_controls_registered = True
    app.state.request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    app.state.rate_limit_hits = {}
    app.state.rate_limit_lock = asyncio.Lock()

    @app.middleware("http")
    async def lightweight_security_controls(request: Request, call_next):
        size_response = _reject_oversized_request(request, config.max_request_bytes)
        if size_response is not None:
            return size_response

        auth_response = _reject_unauthorized_request(request, config.api_key)
        if auth_response is not None:
            return auth_response

        rate_response = await _reject_rate_limited_request(app, request, config.rate_limit_per_minute)
        if rate_response is not None:
            return rate_response

        semaphore: asyncio.Semaphore = app.state.request_semaphore
        if semaphore.locked():
            return JSONResponse(
                status_code=503,
                content=structured_error_response(
                    error_code="concurrency_limit_reached",
                    message="API is handling the maximum allowed concurrent requests",
                    details={"path": request.url.path},
                ),
            )

        await semaphore.acquire()
        try:
            return await call_next(request)
        finally:
            semaphore.release()


def register_request_logging_middleware(app: FastAPI) -> None:
    app.state.request_logging_registered = True

    @app.middleware("http")
    async def structured_request_logging(request: Request, call_next):
        state = get_app_state(app)
        started = time.monotonic()
        try:
            response = await call_next(request)
        except Exception as exc:
            duration = time.monotonic() - started
            state.runtime_event_logger.write(
                "errors",
                "api_request_exception",
                {
                    "method": request.method,
                    "path": request.url.path,
                    "duration_seconds": round(duration, 6),
                    "error": str(exc),
                },
                status="failed",
                level="ERROR",
            )
            raise
        duration = time.monotonic() - started
        state.runtime_event_logger.write(
            "api",
            "request",
            {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_seconds": round(duration, 6),
                "client": request.client.host if request.client else "unknown",
            },
            status="completed" if response.status_code < 400 else "failed",
            level="INFO" if response.status_code < 500 else "ERROR",
        )
        return response


def _reject_oversized_request(request: Request, max_request_bytes: int) -> JSONResponse | None:
    content_length = request.headers.get("content-length")
    if content_length is None:
        return None
    try:
        request_bytes = int(content_length)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content=structured_error_response(
                error_code="invalid_content_length",
                message="Content-Length must be a valid integer",
                details={"path": request.url.path},
            ),
        )
    if request_bytes <= max_request_bytes:
        return None
    return JSONResponse(
        status_code=413,
        content=structured_error_response(
            error_code="request_too_large",
            message="Request body exceeds the configured size limit",
            details={
                "path": request.url.path,
                "content_length": request_bytes,
                "max_request_bytes": max_request_bytes,
            },
        ),
    )


def _reject_unauthorized_request(request: Request, api_key: str | None) -> JSONResponse | None:
    if not api_key or _is_public_path(request.url.path):
        return None

    supplied_key = request.headers.get("x-api-key")
    authorization = request.headers.get("authorization", "")
    if not supplied_key and authorization.lower().startswith("bearer "):
        supplied_key = authorization[7:].strip()

    if not supplied_key:
        return JSONResponse(
            status_code=401,
            content=structured_error_response(
                error_code="missing_api_key",
                message="API key is required",
                details={"path": request.url.path},
            ),
        )
    if supplied_key != api_key:
        return JSONResponse(
            status_code=403,
            content=structured_error_response(
                error_code="invalid_api_key",
                message="API key is invalid",
                details={"path": request.url.path},
            ),
        )
    return None


async def _reject_rate_limited_request(app: FastAPI, request: Request, limit_per_minute: int) -> JSONResponse | None:
    client_key = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - 60.0
    async with app.state.rate_limit_lock:
        hits_by_client: dict[str, deque[float]] = app.state.rate_limit_hits
        hits = hits_by_client.setdefault(client_key, deque())
        while hits and hits[0] < window_start:
            hits.popleft()
        if len(hits) >= limit_per_minute:
            return JSONResponse(
                status_code=429,
                content=structured_error_response(
                    error_code="rate_limited",
                    message="Too many requests in the current rate window",
                    details={
                        "path": request.url.path,
                        "client": client_key,
                        "limit_per_minute": limit_per_minute,
                    },
                ),
            )
        hits.append(now)
    return None


def _is_public_path(path: str) -> bool:
    return path in {"/health", "/ready", "/docs", "/redoc", "/openapi.json"}


def register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        state = get_app_state(app)
        uptime = time.monotonic() - state.started_at
        return HealthResponse(
            status="healthy",
            model_loaded=state.model_loaded,
            model_version=state.model_version,
            scheduler_running=state.scheduler.running,
            uptime_seconds=uptime,
        )

    @app.get("/ready", response_model=ReadinessResponse)
    def ready() -> ReadinessResponse:
        state = get_app_state(app)
        roads_loaded = state.roads is not None and not state.roads.empty
        report = state.readiness_report()
        resources = report.by_name()
        return ReadinessResponse(
            ready=report.ready,
            model_loaded=state.model_loaded,
            roads_loaded=roads_loaded,
            buffer_available=state.buffer_available,
            tomtom_configured=resources["tomtom"].ready,
            scheduler_registered=all(
                name in state.scheduler.jobs
                for name in [
                    "tomtom_ingestion",
                    "buffer_persistence",
                    "prediction_cache_refresh",
                    "data_quality_summary",
                    "drift_check",
                    "retraining_candidate",
                ]
            ),
            scheduler_running=state.scheduler.running,
            resources=report.statuses(),
            details=report.details(),
        )

    @app.get("/data-quality", response_model=DataQualityResponse)
    def data_quality() -> DataQualityResponse:
        state = get_app_state(app)
        return state.data_quality()

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        state = get_app_state(app)
        return state.metrics()

    @app.get("/model/version", response_model=ModelVersionResponse)
    def model_version() -> ModelVersionResponse:
        state = get_app_state(app)
        return ModelVersionResponse(
            model_version=state.model_version,
            model_loaded=state.model_loaded,
            artifact_path=str(state.model_artifact_path) if state.model_artifact_path else None,
        )

    @app.post("/model/reload", response_model=ModelReloadResponse)
    def reload_model() -> ModelReloadResponse:
        state = get_app_state(app)
        return state.reload_model()

    @app.post("/ingest/manual", response_model=ManualIngestResponse)
    def ingest_manual(request: ManualIngestRequest) -> ManualIngestResponse:
        state = get_app_state(app)
        return state.ingest_manual(request)

    @app.post("/ingest/tomtom", response_model=JobTriggerResponse)
    def ingest_tomtom() -> JobTriggerResponse:
        state = get_app_state(app)
        return state.trigger_job("tomtom_ingestion")

    @app.post("/jobs/{job_name}/trigger", response_model=JobTriggerResponse)
    def trigger_job(job_name: str) -> JobTriggerResponse:
        state = get_app_state(app)
        return state.trigger_job(job_name)

    @app.get("/scheduler/status")
    def scheduler_status() -> dict:
        state = get_app_state(app)
        return state.scheduler.status()

    @app.get("/roads", response_model=list[RoadResponse])
    def roads() -> list[RoadResponse]:
        state = get_app_state(app)
        if state.roads is None:
            return []
        columns = [
            "road_id",
            "road_name",
            "city",
            "road_weight",
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
            "mid_lat",
            "mid_lon",
        ]
        available = [column for column in columns if column in state.roads.columns]
        return [RoadResponse(**record) for record in state.roads[available].to_dict("records")]


    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        state = get_app_state(app)
        return state.predict(request)

    @app.post("/predict/batch", response_model=PredictionBatchResponse)
    def predict_batch(request: PredictionBatchRequest, response: Response) -> PredictionBatchResponse:
        state = get_app_state(app)
        result = state.predict_batch(request)
        if result.successful_count > 0 and result.failed_count > 0:
            response.status_code = 206
        return result


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        get_app_state(app).runtime_event_logger.write(
            "errors",
            "http_exception",
            {
                "path": request.url.path,
                "status_code": exc.status_code,
                "detail": str(exc.detail),
            },
            status="failed",
            level="WARNING" if exc.status_code < 500 else "ERROR",
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=structured_error_response(
                error_code=http_error_code(exc.status_code, exc.detail),
                message=str(exc.detail),
                details={
                    "status_code": exc.status_code,
                    "path": request.url.path,
                },
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        get_app_state(app).runtime_event_logger.write(
            "errors",
            "request_validation_error",
            {
                "path": request.url.path,
                "status_code": 422,
                "validation_errors": jsonable_encoder(errors),
            },
            status="failed",
            level="WARNING",
        )
        return JSONResponse(
            status_code=422,
            content=structured_error_response(
                error_code=validation_error_code(errors),
                message="Request validation failed",
                details={
                    "status_code": 422,
                    "path": request.url.path,
                    "validation_errors": jsonable_encoder(errors),
                },
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        get_app_state(app).runtime_event_logger.write(
            "errors",
            "unhandled_exception",
            {
                "path": request.url.path,
                "status_code": 500,
                "error": str(exc),
            },
            status="failed",
            level="ERROR",
        )
        return JSONResponse(
            status_code=500,
            content=structured_error_response(
                error_code="prediction_failed" if request.url.path.startswith("/predict") else "internal_error",
                message="Internal server error",
                details={
                    "status_code": 500,
                    "path": request.url.path,
                    "error": str(exc),
                },
            ),
        )


def register_lifecycle_hooks(app: FastAPI) -> None:
    app.state.lifecycle_hooks_registered = True


@asynccontextmanager
async def api_lifespan(app: FastAPI) -> AsyncIterator[None]:
    get_app_state(app).readiness_report()
    try:
        yield
    finally:
        get_app_state(app).shutdown()


def get_app_state(app: FastAPI) -> AppState:
    return app.state.app_state


def structured_error_response(error_code: str, message: str, details: dict | None = None) -> dict:
    return ErrorResponse(
        error_code=error_code,
        message=message,
        details=details or {},
    ).model_dump(mode="json")


def http_error_code(status_code: int, detail: object) -> str:
    text = str(detail).lower()
    if status_code == 404 and "road_id" in text:
        return "unknown_road"
    if status_code == 404 and "job_name" in text:
        return "unknown_job"
    if status_code == 404:
        return "not_found"
    if status_code == 422 and "horizon" in text:
        return "invalid_horizon"
    if status_code == 422:
        return "invalid_request"
    if status_code == 503:
        return "service_unavailable"
    if status_code >= 500:
        return "internal_error"
    return "http_error"


def validation_error_code(errors: list[dict]) -> str:
    fields = " ".join(".".join(str(part) for part in error.get("loc", ())) for error in errors)
    if "horizon_minutes" in fields:
        return "invalid_horizon"
    return "invalid_request"


app = create_app()
