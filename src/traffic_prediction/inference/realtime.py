from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd

from traffic_prediction.api.schemas import PredictionRequest, PredictionResponse
from traffic_prediction.features.online import OnlineFeatureEngineer, OnlineFeatureResult
from traffic_prediction.inference.cache import PredictionCache
from traffic_prediction.inference.confidence import ConfidenceAdjuster
from traffic_prediction.inference.congestion import classify_congestion
from traffic_prediction.inference.fallback import FallbackPrediction, FallbackPredictor
from traffic_prediction.ingestion.buffer import LiveBufferManager
from traffic_prediction.monitoring.data_quality import DataQualityMonitor


class PredictionModelRunner(Protocol):
    """Runtime model adapter contract; deep learning implementations can live outside this module."""

    def predict_kmh(self, sequence: np.ndarray) -> np.ndarray:
        """Return a horizon vector in original km/h scale."""


@dataclass(frozen=True)
class RealtimePredictionContext:
    request: PredictionRequest
    road_record: dict
    requested_at: datetime
    target_time: datetime
    cache_key: str
    model_version: str | None


class RealtimePredictionPipeline:
    """Coordinates real-time prediction concerns without owning deep learning model code."""

    def __init__(
        self,
        *,
        live_buffer: LiveBufferManager,
        roads: pd.DataFrame,
        prediction_cache: PredictionCache,
        fallback_predictor: FallbackPredictor,
        confidence_adjuster: ConfidenceAdjuster,
        data_quality_monitor: DataQualityMonitor,
        online_feature_engineer: OnlineFeatureEngineer | None = None,
        model_runner: PredictionModelRunner | None = None,
    ) -> None:
        self.live_buffer = live_buffer
        self.roads = roads
        self.prediction_cache = prediction_cache
        self.fallback_predictor = fallback_predictor
        self.confidence_adjuster = confidence_adjuster
        self.data_quality_monitor = data_quality_monitor
        self.online_feature_engineer = online_feature_engineer
        self.model_runner = model_runner

    def predict(self, context: RealtimePredictionContext) -> PredictionResponse:
        cached = self.prediction_cache.get(context.cache_key, now=context.requested_at)
        if cached is not None:
            response = cached.model_copy(deep=True)
            response.metadata["cache_hit"] = True
            return response

        quality_report = self._quality_report(context.requested_at)
        feature_result, feature_error = self._build_online_features(context.request.road_id)
        response = self._predict_live_or_fallback(
            context=context,
            quality_report=quality_report,
            feature_result=feature_result,
            feature_error=feature_error,
        )
        self.prediction_cache.set(context.cache_key, response.model_copy(deep=True), now=context.requested_at)
        return response

    def _predict_live_or_fallback(
        self,
        *,
        context: RealtimePredictionContext,
        quality_report,
        feature_result: OnlineFeatureResult | None,
        feature_error: str | None,
    ) -> PredictionResponse:
        model_inference_failed = True
        predicted_speed = 30.0
        uncertainty_margin = 1.0
        active_error = feature_error

        if self.model_runner is not None and feature_result is not None:
            try:
                horizon_values = np.asarray(self.model_runner.predict_kmh(feature_result.sequence), dtype=float).reshape(-1)
                horizon_index = self._horizon_index(context.request.horizon_minutes)
                if horizon_index >= len(horizon_values):
                    raise ValueError("model runner returned fewer horizon predictions than requested")
                predicted_speed = float(horizon_values[horizon_index])
                uncertainty_margin = max(float(np.std(horizon_values)), 1.0)
                model_inference_failed = False
            except Exception as exc:
                active_error = f"model_inference_error: {exc}"

        if not model_inference_failed:
            return self._format_live_response(
                context=context,
                predicted_speed=predicted_speed,
                uncertainty_margin=uncertainty_margin,
                quality_report=quality_report,
                feature_result=feature_result,
            )

        # Fallback path (LSTM failed or wasn't run)
        latest_record = self._latest_live_record(context.request.road_id)
        fallback = self.fallback_predictor.predict(
            road_id=context.request.road_id,
            target_time=context.target_time,
            horizon_minutes=context.request.horizon_minutes,
            # If we have any live data at all, use the latest observation as the prediction base
            # rather than the hardcoded global_default (30.0 km/h).
            latest_live_record=latest_record,
            prefer_persistence=latest_record is not None and context.request.horizon_minutes <= 30,  # Prefer persistence for short horizons if any live record is available
        )
        return self._format_fallback_response(
            context=context,
            fallback=fallback,
            quality_report=quality_report,
            feature_result=feature_result,
            feature_error=active_error,
        )


    def _format_live_response(
        self,
        *,
        context: RealtimePredictionContext,
        predicted_speed: float,
        uncertainty_margin: float,
        quality_report,
        feature_result: OnlineFeatureResult,
    ) -> PredictionResponse:
        predicted_speed = self._clip_speed(predicted_speed)
        free_flow_speed = self._free_flow_speed(context.road_record, predicted_speed)
        source_confidence = self._source_confidence(context.request.road_id)
        confidence = self.confidence_adjuster.adjust(
            base_confidence=0.90,
            uncertainty_margin=uncertainty_margin,
            source_confidence=source_confidence,
            data_quality=quality_report,
        )
        degraded = feature_result.quality.status != "healthy" or getattr(quality_report, "status", None) != "healthy"
        # Get actual current speed from live buffer for accurate real-time classification
        current_speed = self._current_speed_from_buffer(context.request.road_id)
        current_congestion_level = classify_congestion(current_speed, free_flow_speed) if current_speed is not None else None
        return PredictionResponse(
            road_id=context.request.road_id,
            horizon_minutes=context.request.horizon_minutes,
            predicted_speed=round(predicted_speed, 3),
            congestion_level=classify_congestion(predicted_speed, free_flow_speed),
            current_speed=round(current_speed, 3) if current_speed is not None else None,
            current_congestion_level=current_congestion_level,
            free_flow_speed=round(free_flow_speed, 3),
            uncertainty_lower=round(max(0.0, predicted_speed - uncertainty_margin), 3),
            uncertainty_upper=round(min(120.0, predicted_speed + uncertainty_margin), 3),
            confidence_score=confidence.adjusted_confidence,
            model_version=context.model_version,
            prediction_method="live_lstm_runtime",
            degraded=degraded,
            data_quality=self._data_quality_payload(quality_report, degraded=degraded, reason="live_lstm_runtime"),
            metadata={
                **self._base_metadata(context, free_flow_speed),
                **confidence.metadata(),
                "model_runner_available": True,
                "online_features_built": True,
                "feature_quality": feature_result.quality.to_dict(),
                "source_confidence": source_confidence,
            },
        )

    def _format_fallback_response(
        self,
        *,
        context: RealtimePredictionContext,
        fallback: FallbackPrediction,
        quality_report,
        feature_result: OnlineFeatureResult | None,
        feature_error: str | None,
    ) -> PredictionResponse:
        predicted_speed = self._clip_speed(fallback.predicted_speed)
        free_flow_speed = self._free_flow_speed(context.road_record, predicted_speed)
        confidence = self.confidence_adjuster.adjust(
            base_confidence=fallback.confidence_score,
            uncertainty_margin=fallback.uncertainty_margin,
            data_quality=quality_report,
        )
        metadata = {
            **self._base_metadata(context, free_flow_speed),
            **confidence.metadata(),
            "model_runner_available": self.model_runner is not None,
            "online_features_built": feature_result is not None,
            "feature_quality": feature_result.quality.to_dict() if feature_result is not None else None,
            "online_feature_error": feature_error,
            "fallback_reason": fallback.reason if feature_error is None else feature_error,
        }
        # Get actual current speed from live buffer for accurate real-time classification
        current_speed = self._current_speed_from_buffer(context.request.road_id)
        current_congestion_level = classify_congestion(current_speed, free_flow_speed) if current_speed is not None else None
        return PredictionResponse(
            road_id=context.request.road_id,
            horizon_minutes=context.request.horizon_minutes,
            predicted_speed=round(predicted_speed, 3),
            congestion_level=classify_congestion(predicted_speed, free_flow_speed),
            current_speed=round(current_speed, 3) if current_speed is not None else None,
            current_congestion_level=current_congestion_level,
            free_flow_speed=round(free_flow_speed, 3),
            uncertainty_lower=round(max(0.0, predicted_speed - fallback.uncertainty_margin), 3),
            uncertainty_upper=round(min(120.0, predicted_speed + fallback.uncertainty_margin), 3),
            confidence_score=confidence.adjusted_confidence,
            model_version=context.model_version,
            prediction_method=fallback.method,
            degraded=fallback.degraded,
            data_quality=self._data_quality_payload(quality_report, degraded=fallback.degraded, reason=fallback.reason),
            metadata=metadata,
        )

    def _build_online_features(self, road_id: str) -> tuple[OnlineFeatureResult | None, str | None]:
        if self.online_feature_engineer is None:
            return None, "online_feature_engineer_unavailable"
        try:
            return self.online_feature_engineer.build_for_road(road_id), None
        except Exception as exc:
            return None, str(exc)

    def _quality_report(self, now: datetime):
        expected_roads = set(self.roads["road_id"].astype(str)) if self.roads is not None and not self.roads.empty else set()
        latest_records = [
            records[-1]
            for road_id in sorted(self.live_buffer.buffers)
            if (records := self.live_buffer.get_latest(road_id))
        ]
        now = self._align_now_to_record_timezone(now, latest_records)
        return self.data_quality_monitor.evaluate(latest_records, expected_road_ids=expected_roads, now=now)

    def _base_metadata(self, context: RealtimePredictionContext, free_flow_speed: float) -> dict:
        return {
            "requested_at": context.requested_at.isoformat(),
            "target_time": context.target_time.isoformat(),
            "free_flow_speed": free_flow_speed,
            "active_model_loaded": context.model_version is not None,
            "cache_hit": False,
            "cache_key": context.cache_key,
            "realtime_pipeline": True,
        }

    @staticmethod
    def _data_quality_payload(quality_report, *, degraded: bool, reason: str) -> dict:
        return {
            "status": "degraded" if degraded else getattr(quality_report, "status", "healthy"),
            "live_quality_status": getattr(quality_report, "status", "unknown"),
            "reason": reason,
            "buffer_available": bool(getattr(quality_report, "road_count", 0) or getattr(quality_report, "completeness", 0.0)),
            "completeness": getattr(quality_report, "completeness", 0.0),
            "quality_issues": getattr(quality_report, "quality_issues", {}),
            "fallback_recommendation": getattr(quality_report, "fallback_recommendation", "unknown"),
        }

    def _source_confidence(self, road_id: str) -> float | None:
        records = self.live_buffer.get_latest(road_id)
        if not records:
            return None
        return float(np.mean([record.confidence for record in records]))

    def _current_speed_from_buffer(self, road_id: str) -> float | None:
        """Return the most recent observed current_speed from the live buffer, or None if unavailable."""
        records = self.live_buffer.get_latest(road_id, n=1)
        if not records:
            return None
        return float(records[-1].current_speed)

    def _latest_live_record(self, road_id: str):
        """Return the most recent LiveTrafficRecord for a road, or None if the buffer is empty."""
        records = self.live_buffer.get_latest(road_id, n=1)
        return records[-1] if records else None

    @staticmethod
    def _free_flow_speed(road_record: dict, predicted_speed: float) -> float:
        return float(road_record.get("free_flow_speed", predicted_speed) or predicted_speed)

    @staticmethod
    def _horizon_index(horizon_minutes: int) -> int:
        return max(int(horizon_minutes // 15) - 1, 0)

    @staticmethod
    def _clip_speed(value: float) -> float:
        return float(np.clip(value, 0.0, 120.0))

    @staticmethod
    def _align_now_to_record_timezone(now: datetime, latest_records) -> datetime:
        if not latest_records:
            return now
        record_timestamp = pd.Timestamp(latest_records[0].timestamp)
        current = pd.Timestamp(now)
        if record_timestamp.tzinfo is None:
            if current.tzinfo is not None:
                return current.tz_convert(None).to_pydatetime()
            return current.to_pydatetime()
        if current.tzinfo is None:
            return current.tz_localize(record_timestamp.tzinfo).to_pydatetime()
        return current.tz_convert(record_timestamp.tzinfo).to_pydatetime()
