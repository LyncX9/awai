from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_prediction.data.scalers import ScalerStore
from traffic_prediction.data.schemas import FeatureManifest, LiveTrafficRecord
from traffic_prediction.features.offline import FeatureEngineer
from traffic_prediction.ingestion.buffer import LiveBufferManager


@dataclass(frozen=True)
class OnlineFeatureQuality:
    road_id: str
    observed_timesteps: int
    required_timesteps: int
    has_minimum_history: bool
    missing_feature_columns: list[str] = field(default_factory=list)
    padded_timesteps: int = 0
    status: str = "healthy"

    def to_dict(self) -> dict[str, int | bool | str | list[str]]:
        return {
            "road_id": self.road_id,
            "observed_timesteps": self.observed_timesteps,
            "required_timesteps": self.required_timesteps,
            "has_minimum_history": self.has_minimum_history,
            "missing_feature_columns": self.missing_feature_columns,
            "padded_timesteps": self.padded_timesteps,
            "status": self.status,
        }


@dataclass(frozen=True)
class OnlineFeatureResult:
    road_id: str
    feature_frame: pd.DataFrame
    sequence: np.ndarray
    quality: OnlineFeatureQuality


class OnlineFeatureEngineer:
    """Builds online inference features from the live buffer using the offline feature contract."""

    def __init__(
        self,
        manifest: FeatureManifest,
        buffer_manager: LiveBufferManager,
        roads: pd.DataFrame,
        scaler_store: ScalerStore | None = None,
        neighbor_mapping: dict[str, list[str]] | None = None,
    ) -> None:
        self.manifest = manifest
        self.buffer_manager = buffer_manager
        self.roads = roads.copy()
        self.scaler_store = scaler_store
        self.feature_engineer = FeatureEngineer(neighbor_mapping=neighbor_mapping or {})

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        buffer_manager: LiveBufferManager,
        roads: pd.DataFrame,
        neighbor_mapping: dict[str, list[str]] | None = None,
    ) -> "OnlineFeatureEngineer":
        artifact_path = Path(artifact_dir)
        manifest = FeatureManifest(**json.loads((artifact_path / "feature_manifest.json").read_text(encoding="utf-8")))
        scaler_path = artifact_path / "scaler_params.joblib"
        scaler_store = ScalerStore.load(scaler_path) if scaler_path.exists() else None
        return cls(
            manifest=manifest,
            buffer_manager=buffer_manager,
            roads=roads,
            scaler_store=scaler_store,
            neighbor_mapping=neighbor_mapping,
        )

    def build_for_road(self, road_id: str) -> OnlineFeatureResult:
        records = self._all_buffer_records()
        if not any(record.road_id == road_id for record in records):
            raise ValueError(f"No live buffer records available for road_id: {road_id}")

        base = self._records_to_frame(records)
        engineered = self.feature_engineer.extract_features(base)
        missing_columns = [column for column in self.manifest.feature_columns if column not in engineered.columns]
        for column in missing_columns:
            engineered[column] = 0.0

        selected = engineered[engineered["road_id"].astype(str) == str(road_id)].tail(self.manifest.lookback).copy()
        observed_timesteps = len(selected)
        selected = self._pad_to_lookback(selected)
        selected = selected[self.manifest.feature_columns].astype(float)
        if self.scaler_store is not None:
            selected = self.scaler_store.transform(selected)

        sequence = selected.to_numpy(dtype=np.float32).reshape(1, self.manifest.lookback, len(self.manifest.feature_columns))
        padded_timesteps = max(self.manifest.lookback - observed_timesteps, 0)
        status = "healthy" if observed_timesteps >= self.manifest.lookback and not missing_columns else "degraded"
        quality = OnlineFeatureQuality(
            road_id=road_id,
            observed_timesteps=observed_timesteps,
            required_timesteps=self.manifest.lookback,
            has_minimum_history=observed_timesteps >= self.manifest.lookback,
            missing_feature_columns=missing_columns,
            padded_timesteps=padded_timesteps,
            status=status,
        )
        return OnlineFeatureResult(road_id=road_id, feature_frame=selected, sequence=sequence, quality=quality)

    def _all_buffer_records(self) -> list[LiveTrafficRecord]:
        records: list[LiveTrafficRecord] = []
        for road_id in sorted(self.buffer_manager.buffers):
            records.extend(self.buffer_manager.get_latest(road_id))
        return records

    def _records_to_frame(self, records: list[LiveTrafficRecord]) -> pd.DataFrame:
        rows = [
            {
                "road_id": record.road_id,
                "current_speed": record.current_speed,
                "confidence": record.confidence,
                "collected_at_wib": record.timestamp,
            }
            for record in records
        ]
        frame = pd.DataFrame(rows)
        frame["collected_at_wib"] = pd.to_datetime(frame["collected_at_wib"], errors="coerce")
        if not self.roads.empty:
            road_columns = [column for column in ["road_id", "road_weight", "mid_lat", "mid_lon"] if column in self.roads.columns]
            frame = frame.merge(self.roads[road_columns], on="road_id", how="left")
        return self._ensure_base_columns(frame)

    def _ensure_base_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["road_weight"] = self._column_or_default(out, "road_weight", 0.0).fillna(0.0)
        out["free_flow_speed"] = self._column_or_default(out, "free_flow_speed", out["current_speed"]).fillna(out["current_speed"])
        out["current_travel_time"] = self._column_or_default(out, "current_travel_time", 0.0)
        out["free_flow_travel_time"] = self._column_or_default(out, "free_flow_travel_time", 0.0)
        out["road_closure"] = self._column_or_default(out, "road_closure", 0.0)
        out["sample_lat"] = self._column_or_default(out, "sample_lat", self._column_or_default(out, "mid_lat", 0.0)).fillna(0.0)
        out["sample_lon"] = self._column_or_default(out, "sample_lon", self._column_or_default(out, "mid_lon", 0.0)).fillna(0.0)
        out["speed_ratio"] = out["current_speed"] / out["free_flow_speed"].replace(0, np.nan)
        out["speed_ratio"] = out["speed_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def _column_or_default(self, frame: pd.DataFrame, column: str, default) -> pd.Series:
        if column in frame.columns:
            return frame[column]
        if isinstance(default, pd.Series):
            return default
        return pd.Series([default] * len(frame), index=frame.index)

    def _pad_to_lookback(self, frame: pd.DataFrame) -> pd.DataFrame:
        if len(frame) >= self.manifest.lookback:
            return frame
        if frame.empty:
            raise ValueError("Cannot pad an empty feature frame")
        pad_count = self.manifest.lookback - len(frame)
        padding = pd.concat([frame.iloc[[0]].copy() for _ in range(pad_count)], ignore_index=True)
        return pd.concat([padding, frame], ignore_index=True)
