from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from traffic_prediction.data.schemas import LiveTrafficRecord


@dataclass(frozen=True)
class RetrainingDatasetConfig:
    retention_days: int = 60
    min_retention_days: int = 30
    max_retention_days: int = 180
    max_live_fraction: float = 0.35
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    keep_versions: int = 3


@dataclass(frozen=True)
class RetrainingDiversityReport:
    expected_road_count: int
    observed_road_count: int
    road_coverage: float
    covered_hours: list[int]
    covered_days_of_week: list[int]
    congestion_levels: list[str]
    is_valid: bool
    issues: list[str]


@dataclass(frozen=True)
class RetrainingDatasetManifest:
    version: str
    created_at: str
    dataset_path: str
    manifest_path: str
    retention_days: int
    date_range_start: str | None
    date_range_end: str | None
    total_records: int
    historical_records: int
    live_records: int
    train_rows: int
    validation_rows: int
    test_rows: int
    diversity: RetrainingDiversityReport

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["diversity"] = asdict(self.diversity)
        return payload


class RetrainingDatasetManager:
    """Builds balanced retraining candidate datasets from historical and live records."""

    def __init__(self, config: RetrainingDatasetConfig | None = None) -> None:
        self.config = config or RetrainingDatasetConfig()
        self._validate_config()

    def status(
        self,
        historical_csv: str | Path,
        live_records: list[LiveTrafficRecord] | None = None,
        roads: pd.DataFrame | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        dataset = self._combine_sources(historical_csv, live_records or [], now=now)
        diversity = self.validate_diversity(dataset, roads)
        return {
            "retention_days": self._retention_days(),
            "date_range_start": self._format_timestamp(dataset["collected_at_wib"].min()) if not dataset.empty else None,
            "date_range_end": self._format_timestamp(dataset["collected_at_wib"].max()) if not dataset.empty else None,
            "total_records": int(len(dataset)),
            "historical_records": int((dataset["source"] == "historical").sum()) if not dataset.empty else 0,
            "live_records": int((dataset["source"] == "live").sum()) if not dataset.empty else 0,
            "diversity": asdict(diversity),
        }

    def build_candidate(
        self,
        historical_csv: str | Path,
        live_records: list[LiveTrafficRecord],
        output_root: str | Path,
        roads: pd.DataFrame | None = None,
        now: datetime | None = None,
    ) -> RetrainingDatasetManifest:
        created_at = pd.Timestamp(now or datetime.now()).isoformat()
        version = f"retraining-{pd.Timestamp(now or datetime.now()).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        version_dir = Path(output_root) / version
        version_dir.mkdir(parents=True, exist_ok=True)

        dataset = self._combine_sources(historical_csv, live_records, now=now)
        dataset = self._assign_chronological_splits(dataset)
        diversity = self.validate_diversity(dataset, roads)

        dataset_path = version_dir / "candidate_dataset.csv"
        manifest_path = version_dir / "manifest.json"
        dataset.to_csv(dataset_path, index=False)

        manifest = RetrainingDatasetManifest(
            version=version,
            created_at=created_at,
            dataset_path=str(dataset_path.resolve()),
            manifest_path=str(manifest_path.resolve()),
            retention_days=self._retention_days(),
            date_range_start=self._format_timestamp(dataset["collected_at_wib"].min()) if not dataset.empty else None,
            date_range_end=self._format_timestamp(dataset["collected_at_wib"].max()) if not dataset.empty else None,
            total_records=int(len(dataset)),
            historical_records=int((dataset["source"] == "historical").sum()) if not dataset.empty else 0,
            live_records=int((dataset["source"] == "live").sum()) if not dataset.empty else 0,
            train_rows=int((dataset["split"] == "train").sum()) if not dataset.empty else 0,
            validation_rows=int((dataset["split"] == "validation").sum()) if not dataset.empty else 0,
            test_rows=int((dataset["split"] == "test").sum()) if not dataset.empty else 0,
            diversity=diversity,
        )
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        self._prune_old_versions(Path(output_root))
        return manifest

    def list_versions(self, output_root: str | Path, include_archived: bool = False) -> list[Path]:
        versions = sorted(
            [path for path in Path(output_root).glob("retraining-*") if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if include_archived:
            return versions
        return [path for path in versions if not (path / "ARCHIVED").exists()]

    def validate_diversity(
        self,
        dataset: pd.DataFrame,
        roads: pd.DataFrame | None = None,
    ) -> RetrainingDiversityReport:
        if dataset.empty:
            return RetrainingDiversityReport(
                expected_road_count=0,
                observed_road_count=0,
                road_coverage=0.0,
                covered_hours=[],
                covered_days_of_week=[],
                congestion_levels=[],
                is_valid=False,
                issues=["dataset_empty"],
            )

        expected_roads = int(roads["road_id"].nunique()) if roads is not None and "road_id" in roads else int(dataset["road_id"].nunique())
        observed_roads = int(dataset["road_id"].nunique())
        road_coverage = observed_roads / expected_roads if expected_roads else 0.0
        timestamps = self._coerce_timestamps(dataset["collected_at_wib"])
        covered_hours = sorted(int(value) for value in timestamps.dt.hour.dropna().unique())
        covered_days = sorted(int(value) for value in timestamps.dt.dayofweek.dropna().unique())
        congestion_levels = self._congestion_levels(dataset)

        issues: list[str] = []
        if road_coverage < 0.90:
            issues.append("road_coverage_below_90_percent")
        if len(covered_hours) < 24:
            issues.append("missing_hours_of_day")
        if len(covered_days) < 7:
            issues.append("missing_days_of_week")
        if set(congestion_levels) != {"free_flow", "moderate", "congested"}:
            issues.append("missing_congestion_levels")

        return RetrainingDiversityReport(
            expected_road_count=expected_roads,
            observed_road_count=observed_roads,
            road_coverage=round(road_coverage, 6),
            covered_hours=covered_hours,
            covered_days_of_week=covered_days,
            congestion_levels=congestion_levels,
            is_valid=not issues,
            issues=issues,
        )

    def _combine_sources(
        self,
        historical_csv: str | Path,
        live_records: list[LiveTrafficRecord],
        now: datetime | None = None,
    ) -> pd.DataFrame:
        historical = self._load_historical(historical_csv)
        live = self._live_records_to_frame(live_records, historical)
        if live.empty:
            combined = historical.copy()
        else:
            combined = pd.concat([historical, live], ignore_index=True, sort=False)
        if combined.empty:
            return combined
        combined["collected_at_wib"] = self._coerce_timestamps(combined["collected_at_wib"])
        combined = combined.dropna(subset=["road_id", "current_speed", "confidence", "collected_at_wib"]).copy()
        combined = self._apply_retention(combined, now=now)
        combined = self._balance_live_fraction(combined)
        return combined.sort_values(["collected_at_wib", "road_id", "source"]).reset_index(drop=True)

    def _load_historical(self, historical_csv: str | Path) -> pd.DataFrame:
        path = Path(historical_csv)
        if not path.exists():
            raise FileNotFoundError(f"Historical CSV not found: {path}")
        historical = pd.read_csv(path)
        required = {"road_id", "current_speed", "confidence", "collected_at_wib"}
        missing = sorted(required - set(historical.columns))
        if missing:
            raise ValueError(f"Historical CSV missing required columns: {missing}")
        historical = historical.copy()
        historical["collected_at_wib"] = self._coerce_timestamps(historical["collected_at_wib"])
        historical = historical.dropna(subset=["road_id", "current_speed", "confidence", "collected_at_wib"])
        historical["source"] = "historical"
        return historical

    def _live_records_to_frame(self, records: list[LiveTrafficRecord], historical: pd.DataFrame) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=[*historical.columns])

        profiles = (
            historical.sort_values("collected_at_wib")
            .drop_duplicates(subset=["road_id"], keep="last")
            .set_index("road_id", drop=False)
        )
        rows: list[dict[str, Any]] = []
        for record in records:
            base = profiles.loc[record.road_id].to_dict() if record.road_id in profiles.index else {}
            base.update(
                {
                    "road_id": record.road_id,
                    "current_speed": record.current_speed,
                    "confidence": record.confidence,
                    "collected_at_wib": record.timestamp,
                    "source": "live",
                }
            )
            if "id" in base:
                base["id"] = None
            rows.append(base)
        live = pd.DataFrame(rows)
        return live.reindex(columns=historical.columns.union(live.columns, sort=False))

    def _apply_retention(self, dataset: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
        retention_days = self._retention_days()
        reference = self._coerce_timestamp(now) if now is not None else pd.Timestamp(dataset["collected_at_wib"].max())
        cutoff = reference - pd.Timedelta(days=retention_days)
        return dataset[dataset["collected_at_wib"] >= cutoff].copy()

    def _balance_live_fraction(self, dataset: pd.DataFrame) -> pd.DataFrame:
        historical = dataset[dataset["source"] == "historical"].copy()
        live = dataset[dataset["source"] == "live"].copy()
        if historical.empty or live.empty:
            return dataset

        max_live_rows = int(len(historical) * self.config.max_live_fraction / (1.0 - self.config.max_live_fraction))
        max_live_rows = max(max_live_rows, 1)
        if len(live) <= max_live_rows:
            return dataset
        live = live.sort_values("collected_at_wib").tail(max_live_rows)
        return pd.concat([historical, live], ignore_index=True, sort=False)

    def _assign_chronological_splits(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.sort_values(["collected_at_wib", "road_id", "source"]).reset_index(drop=True).copy()
        total = len(dataset)
        train_end = int(total * self.config.train_fraction)
        validation_end = train_end + int(total * self.config.validation_fraction)
        if total >= 3:
            train_end = max(train_end, 1)
            validation_end = max(validation_end, train_end + 1)
            validation_end = min(validation_end, total - 1)
        splits = ["train"] * train_end + ["validation"] * (validation_end - train_end) + ["test"] * (total - validation_end)
        dataset["split"] = splits
        return dataset

    def _congestion_levels(self, dataset: pd.DataFrame) -> list[str]:
        if "free_flow_speed" in dataset.columns:
            free_flow = pd.to_numeric(dataset["free_flow_speed"], errors="coerce").replace(0, pd.NA)
            ratio = pd.to_numeric(dataset["current_speed"], errors="coerce") / free_flow
        else:
            ratio = pd.Series([1.0] * len(dataset), index=dataset.index)
        levels = []
        if (ratio >= 0.80).any():
            levels.append("free_flow")
        if ((ratio >= 0.50) & (ratio < 0.80)).any():
            levels.append("moderate")
        if (ratio < 0.50).any():
            levels.append("congested")
        return levels

    def _prune_old_versions(self, output_root: Path) -> None:
        versions = self.list_versions(output_root, include_archived=False)
        for old_version in versions[self.config.keep_versions :]:
            (old_version / "ARCHIVED").write_text("archived by retraining dataset manager\n", encoding="utf-8")

    def _retention_days(self) -> int:
        return min(max(self.config.retention_days, self.config.min_retention_days), self.config.max_retention_days)

    def _validate_config(self) -> None:
        if not 0 < self.config.max_live_fraction < 1:
            raise ValueError("max_live_fraction must be between 0 and 1")
        if self.config.train_fraction <= 0 or self.config.validation_fraction <= 0:
            raise ValueError("split fractions must be greater than 0")
        if self.config.train_fraction + self.config.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be less than 1")
        if self.config.keep_versions <= 0:
            raise ValueError("keep_versions must be greater than 0")

    def _format_timestamp(self, value: Any) -> str:
        return pd.Timestamp(value).isoformat()

    def _coerce_timestamps(self, values: Any) -> pd.Series:
        return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(None)

    def _coerce_timestamp(self, value: Any) -> pd.Timestamp:
        return pd.Timestamp(value).tz_convert("UTC").tz_localize(None) if pd.Timestamp(value).tzinfo else pd.Timestamp(value)
