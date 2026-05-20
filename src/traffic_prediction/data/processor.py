from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_prediction.config.settings import DataConfig, FeatureConfig
from traffic_prediction.data.exceptions import ValidationError
from traffic_prediction.data.scalers import ScalerStore
from traffic_prediction.data.schemas import FeatureManifest, SequenceMetadata, SplitStatistics, ValidationReport
from traffic_prediction.data.validation import (
    REQUIRED_TRAFFIC_COLUMNS,
    build_traffic_validation_report,
    validate_required_columns,
    validate_roads_schema,
    validate_traffic_values,
)
from traffic_prediction.features.leakage import LeakageValidator


class DataProcessor:
    """Historical data pipeline for validation, cleaning, split, scaling, and sequences."""

    STATIC_FILL_COLUMNS = [
        "road_name",
        "city",
        "road_weight",
        "free_flow_speed",
        "current_travel_time",
        "free_flow_travel_time",
        "road_closure",
        "frc",
        "sample_lat",
        "sample_lon",
    ]

    def __init__(
        self,
        data_config: DataConfig | None = None,
        feature_config: FeatureConfig | None = None,
        scaler_store: ScalerStore | None = None,
    ) -> None:
        self.data_config = data_config or DataConfig()
        self.feature_config = feature_config or FeatureConfig()
        self.scaler_store = scaler_store
        self.leakage_validator = LeakageValidator()

    def load_traffic_csv(self, path: str | Path) -> tuple[pd.DataFrame, ValidationReport]:
        df = pd.read_csv(path)
        validate_required_columns(df, REQUIRED_TRAFFIC_COLUMNS, "traffic_data")
        report = build_traffic_validation_report(
            df,
            min_speed=self.data_config.min_speed,
            max_speed=self.data_config.max_speed,
        )
        validate_traffic_values(report)
        return df, report

    def load_roads_csv(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        validate_roads_schema(df)
        return df

    def validate_traffic(self, df: pd.DataFrame) -> ValidationReport:
        report = build_traffic_validation_report(
            df,
            min_speed=self.data_config.min_speed,
            max_speed=self.data_config.max_speed,
        )
        validate_traffic_values(report)
        return report

    def validate_roads(self, df: pd.DataFrame) -> ValidationReport:
        validate_roads_schema(df)
        return ValidationReport(
            row_count=int(len(df)),
            road_count=int(df["road_id"].nunique()),
            date_range_start=None,
            date_range_end=None,
            missing_values={column: int(value) for column, value in df.isna().sum().to_dict().items()},
            invalid_speed_count=0,
            invalid_confidence_count=0,
            duplicate_count=int(df.duplicated(subset=["road_id"]).sum()),
            is_chronological_per_road=True,
        )

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned["collected_at_wib"] = self._parse_timestamp(cleaned["collected_at_wib"])
        cleaned["collected_at_wib"] = cleaned["collected_at_wib"].dt.floor(self.data_config.frequency)
        cleaned = cleaned.sort_values(["road_id", "collected_at_wib", "confidence"])
        cleaned = cleaned.drop_duplicates(subset=["road_id", "collected_at_wib"], keep="last")
        cleaned = cleaned[cleaned["confidence"] >= self.data_config.min_confidence].copy()
        cleaned = self._create_uniform_grid(cleaned)
        cleaned = self._impute_missing_values(cleaned)
        cleaned["speed_ratio"] = cleaned["current_speed"] / cleaned["free_flow_speed"].replace(0, np.nan)
        cleaned["speed_ratio"] = cleaned["speed_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
        self._assert_clean(cleaned)
        return cleaned.sort_values(["road_id", "collected_at_wib"]).reset_index(drop=True)

    def clean_traffic(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.clean(df)

    def chronological_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitStatistics]:
        ordered = df.sort_values(["collected_at_wib", "road_id"]).reset_index(drop=True)
        start = ordered["collected_at_wib"].min()
        train_end = start + pd.Timedelta(days=self.data_config.train_days)
        validation_end = train_end + pd.Timedelta(days=self.data_config.validation_days)

        train = ordered[ordered["collected_at_wib"] < train_end].copy()
        validation = ordered[
            (ordered["collected_at_wib"] >= train_end) & (ordered["collected_at_wib"] < validation_end)
        ].copy()
        test = ordered[ordered["collected_at_wib"] >= validation_end].copy()

        self.leakage_validator.validate_temporal_split(train, validation, test)
        stats = self._build_split_statistics(train, validation, test)
        return train, validation, test, stats

    def build_default_scaler(self, feature_columns: list[str]) -> ScalerStore:
        temporal_minmax = [
            "hour_of_day",
            "day_of_week",
            "time_since_midnight",
            "road_weight",
            "free_flow_speed",
            "speed_ratio",
        ]
        standard = [
            column
            for column in feature_columns
            if column.startswith("lag_")
            or column.startswith("rolling_")
            or column
            in {
                "current_speed",
                "speed_delta",
                "speed_acceleration",
                "speed_volatility",
                "neighbor_speed_mean",
                "neighbor_speed_std",
                "neighbor_speed_min",
                "neighbor_speed_max",
            }
        ]
        return ScalerStore(
            speed_columns=["current_speed"],
            minmax_columns=[column for column in temporal_minmax if column in feature_columns],
            standard_columns=[column for column in standard if column in feature_columns and column != "current_speed"],
        )

    def fit_transform_train(self, train: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        self.scaler_store = self.build_default_scaler(feature_columns)
        return self.scaler_store.fit_transform(train)

    def transform_eval(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_store is None:
            raise ValidationError("Scaler store is not fitted; call fit_transform_train first")
        return self.scaler_store.transform(df)

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "current_speed",
    ) -> tuple[np.ndarray, np.ndarray, list[SequenceMetadata]]:
        lookback = self.feature_config.lookback
        horizon = self.feature_config.horizon
        sequences: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        metadata: list[SequenceMetadata] = []

        self.leakage_validator.validate_feature_columns(df, feature_columns)
        frequency = pd.Timedelta(self.data_config.frequency)

        for road_id, road_df in df.groupby("road_id", sort=False):
            road_df = road_df.sort_values("collected_at_wib").reset_index(drop=True)
            gap_id = road_df["collected_at_wib"].diff().gt(frequency).cumsum()
            for _, block in road_df.groupby(gap_id):
                if len(block) < lookback + horizon:
                    continue
                for index in range(0, len(block) - lookback - horizon + 1):
                    x = block.iloc[index : index + lookback][feature_columns].to_numpy(dtype=np.float32)
                    y = block.iloc[index + lookback : index + lookback + horizon][target_column].to_numpy(
                        dtype=np.float32
                    )
                    sequences.append(x)
                    targets.append(y.reshape(horizon, 1))
                    metadata.append(
                        SequenceMetadata(
                            road_id=str(road_id),
                            input_start=block.iloc[index]["collected_at_wib"].to_pydatetime(),
                            input_end=block.iloc[index + lookback - 1]["collected_at_wib"].to_pydatetime(),
                            target_start=block.iloc[index + lookback]["collected_at_wib"].to_pydatetime(),
                            target_end=block.iloc[index + lookback + horizon - 1]["collected_at_wib"].to_pydatetime(),
                        )
                    )

        return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32), metadata

    def build_feature_manifest(self, feature_columns: list[str]) -> FeatureManifest:
        return FeatureManifest(
            feature_columns=feature_columns,
            target_column="current_speed",
            lookback=self.feature_config.lookback,
            horizon=self.feature_config.horizon,
        )

    def save_feature_manifest(self, manifest: FeatureManifest, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    def _parse_timestamp(self, values: pd.Series) -> pd.Series:
        parsed = pd.to_datetime(values, errors="coerce")
        if parsed.isna().any():
            raise ValidationError("collected_at_wib contains unparsable timestamps")
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize(self.data_config.timezone)
        else:
            parsed = parsed.dt.tz_convert(self.data_config.timezone)
        return parsed.dt.floor(self.data_config.frequency)

    def _create_uniform_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        frames = []
        frequency = self.data_config.frequency
        for road_id, group in df.groupby("road_id", sort=False):
            group = group.sort_values("collected_at_wib").set_index("collected_at_wib")
            full_index = pd.date_range(
                group.index.min().floor(frequency),
                group.index.max().ceil(frequency),
                freq=frequency,
                tz=self.data_config.timezone,
            )
            reindexed = group.reindex(full_index)
            reindexed.index.name = "collected_at_wib"
            reindexed["road_id"] = road_id
            frames.append(reindexed.reset_index())
        return pd.concat(frames, ignore_index=True)

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.sort_values(["road_id", "collected_at_wib"]).copy()
        numeric_by_road = [
            "current_speed",
            "free_flow_speed",
            "current_travel_time",
            "free_flow_travel_time",
            "confidence",
            "road_weight",
            "sample_lat",
            "sample_lon",
        ]
        present_numeric = [column for column in numeric_by_road if column in out.columns]
        for column in present_numeric:
            out[column] = out.groupby("road_id", sort=False)[column].transform(
                lambda group: group.interpolate(limit_direction="both").ffill().bfill()
            )

        static_columns = [column for column in self.STATIC_FILL_COLUMNS if column in out.columns]
        if static_columns:
            out[static_columns] = out.groupby("road_id", sort=False)[static_columns].transform(
                lambda group: group.ffill().bfill()
            )
        out["confidence"] = out["confidence"].fillna(self.data_config.min_confidence)
        return out

    def _assert_clean(self, df: pd.DataFrame) -> None:
        critical = ["road_id", "current_speed", "free_flow_speed", "confidence", "collected_at_wib"]
        missing = df[critical].isna().sum()
        failed = {column: int(count) for column, count in missing.items() if count}
        if failed:
            raise ValidationError(f"Cleaned dataset still has missing critical values: {failed}")
        if not df["current_speed"].between(self.data_config.min_speed, self.data_config.max_speed).all():
            raise ValidationError("Cleaned dataset contains out-of-range current_speed values")
        if not df["confidence"].between(0.0, 1.0).all():
            raise ValidationError("Cleaned dataset contains out-of-range confidence values")

    def _build_split_statistics(
        self, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
    ) -> SplitStatistics:
        def min_ts(frame: pd.DataFrame):
            return frame["collected_at_wib"].min().to_pydatetime() if not frame.empty else None

        def max_ts(frame: pd.DataFrame):
            return frame["collected_at_wib"].max().to_pydatetime() if not frame.empty else None

        return SplitStatistics(
            train_rows=int(len(train)),
            validation_rows=int(len(validation)),
            test_rows=int(len(test)),
            train_start=min_ts(train),
            train_end=max_ts(train),
            validation_start=min_ts(validation),
            validation_end=max_ts(validation),
            test_start=min_ts(test),
            test_end=max_ts(test),
            train_road_count=int(train["road_id"].nunique()),
            validation_road_count=int(validation["road_id"].nunique()),
            test_road_count=int(test["road_id"].nunique()),
        )
