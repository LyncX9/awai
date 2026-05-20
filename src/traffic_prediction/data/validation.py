from __future__ import annotations

import pandas as pd

from traffic_prediction.data.exceptions import ValidationError
from traffic_prediction.data.schemas import ValidationReport


REQUIRED_TRAFFIC_COLUMNS = {
    "road_id",
    "current_speed",
    "collected_at_wib",
    "free_flow_speed",
    "confidence",
}

REQUIRED_ROAD_COLUMNS = {
    "road_id",
    "start_lat",
    "start_lon",
    "end_lat",
    "end_lon",
    "mid_lat",
    "mid_lon",
}


def validate_required_columns(df: pd.DataFrame, required: set[str], dataset_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValidationError(f"{dataset_name} missing required columns: {', '.join(missing)}")


def validate_roads_schema(df: pd.DataFrame) -> None:
    validate_required_columns(df, REQUIRED_ROAD_COLUMNS, "roads_master")
    missing_coordinates = df[list(REQUIRED_ROAD_COLUMNS - {"road_id"})].isna().any(axis=1)
    if missing_coordinates.any():
        count = int(missing_coordinates.sum())
        raise ValidationError(f"roads_master has {count} rows with incomplete coordinate data")


def build_traffic_validation_report(
    df: pd.DataFrame,
    min_speed: float,
    max_speed: float,
) -> ValidationReport:
    timestamps = pd.to_datetime(df["collected_at_wib"], errors="coerce")
    duplicate_count = int(df.duplicated(subset=["road_id", "collected_at_wib"]).sum())
    chronological_flags = []
    temp = df.assign(_timestamp=timestamps).sort_values(["road_id", "_timestamp"])
    for _, group in temp.groupby("road_id", sort=False):
        original = df.loc[group.index, "collected_at_wib"]
        chronological_flags.append(original.is_monotonic_increasing)

    invalid_speed = ~df["current_speed"].between(min_speed, max_speed)
    invalid_confidence = ~df["confidence"].between(0.0, 1.0)

    return ValidationReport(
        row_count=int(len(df)),
        road_count=int(df["road_id"].nunique()),
        date_range_start=timestamps.min().to_pydatetime() if timestamps.notna().any() else None,
        date_range_end=timestamps.max().to_pydatetime() if timestamps.notna().any() else None,
        missing_values={column: int(value) for column, value in df.isna().sum().to_dict().items()},
        invalid_speed_count=int(invalid_speed.sum()),
        invalid_confidence_count=int(invalid_confidence.sum()),
        duplicate_count=duplicate_count,
        is_chronological_per_road=all(chronological_flags) if chronological_flags else True,
    )


def validate_traffic_values(report: ValidationReport) -> None:
    if report.invalid_speed_count:
        raise ValidationError(f"traffic data has {report.invalid_speed_count} invalid speed values")
    if report.invalid_confidence_count:
        raise ValidationError(f"traffic data has {report.invalid_confidence_count} invalid confidence values")

