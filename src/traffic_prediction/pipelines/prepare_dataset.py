from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_prediction.config.settings import load_config
from traffic_prediction.data.augmentation import AugmentationParameters, AugmentationValidator, DataAugmenter
from traffic_prediction.data.processor import DataProcessor
from traffic_prediction.data.schemas import DatasetBundle, FeatureManifest, PipelineSummary
from traffic_prediction.features.leakage import LeakageValidator
from traffic_prediction.features.offline import FeatureEngineer
from traffic_prediction.features.spatial import build_neighbor_mapping


def run_prepare_dataset(project_root: str | Path | None = None, augment: bool = False) -> PipelineSummary:
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    config = load_config(project_root=root)
    config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    config.paths.models_dir.mkdir(parents=True, exist_ok=True)

    processor = DataProcessor(config.data, config.features)
    traffic, traffic_report = processor.load_traffic_csv(config.paths.traffic_csv)
    roads = processor.load_roads_csv(config.paths.roads_csv)

    road_report = processor.validate_roads(roads)

    cleaned = processor.clean(traffic)
    train, validation, test, split_report = processor.chronological_split(cleaned)

    neighbor_mapping = build_neighbor_mapping(roads, config.features.spatial_neighbor_count)
    feature_engineer = FeatureEngineer(
        neighbor_mapping=neighbor_mapping,
        lag_steps=config.features.lag_steps,
        rolling_windows=config.features.rolling_windows,
    )

    train_features = feature_engineer.extract_features(train)
    validation_features = feature_engineer.extract_features(validation)
    test_features = feature_engineer.extract_features(test)

    leakage = LeakageValidator()
    leakage_report = _run_leakage_gates(
        leakage=leakage,
        train=train,
        validation=validation,
        test=test,
        train_features=train_features,
        validation_features=validation_features,
        test_features=test_features,
        neighbor_mapping=neighbor_mapping,
        lag_steps=config.features.lag_steps,
        rolling_windows=config.features.rolling_windows,
        augment=augment,
    )

    feature_columns = _infer_feature_columns(train_features)
    leakage.validate_feature_columns(train_features, feature_columns)
    leakage.validate_scaler_fit_source("train")
    train_scaled = processor.fit_transform_train(train_features, feature_columns)
    validation_scaled = processor.transform_eval(validation_features)
    test_scaled = processor.transform_eval(test_features)

    manifest = FeatureManifest(
        feature_columns=feature_columns,
        target_column="current_speed",
        lookback=config.features.lookback,
        horizon=config.features.horizon,
    )

    X_train, y_train, _ = processor.create_sequences(train_scaled, manifest.feature_columns, manifest.target_column)
    X_validation, y_validation, _ = processor.create_sequences(validation_scaled, manifest.feature_columns, manifest.target_column)
    X_test, y_test, _ = processor.create_sequences(test_scaled, manifest.feature_columns, manifest.target_column)

    augmentation_report: dict[str, Any] | None = None
    if augment:
        original_X_train = X_train.copy()
        original_y_train = y_train.copy()
        preserved_indices = _preserved_temporal_feature_indices(manifest.feature_columns)
        augmentation_params = AugmentationParameters(
            factor=2,
            noise_std=0.01,
            magnitude_min=0.97,
            magnitude_max=1.03,
            max_speed_jump=80.0,
            min_speed=config.data.min_speed,
            max_speed=config.data.max_speed,
            preserve_feature_indices=preserved_indices,
        )
        augmenter = DataAugmenter(random_seed=config.training.random_seed)
        X_train, y_train = augmenter.augment(
            X_train,
            y_train,
            factor=augmentation_params.factor,
            noise_std=augmentation_params.noise_std,
            magnitude_min=augmentation_params.magnitude_min,
            magnitude_max=augmentation_params.magnitude_max,
            preserve_feature_indices=augmentation_params.preserve_feature_indices,
        )
        speed_feature_index = manifest.feature_columns.index("current_speed") if "current_speed" in manifest.feature_columns else None
        validator = AugmentationValidator(augmentation_params)
        validation_report = validator.validate(
            original_X_train,
            original_y_train,
            X_train,
            y_train,
            speed_feature_index=speed_feature_index,
            inverse_speed_transform=processor.scaler_store.inverse_transform_speed if processor.scaler_store is not None else None,
        )
        augmentation_report = validation_report.to_dict()

    bundle = DatasetBundle(
        X_train_shape=tuple(X_train.shape),
        y_train_shape=tuple(y_train.shape),
        X_validation_shape=tuple(X_validation.shape),
        y_validation_shape=tuple(y_validation.shape),
        X_test_shape=tuple(X_test.shape),
        y_test_shape=tuple(y_test.shape),
        feature_count=len(manifest.feature_columns),
        train_samples=int(len(X_train)),
        validation_samples=int(len(X_validation)),
        test_samples=int(len(X_test)),
    )

    summary = PipelineSummary(
        traffic_validation=traffic_report,
        road_validation=road_report,
        split_report=split_report,
        feature_manifest=manifest,
        dataset_bundle=bundle,
        leakage_status="passed",
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = config.paths.reports_dir / f"prepare_dataset_summary_{timestamp}.json"
    leakage_report_path = config.paths.reports_dir / f"leakage_report_{timestamp}.json"
    augmentation_report_path = config.paths.reports_dir / f"augmentation_report_{timestamp}.json"
    manifest_path = config.paths.models_dir / "feature_manifest.json"
    scaler_path = config.paths.models_dir / "scaler_params.joblib"

    _write_json(report_path, _to_jsonable(summary))
    _write_json(leakage_report_path, _to_jsonable(leakage_report))
    if augmentation_report is not None:
        _write_json(augmentation_report_path, _to_jsonable(augmentation_report))
    _write_json(manifest_path, _to_jsonable(manifest))
    if processor.scaler_store is not None:
        processor.scaler_store.save(scaler_path)

    arrays_path = config.paths.reports_dir / f"dataset_shapes_{timestamp}.npz"
    np.savez(
        arrays_path,
        X_train_shape=np.array(X_train.shape),
        y_train_shape=np.array(y_train.shape),
        X_validation_shape=np.array(X_validation.shape),
        y_validation_shape=np.array(y_validation.shape),
        X_test_shape=np.array(X_test.shape),
        y_test_shape=np.array(y_test.shape),
    )

    return summary


def _run_leakage_gates(
    leakage: LeakageValidator,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    test_features: pd.DataFrame,
    neighbor_mapping: dict[str, list[str]],
    lag_steps: tuple[int, ...],
    rolling_windows: tuple[int, ...],
    augment: bool,
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []

    def run_check(name: str, action) -> None:
        action()
        checks.append({"name": name, "status": "passed"})

    run_check("temporal_split_order", lambda: leakage.validate_temporal_split(train, validation, test))
    run_check("split_key_isolation", lambda: leakage.validate_split_isolation(train, validation, test))
    for split_name, frame in [
        ("train", train_features),
        ("validation", validation_features),
        ("test", test_features),
    ]:
        run_check(f"{split_name}_feature_monotonicity", lambda frame=frame: leakage.validate_feature_monotonicity(frame))
        run_check(f"{split_name}_lag_causality", lambda frame=frame: leakage.validate_lag_causality(frame, lag_steps))
        run_check(
            f"{split_name}_rolling_causality",
            lambda frame=frame: leakage.validate_rolling_causality(frame, rolling_windows),
        )
        run_check(
            f"{split_name}_spatial_same_timestamp",
            lambda frame=frame: leakage.validate_spatial_same_timestamp(frame, neighbor_mapping),
        )
    if augment:
        run_check("augmentation_train_only", lambda: leakage.validate_augmentation_boundary("train"))

    return {
        "created_at": datetime.now().isoformat(),
        "status": "passed",
        "checks": checks,
    }


def _infer_feature_columns(df) -> list[str]:
    excluded = {"id", "road_id", "road_name", "city", "frc", "collected_at_wib"}
    return [
        column for column in df.columns
        if column not in excluded and hasattr(df[column], "dtype") and df[column].dtype.kind in "biufc"
    ]


def _preserved_temporal_feature_indices(feature_columns: list[str]) -> tuple[int, ...]:
    preserved = {"is_weekend", "is_rush_hour", "is_morning_peak", "is_evening_peak"}
    return tuple(index for index, column in enumerate(feature_columns) if column in preserved)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LSTM-ready traffic dataset.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--augment", action="store_true", help="Apply conservative train-only augmentation.")
    args = parser.parse_args()
    summary = run_prepare_dataset(project_root=args.project_root, augment=args.augment)
    print(json.dumps(_to_jsonable(summary.dataset_bundle), indent=2))


if __name__ == "__main__":
    main()
