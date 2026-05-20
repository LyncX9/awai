from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from traffic_prediction.config.settings import AppConfig, load_config
from traffic_prediction.data.processor import DataProcessor
from traffic_prediction.features.offline import FeatureEngineer
from traffic_prediction.features.spatial import build_neighbor_mapping


DEFAULT_EXCLUDED_FEATURE_COLUMNS = {
    "id",
    "road_id",
    "road_name",
    "city",
    "collected_at_wib",
    "frc",
}


def select_feature_columns(df) -> list[str]:
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [column for column in numeric_columns if column not in DEFAULT_EXCLUDED_FEATURE_COLUMNS]


def run_offline_data_pipeline(config: AppConfig | None = None) -> dict:
    config = config or load_config()
    processor = DataProcessor(config.data, config.features)

    traffic_raw, validation_report = processor.load_traffic_csv(config.paths.traffic_csv)
    roads = processor.load_roads_csv(config.paths.roads_csv)
    cleaned = processor.clean(traffic_raw)

    neighbor_mapping = build_neighbor_mapping(roads, neighbor_count=config.features.spatial_neighbor_count)
    engineer = FeatureEngineer(
        neighbor_mapping=neighbor_mapping,
        lag_steps=config.features.lag_steps,
        rolling_windows=config.features.rolling_windows,
    )
    featured = engineer.extract_features(cleaned)
    feature_columns = select_feature_columns(featured)
    manifest = processor.build_feature_manifest(feature_columns)

    train, validation, test, split_stats = processor.chronological_split(featured)
    train_scaled = processor.fit_transform_train(train, feature_columns)
    validation_scaled = processor.transform_eval(validation)
    test_scaled = processor.transform_eval(test)

    x_train, y_train, train_metadata = processor.create_sequences(train_scaled, feature_columns)
    x_validation, y_validation, validation_metadata = processor.create_sequences(validation_scaled, feature_columns)
    x_test, y_test, test_metadata = processor.create_sequences(test_scaled, feature_columns)

    run_id = datetime.now().strftime("offline-%Y%m%d-%H%M%S")
    report_dir = config.paths.reports_dir / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir = config.paths.models_dir / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = report_dir / "cleaned_dataset.pkl"
    featured_path = report_dir / "featured_dataset.pkl"
    scaler_path = model_dir / "scaler_params.joblib"
    manifest_path = model_dir / "feature_manifest.json"
    summary_path = report_dir / "offline_pipeline_summary.json"

    cleaned.to_pickle(cleaned_path)
    featured.to_pickle(featured_path)
    processor.scaler_store.save(scaler_path)
    processor.save_feature_manifest(manifest, manifest_path)

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "traffic_validation": validation_report.to_dict(),
        "split_statistics": asdict(split_stats),
        "road_count": int(roads["road_id"].nunique()),
        "neighbor_mapping_count": len(neighbor_mapping),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "sequence_shapes": {
            "X_train": list(x_train.shape),
            "y_train": list(y_train.shape),
            "X_validation": list(x_validation.shape),
            "y_validation": list(y_validation.shape),
            "X_test": list(x_test.shape),
            "y_test": list(y_test.shape),
        },
        "sequence_counts": {
            "train": len(train_metadata),
            "validation": len(validation_metadata),
            "test": len(test_metadata),
        },
        "artifacts": {
            "cleaned_dataset": str(cleaned_path),
            "featured_dataset": str(featured_path),
            "scaler_params": str(scaler_path),
            "feature_manifest": str(manifest_path),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    summary = run_offline_data_pipeline()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

