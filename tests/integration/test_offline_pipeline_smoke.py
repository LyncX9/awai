from __future__ import annotations

from traffic_prediction.config.settings import load_config
from traffic_prediction.pipelines.offline import run_offline_data_pipeline


def test_offline_pipeline_can_build_artifacts_from_real_dataset() -> None:
    config = load_config(project_root=".")
    summary = run_offline_data_pipeline(config)

    assert summary["road_count"] == 50
    assert summary["feature_count"] > 10
    assert summary["sequence_shapes"]["X_train"][0] > 0
    assert summary["sequence_shapes"]["X_validation"][0] > 0
    assert summary["sequence_shapes"]["X_test"][0] > 0

