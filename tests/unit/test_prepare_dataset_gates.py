from __future__ import annotations

import pandas as pd
import pytest

from traffic_prediction.data.exceptions import DataLeakageError
from traffic_prediction.features.leakage import LeakageValidator
from traffic_prediction.features.offline import FeatureEngineer
from traffic_prediction.pipelines.prepare_dataset import _run_leakage_gates


def test_prepare_dataset_leakage_gates_report_passed_checks() -> None:
    train = _frame("2026-04-01", [10, 11, 12, 13])
    validation = _frame("2026-04-02", [14, 15, 16, 17])
    test = _frame("2026-04-03", [18, 19, 20, 21])
    engineer = FeatureEngineer(lag_steps=(1,), rolling_windows=(3,))

    report = _run_leakage_gates(
        leakage=LeakageValidator(),
        train=train,
        validation=validation,
        test=test,
        train_features=engineer.extract_features(train),
        validation_features=engineer.extract_features(validation),
        test_features=engineer.extract_features(test),
        neighbor_mapping={},
        lag_steps=(1,),
        rolling_windows=(3,),
        augment=True,
    )

    assert report["status"] == "passed"
    assert {item["status"] for item in report["checks"]} == {"passed"}
    assert "augmentation_train_only" in {item["name"] for item in report["checks"]}


def test_prepare_dataset_leakage_gates_fail_on_overlapping_split_keys() -> None:
    train = _frame("2026-04-01", [10, 11, 12, 13])
    validation = train.iloc[[0]].copy()
    test = _frame("2026-04-03", [18, 19, 20, 21])
    engineer = FeatureEngineer(lag_steps=(1,), rolling_windows=(3,))

    with pytest.raises(DataLeakageError, match="overlaps validation|share row keys"):
        _run_leakage_gates(
            leakage=LeakageValidator(),
            train=train,
            validation=validation,
            test=test,
            train_features=engineer.extract_features(train),
            validation_features=engineer.extract_features(validation),
            test_features=engineer.extract_features(test),
            neighbor_mapping={},
            lag_steps=(1,),
            rolling_windows=(3,),
            augment=False,
        )


def _frame(start: str, speeds: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(speeds), freq="15min", tz="Asia/Jakarta")
    return pd.DataFrame(
        {
            "road_id": ["R1"] * len(speeds),
            "collected_at_wib": timestamps,
            "current_speed": speeds,
            "free_flow_speed": [40.0] * len(speeds),
            "confidence": [0.9] * len(speeds),
            "speed_ratio": [0.5] * len(speeds),
        }
    )
