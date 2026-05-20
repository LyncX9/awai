from __future__ import annotations

import numpy as np
import pytest

from traffic_prediction.data.augmentation import AugmentationParameters, AugmentationValidator, DataAugmenter
from traffic_prediction.data.exceptions import ValidationError


def test_augmenter_preserves_temporal_indicator_features() -> None:
    x, y = _sequence_arrays()
    params = AugmentationParameters(factor=3, preserve_feature_indices=(1,))

    augmented_x, augmented_y = DataAugmenter(random_seed=7).augment(
        x,
        y,
        factor=params.factor,
        preserve_feature_indices=params.preserve_feature_indices,
    )
    report = AugmentationValidator(params).validate(
        x,
        y,
        augmented_x,
        augmented_y,
        speed_feature_index=0,
    )

    assert report.status == "passed"
    assert report.augmented_sample_count == len(x) * 3
    assert np.allclose(augmented_x[:, :, 1], np.tile(x[:, :, 1], (3, 1)))
    assert "rush_hour_preservation" in {check["name"] for check in report.checks}
    assert "speed_bounds" in {check["name"] for check in report.checks}


def test_augmentation_validator_rejects_speed_bounds_violation() -> None:
    x, y = _sequence_arrays()
    augmented_x = np.concatenate([x, x.copy()], axis=0)
    augmented_y = np.concatenate([y, y.copy()], axis=0)
    augmented_x[-1, -1, 0] = 200.0
    params = AugmentationParameters(factor=2, max_speed=120.0)

    with pytest.raises(ValidationError, match="out of bounds"):
        AugmentationValidator(params).validate(
            x,
            y,
            augmented_x,
            augmented_y,
            speed_feature_index=0,
        )


def test_augmentation_validator_rejects_large_speed_jump() -> None:
    x, y = _sequence_arrays()
    augmented_x = np.concatenate([x, x.copy()], axis=0)
    augmented_y = np.concatenate([y, y.copy()], axis=0)
    augmented_x[-1, :, 0] = np.array([10.0, 90.0, 10.0, 90.0], dtype=np.float32)
    params = AugmentationParameters(factor=2, max_speed_jump=30.0)

    with pytest.raises(ValidationError, match="speed jump"):
        AugmentationValidator(params).validate(
            x,
            y,
            augmented_x,
            augmented_y,
            speed_feature_index=0,
        )


def test_augmentation_validator_rejects_changed_target_values() -> None:
    x, y = _sequence_arrays()
    augmented_x = np.concatenate([x, x.copy()], axis=0)
    augmented_y = np.concatenate([y, y.copy()], axis=0)
    augmented_y[-1, 0, 0] += 1.0
    params = AugmentationParameters(factor=2)

    with pytest.raises(ValidationError, match="targets"):
        AugmentationValidator(params).validate(x, y, augmented_x, augmented_y)


def _sequence_arrays() -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [
            [[20.0, 1.0], [21.0, 1.0], [22.0, 1.0], [23.0, 1.0]],
            [[30.0, 0.0], [31.0, 0.0], [32.0, 0.0], [33.0, 0.0]],
        ],
        dtype=np.float32,
    )
    y = np.array([[[24.0], [25.0]], [[34.0], [35.0]]], dtype=np.float32)
    return x, y
