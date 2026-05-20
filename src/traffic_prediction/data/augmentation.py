from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from traffic_prediction.data.exceptions import ValidationError


InverseTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class AugmentationParameters:
    factor: int = 2
    noise_std: float = 0.01
    magnitude_min: float = 0.97
    magnitude_max: float = 1.03
    max_speed_jump: float = 80.0
    min_speed: float = 0.0
    max_speed: float = 120.0
    preserve_feature_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["preserve_feature_indices"] = list(self.preserve_feature_indices)
        return payload


@dataclass(frozen=True)
class AugmentationValidationReport:
    status: str
    original_sample_count: int
    augmented_sample_count: int
    parameters: dict
    checks: tuple[dict[str, str | float | int], ...]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "original_sample_count": self.original_sample_count,
            "augmented_sample_count": self.augmented_sample_count,
            "parameters": self.parameters,
            "checks": [dict(check) for check in self.checks],
        }


class DataAugmenter:
    """Conservative sequence augmentation for the training split only."""

    def __init__(self, random_seed: int = 42) -> None:
        self.rng = np.random.default_rng(random_seed)

    def augment(
        self,
        x: np.ndarray,
        y: np.ndarray,
        factor: int = 2,
        noise_std: float = 0.01,
        magnitude_min: float = 0.97,
        magnitude_max: float = 1.03,
        preserve_feature_indices: tuple[int, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        if factor <= 1 or len(x) == 0:
            return x, y

        augmented_x = [x]
        augmented_y = [y]
        needed = factor - 1
        for _ in range(needed):
            noisy_x = x.copy()
            scale = self.rng.uniform(magnitude_min, magnitude_max, size=(len(x), 1, 1))
            noise = self.rng.normal(0, noise_std, size=noisy_x.shape)
            noisy_x = noisy_x * scale + noise
            if preserve_feature_indices:
                noisy_x[:, :, list(preserve_feature_indices)] = x[:, :, list(preserve_feature_indices)]
            augmented_x.append(noisy_x.astype(np.float32))
            augmented_y.append(y.copy())

        return np.concatenate(augmented_x, axis=0), np.concatenate(augmented_y, axis=0)


class AugmentationValidator:
    """Validates conservative train-only sequence augmentation output."""

    def __init__(self, parameters: AugmentationParameters | None = None) -> None:
        self.parameters = parameters or AugmentationParameters()

    def validate(
        self,
        original_x: np.ndarray,
        original_y: np.ndarray,
        augmented_x: np.ndarray,
        augmented_y: np.ndarray,
        *,
        speed_feature_index: int | None = None,
        inverse_speed_transform: InverseTransform | None = None,
    ) -> AugmentationValidationReport:
        checks: list[dict[str, str | float | int]] = []
        self._validate_shapes(original_x, original_y, augmented_x, augmented_y, checks)
        self._validate_finite(augmented_x, augmented_y, checks)
        self._validate_original_prefix(original_x, original_y, augmented_x, augmented_y, checks)
        self._validate_targets_preserved(original_y, augmented_y, checks)
        self._validate_preserved_features(original_x, augmented_x, checks)
        if speed_feature_index is not None:
            self._validate_speed_bounds(
                augmented_x,
                augmented_y,
                speed_feature_index,
                inverse_speed_transform,
                checks,
            )
            self._validate_speed_jumps(
                original_x,
                augmented_x,
                speed_feature_index,
                inverse_speed_transform,
                checks,
            )
        return AugmentationValidationReport(
            status="passed",
            original_sample_count=int(len(original_x)),
            augmented_sample_count=int(len(augmented_x)),
            parameters=self.parameters.to_dict(),
            checks=tuple(checks),
        )

    def _validate_shapes(
        self,
        original_x: np.ndarray,
        original_y: np.ndarray,
        augmented_x: np.ndarray,
        augmented_y: np.ndarray,
        checks: list[dict[str, str | float | int]],
    ) -> None:
        if original_x.ndim != 3:
            raise ValidationError("original_x must be a 3D sequence array")
        if original_y.ndim != 3:
            raise ValidationError("original_y must be a 3D target array")
        if augmented_x.shape[1:] != original_x.shape[1:]:
            raise ValidationError("Augmented sequence dimensions do not match original sequence dimensions")
        if augmented_y.shape[1:] != original_y.shape[1:]:
            raise ValidationError("Augmented target dimensions do not match original target dimensions")
        expected_samples = len(original_x) if self.parameters.factor <= 1 else len(original_x) * self.parameters.factor
        if len(augmented_x) != expected_samples or len(augmented_y) != expected_samples:
            raise ValidationError("Augmented sample count does not match requested augmentation factor")
        checks.append({"name": "shape_and_factor", "status": "passed", "expected_samples": int(expected_samples)})

    def _validate_finite(
        self,
        augmented_x: np.ndarray,
        augmented_y: np.ndarray,
        checks: list[dict[str, str | float | int]],
    ) -> None:
        if not np.isfinite(augmented_x).all() or not np.isfinite(augmented_y).all():
            raise ValidationError("Augmentation produced non-finite values")
        checks.append({"name": "finite_values", "status": "passed"})

    def _validate_original_prefix(
        self,
        original_x: np.ndarray,
        original_y: np.ndarray,
        augmented_x: np.ndarray,
        augmented_y: np.ndarray,
        checks: list[dict[str, str | float | int]],
    ) -> None:
        if not np.allclose(augmented_x[: len(original_x)], original_x):
            raise ValidationError("Augmentation changed original sequence prefix")
        if not np.allclose(augmented_y[: len(original_y)], original_y):
            raise ValidationError("Augmentation changed original target prefix")
        checks.append({"name": "temporal_consistency_original_prefix", "status": "passed"})

    def _validate_targets_preserved(
        self,
        original_y: np.ndarray,
        augmented_y: np.ndarray,
        checks: list[dict[str, str | float | int]],
    ) -> None:
        for start in range(0, len(augmented_y), len(original_y)):
            if not np.allclose(augmented_y[start : start + len(original_y)], original_y):
                raise ValidationError("Augmented targets must preserve original target values")
        checks.append({"name": "target_preservation", "status": "passed"})

    def _validate_preserved_features(
        self,
        original_x: np.ndarray,
        augmented_x: np.ndarray,
        checks: list[dict[str, str | float | int]],
    ) -> None:
        indices = self.parameters.preserve_feature_indices
        if not indices:
            checks.append({"name": "rush_hour_preservation", "status": "skipped"})
            return
        for index in indices:
            for start in range(0, len(augmented_x), len(original_x)):
                actual = augmented_x[start : start + len(original_x), :, index]
                expected = original_x[:, :, index]
                if not np.allclose(actual, expected):
                    raise ValidationError(f"Preserved temporal feature changed during augmentation at index {index}")
        checks.append({"name": "rush_hour_preservation", "status": "passed", "preserved_feature_count": len(indices)})

    def _validate_speed_bounds(
        self,
        augmented_x: np.ndarray,
        augmented_y: np.ndarray,
        speed_feature_index: int,
        inverse_speed_transform: InverseTransform | None,
        checks: list[dict[str, str | float | int]],
    ) -> None:
        x_speed = augmented_x[:, :, speed_feature_index].reshape(-1, 1)
        y_speed = augmented_y.reshape(-1, 1)
        if inverse_speed_transform is not None:
            x_speed = inverse_speed_transform(x_speed)
            y_speed = inverse_speed_transform(y_speed)
        min_value = float(min(np.min(x_speed), np.min(y_speed)))
        max_value = float(max(np.max(x_speed), np.max(y_speed)))
        if min_value < self.parameters.min_speed or max_value > self.parameters.max_speed:
            raise ValidationError(
                f"Augmented speed values out of bounds: min={min_value:.3f}, max={max_value:.3f}"
            )
        checks.append(
            {
                "name": "speed_bounds",
                "status": "passed",
                "min_speed": round(min_value, 6),
                "max_speed": round(max_value, 6),
            }
        )

    def _validate_speed_jumps(
        self,
        original_x: np.ndarray,
        augmented_x: np.ndarray,
        speed_feature_index: int,
        inverse_speed_transform: InverseTransform | None,
        checks: list[dict[str, str | float | int]],
    ) -> None:
        original_speed = original_x[:, :, speed_feature_index]
        augmented_speed = augmented_x[:, :, speed_feature_index]
        if inverse_speed_transform is not None:
            original_speed = inverse_speed_transform(original_speed.reshape(-1, 1)).reshape(original_speed.shape)
            augmented_speed = inverse_speed_transform(augmented_speed.reshape(-1, 1)).reshape(augmented_speed.shape)
        original_max_jump = _max_temporal_jump(original_speed)
        augmented_max_jump = _max_temporal_jump(augmented_speed)
        allowed_jump = max(float(original_max_jump) + 1e-6, self.parameters.max_speed_jump)
        if augmented_max_jump > allowed_jump:
            raise ValidationError(
                f"Augmented speed jump exceeds limit: max_jump={augmented_max_jump:.3f}, limit={allowed_jump:.3f}"
            )
        checks.append(
            {
                "name": "maximum_speed_jump",
                "status": "passed",
                "max_jump": round(float(augmented_max_jump), 6),
                "limit": round(float(allowed_jump), 6),
            }
        )


def _max_temporal_jump(values: np.ndarray) -> float:
    if values.shape[1] < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(values, axis=1))))
