from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConfidenceAdjustment:
    base_confidence: float
    adjusted_confidence: float
    total_penalty: float
    uncertainty_penalty: float
    source_penalty: float
    quality_penalty: float
    completeness_penalty: float
    issue_penalty: float
    reason: str

    def metadata(self) -> dict[str, float | str]:
        return {
            "confidence_base": round(self.base_confidence, 6),
            "confidence_adjusted": round(self.adjusted_confidence, 6),
            "confidence_total_penalty": round(self.total_penalty, 6),
            "confidence_uncertainty_penalty": round(self.uncertainty_penalty, 6),
            "confidence_source_penalty": round(self.source_penalty, 6),
            "confidence_quality_penalty": round(self.quality_penalty, 6),
            "confidence_completeness_penalty": round(self.completeness_penalty, 6),
            "confidence_issue_penalty": round(self.issue_penalty, 6),
            "confidence_adjustment_reason": self.reason,
        }


class ConfidenceAdjuster:
    """Adjusts prediction confidence using uncertainty and live-data quality signals."""

    def __init__(self, minimum_confidence: float = 0.05) -> None:
        self.minimum_confidence = minimum_confidence

    def adjust(
        self,
        base_confidence: float,
        uncertainty_margin: float,
        source_confidence: float | None = None,
        data_quality: Any | None = None,
    ) -> ConfidenceAdjustment:
        base = _clamp(base_confidence)
        uncertainty_penalty = min(max(uncertainty_margin, 0.0) / 40.0, 0.25)
        source_penalty = 0.0 if source_confidence is None else max(0.0, 1.0 - _clamp(source_confidence)) * 0.20
        quality_penalty = self._quality_status_penalty(data_quality)
        completeness_penalty = self._completeness_penalty(data_quality)
        issue_penalty = self._issue_penalty(data_quality)
        total_penalty = min(
            uncertainty_penalty + source_penalty + quality_penalty + completeness_penalty + issue_penalty,
            0.90,
        )
        adjusted = max(self.minimum_confidence, base * (1.0 - total_penalty))
        return ConfidenceAdjustment(
            base_confidence=base,
            adjusted_confidence=_clamp(adjusted),
            total_penalty=total_penalty,
            uncertainty_penalty=uncertainty_penalty,
            source_penalty=source_penalty,
            quality_penalty=quality_penalty,
            completeness_penalty=completeness_penalty,
            issue_penalty=issue_penalty,
            reason=self._reason(quality_penalty, completeness_penalty, issue_penalty, source_penalty),
        )

    def _quality_status_penalty(self, data_quality: Any | None) -> float:
        status = _get_value(data_quality, "status")
        if status == "unavailable":
            return 0.25
        if status == "degraded":
            return 0.10
        return 0.0

    def _completeness_penalty(self, data_quality: Any | None) -> float:
        completeness = _get_float(data_quality, "completeness", default=1.0)
        return max(0.0, 1.0 - completeness) * 0.20

    def _issue_penalty(self, data_quality: Any | None) -> float:
        issues = _get_value(data_quality, "quality_issues") or {}
        if not isinstance(issues, dict):
            return 0.0
        issue_count = sum(max(int(value), 0) for value in issues.values() if isinstance(value, (int, float)))
        return min(issue_count * 0.015, 0.20)

    def _reason(
        self,
        quality_penalty: float,
        completeness_penalty: float,
        issue_penalty: float,
        source_penalty: float,
    ) -> str:
        if quality_penalty or completeness_penalty or issue_penalty:
            return "data_quality_penalty_applied"
        if source_penalty:
            return "source_confidence_penalty_applied"
        return "uncertainty_only"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(max(float(value), lower), upper)


def _get_value(source: Any | None, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _get_float(source: Any | None, name: str, default: float) -> float:
    value = _get_value(source, name)
    if value is None:
        return default
    return float(value)
