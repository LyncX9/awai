from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from traffic_prediction.data.schemas import DataQualityReport
from traffic_prediction.inference.confidence import ConfidenceAdjuster


def test_confidence_adjuster_keeps_healthy_quality_mostly_intact() -> None:
    report = _quality_report(status="healthy", completeness=1.0)
    adjustment = ConfidenceAdjuster().adjust(
        base_confidence=0.80,
        uncertainty_margin=2.0,
        source_confidence=0.95,
        data_quality=report,
    )

    assert 0.70 < adjustment.adjusted_confidence < 0.80
    assert adjustment.reason == "source_confidence_penalty_applied"
    assert adjustment.metadata()["confidence_adjusted"] == round(adjustment.adjusted_confidence, 6)


def test_confidence_adjuster_penalizes_unavailable_quality_and_issues() -> None:
    report = replace(
        _quality_report(status="unavailable", completeness=0.40),
        quality_issues={
            "missing_roads": 10,
            "stale_roads": 5,
            "delayed_roads": 3,
            "low_confidence_roads": 2,
            "outlier_roads": 1,
            "api_failures": 4,
        },
    )
    adjustment = ConfidenceAdjuster().adjust(
        base_confidence=0.80,
        uncertainty_margin=8.0,
        source_confidence=0.50,
        data_quality=report,
    )

    assert adjustment.adjusted_confidence < 0.30
    assert adjustment.quality_penalty == 0.25
    assert adjustment.completeness_penalty > 0.0
    assert adjustment.issue_penalty > 0.0
    assert adjustment.reason == "data_quality_penalty_applied"


def test_confidence_adjuster_enforces_minimum_confidence() -> None:
    report = _quality_report(status="unavailable", completeness=0.0)
    adjustment = ConfidenceAdjuster(minimum_confidence=0.10).adjust(
        base_confidence=0.20,
        uncertainty_margin=100.0,
        source_confidence=0.0,
        data_quality=report,
    )

    assert adjustment.adjusted_confidence >= 0.10


def _quality_report(status: str, completeness: float) -> DataQualityReport:
    return DataQualityReport(
        timestamp=datetime(2026, 5, 19, 8, 0),
        completeness=completeness,
        average_confidence=0.9,
        stale_roads=[],
        missing_roads=[],
        delayed_roads=[],
        low_confidence_roads=[],
        outlier_roads=[],
        api_uptime=1.0,
        fallback_recommendation="use_live_lstm",
        quality_issues={
            "missing_roads": 0,
            "stale_roads": 0,
            "delayed_roads": 0,
            "low_confidence_roads": 0,
            "outlier_roads": 0,
            "api_failures": 0,
        },
        status=status,
    )
