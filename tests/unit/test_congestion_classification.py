from __future__ import annotations

import pytest

from traffic_prediction.inference.congestion import classify_congestion, congestion_details


def test_congestion_classification_boundaries() -> None:
    assert classify_congestion(75.0, 100.0) == "free_flow"
    assert classify_congestion(50.0, 100.0) == "moderate"
    assert classify_congestion(30.0, 100.0) == "congested"
    assert classify_congestion(29.9, 100.0) == "severe"


def test_congestion_details_include_ratio_and_description() -> None:
    details = congestion_details(35.0, 100.0)

    assert details.level == "congested"
    assert details.speed_ratio == pytest.approx(0.35)
    assert "slower" in details.description


def test_congestion_classification_handles_zero_free_flow_speed() -> None:
    details = congestion_details(10.0, 0.0)

    assert details.level == "free_flow"
    assert details.speed_ratio > 1.0
