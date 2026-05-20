from __future__ import annotations

from datetime import datetime, timedelta

from traffic_prediction.data.schemas import LiveTrafficRecord
from traffic_prediction.ingestion.buffer import LiveBufferManager
from traffic_prediction.inference.cache import PredictionCache
from traffic_prediction.monitoring.data_quality import DataQualityMonitor


def test_live_buffer_keeps_latest_records_and_reports_stats() -> None:
    now = datetime(2026, 5, 17, 12, 0, 0)
    manager = LiveBufferManager(min_timesteps=2, max_timesteps=3)
    for index in range(4):
        manager.append(
            LiveTrafficRecord(
                road_id="R1",
                current_speed=20 + index,
                confidence=1.0,
                timestamp=now + timedelta(minutes=15 * index),
            )
        )

    latest = manager.get_latest("R1")
    assert len(latest) == 3
    assert latest[0].current_speed == 21
    assert manager.has_minimum_history("R1")


def test_data_quality_monitor_detects_missing_stale_and_low_confidence() -> None:
    now = datetime(2026, 5, 17, 12, 0, 0)
    monitor = DataQualityMonitor()
    records = [
        LiveTrafficRecord("R1", 20.0, 0.2, now - timedelta(minutes=35)),
        LiveTrafficRecord("R2", 30.0, 1.0, now),
    ]

    report = monitor.evaluate(records, expected_road_ids={"R1", "R2", "R3"}, now=now)

    assert report.status == "unavailable"
    assert report.completeness == 1 / 3
    assert report.stale_roads == ["R1"]
    assert report.missing_roads == ["R3"]
    assert report.delayed_roads == ["R1"]
    assert report.low_confidence_roads == ["R1"]
    assert report.fallback_recommendation == "use_historical_average_fallback"
    assert report.quality_issues["missing_roads"] == 1


def test_data_quality_monitor_tracks_api_uptime_and_recommendation() -> None:
    now = datetime(2026, 5, 17, 12, 0, 0)
    monitor = DataQualityMonitor()
    records = [
        LiveTrafficRecord("R1", 20.0, 0.9, now, freshness_indicator=timedelta(seconds=20)),
        LiveTrafficRecord("R2", 30.0, 0.9, now, freshness_indicator=timedelta(seconds=20)),
    ]

    monitor.record_api_result(False)
    report = monitor.evaluate(records, expected_road_ids={"R1", "R2"}, now=now, api_success=False)

    assert report.status == "unavailable"
    assert report.api_uptime == 0.0
    assert report.quality_issues["api_failures"] == 2
    assert report.fallback_recommendation == "use_historical_average_fallback"


def test_data_quality_monitor_detects_outlier_speed_change() -> None:
    now = datetime(2026, 5, 17, 12, 0, 0)
    monitor = DataQualityMonitor(outlier_speed_delta=10.0)
    first = [LiveTrafficRecord("R1", 20.0, 0.9, now)]
    second = [LiveTrafficRecord("R1", 45.0, 0.9, now + timedelta(minutes=1))]

    monitor.evaluate(first, expected_road_ids={"R1"}, now=now)
    report = monitor.evaluate(second, expected_road_ids={"R1"}, now=now + timedelta(minutes=1))

    assert report.status == "degraded"
    assert report.outlier_roads == ["R1"]
    assert report.fallback_recommendation == "use_live_prediction_with_quality_penalty"


def test_prediction_cache_expires_entries() -> None:
    now = datetime(2026, 5, 17, 12, 0, 0)
    cache = PredictionCache(ttl_seconds=60)
    key = cache.make_key("v1", "R1", 60, now)
    cache.set(key, {"ok": True}, now=now)

    assert cache.get(key, now=now + timedelta(seconds=30)) == {"ok": True}
    assert cache.get(key, now=now + timedelta(seconds=61)) is None
