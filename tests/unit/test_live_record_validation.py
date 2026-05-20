from datetime import datetime, timedelta, timezone

UTC = timezone.utc

import pytest

from traffic_prediction.data.schemas import LiveTrafficRecord
from traffic_prediction.ingestion.buffer import LiveBufferManager
from traffic_prediction.ingestion.validation import LiveRecordValidationConfig, LiveRecordValidator


def test_live_record_validator_normalizes_timestamp_and_freshness() -> None:
    now = datetime(2026, 5, 18, 7, 0)
    record = LiveTrafficRecord("R1", 25.0, 0.9, datetime(2026, 5, 18, 6, 45))
    validator = LiveRecordValidator({"R1"})

    validated = validator.validate(record, now)

    assert validated.timestamp.tzinfo is not None
    assert validated.timestamp.utcoffset().total_seconds() == 7 * 60 * 60
    assert validated.freshness_indicator == timedelta(minutes=15)


def test_live_record_validator_rejects_unknown_road_mapping() -> None:
    now = datetime(2026, 5, 18, 7, 0, tzinfo=UTC)
    validator = LiveRecordValidator({"R1"})

    with pytest.raises(ValueError, match="Unknown road mapping"):
        validator.validate(LiveTrafficRecord("missing-road", 25.0, 0.9, now), now)


def test_live_record_validator_rejects_future_records() -> None:
    now = datetime(2026, 5, 18, 7, 0, tzinfo=UTC)
    validator = LiveRecordValidator(
        {"R1"},
        config=LiveRecordValidationConfig(future_tolerance=timedelta(minutes=1)),
    )

    with pytest.raises(ValueError, match="Future TomTom record"):
        validator.validate(LiveTrafficRecord("R1", 25.0, 0.9, now + timedelta(minutes=5)), now)


def test_live_record_validator_rejects_duplicate_buffer_timestamp() -> None:
    now = datetime(2026, 5, 18, 7, 0, tzinfo=UTC)
    buffer = LiveBufferManager()
    existing = LiveTrafficRecord("R1", 20.0, 0.9, now - timedelta(minutes=5))
    buffer.append(existing)
    validator = LiveRecordValidator({"R1"}, buffer_manager=buffer)

    with pytest.raises(ValueError, match="Duplicate TomTom record"):
        validator.validate(LiveTrafficRecord("R1", 22.0, 0.8, existing.timestamp), now)
