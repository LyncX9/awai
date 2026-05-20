from datetime import timezone

UTC = timezone.utc

from traffic_prediction.config.settings import load_config
from traffic_prediction.ingestion.tomtom_client import (
    TomTomSegmentQuery,
    TomTomTrafficClient,
)


def test_tomtom_client_parses_flow_segment_payload() -> None:
    payload = {
        "flowSegmentData": {
            "currentSpeed": 27.5,
            "confidence": 0.82,
            "timestamp": "2026-05-18T00:15:00Z",
        }
    }

    observation = TomTomTrafficClient.parse_flow_segment_response(payload, "tt-segment-1")

    assert observation.tomtom_segment_id == "tt-segment-1"
    assert observation.current_speed == 27.5
    assert observation.confidence == 0.82
    assert observation.timestamp_utc.tzinfo == UTC
    assert observation.timestamp_utc.isoformat() == "2026-05-18T00:15:00+00:00"


def test_tomtom_client_retries_transient_transport_errors() -> None:
    calls: list[str] = []

    def flaky_transport(url: str, timeout_seconds: float) -> dict:
        calls.append(url)
        if len(calls) < 3:
            raise TimeoutError("temporary timeout")
        return {
            "flowSegmentData": {
                "currentSpeed": 31.0,
                "confidence": 0.9,
                "lastUpdated": "2026-05-18T00:30:00Z",
            }
        }

    client = TomTomTrafficClient(
        api_key="test-key",
        base_url="https://example.test/flow",
        max_retries=2,
        backoff_seconds=0.0,
        transport=flaky_transport,
        sleep=lambda _: None,
    )

    observation = client.fetch_flow_segment(TomTomSegmentQuery("tt-segment-2", -6.9, 106.9))

    assert len(calls) == 3
    assert "point=-6.9%2C106.9" in calls[-1]
    assert "unit=KMPH" in calls[-1]
    assert "key=test-key" in calls[-1]
    assert observation.current_speed == 31.0


def test_tomtom_client_collects_failed_segment_errors_without_crashing() -> None:
    def failing_transport(url: str, timeout_seconds: float) -> dict:
        raise TimeoutError("network down")

    client = TomTomTrafficClient(
        api_key="test-key",
        base_url="https://example.test/flow",
        max_retries=0,
        transport=failing_transport,
        sleep=lambda _: None,
    )

    result = client.fetch_flow_segments([TomTomSegmentQuery("tt-down", -6.9, 106.9)])

    assert result.success_count == 0
    assert result.failure_count == 1
    assert "tt-down" in result.errors
    assert "TomTom request failed" in result.errors["tt-down"]


def test_tomtom_config_reads_api_key_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("TOMTOM_API_KEY", "env-tomtom-key")
    monkeypatch.delenv("TOMTOM_API_KEYS", raising=False)

    config = load_config(project_root=".")

    assert config.tomtom.api_key == "env-tomtom-key"
    assert config.tomtom.api_keys[0] == "env-tomtom-key"
    assert config.tomtom.max_retries == 3
    assert config.tomtom.timeout_seconds == 5.0
    assert config.tomtom.key_cooldown_seconds == 300.0


def test_tomtom_config_reads_multiple_api_keys_from_environment(monkeypatch) -> None:
    monkeypatch.delenv("TOMTOM_API_KEY", raising=False)
    monkeypatch.setenv("TOMTOM_API_KEYS", "key-a,key-b,key-c,key-a")

    config = load_config(project_root=".")

    assert config.tomtom.api_key == "key-a"
    assert config.tomtom.api_keys == ("key-a", "key-b", "key-c")


def test_tomtom_client_requires_api_key() -> None:
    client = TomTomTrafficClient(api_key=None, base_url="https://example.test/flow")

    result = client.fetch_flow_segments([TomTomSegmentQuery("tt-missing-key", -6.9, 106.9)])

    assert result.success_count == 0
    assert result.failure_count == 1
    assert "TOMTOM_API_KEY or TOMTOM_API_KEYS is not configured" in result.errors["tt-missing-key"]


def test_tomtom_client_rotates_to_next_api_key_after_failure() -> None:
    urls: list[str] = []

    def limit_first_key(url: str, timeout_seconds: float) -> dict:
        urls.append(url)
        if "key=limited-key" in url:
            raise RuntimeError("rate limit for limited-key")
        return {
            "flowSegmentData": {
                "currentSpeed": 29.0,
                "confidence": 0.95,
                "timestamp": "2026-05-18T00:45:00Z",
            }
        }

    client = TomTomTrafficClient(
        api_key=None,
        api_keys=("limited-key", "fallback-key"),
        base_url="https://example.test/flow",
        max_retries=0,
        transport=limit_first_key,
        sleep=lambda _: None,
    )

    observation = client.fetch_flow_segment(TomTomSegmentQuery("tt-rotate", -6.9, 106.9))

    assert observation.current_speed == 29.0
    assert len(urls) == 2
    assert "key=limited-key" in urls[0]
    assert "key=fallback-key" in urls[1]


def test_tomtom_client_redacts_api_keys_in_failure_messages() -> None:
    def always_fail(url: str, timeout_seconds: float) -> dict:
        raise RuntimeError(f"limit hit while calling {url}")

    client = TomTomTrafficClient(
        api_key=None,
        api_keys=("secret-a", "secret-b"),
        base_url="https://example.test/flow",
        max_retries=0,
        transport=always_fail,
        sleep=lambda _: None,
    )

    result = client.fetch_flow_segments([TomTomSegmentQuery("tt-redact", -6.9, 106.9)])
    error = result.errors["tt-redact"]

    assert "secret-a" not in error
    assert "secret-b" not in error
    assert "[REDACTED]" in error


def test_tomtom_client_skips_limited_key_during_cooldown() -> None:
    urls: list[str] = []
    now = 10.0

    def fake_time() -> float:
        return now

    def transport(url: str, timeout_seconds: float) -> dict:
        urls.append(url)
        if "key=limited-key" in url:
            raise RuntimeError("429 rate limit for limited-key")
        return {
            "flowSegmentData": {
                "currentSpeed": 30.0,
                "confidence": 0.9,
                "timestamp": "2026-05-18T01:00:00Z",
            }
        }

    client = TomTomTrafficClient(
        api_key=None,
        api_keys=("limited-key", "fallback-key"),
        base_url="https://example.test/flow",
        max_retries=0,
        key_cooldown_seconds=60.0,
        transport=transport,
        sleep=lambda _: None,
        time_fn=fake_time,
    )

    first = client.fetch_flow_segment(TomTomSegmentQuery("tt-first", -6.9, 106.9))
    second = client.fetch_flow_segment(TomTomSegmentQuery("tt-second", -6.9, 106.9))

    assert first.current_speed == 30.0
    assert second.current_speed == 30.0
    assert "key=limited-key" in urls[0]
    assert "key=fallback-key" in urls[1]
    assert "key=fallback-key" in urls[2]
    assert len(urls) == 3


def test_tomtom_client_reports_all_keys_in_cooldown() -> None:
    def fake_time() -> float:
        return 20.0

    def transport(url: str, timeout_seconds: float) -> dict:
        raise RuntimeError("429 too many requests")

    client = TomTomTrafficClient(
        api_key=None,
        api_keys=("limited-a", "limited-b"),
        base_url="https://example.test/flow",
        max_retries=0,
        key_cooldown_seconds=60.0,
        transport=transport,
        sleep=lambda _: None,
        time_fn=fake_time,
    )

    first = client.fetch_flow_segments([TomTomSegmentQuery("tt-first", -6.9, 106.9)])
    second = client.fetch_flow_segments([TomTomSegmentQuery("tt-second", -6.9, 106.9)])

    assert "TomTom request failed" in first.errors["tt-first"]
    assert "all 2 API key(s) are in cooldown" in second.errors["tt-second"]
