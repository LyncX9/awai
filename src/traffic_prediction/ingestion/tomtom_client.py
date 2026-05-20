from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

UTC = timezone.utc
from urllib.parse import urlencode
from urllib.request import Request, urlopen


JsonTransport = Callable[[str, float], dict[str, Any]]
SleepFn = Callable[[float], None]
TimeFn = Callable[[], float]


class TomTomClientError(RuntimeError):
    """Raised when TomTom traffic data cannot be fetched or parsed."""


@dataclass(frozen=True)
class TomTomSegmentQuery:
    tomtom_segment_id: str
    latitude: float
    longitude: float
    zoom: int = 10


@dataclass(frozen=True)
class TomTomTrafficObservation:
    tomtom_segment_id: str
    current_speed: float
    confidence: float
    timestamp_utc: datetime


@dataclass(frozen=True)
class TomTomFetchResult:
    observations: list[TomTomTrafficObservation]
    errors: dict[str, str] = field(default_factory=dict)
    response_time_seconds: float = 0.0

    @property
    def success_count(self) -> int:
        return len(self.observations)

    @property
    def failure_count(self) -> int:
        return len(self.errors)


class TomTomTrafficClient:
    """Small TomTom Traffic Flow client with retry and testable transport injection."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        api_keys: list[str] | tuple[str, ...] | None = None,
        timeout_seconds: float = 5.0,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        key_cooldown_seconds: float = 300.0,
        transport: JsonTransport | None = None,
        sleep: SleepFn = time.sleep,
        time_fn: TimeFn = time.monotonic,
    ) -> None:
        keys = list(api_keys or [])
        if api_key and api_key not in keys:
            keys.insert(0, api_key)
        self.api_keys = tuple(key for key in keys if key)
        self.api_key = self.api_keys[0] if self.api_keys else None
        self._next_key_index = 0
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.key_cooldown_seconds = key_cooldown_seconds
        self.transport = transport or self._default_transport
        self.sleep = sleep
        self.time_fn = time_fn
        self._cooldown_until_by_key: dict[str, float] = {}

    def fetch_flow_segment(self, query: TomTomSegmentQuery) -> TomTomTrafficObservation:
        payload = self._request_with_retry(query)
        return self.parse_flow_segment_response(payload, query.tomtom_segment_id)

    def fetch_flow_segments(self, queries: list[TomTomSegmentQuery]) -> TomTomFetchResult:
        started_at = time.monotonic()
        observations: list[TomTomTrafficObservation] = []
        errors: dict[str, str] = {}
        for query in queries:
            try:
                observations.append(self.fetch_flow_segment(query))
            except TomTomClientError as exc:
                errors[query.tomtom_segment_id] = str(exc)
        return TomTomFetchResult(
            observations=observations,
            errors=errors,
            response_time_seconds=time.monotonic() - started_at,
        )

    def _build_url(self, query: TomTomSegmentQuery) -> str:
        return self._build_url_with_key(query, self._next_api_key())

    def _build_url_with_key(self, query: TomTomSegmentQuery, api_key: str) -> str:
        params = urlencode(
            {
                "point": f"{query.latitude},{query.longitude}",
                "unit": "KMPH",
                "key": api_key,
            }
        )
        separator = "&" if "?" in self.base_url else "?"
        return f"{self.base_url}{separator}{params}"

    def _request_with_retry(self, query: TomTomSegmentQuery) -> dict[str, Any]:
        if not self.api_keys:
            raise TomTomClientError("TOMTOM_API_KEY or TOMTOM_API_KEYS is not configured")
        last_error: Exception | None = None
        attempt_count = 0
        cooldown_only_rounds = 0
        for retry_round in range(self.max_retries + 1):
            for _ in self.api_keys:
                api_key = self._next_available_api_key()
                if api_key is None:
                    cooldown_only_rounds += 1
                    break
                attempt_count += 1
                try:
                    return self.transport(self._build_url_with_key(query, api_key), self.timeout_seconds)
                except Exception as exc:  # noqa: BLE001 - external transport errors vary by backend.
                    last_error = exc
                    if _is_limit_error(exc):
                        self._cooldown_until_by_key[api_key] = self.time_fn() + self.key_cooldown_seconds
                    continue
            if retry_round < self.max_retries:
                self.sleep(self.backoff_seconds * (2**retry_round))
        if attempt_count == 0 and cooldown_only_rounds > 0:
            raise TomTomClientError(
                f"TomTom request skipped because all {len(self.api_keys)} API key(s) are in cooldown"
            )
        raise TomTomClientError(
            f"TomTom request failed after {attempt_count} attempts across {len(self.api_keys)} API key(s): "
            f"{_redact_secret(str(last_error), self.api_keys)}"
        )

    def _next_api_key(self) -> str:
        if not self.api_keys:
            raise TomTomClientError("TOMTOM_API_KEY or TOMTOM_API_KEYS is not configured")
        key = self.api_keys[self._next_key_index % len(self.api_keys)]
        self._next_key_index = (self._next_key_index + 1) % len(self.api_keys)
        return key

    def _next_available_api_key(self) -> str | None:
        if not self.api_keys:
            raise TomTomClientError("TOMTOM_API_KEY or TOMTOM_API_KEYS is not configured")
        now = self.time_fn()
        for _ in self.api_keys:
            key = self._next_api_key()
            cooldown_until = self._cooldown_until_by_key.get(key, 0.0)
            if cooldown_until <= now:
                return key
        return None

    @staticmethod
    def _default_transport(url: str, timeout_seconds: float) -> dict[str, Any]:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - configured trusted API URL.
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def parse_flow_segment_response(
        payload: dict[str, Any],
        tomtom_segment_id: str,
        fetched_at_utc: datetime | None = None,
    ) -> TomTomTrafficObservation:
        data = payload.get("flowSegmentData", payload)
        current_speed = _required_float(data, "currentSpeed", "current_speed")
        confidence = _required_float(data, "confidence")
        timestamp = _timestamp_from_payload(data, fetched_at_utc=fetched_at_utc)
        return TomTomTrafficObservation(
            tomtom_segment_id=tomtom_segment_id,
            current_speed=current_speed,
            confidence=confidence,
            timestamp_utc=timestamp,
        )


def _required_float(data: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return float(value)
    raise TomTomClientError(f"Missing required TomTom field: {'/'.join(keys)}")


def _timestamp_from_payload(data: dict[str, Any], fetched_at_utc: datetime | None = None) -> datetime:
    value = data.get("timestamp") or data.get("lastUpdated") or data.get("last_updated")
    if value is None:
        return fetched_at_utc or datetime.now(UTC)
    if isinstance(value, datetime):
        timestamp = value
    else:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _redact_secret(message: str, secrets: tuple[str, ...]) -> str:
    redacted = message
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _is_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "429",
            "rate limit",
            "too many requests",
            "quota",
            "limit exceeded",
            "usage limit",
        )
    )
