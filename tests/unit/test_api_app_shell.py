from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import HTTPException

from traffic_prediction.api.app import AppState, create_app, get_app_state
from traffic_prediction.config.settings import load_config


def test_create_app_wires_state_routes_and_exception_handlers() -> None:
    app = create_app(load_config(project_root=".", load_dotenv_file=False))

    state = get_app_state(app)
    routes = {route.path for route in app.routes}

    assert isinstance(state, AppState)
    assert app.state.lifecycle_hooks_registered is True
    assert app.state.security_controls_registered is True
    assert "/health" in routes
    assert "/ready" in routes
    assert HTTPException in app.exception_handlers


def test_api_key_validation_is_opt_in_and_exempts_health_checks() -> None:
    config, _ = _config_with_buffer_dir()
    app = create_app(replace(config, api=replace(config.api, api_key="secret-token")))

    async def call_requests() -> tuple[httpx.Response, httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            public_health = await client.get("/health")
            missing_key = await client.get("/model/version")
            wrong_key = await client.get("/model/version", headers={"X-API-Key": "wrong"})
            valid_key = await client.get("/model/version", headers={"Authorization": "Bearer secret-token"})
            return public_health, missing_key, wrong_key, valid_key

    public_health, missing_key, wrong_key, valid_key = asyncio.run(call_requests())

    assert public_health.status_code == 200
    assert missing_key.status_code == 401
    assert missing_key.json()["error_code"] == "missing_api_key"
    assert wrong_key.status_code == 403
    assert wrong_key.json()["error_code"] == "invalid_api_key"
    assert valid_key.status_code == 200


def test_request_size_limit_rejects_large_payload_before_validation() -> None:
    config, _ = _config_with_buffer_dir()
    app = create_app(replace(config, api=replace(config.api, max_request_bytes=5)))

    async def call_predict() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/predict", json={"road_id": "SBM_BHY_01", "horizon_minutes": 15})

    response = asyncio.run(call_predict())

    assert response.status_code == 413
    assert response.json()["error_code"] == "request_too_large"


def test_rate_limit_rejects_requests_after_configured_minute_budget() -> None:
    config, _ = _config_with_buffer_dir()
    app = create_app(replace(config, api=replace(config.api, rate_limit_per_minute=1)))

    async def call_twice() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/health"), await client.get("/health")

    first, second = asyncio.run(call_twice())

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error_code"] == "rate_limited"


def test_concurrency_guard_returns_service_unavailable_when_slots_are_full() -> None:
    config, _ = _config_with_buffer_dir()
    app = create_app(replace(config, api=replace(config.api, max_concurrent_requests=1)))
    app.state.request_semaphore = asyncio.Semaphore(0)

    async def call_health() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/health")

    response = asyncio.run(call_health())

    assert response.status_code == 503
    assert response.json()["error_code"] == "concurrency_limit_reached"


def test_lifecycle_shutdown_stops_scheduler_and_persists_buffer() -> None:
    config, buffer_dir = _config_with_buffer_dir()
    app = create_app(config)

    async def run_lifecycle() -> None:
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(run_lifecycle())

    assert (buffer_dir / "live_buffer.pkl").exists()
    assert get_app_state(app).scheduler.running is False


def test_http_exception_handler_returns_structured_error() -> None:
    config, _ = _config_with_buffer_dir()
    app = create_app(config)

    @app.get("/forced-error")
    def forced_error() -> None:
        raise HTTPException(status_code=418, detail="forced")

    async def call_error() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/forced-error")

    response = asyncio.run(call_error())
    payload = response.json()

    assert response.status_code == 418
    assert payload["error_code"] == "http_error"
    assert payload["message"] == "forced"
    assert payload["details"]["status_code"] == 418
    assert payload["details"]["path"] == "/forced-error"
    assert "timestamp" in payload


def _config_with_buffer_dir():
    config = load_config(project_root=".", load_dotenv_file=False)
    buffer_dir = Path("artifacts/test_runs/api_app_shell") / uuid4().hex / "buffers"
    buffer_dir.mkdir(parents=True, exist_ok=True)
    return replace(config, paths=replace(config.paths, buffers_dir=buffer_dir.resolve())), buffer_dir
