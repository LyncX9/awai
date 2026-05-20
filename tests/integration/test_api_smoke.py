from __future__ import annotations

import asyncio

import httpx

from traffic_prediction.api.app import create_app
from traffic_prediction.config.settings import load_config


def test_api_health_ready_roads_and_model_version_endpoints() -> None:
    app = create_app(load_config(project_root="."))

    async def call_endpoints() -> tuple[
        httpx.Response,
        httpx.Response,
        httpx.Response,
        httpx.Response,
        httpx.Response,
        httpx.Response,
    ]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return (
                await client.get("/health"),
                await client.get("/ready"),
                await client.get("/roads"),
                await client.get("/model/version"),
                await client.get("/data-quality"),
                await client.get("/metrics"),
            )

    health, ready, roads, model, data_quality, metrics = asyncio.run(call_endpoints())
    quality_payload = data_quality.json()
    ready_payload = ready.json()
    metrics_payload = metrics.json()

    assert health.status_code == 200
    assert ready.status_code == 200
    assert roads.status_code == 200
    assert model.status_code == 200
    assert data_quality.status_code == 200
    assert metrics.status_code == 200
    assert ready_payload["ready"] is True
    assert ready_payload["model_loaded"] is True
    assert ready_payload["roads_loaded"] is True
    assert ready_payload["buffer_available"] is True
    assert ready_payload["scheduler_registered"] is True
    assert ready_payload["scheduler_running"] is False
    assert "tomtom" in ready_payload["resources"]
    assert "scheduler" in ready_payload["resources"]
    assert ready_payload["details"]["buffer"] == "live buffer available"
    assert len(roads.json()) == 50
    assert quality_payload["buffer_available"] is True
    assert quality_payload["seeded_from_history"] is True
    assert quality_payload["status"] in {"healthy", "degraded", "unavailable"}
    assert 0.0 <= quality_payload["completeness"] <= 1.0
    assert 0.0 <= quality_payload["api_uptime"] <= 1.0
    assert quality_payload["fallback_recommendation"] in {
        "use_live_lstm",
        "use_live_prediction_with_quality_penalty",
        "use_historical_average_fallback",
    }
    assert "quality_issues" in quality_payload
    assert "buffer_stats" in quality_payload
    assert metrics_payload["model_loaded"] is True
    assert metrics_payload["model_version"] is not None
    assert metrics_payload["roads_total"] == 50
    assert metrics_payload["buffer_available"] is True
    assert metrics_payload["buffer_fresh_roads"] + metrics_payload["buffer_stale_roads"] == 50
    assert 0.0 <= metrics_payload["buffer_average_fill_rate"] <= 1.0
    assert metrics_payload["scheduler_job_count"] >= 6
    assert "buffer_persistence" in metrics_payload["scheduler_jobs"]
    assert metrics_payload["data_quality_status"] in {"healthy", "degraded", "unavailable"}


def test_predict_endpoint_returns_live_lstm_response() -> None:
    """With model loaded and seeded live buffer, predict should use live_lstm_runtime."""
    app = create_app(load_config(project_root="."))

    async def call_predict() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/predict",
                json={
                    "road_id": "SBM_BHY_01",
                    "horizon_minutes": 60,
                    "requested_at": "2026-04-29T08:00:00",
                },
            )

    response = asyncio.run(call_predict())
    payload = response.json()

    assert response.status_code == 200
    assert payload["road_id"] == "SBM_BHY_01"
    assert payload["horizon_minutes"] == 60
    # Live LSTM is active when model is loaded and buffer is seeded from history
    assert payload["prediction_method"] == "live_lstm_runtime"
    assert 0.0 <= payload["predicted_speed"] <= 120.0
    assert payload["uncertainty_lower"] <= payload["predicted_speed"] <= payload["uncertainty_upper"]
    assert payload["model_version"] is not None
    assert payload["metadata"]["cache_hit"] is False
    assert 0.0 <= payload["confidence_score"] <= 1.0
    assert payload["metadata"]["model_runner_available"] is True
    assert payload["metadata"]["online_features_built"] is True
    assert "feature_quality" in payload["metadata"]
    assert payload["metadata"]["feature_quality"]["road_id"] == "SBM_BHY_01"
    assert "quality_issues" in payload["data_quality"]
    assert "confidence_adjustment_reason" in payload["metadata"]


def test_predict_endpoint_falls_back_to_historical_when_model_runner_unavailable() -> None:
    """When model_runner is None (degraded state), predict falls back to historical_average_fallback."""
    from traffic_prediction.api.app import get_app_state

    app = create_app(load_config(project_root="."))

    async def call_predict_no_runner() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Force model_runner to None to simulate degraded mode
            state = get_app_state(app)
            original_runner = state.model_runner
            state.model_runner = None
            try:
                return await client.post(
                    "/predict",
                    json={
                        "road_id": "SBM_BHY_01",
                        "horizon_minutes": 60,
                        "requested_at": "2026-04-29T08:00:00",
                    },
                )
            finally:
                state.model_runner = original_runner

    response = asyncio.run(call_predict_no_runner())
    payload = response.json()

    assert response.status_code == 200
    assert payload["road_id"] == "SBM_BHY_01"
    assert payload["prediction_method"] == "historical_average_fallback"
    assert payload["degraded"] is True
    assert payload["data_quality"]["status"] == "degraded"
    assert 0.0 <= payload["predicted_speed"] <= 120.0
    assert payload["uncertainty_lower"] <= payload["predicted_speed"] <= payload["uncertainty_upper"]
    assert payload["metadata"]["model_runner_available"] is False
    assert 0.0 <= payload["confidence_score"] <= 1.0


def test_predict_endpoint_uses_prediction_cache_for_repeated_request() -> None:
    app = create_app(load_config(project_root="."))
    request_payload = {
        "road_id": "SBM_BHY_01",
        "horizon_minutes": 60,
        "requested_at": "2026-04-29T08:00:00",
    }

    async def call_twice() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            first = await client.post("/predict", json=request_payload)
            second = await client.post("/predict", json=request_payload)
            return first, second

    first, second = asyncio.run(call_twice())
    first_payload = first.json()
    second_payload = second.json()

    assert first.status_code == 200
    assert second.status_code == 200
    assert first_payload["metadata"]["cache_hit"] is False
    assert second_payload["metadata"]["cache_hit"] is True
    assert first_payload["predicted_speed"] == second_payload["predicted_speed"]
    assert first_payload["metadata"]["cache_key"] == second_payload["metadata"]["cache_key"]


def test_model_reload_invalidates_prediction_cache() -> None:
    app = create_app(load_config(project_root="."))
    request_payload = {
        "road_id": "SBM_BHY_01",
        "horizon_minutes": 60,
        "requested_at": "2026-04-29T08:00:00",
    }

    async def call_sequence() -> tuple[httpx.Response, httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            first = await client.post("/predict", json=request_payload)
            cached = await client.post("/predict", json=request_payload)
            reload_response = await client.post("/model/reload")
            after_reload = await client.post("/predict", json=request_payload)
            return first, cached, reload_response, after_reload

    first, cached, reload_response, after_reload = asyncio.run(call_sequence())

    assert first.status_code == 200
    assert cached.status_code == 200
    assert reload_response.status_code == 200
    assert after_reload.status_code == 200
    assert first.json()["metadata"]["cache_hit"] is False
    assert cached.json()["metadata"]["cache_hit"] is True
    assert reload_response.json()["cache_invalidated"] is True
    assert reload_response.json()["model_loaded"] is True
    assert after_reload.json()["metadata"]["cache_hit"] is False


def test_manual_ingest_appends_live_buffer_and_invalidates_prediction_cache() -> None:
    app = create_app(load_config(project_root="."))
    prediction_payload = {
        "road_id": "SBM_BHY_01",
        "horizon_minutes": 60,
        "requested_at": "2026-04-29T08:00:00",
    }
    ingest_payload = {
        "records": [
            {
                "road_id": "SBM_BHY_01",
                "current_speed": 23.5,
                "confidence": 0.91,
                "timestamp": "2026-05-18T06:45:00+07:00",
            }
        ]
    }

    async def call_sequence() -> tuple[httpx.Response, httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            first = await client.post("/predict", json=prediction_payload)
            cached = await client.post("/predict", json=prediction_payload)
            ingest = await client.post("/ingest/manual", json=ingest_payload)
            after_ingest = await client.post("/predict", json=prediction_payload)
            return first, cached, ingest, after_ingest

    first, cached, ingest, after_ingest = asyncio.run(call_sequence())

    assert first.status_code == 200
    assert cached.status_code == 200
    assert ingest.status_code == 200
    assert after_ingest.status_code == 200
    assert cached.json()["metadata"]["cache_hit"] is True
    assert ingest.json()["accepted_count"] == 1
    assert ingest.json()["cache_invalidated"] is True
    assert ingest.json()["buffer_available"] is True
    assert after_ingest.json()["metadata"]["cache_hit"] is False


def test_manual_ingest_validates_records() -> None:
    app = create_app(load_config(project_root="."))

    async def call_bad_records() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            unknown_road = await client.post(
                "/ingest/manual",
                json={
                    "records": [
                        {
                            "road_id": "missing-road",
                            "current_speed": 20.0,
                            "confidence": 0.8,
                            "timestamp": "2026-05-18T06:45:00+07:00",
                        }
                    ]
                },
            )
            bad_speed = await client.post(
                "/ingest/manual",
                json={
                    "records": [
                        {
                            "road_id": "SBM_BHY_01",
                            "current_speed": 121.0,
                            "confidence": 0.8,
                            "timestamp": "2026-05-18T06:45:00+07:00",
                        }
                    ]
                },
            )
            return unknown_road, bad_speed

    unknown_road, bad_speed = asyncio.run(call_bad_records())

    assert unknown_road.status_code == 404
    assert bad_speed.status_code == 422
    assert unknown_road.json()["error_code"] == "unknown_road"
    assert bad_speed.json()["error_code"] == "invalid_request"


def test_job_trigger_runs_tomtom_ingestion_without_configured_api_key(monkeypatch) -> None:
    monkeypatch.delenv("TOMTOM_API_KEY", raising=False)
    monkeypatch.delenv("TOMTOM_API_KEYS", raising=False)
    app = create_app(load_config(project_root=".", load_dotenv_file=False))

    async def call_job() -> tuple[httpx.Response, httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            trigger = await client.post("/jobs/tomtom_ingestion/trigger")
            alias = await client.post("/ingest/tomtom")
            scheduler = await client.get("/scheduler/status")
            unknown = await client.post("/jobs/unknown_job/trigger")
            return trigger, alias, scheduler, unknown

    trigger, alias, scheduler, unknown = asyncio.run(call_job())
    payload = trigger.json()
    alias_payload = alias.json()
    scheduler_payload = scheduler.json()

    assert trigger.status_code == 200
    assert alias.status_code == 200
    assert scheduler.status_code == 200
    assert payload["job_name"] == "tomtom_ingestion"
    assert payload["status"] == "completed_with_errors"
    assert alias_payload["job_name"] == "tomtom_ingestion"
    assert alias_payload["status"] == "completed_with_errors"
    assert payload["accepted_count"] == 0
    assert payload["fetch_error_count"] == 50
    assert payload["rejected_count"] == 50
    assert payload["cache_invalidated"] is False
    assert payload["buffer_available"] is True
    assert payload["event_log_path"] is not None
    assert payload["event_log_path"].endswith(".jsonl")
    assert any("TOMTOM_API_KEY or TOMTOM_API_KEYS is not configured" in message for message in payload["errors"].values())
    assert scheduler_payload["job_count"] >= 6
    assert "tomtom_ingestion" in scheduler_payload["jobs"]
    assert unknown.status_code == 404
    assert unknown.json()["error_code"] == "unknown_job"


def test_predict_endpoint_validates_road_and_horizon() -> None:
    app = create_app(load_config(project_root="."))

    async def call_requests() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            unknown_road = await client.post(
                "/predict",
                json={"road_id": "missing-road", "horizon_minutes": 15},
            )
            bad_horizon = await client.post(
                "/predict",
                json={"road_id": "SBM_BHY_01", "horizon_minutes": 10},
            )
            return unknown_road, bad_horizon

    unknown_road, bad_horizon = asyncio.run(call_requests())

    assert unknown_road.status_code == 404
    assert bad_horizon.status_code == 422
    assert unknown_road.json()["error_code"] == "unknown_road"
    assert bad_horizon.json()["error_code"] == "invalid_horizon"


def test_predict_batch_endpoint_returns_multiple_predictions() -> None:
    app = create_app(load_config(project_root="."))

    async def call_predict_batch() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/predict/batch",
                json={
                    "predictions": [
                        {
                            "road_id": "SBM_BHY_01",
                            "horizon_minutes": 15,
                            "requested_at": "2026-04-29T08:00:00",
                        },
                        {
                            "road_id": "SBM_BHY_02",
                            "horizon_minutes": 60,
                            "requested_at": "2026-04-29T08:00:00",
                        },
                    ]
                },
            )

    response = asyncio.run(call_predict_batch())
    payload = response.json()

    assert response.status_code == 200
    assert payload["requested_count"] == 2
    assert payload["successful_count"] == 2
    assert payload["failed_count"] == 0
    assert [item["road_id"] for item in payload["predictions"]] == ["SBM_BHY_01", "SBM_BHY_02"]
    # Accept live_lstm_runtime (model loaded + buffer seeded) or historical_average_fallback (degraded)
    valid_methods = {"live_lstm_runtime", "historical_average_fallback"}
    assert all(item["prediction_method"] in valid_methods for item in payload["predictions"])
    assert all(0.0 <= item["predicted_speed"] <= 120.0 for item in payload["predictions"])


def test_predict_batch_endpoint_returns_partial_success_for_mixed_batch() -> None:
    app = create_app(load_config(project_root="."))

    async def call_predict_batch() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/predict/batch",
                json={
                    "predictions": [
                        {
                            "road_id": "SBM_BHY_01",
                            "horizon_minutes": 15,
                            "requested_at": "2026-04-29T08:00:00",
                        },
                        {
                            "road_id": "missing-road",
                            "horizon_minutes": 60,
                            "requested_at": "2026-04-29T08:00:00",
                        },
                    ]
                },
            )

    response = asyncio.run(call_predict_batch())
    payload = response.json()

    assert response.status_code == 206
    assert payload["requested_count"] == 2
    assert payload["successful_count"] == 1
    assert payload["failed_count"] == 1
    assert payload["predictions"][0]["road_id"] == "SBM_BHY_01"
    assert payload["failures"][0]["error_code"] == "unknown_road"
    assert payload["failures"][0]["details"]["index"] == 1


def test_predict_batch_endpoint_rejects_empty_batch() -> None:
    app = create_app(load_config(project_root="."))

    async def call_empty_batch() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/predict/batch", json={"predictions": []})

    response = asyncio.run(call_empty_batch())

    assert response.status_code == 422
    assert response.json()["error_code"] == "invalid_request"
