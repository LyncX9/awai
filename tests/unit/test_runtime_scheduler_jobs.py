from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from traffic_prediction.api.app import AppState
from traffic_prediction.config.settings import AppConfig, load_config


def test_app_state_registers_full_runtime_scheduler_jobs() -> None:
    state = AppState(_config_with_buffer_dir("registered"))

    state.load_static_resources()

    assert set(state.scheduler.jobs) >= {
        "tomtom_ingestion",
        "buffer_persistence",
        "prediction_cache_refresh",
        "data_quality_summary",
        "drift_check",
        "retraining_candidate",
    }
    assert all(job.enabled is False for job in state.scheduler.jobs.values())


def test_cache_refresh_data_quality_drift_and_retraining_jobs_return_summaries() -> None:
    state = AppState(_config_with_buffer_dir("summaries"))
    state.load_static_resources()

    cache = state.trigger_job("cache_refresh")
    quality = state.trigger_job("data_quality_summary")
    drift = state.trigger_job("drift_check")
    retraining = state.trigger_job("retraining_candidate")

    assert cache.status == "completed"
    assert cache.cache_invalidated is True
    assert quality.status == "completed"
    assert quality.data_quality is not None
    assert quality.data_quality["status"] in {"healthy", "degraded", "unavailable"}
    assert drift.status in {"completed", "completed_with_warnings"}
    assert drift.data_quality is not None
    assert "drift_road_count" in drift.data_quality
    assert retraining.status in {"completed", "candidate_found"}
    assert retraining.data_quality is not None
    assert "should_retrain" in retraining.data_quality


def test_scheduler_run_due_executes_enabled_non_network_jobs() -> None:
    state = AppState(_config_with_buffer_dir("enabled"))
    state.load_static_resources()

    due_at = state.scheduler.now_fn()
    for name in [
        "buffer_persistence",
        "prediction_cache_refresh",
        "data_quality_summary",
        "drift_check",
        "retraining_candidate",
    ]:
        job = state.scheduler.jobs[name]
        job.enabled = True
        job.next_run_at = due_at
    state.scheduler.jobs["tomtom_ingestion"].enabled = False

    results = state.scheduler.run_due(now=due_at)
    completed_jobs = {result.job_name for result in results if result.status == "completed"}

    assert "buffer_persistence" in completed_jobs
    assert "prediction_cache_refresh" in completed_jobs
    assert "data_quality_summary" in completed_jobs


def _config_with_buffer_dir(name: str) -> AppConfig:
    config = load_config(project_root=".", load_dotenv_file=False)
    buffer_dir = Path("artifacts/test_runs/runtime_scheduler_jobs") / f"{name}-{uuid4().hex}"
    paths = replace(config.paths, buffers_dir=buffer_dir.resolve())
    paths.buffers_dir.mkdir(parents=True, exist_ok=True)
    return replace(config, paths=paths)
