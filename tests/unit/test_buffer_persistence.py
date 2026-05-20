from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from traffic_prediction.api.app import AppState
from traffic_prediction.config.settings import AppConfig, load_config
from traffic_prediction.data.schemas import LiveTrafficRecord


def test_app_state_persists_and_restores_live_buffer() -> None:
    config = _config_with_buffer_dir("restore")
    path = config.paths.buffers_dir / "live_buffer.pkl"
    state = AppState(config)
    state.load_static_resources()
    state.live_buffer.append(
        LiveTrafficRecord(
            road_id="SBM_BHY_01",
            current_speed=22.5,
            confidence=0.9,
            timestamp=datetime.fromisoformat("2026-05-18T06:45:00+07:00"),
        )
    )

    response = state.persist_live_buffer()
    restored = AppState(config)
    restored.load_static_resources()

    assert path.exists()
    assert response.status == "completed"
    assert response.buffer_available is True
    assert restored.buffer_available is True
    assert restored.live_buffer.get_latest("SBM_BHY_01")[-1].current_speed == 22.5


def test_app_state_registers_buffer_persistence_scheduler_job() -> None:
    config = _config_with_buffer_dir("scheduler")
    state = AppState(config)
    state.load_static_resources()

    result = state.scheduler.trigger("buffer_persistence")

    assert result.status == "completed"
    assert result.result.status == "completed"
    assert (config.paths.buffers_dir / "live_buffer.pkl").exists()
    assert state.scheduler.status()["jobs"]["buffer_persistence"]["run_count"] == 1


def test_app_state_seeds_live_buffer_from_history_when_no_persisted_buffer() -> None:
    config = _config_with_buffer_dir("seed")
    state = AppState(config)

    state.load_static_resources()

    assert state.buffer_available is True
    assert state.live_buffer_seeded_from_history is True
    assert len(state.live_buffer.buffers) == 50
    assert state.live_buffer.has_minimum_history("SBM_BHY_01")
    assert len(state.live_buffer.get_latest("SBM_BHY_01")) <= state.live_buffer.max_timesteps


def _config_with_buffer_dir(name: str) -> AppConfig:
    config = load_config(project_root=".")
    buffer_dir = Path("artifacts/test_runs/buffer_persistence") / f"{name}-{uuid4().hex}"
    paths = replace(config.paths, buffers_dir=buffer_dir.resolve())
    paths.buffers_dir.mkdir(parents=True, exist_ok=True)
    return replace(config, paths=paths)
