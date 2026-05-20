from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from traffic_prediction.config.settings import ConfigError, load_config, validate_config


def test_load_config_applies_environment_overrides(monkeypatch) -> None:
    scratch = _scratch_dir("env_overrides")
    monkeypatch.setenv("ARTIFACT_DIR", str(scratch / "artifacts"))
    monkeypatch.setenv("REPORTS_DIR", str(scratch / "reports"))
    monkeypatch.setenv("API_PORT", "8088")
    monkeypatch.setenv("API_MAX_REQUEST_BYTES", "2048")
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "12")
    monkeypatch.setenv("API_MAX_CONCURRENT_REQUESTS", "3")
    monkeypatch.setenv("PREDICTION_CACHE_TTL_SECONDS", "60")
    monkeypatch.setenv("TRAINING_BATCH_SIZE", "16")
    monkeypatch.setenv("FEATURE_LAG_STEPS", "1,3,6")
    monkeypatch.setenv("TOMTOM_API_KEYS", "key-a,key-b,key-a")
    monkeypatch.setenv("ACTIVE_MODEL_VERSION", "lstm-rollback")

    config = load_config(project_root=".", load_dotenv_file=False)

    assert config.paths.artifact_dir == (scratch / "artifacts").resolve()
    assert config.paths.reports_dir == (scratch / "reports").resolve()
    assert config.api.port == 8088
    assert config.api.max_request_bytes == 2048
    assert config.api.rate_limit_per_minute == 12
    assert config.api.max_concurrent_requests == 3
    assert config.runtime.prediction_cache_ttl_seconds == 60
    assert config.training.batch_size == 16
    assert config.features.lag_steps == (1, 3, 6)
    assert config.tomtom.api_key == "key-a"
    assert config.tomtom.api_keys == ("key-a", "key-b")
    assert config.runtime.active_model_version == "lstm-rollback"


def test_load_config_reads_dotenv_when_enabled(monkeypatch) -> None:
    scratch = _scratch_dir("dotenv")
    dotenv = scratch / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "API_KEY=local-api-key",
                "TOMTOM_API_KEYS=tomtom-a,tomtom-b",
                "LOG_LEVEL=DEBUG",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("TOMTOM_API_KEYS", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    config = load_config(project_root=scratch)

    assert config.api.api_key == "local-api-key"
    assert config.tomtom.api_keys == ("tomtom-a", "tomtom-b")
    assert config.runtime.log_level == "DEBUG"


def test_validate_config_reports_startup_path_and_secret_issues() -> None:
    scratch = _scratch_dir("startup_validation")
    config = load_config(project_root=scratch, load_dotenv_file=False)

    report = validate_config(
        config,
        require_input_paths=True,
        require_tomtom_secret=True,
        require_api_secret=True,
    )

    fields = {(issue.section, issue.field) for issue in report.errors}
    assert ("paths", "dataset_dir") in fields
    assert ("paths", "traffic_csv") in fields
    assert ("paths", "roads_csv") in fields
    assert ("tomtom", "api_keys") in fields
    assert ("api", "api_key") in fields


def test_load_config_rejects_invalid_runtime_values(monkeypatch) -> None:
    monkeypatch.setenv("API_PORT", "70000")

    with pytest.raises(ConfigError, match="api.port"):
        load_config(project_root=".", load_dotenv_file=False)


def test_load_config_rejects_invalid_api_guard_values(monkeypatch) -> None:
    monkeypatch.setenv("API_MAX_CONCURRENT_REQUESTS", "0")

    with pytest.raises(ConfigError, match="api.max_concurrent_requests"):
        load_config(project_root=".", load_dotenv_file=False)


def _scratch_dir(name: str) -> Path:
    path = Path("artifacts/test_runs/config_settings") / name / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()
