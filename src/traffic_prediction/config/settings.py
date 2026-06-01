from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True)
class ConfigValidationIssue:
    section: str
    field: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class ConfigValidationReport:
    issues: tuple[ConfigValidationIssue, ...]

    @property
    def errors(self) -> tuple[ConfigValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ConfigValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_ready(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            details = "; ".join(
                f"{issue.section}.{issue.field}: {issue.message}" for issue in self.errors
            )
            raise ConfigError(details)


@dataclass(frozen=True)
class PathConfig:
    dataset_dir: Path
    traffic_csv: Path
    roads_csv: Path
    tomtom_mapping_csv: Path
    artifact_dir: Path
    reports_dir: Path
    models_dir: Path
    buffers_dir: Path
    figures_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class DataConfig:
    timezone: str = "Asia/Jakarta"
    frequency: str = "15min"
    min_speed: float = 0.0
    max_speed: float = 120.0
    min_confidence: float = 0.5
    train_days: int = 18
    validation_days: int = 6
    test_days: int = 6


@dataclass(frozen=True)
class FeatureConfig:
    lookback: int = 12
    horizon: int = 4
    lag_steps: tuple[int, ...] = (1, 2, 4, 8)
    rolling_windows: tuple[int, ...] = (3, 6)
    spatial_neighbor_count: int = 4


@dataclass(frozen=True)
class TrainingConfig:
    random_seed: int = 42
    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 0.001


@dataclass(frozen=True)
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    max_request_bytes: int = 1_000_000
    rate_limit_per_minute: int = 100
    max_concurrent_requests: int = 8
    database_url: str | None = None



@dataclass(frozen=True)
class TomTomConfig:
    base_url: str = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    api_key: str | None = None
    api_keys: tuple[str, ...] = ()
    timeout_seconds: float = 5.0
    max_retries: int = 3
    backoff_seconds: float = 0.5
    key_cooldown_seconds: float = 300.0


@dataclass(frozen=True)
class RuntimeConfig:
    prediction_cache_ttl_seconds: int = 900
    ingestion_interval_seconds: int = 900
    log_level: str = "INFO"
    log_retention_days: int = 30
    scheduler_enabled: bool = False
    active_model_version: str | None = None
    # Database retention — keeps Supabase free-plan storage (0.5 GB) under control.
    # live_traffic_records: only last N hours are needed for LSTM inference (model
    # uses at most 24 timesteps = 6 h at 15-min frequency).
    # predictions: old prediction logs are only useful for short-term drift checks.
    db_live_records_retention_hours: int = 12
    db_predictions_retention_days: int = 3
    db_cleanup_interval_hours: int = 6


@dataclass(frozen=True)
class AppConfig:
    paths: PathConfig
    data: DataConfig
    features: FeatureConfig
    training: TrainingConfig
    api: ApiConfig
    tomtom: TomTomConfig
    runtime: RuntimeConfig


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _load_dotenv(root: Path) -> dict[str, str]:
    env_path = root / ".env"
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _env_value(name: str, fallback: Any, dotenv: dict[str, str]) -> Any:
    return os.getenv(name) or dotenv.get(name) or fallback


def _env_path(root: Path, name: str, fallback: Any, dotenv: dict[str, str]) -> Path:
    return _resolve_path(root, _env_value(name, fallback, dotenv))


def _env_int(name: str, fallback: Any, dotenv: dict[str, str]) -> int:
    return int(_env_value(name, fallback, dotenv))


def _env_float(name: str, fallback: Any, dotenv: dict[str, str]) -> float:
    return float(_env_value(name, fallback, dotenv))


def _env_str(name: str, fallback: Any, dotenv: dict[str, str]) -> str:
    return str(_env_value(name, fallback, dotenv))


def _env_bool(name: str, fallback: Any, dotenv: dict[str, str]) -> bool:
    value = _env_value(name, fallback, dotenv)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _split_keys(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_values = value.replace("\n", ",").split(",")
    elif isinstance(value, (list, tuple)):
        raw_values = [str(item) for item in value]
    else:
        raw_values = [str(value)]
    seen: set[str] = set()
    keys: list[str] = []
    for item in raw_values:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return tuple(keys)


def validate_config(
    config: AppConfig,
    *,
    require_input_paths: bool = True,
    require_tomtom_secret: bool = False,
    require_api_secret: bool = False,
) -> ConfigValidationReport:
    issues: list[ConfigValidationIssue] = []

    if require_input_paths:
        for field_name in ("dataset_dir", "traffic_csv", "roads_csv"):
            path = getattr(config.paths, field_name)
            if not path.exists():
                issues.append(
                    ConfigValidationIssue(
                        section="paths",
                        field=field_name,
                        message=f"required path does not exist: {path}",
                    )
                )

    for field_name in ("artifact_dir", "reports_dir", "models_dir", "buffers_dir", "figures_dir", "logs_dir"):
        path = getattr(config.paths, field_name)
        if not path.exists():
            issues.append(
                ConfigValidationIssue(
                    section="paths",
                    field=field_name,
                    message=f"runtime directory does not exist: {path}",
                )
            )

    if config.data.min_speed < 0:
        issues.append(ConfigValidationIssue("data", "min_speed", "must be greater than or equal to 0"))
    if config.data.max_speed <= config.data.min_speed:
        issues.append(ConfigValidationIssue("data", "max_speed", "must be greater than min_speed"))
    if not 0 <= config.data.min_confidence <= 1:
        issues.append(ConfigValidationIssue("data", "min_confidence", "must be between 0 and 1"))
    for field_name in ("train_days", "validation_days", "test_days"):
        if getattr(config.data, field_name) <= 0:
            issues.append(ConfigValidationIssue("data", field_name, "must be greater than 0"))

    if config.features.lookback <= 0:
        issues.append(ConfigValidationIssue("features", "lookback", "must be greater than 0"))
    if config.features.horizon <= 0:
        issues.append(ConfigValidationIssue("features", "horizon", "must be greater than 0"))
    if any(step <= 0 for step in config.features.lag_steps):
        issues.append(ConfigValidationIssue("features", "lag_steps", "all lag steps must be greater than 0"))
    if any(window <= 0 for window in config.features.rolling_windows):
        issues.append(ConfigValidationIssue("features", "rolling_windows", "all windows must be greater than 0"))
    if config.features.spatial_neighbor_count < 0:
        issues.append(
            ConfigValidationIssue("features", "spatial_neighbor_count", "must be greater than or equal to 0")
        )

    if config.training.random_seed < 0:
        issues.append(ConfigValidationIssue("training", "random_seed", "must be greater than or equal to 0"))
    if config.training.batch_size <= 0:
        issues.append(ConfigValidationIssue("training", "batch_size", "must be greater than 0"))
    if config.training.max_epochs <= 0:
        issues.append(ConfigValidationIssue("training", "max_epochs", "must be greater than 0"))
    if config.training.learning_rate <= 0:
        issues.append(ConfigValidationIssue("training", "learning_rate", "must be greater than 0"))

    if config.api.port <= 0 or config.api.port > 65535:
        issues.append(ConfigValidationIssue("api", "port", "must be between 1 and 65535"))
    if config.api.max_request_bytes <= 0:
        issues.append(ConfigValidationIssue("api", "max_request_bytes", "must be greater than 0"))
    if config.api.rate_limit_per_minute <= 0:
        issues.append(ConfigValidationIssue("api", "rate_limit_per_minute", "must be greater than 0"))
    if config.api.max_concurrent_requests <= 0:
        issues.append(ConfigValidationIssue("api", "max_concurrent_requests", "must be greater than 0"))
    if require_api_secret and not config.api.api_key:
        issues.append(ConfigValidationIssue("api", "api_key", "API_KEY is required"))

    if not config.tomtom.base_url:
        issues.append(ConfigValidationIssue("tomtom", "base_url", "must not be empty"))
    if config.tomtom.timeout_seconds <= 0:
        issues.append(ConfigValidationIssue("tomtom", "timeout_seconds", "must be greater than 0"))
    if config.tomtom.max_retries < 0:
        issues.append(ConfigValidationIssue("tomtom", "max_retries", "must be greater than or equal to 0"))
    if config.tomtom.backoff_seconds < 0:
        issues.append(ConfigValidationIssue("tomtom", "backoff_seconds", "must be greater than or equal to 0"))
    if config.tomtom.key_cooldown_seconds < 0:
        issues.append(
            ConfigValidationIssue("tomtom", "key_cooldown_seconds", "must be greater than or equal to 0")
        )
    if require_tomtom_secret and not config.tomtom.api_keys:
        issues.append(ConfigValidationIssue("tomtom", "api_keys", "TOMTOM_API_KEY or TOMTOM_API_KEYS is required"))
    elif not config.tomtom.api_keys:
        issues.append(
            ConfigValidationIssue(
                section="tomtom",
                field="api_keys",
                message="live TomTom ingestion will be unavailable until TOMTOM_API_KEY or TOMTOM_API_KEYS is set",
                severity="warning",
            )
        )

    if config.runtime.prediction_cache_ttl_seconds <= 0:
        issues.append(ConfigValidationIssue("runtime", "prediction_cache_ttl_seconds", "must be greater than 0"))
    if config.runtime.ingestion_interval_seconds <= 0:
        issues.append(ConfigValidationIssue("runtime", "ingestion_interval_seconds", "must be greater than 0"))
    if not config.runtime.log_level:
        issues.append(ConfigValidationIssue("runtime", "log_level", "must not be empty"))
    if config.runtime.log_retention_days <= 0:
        issues.append(ConfigValidationIssue("runtime", "log_retention_days", "must be greater than 0"))
    if config.runtime.active_model_version is not None and not config.runtime.active_model_version.strip():
        issues.append(ConfigValidationIssue("runtime", "active_model_version", "must not be empty when set"))

    return ConfigValidationReport(tuple(issues))


def load_config(
    config_path: str | Path | None = None,
    project_root: str | Path | None = None,
    load_dotenv_file: bool = True,
    validate: bool = True,
    require_input_paths: bool = False,
    require_tomtom_secret: bool = False,
    require_api_secret: bool = False,
) -> AppConfig:
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    dotenv = _load_dotenv(root) if load_dotenv_file else {}
    path = Path(config_path) if config_path else Path(__file__).with_name("default.yaml")
    with path.open("r", encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream)

    paths = raw.get("paths", {})
    path_config = PathConfig(
        dataset_dir=_env_path(root, "DATASET_DIR", paths["dataset_dir"], dotenv),
        traffic_csv=_env_path(root, "TRAFFIC_CSV", paths["traffic_csv"], dotenv),
        roads_csv=_env_path(root, "ROADS_CSV", paths["roads_csv"], dotenv),
        tomtom_mapping_csv=_env_path(
            root,
            "TOMTOM_MAPPING_CSV",
            paths.get("tomtom_mapping_csv", "dataset/tomtom_road_mapping.csv"),
            dotenv,
        ),
        artifact_dir=_env_path(root, "ARTIFACT_DIR", paths["artifact_dir"], dotenv),
        reports_dir=_env_path(root, "REPORTS_DIR", paths["reports_dir"], dotenv),
        models_dir=_env_path(root, "MODELS_DIR", paths["models_dir"], dotenv),
        buffers_dir=_env_path(root, "BUFFERS_DIR", paths["buffers_dir"], dotenv),
        figures_dir=_env_path(root, "FIGURES_DIR", paths.get("figures_dir", "artifacts/figures"), dotenv),
        logs_dir=_env_path(root, "LOGS_DIR", paths.get("logs_dir", "artifacts/logs"), dotenv),
    )

    data = raw.get("data", {})
    feature = raw.get("features", {})
    training = raw.get("training", {})
    api = raw.get("api", {})
    tomtom = raw.get("tomtom", {})
    runtime = raw.get("runtime", {})
    configured_model_version = _env_value(
        "ACTIVE_MODEL_VERSION",
        _env_value("MODEL_VERSION", runtime.get("active_model_version"), dotenv),
        dotenv,
    )

    tomtom_api_keys = _split_keys(_env_value("TOMTOM_API_KEYS", tomtom.get("api_keys"), dotenv))
    tomtom_api_key = _env_value("TOMTOM_API_KEY", tomtom.get("api_key"), dotenv)
    if tomtom_api_key and tomtom_api_key not in tomtom_api_keys:
        tomtom_api_keys = (str(tomtom_api_key), *tomtom_api_keys)

    app_config = AppConfig(
        paths=path_config,
        data=DataConfig(
            timezone=_env_str("DATA_TIMEZONE", data.get("timezone", DataConfig.timezone), dotenv),
            frequency=_env_str("DATA_FREQUENCY", data.get("frequency", DataConfig.frequency), dotenv),
            min_speed=_env_float("DATA_MIN_SPEED", data.get("min_speed", DataConfig.min_speed), dotenv),
            max_speed=_env_float("DATA_MAX_SPEED", data.get("max_speed", DataConfig.max_speed), dotenv),
            min_confidence=_env_float(
                "DATA_MIN_CONFIDENCE",
                data.get("min_confidence", DataConfig.min_confidence),
                dotenv,
            ),
            train_days=_env_int("DATA_TRAIN_DAYS", data.get("train_days", DataConfig.train_days), dotenv),
            validation_days=_env_int(
                "DATA_VALIDATION_DAYS",
                data.get("validation_days", DataConfig.validation_days),
                dotenv,
            ),
            test_days=_env_int("DATA_TEST_DAYS", data.get("test_days", DataConfig.test_days), dotenv),
        ),
        features=FeatureConfig(
            lookback=_env_int("FEATURE_LOOKBACK", feature.get("lookback", 12), dotenv),
            horizon=_env_int("FEATURE_HORIZON", feature.get("horizon", 4), dotenv),
            lag_steps=tuple(
                int(value) for value in _split_keys(_env_value("FEATURE_LAG_STEPS", feature.get("lag_steps", [1, 2, 4, 8]), dotenv))
            ),
            rolling_windows=tuple(
                int(value)
                for value in _split_keys(
                    _env_value("FEATURE_ROLLING_WINDOWS", feature.get("rolling_windows", [3, 6]), dotenv)
                )
            ),
            spatial_neighbor_count=_env_int(
                "FEATURE_SPATIAL_NEIGHBOR_COUNT",
                feature.get("spatial_neighbor_count", 4),
                dotenv,
            ),
        ),
        training=TrainingConfig(
            random_seed=_env_int("TRAINING_RANDOM_SEED", training.get("random_seed", 42), dotenv),
            batch_size=_env_int("TRAINING_BATCH_SIZE", training.get("batch_size", 64), dotenv),
            max_epochs=_env_int("TRAINING_MAX_EPOCHS", training.get("max_epochs", 100), dotenv),
            learning_rate=_env_float("TRAINING_LEARNING_RATE", training.get("learning_rate", 0.001), dotenv),
        ),
        api=ApiConfig(
            host=_env_str("API_HOST", api.get("host", "0.0.0.0"), dotenv),
            port=_env_int("API_PORT", api.get("port", 8000), dotenv),
            api_key=_env_value("API_KEY", api.get("api_key"), dotenv),
            max_request_bytes=_env_int(
                "API_MAX_REQUEST_BYTES",
                api.get("max_request_bytes", ApiConfig.max_request_bytes),
                dotenv,
            ),
            rate_limit_per_minute=_env_int(
                "API_RATE_LIMIT_PER_MINUTE",
                api.get("rate_limit_per_minute", ApiConfig.rate_limit_per_minute),
                dotenv,
            ),
            max_concurrent_requests=_env_int(
                "API_MAX_CONCURRENT_REQUESTS",
                api.get("max_concurrent_requests", ApiConfig.max_concurrent_requests),
                dotenv,
            ),
            database_url=_env_value("DATABASE_URL", api.get("database_url"), dotenv),
        ),
        tomtom=TomTomConfig(
            base_url=_env_str("TOMTOM_BASE_URL", tomtom.get("base_url", TomTomConfig.base_url), dotenv),
            api_key=str(tomtom_api_key) if tomtom_api_key else (tomtom_api_keys[0] if tomtom_api_keys else None),
            api_keys=tomtom_api_keys,
            timeout_seconds=_env_float("TOMTOM_TIMEOUT_SECONDS", tomtom.get("timeout_seconds", 5.0), dotenv),
            max_retries=_env_int("TOMTOM_MAX_RETRIES", tomtom.get("max_retries", 3), dotenv),
            backoff_seconds=_env_float("TOMTOM_BACKOFF_SECONDS", tomtom.get("backoff_seconds", 0.5), dotenv),
            key_cooldown_seconds=_env_float(
                "TOMTOM_KEY_COOLDOWN_SECONDS",
                tomtom.get("key_cooldown_seconds", 300.0),
                dotenv,
            ),
        ),
        runtime=RuntimeConfig(
            prediction_cache_ttl_seconds=_env_int(
                "PREDICTION_CACHE_TTL_SECONDS",
                runtime.get("prediction_cache_ttl_seconds", 900),
                dotenv,
            ),
            ingestion_interval_seconds=_env_int(
                "INGESTION_INTERVAL_SECONDS",
                runtime.get("ingestion_interval_seconds", 900),
                dotenv,
            ),
            log_level=_env_str("LOG_LEVEL", runtime.get("log_level", "INFO"), dotenv),
            log_retention_days=_env_int(
                "LOG_RETENTION_DAYS",
                runtime.get("log_retention_days", RuntimeConfig.log_retention_days),
                dotenv,
            ),
            scheduler_enabled=_env_bool(
                "SCHEDULER_ENABLED",
                runtime.get("scheduler_enabled", RuntimeConfig.scheduler_enabled),
                dotenv,
            ),
            active_model_version=str(configured_model_version).strip() if configured_model_version else None,
            db_live_records_retention_hours=_env_int(
                "DB_LIVE_RECORDS_RETENTION_HOURS",
                runtime.get("db_live_records_retention_hours", RuntimeConfig.db_live_records_retention_hours),
                dotenv,
            ),
            db_predictions_retention_days=_env_int(
                "DB_PREDICTIONS_RETENTION_DAYS",
                runtime.get("db_predictions_retention_days", RuntimeConfig.db_predictions_retention_days),
                dotenv,
            ),
            db_cleanup_interval_hours=_env_int(
                "DB_CLEANUP_INTERVAL_HOURS",
                runtime.get("db_cleanup_interval_hours", RuntimeConfig.db_cleanup_interval_hours),
                dotenv,
            ),
        ),
    )

    for directory in [
        path_config.artifact_dir,
        path_config.reports_dir,
        path_config.models_dir,
        path_config.buffers_dir,
        path_config.figures_dir,
        path_config.logs_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    if validate:
        report = validate_config(
            app_config,
            require_input_paths=require_input_paths,
            require_tomtom_secret=require_tomtom_secret,
            require_api_secret=require_api_secret,
        )
        report.raise_for_errors()

    return app_config
