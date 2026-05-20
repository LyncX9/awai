from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

UTC = timezone.utc

from traffic_prediction.config.settings import PathConfig
from traffic_prediction.data.schemas import FeatureManifest


LATEST_MODEL_POINTER = "latest_model.txt"
ARTIFACT_METADATA_KEY = "artifact_metadata"


@dataclass(frozen=True)
class ArtifactMetadata:
    artifact_type: str
    artifact_version: str
    created_at: str
    schema_version: str = "1.0"

    @classmethod
    def create(cls, artifact_type: str, artifact_version: str) -> "ArtifactMetadata":
        return cls(
            artifact_type=artifact_type,
            artifact_version=artifact_version,
            created_at=datetime.now(UTC).isoformat(),
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactLayout:
    artifact_dir: Path
    reports_dir: Path
    models_dir: Path
    buffers_dir: Path
    figures_dir: Path
    logs_dir: Path

    @classmethod
    def from_paths(cls, paths: PathConfig) -> "ArtifactLayout":
        return cls(
            artifact_dir=paths.artifact_dir,
            reports_dir=paths.reports_dir,
            models_dir=paths.models_dir,
            buffers_dir=paths.buffers_dir,
            figures_dir=paths.figures_dir,
            logs_dir=paths.logs_dir,
        )

    def ensure_directories(self) -> None:
        for path in [
            self.artifact_dir,
            self.reports_dir,
            self.models_dir,
            self.buffers_dir,
            self.figures_dir,
            self.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def model_dir(self, model_version: str) -> Path:
        return self.models_dir / model_version

    def model_checkpoint_path(self, model_version: str) -> Path:
        return self.model_dir(model_version) / "model.pt"

    def feature_manifest_path(self, model_version: str | None = None) -> Path:
        root = self.model_dir(model_version) if model_version else self.models_dir
        return root / "feature_manifest.json"

    def scaler_params_path(self, model_version: str | None = None) -> Path:
        root = self.model_dir(model_version) if model_version else self.models_dir
        return root / "scaler_params.joblib"

    def model_config_path(self, model_version: str | None = None) -> Path:
        root = self.model_dir(model_version) if model_version else self.models_dir
        return root / "model_config.json"

    def training_history_path(self, model_version: str | None = None) -> Path:
        root = self.model_dir(model_version) if model_version else self.models_dir
        return root / "training_history.json"

    def evaluation_report_path(self, model_version: str | None = None) -> Path:
        root = self.model_dir(model_version) if model_version else self.models_dir
        return root / "evaluation_report.json"

    def model_card_path(self, model_version: str | None = None) -> Path:
        root = self.model_dir(model_version) if model_version else self.models_dir
        return root / "model_card.md"

    def report_path(self, stem: str, timestamp: str, suffix: str = ".json") -> Path:
        return self.reports_dir / f"{stem}_{timestamp}{suffix}"

    def figure_dir(self, run_id: str) -> Path:
        return self.figures_dir / run_id

    def log_path(self, name: str, date_stamp: str, suffix: str = ".jsonl") -> Path:
        return self.logs_dir / f"{name}-{date_stamp}{suffix}"

    def buffer_snapshot_path(self, name: str = "live_buffer") -> Path:
        return self.buffers_dir / f"{name}.pkl"

    def latest_model_pointer_path(self) -> Path:
        return self.models_dir / LATEST_MODEL_POINTER

    def write_latest_model_pointer(self, model_path: str | Path) -> Path:
        resolved = Path(model_path).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        try:
            pointer_value = str(resolved.relative_to(self.models_dir.resolve()))
        except ValueError:
            pointer_value = str(resolved)
        pointer_path = self.latest_model_pointer_path()
        pointer_path.write_text(pointer_value, encoding="utf-8")
        return pointer_path

    def resolve_latest_model(self, registry_path: str | Path | None = None) -> Path | None:
        registry_model = self._resolve_registry_active_model(registry_path)
        if registry_model is not None:
            return registry_model

        pointer_model = self._resolve_pointer_model()
        if pointer_model is not None:
            return pointer_model

        candidates = [
            path
            for path in self.models_dir.iterdir()
            if path.is_dir() and self.is_model_artifact_dir(path)
        ] if self.models_dir.exists() else []
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime).resolve()

    @staticmethod
    def is_model_artifact_dir(path: str | Path) -> bool:
        artifact = Path(path)
        has_checkpoint = (artifact / "model.pt").exists()
        has_offline_contract = (artifact / "feature_manifest.json").exists() and (artifact / "scaler_params.joblib").exists()
        return artifact.is_dir() and (has_checkpoint or has_offline_contract)

    def _resolve_registry_active_model(self, registry_path: str | Path | None) -> Path | None:
        path = Path(registry_path) if registry_path else self.models_dir / "registry.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        active_version = payload.get("active_model_version")
        if not active_version:
            return None
        for item in payload.get("models", []):
            if item.get("model_version") != active_version:
                continue
            artifact = Path(str(item.get("artifact_path", ""))).resolve()
            if self.is_model_artifact_dir(artifact):
                return artifact
        return None

    def _resolve_pointer_model(self) -> Path | None:
        pointer = self.latest_model_pointer_path()
        if not pointer.exists():
            return None
        value = pointer.read_text(encoding="utf-8").strip()
        if not value:
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.models_dir / candidate
        candidate = candidate.resolve()
        if self.is_model_artifact_dir(candidate):
            return candidate
        return None


def timestamp_id(prefix: str, now: datetime | None = None) -> str:
    current = now or datetime.now()
    return f"{prefix}-{current.strftime('%Y%m%d-%H%M%S')}"


def with_artifact_metadata(
    payload: dict[str, Any],
    *,
    artifact_type: str,
    artifact_version: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    metadata = ArtifactMetadata(
        artifact_type=artifact_type,
        artifact_version=artifact_version,
        created_at=created_at or datetime.now(UTC).isoformat(),
    )
    return {
        ARTIFACT_METADATA_KEY: metadata.to_dict(),
        **payload,
    }


def write_json_artifact(
    path: str | Path,
    payload: dict[str, Any],
    *,
    artifact_type: str,
    artifact_version: str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            with_artifact_metadata(
                payload,
                artifact_type=artifact_type,
                artifact_version=artifact_version,
            ),
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return target


def read_json_artifact(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_model_config(layout: ArtifactLayout, model_version: str, payload: dict[str, Any]) -> Path:
    return write_json_artifact(
        layout.model_config_path(model_version),
        payload,
        artifact_type="model_config",
        artifact_version=model_version,
    )


def load_model_config(layout: ArtifactLayout, model_version: str) -> dict[str, Any]:
    return read_json_artifact(layout.model_config_path(model_version))


def save_feature_manifest(layout: ArtifactLayout, model_version: str, manifest: FeatureManifest | dict[str, Any]) -> Path:
    payload = manifest.to_dict() if isinstance(manifest, FeatureManifest) else dict(manifest)
    return write_json_artifact(
        layout.feature_manifest_path(model_version),
        payload,
        artifact_type="feature_manifest",
        artifact_version=model_version,
    )


def load_feature_manifest(layout: ArtifactLayout, model_version: str) -> FeatureManifest:
    payload = read_json_artifact(layout.feature_manifest_path(model_version))
    payload.pop(ARTIFACT_METADATA_KEY, None)
    return FeatureManifest(**payload)


def save_scaler_params(layout: ArtifactLayout, model_version: str, scaler_params: Any) -> Path:
    path = layout.scaler_params_path(model_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_params, path)
    return path


def load_scaler_params(layout: ArtifactLayout, model_version: str) -> Any:
    return joblib.load(layout.scaler_params_path(model_version))


def save_training_history(layout: ArtifactLayout, model_version: str, history: dict[str, Any] | list[dict[str, Any]]) -> Path:
    payload = {"history": history} if isinstance(history, list) else history
    return write_json_artifact(
        layout.training_history_path(model_version),
        payload,
        artifact_type="training_history",
        artifact_version=model_version,
    )


def load_training_history(layout: ArtifactLayout, model_version: str) -> dict[str, Any]:
    return read_json_artifact(layout.training_history_path(model_version))


def save_evaluation_report(layout: ArtifactLayout, model_version: str, report: dict[str, Any]) -> Path:
    return write_json_artifact(
        layout.evaluation_report_path(model_version),
        report,
        artifact_type="evaluation_report",
        artifact_version=model_version,
    )


def load_evaluation_report(layout: ArtifactLayout, model_version: str) -> dict[str, Any]:
    return read_json_artifact(layout.evaluation_report_path(model_version))


def save_model_card(layout: ArtifactLayout, model_version: str, content: str) -> Path:
    path = layout.model_card_path(model_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"<!-- artifact_type: model_card; artifact_version: {model_version} -->\n"
    path.write_text(header + content, encoding="utf-8")
    return path


def load_model_card(layout: ArtifactLayout, model_version: str) -> str:
    return layout.model_card_path(model_version).read_text(encoding="utf-8")


def save_buffer_snapshot(layout: ArtifactLayout, snapshot: Any, name: str = "live_buffer") -> Path:
    path = layout.buffer_snapshot_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(snapshot, stream)
    return path


def load_buffer_snapshot(layout: ArtifactLayout, name: str = "live_buffer") -> Any:
    with layout.buffer_snapshot_path(name).open("rb") as stream:
        return pickle.load(stream)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (datetime, Path)):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
