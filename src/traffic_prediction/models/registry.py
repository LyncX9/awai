from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from traffic_prediction.artifacts import ArtifactLayout, timestamp_id


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Metadata for one versioned model artifact."""

    model_version: str
    artifact_path: str
    created_at: str
    model_type: str = "lstm"
    framework: str = "pytorch"
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    is_active: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelRegistryEntry":
        return cls(
            model_version=str(payload["model_version"]),
            artifact_path=str(payload["artifact_path"]),
            created_at=str(payload["created_at"]),
            model_type=str(payload.get("model_type", "lstm")),
            framework=str(payload.get("framework", "pytorch")),
            metrics=dict(payload.get("metrics", {})),
            config=dict(payload.get("config", {})),
            tags=list(payload.get("tags", [])),
            is_active=bool(payload.get("is_active", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """JSON-backed local model registry for lightweight deployment."""

    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_payload({"active_model_version": None, "models": []})

    def register(
        self,
        artifact_path: str | Path,
        model_version: str | None = None,
        model_type: str = "lstm",
        framework: str = "pytorch",
        metrics: dict[str, float] | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        activate: bool = False,
    ) -> ModelRegistryEntry:
        artifact = Path(artifact_path).resolve()
        if not artifact.exists():
            raise FileNotFoundError(f"Model artifact path does not exist: {artifact}")

        version = model_version or self._unique_version(model_type)
        payload = self._read_payload()
        entries = [ModelRegistryEntry.from_dict(item) for item in payload["models"]]
        if any(entry.model_version == version for entry in entries):
            raise ValueError(f"Model version already registered: {version}")

        entry = ModelRegistryEntry(
            model_version=version,
            artifact_path=str(artifact),
            created_at=datetime.now().isoformat(),
            model_type=model_type,
            framework=framework,
            metrics=metrics or {},
            config=config or {},
            tags=tags or [],
            is_active=activate,
        )
        if activate:
            entries = [self._replace_active_flag(item, False) for item in entries]
            payload["active_model_version"] = version
        entries.append(entry)
        payload["models"] = [item.to_dict() for item in entries]
        self._write_payload(payload)
        if activate:
            self._write_latest_pointer(artifact)
        return entry

    def list_models(self) -> list[ModelRegistryEntry]:
        payload = self._read_payload()
        entries = [ModelRegistryEntry.from_dict(item) for item in payload["models"]]
        return sorted(entries, key=lambda item: item.created_at)

    def get(self, model_version: str) -> ModelRegistryEntry:
        for entry in self.list_models():
            if entry.model_version == model_version:
                return entry
        raise KeyError(f"Model version not found: {model_version}")

    def get_active(self) -> ModelRegistryEntry | None:
        payload = self._read_payload()
        active = payload.get("active_model_version")
        if not active:
            return None
        return self.get(str(active))

    def get_latest(self) -> ModelRegistryEntry | None:
        entries = self.list_models()
        if not entries:
            return None
        return max(entries, key=lambda item: item.created_at)

    def activate(self, model_version: str) -> ModelRegistryEntry:
        payload = self._read_payload()
        entries = [ModelRegistryEntry.from_dict(item) for item in payload["models"]]
        if not any(entry.model_version == model_version for entry in entries):
            raise KeyError(f"Model version not found: {model_version}")

        updated = [
            self._replace_active_flag(entry, entry.model_version == model_version)
            for entry in entries
        ]
        payload["active_model_version"] = model_version
        payload["models"] = [entry.to_dict() for entry in updated]
        self._write_payload(payload)
        activated = self.get(model_version)
        self._write_latest_pointer(Path(activated.artifact_path))
        return activated

    def rollback(self, model_version: str) -> ModelRegistryEntry:
        """Alias for activate, used when selecting a previous known-good model."""
        return self.activate(model_version)

    def resolve(self, configured_model_version: str | None = None) -> ModelRegistryEntry | None:
        """Resolve a configured model version, then active model, then latest entry."""
        if configured_model_version:
            return self.get(configured_model_version)
        return self.get_active() or self.get_latest()

    def discover_offline_artifacts(self, models_dir: str | Path) -> list[Path]:
        models_path = Path(models_dir)
        if not models_path.exists():
            return []
        artifacts = []
        for candidate in models_path.glob("offline-*"):
            if not candidate.is_dir():
                continue
            if (candidate / "feature_manifest.json").exists() and (candidate / "scaler_params.joblib").exists():
                artifacts.append(candidate.resolve())
        return sorted(artifacts, key=lambda path: path.stat().st_mtime)

    def register_discovered_offline_artifacts(self, models_dir: str | Path, activate_latest: bool = True) -> list[ModelRegistryEntry]:
        existing_versions = {entry.model_version for entry in self.list_models()}
        discovered = self.discover_offline_artifacts(models_dir)
        registered: list[ModelRegistryEntry] = []
        for artifact in discovered:
            if artifact.name in existing_versions:
                continue
            registered.append(
                self.register(
                    artifact_path=artifact,
                    model_version=artifact.name,
                    model_type="pretraining-artifacts",
                    framework="artifact-only",
                    tags=["offline-pipeline"],
                    activate=False,
                )
            )
        if activate_latest and discovered:
            self.activate(discovered[-1].name)
        return registered

    def _read_payload(self) -> dict[str, Any]:
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_latest_pointer(self, artifact_path: Path) -> None:
        layout = ArtifactLayout(
            artifact_dir=self.registry_path.parent.parent,
            reports_dir=self.registry_path.parent.parent / "reports",
            models_dir=self.registry_path.parent,
            buffers_dir=self.registry_path.parent.parent / "buffers",
            figures_dir=self.registry_path.parent.parent / "figures",
            logs_dir=self.registry_path.parent.parent / "logs",
        )
        layout.write_latest_model_pointer(artifact_path)

    def _unique_version(self, prefix: str) -> str:
        existing = {entry.model_version for entry in self.list_models()}
        candidate = timestamp_id(prefix)
        if candidate not in existing:
            return candidate
        suffix = 1
        while f"{candidate}-{suffix}" in existing:
            suffix += 1
        return f"{candidate}-{suffix}"

    @staticmethod
    def _replace_active_flag(entry: ModelRegistryEntry, is_active: bool) -> ModelRegistryEntry:
        data = entry.to_dict()
        data["is_active"] = is_active
        return ModelRegistryEntry.from_dict(data)
