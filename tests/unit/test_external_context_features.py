from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from traffic_prediction.data.schemas import FeatureManifest
from traffic_prediction.features.external_context import ExternalContextConfig, ExternalContextFeatureHook


@dataclass
class StaticProvider:
    feature_columns: tuple[str, ...]
    values: dict[str, float]

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for column, value in self.values.items():
            out[column] = value
        return out


def test_external_context_hooks_are_disabled_by_default() -> None:
    frame = pd.DataFrame({"road_id": ["R1"], "current_speed": [20.0]})
    hook = ExternalContextFeatureHook(
        weather_provider=StaticProvider(("weather_rain_mm",), {"weather_rain_mm": 2.0}),
    )

    enriched = hook.apply(frame)

    assert list(enriched.columns) == ["road_id", "current_speed"]


def test_external_context_hook_adds_manifest_declared_columns() -> None:
    frame = pd.DataFrame({"road_id": ["R1"], "current_speed": [20.0]})
    manifest = FeatureManifest(
        feature_columns=["current_speed", "weather_rain_mm"],
        target_column="current_speed",
        lookback=12,
        horizon=4,
    )
    hook = ExternalContextFeatureHook(
        config=ExternalContextConfig(weather_enabled=True),
        weather_provider=StaticProvider(("weather_rain_mm",), {"weather_rain_mm": 2.0}),
    )

    enriched = hook.apply(frame, manifest=manifest)

    assert enriched["weather_rain_mm"].tolist() == [2.0]


def test_external_context_hook_rejects_manifest_drift() -> None:
    frame = pd.DataFrame({"road_id": ["R1"], "current_speed": [20.0]})
    manifest = FeatureManifest(
        feature_columns=["current_speed"],
        target_column="current_speed",
        lookback=12,
        horizon=4,
    )
    hook = ExternalContextFeatureHook(
        config=ExternalContextConfig(incidents_enabled=True),
        incident_provider=StaticProvider(("incident_count",), {"incident_count": 1.0}),
    )

    with pytest.raises(ValueError, match="not present in the feature manifest"):
        hook.apply(frame, manifest=manifest)
