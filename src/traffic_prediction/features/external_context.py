from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from traffic_prediction.data.schemas import FeatureManifest


class ExternalContextProvider(Protocol):
    feature_columns: tuple[str, ...]

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


@dataclass(frozen=True)
class ExternalContextConfig:
    weather_enabled: bool = False
    holidays_enabled: bool = False
    incidents_enabled: bool = False

    @property
    def any_enabled(self) -> bool:
        return self.weather_enabled or self.holidays_enabled or self.incidents_enabled


class NoOpExternalContextProvider:
    feature_columns: tuple[str, ...] = ()

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()


class ExternalContextFeatureHook:
    """Optional weather, holiday, and incident feature hooks with manifest protection."""

    def __init__(
        self,
        config: ExternalContextConfig | None = None,
        weather_provider: ExternalContextProvider | None = None,
        holiday_provider: ExternalContextProvider | None = None,
        incident_provider: ExternalContextProvider | None = None,
    ) -> None:
        self.config = config or ExternalContextConfig()
        self.weather_provider = weather_provider or NoOpExternalContextProvider()
        self.holiday_provider = holiday_provider or NoOpExternalContextProvider()
        self.incident_provider = incident_provider or NoOpExternalContextProvider()

    def apply(self, df: pd.DataFrame, manifest: FeatureManifest | None = None) -> pd.DataFrame:
        out = df.copy()
        if not self.config.any_enabled:
            return out

        allowed_columns = set(manifest.feature_columns) if manifest is not None else set(out.columns)
        original_columns = set(out.columns)
        for enabled, provider in [
            (self.config.weather_enabled, self.weather_provider),
            (self.config.holidays_enabled, self.holiday_provider),
            (self.config.incidents_enabled, self.incident_provider),
        ]:
            if not enabled:
                continue
            out = provider.enrich(out)
            self._validate_manifest_contract(
                provider_columns=set(provider.feature_columns),
                allowed_columns=allowed_columns,
                original_columns=original_columns,
            )
        return out

    def _validate_manifest_contract(
        self,
        provider_columns: set[str],
        allowed_columns: set[str],
        original_columns: set[str],
    ) -> None:
        added_columns = provider_columns - original_columns
        unexpected = sorted(added_columns - allowed_columns)
        if unexpected:
            raise ValueError(
                "External context provider attempted to add columns not present in the feature manifest: "
                f"{unexpected}"
            )
