"""Offline and online feature engineering."""

__all__ = [
    "ExternalContextConfig",
    "ExternalContextFeatureHook",
    "FeatureEngineer",
    "NoOpExternalContextProvider",
    "OnlineFeatureEngineer",
    "OnlineFeatureQuality",
    "OnlineFeatureResult",
]


def __getattr__(name: str) -> object:
    if name in {"ExternalContextConfig", "ExternalContextFeatureHook", "NoOpExternalContextProvider"}:
        from traffic_prediction.features.external_context import (
            ExternalContextConfig,
            ExternalContextFeatureHook,
            NoOpExternalContextProvider,
        )

        return {
            "ExternalContextConfig": ExternalContextConfig,
            "ExternalContextFeatureHook": ExternalContextFeatureHook,
            "NoOpExternalContextProvider": NoOpExternalContextProvider,
        }[name]
    if name == "FeatureEngineer":
        from traffic_prediction.features.offline import FeatureEngineer

        return FeatureEngineer
    if name in {"OnlineFeatureEngineer", "OnlineFeatureQuality", "OnlineFeatureResult"}:
        from traffic_prediction.features.online import OnlineFeatureEngineer, OnlineFeatureQuality, OnlineFeatureResult

        return {
            "OnlineFeatureEngineer": OnlineFeatureEngineer,
            "OnlineFeatureQuality": OnlineFeatureQuality,
            "OnlineFeatureResult": OnlineFeatureResult,
        }[name]
    raise AttributeError(f"module 'traffic_prediction.features' has no attribute {name!r}")
