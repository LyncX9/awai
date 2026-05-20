from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Creates offline features with the same causality rules used online."""

    def __init__(
        self,
        neighbor_mapping: dict[str, list[str]] | None = None,
        lag_steps: tuple[int, ...] = (1, 2, 4, 8),
        rolling_windows: tuple[int, ...] = (3, 6),
    ) -> None:
        self.neighbor_mapping = neighbor_mapping or {}
        self.lag_steps = lag_steps
        self.rolling_windows = rolling_windows

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = out.sort_values(["road_id", "collected_at_wib"]).reset_index(drop=True)
        out = self.add_temporal_features(out)
        out = self.add_lag_features(out)
        out = self.add_rolling_features(out)
        out = self.add_traffic_dynamics(out)
        out = self.add_spatial_features(out)
        out = self.fill_feature_edges(out)
        return out

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        ts = out["collected_at_wib"]
        out["hour_of_day"] = ts.dt.hour
        out["day_of_week"] = ts.dt.dayofweek
        out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
        out["is_morning_peak"] = ((out["hour_of_day"] >= 7) & (out["hour_of_day"] < 9)).astype(int)
        out["is_evening_peak"] = ((out["hour_of_day"] >= 17) & (out["hour_of_day"] < 19)).astype(int)
        out["is_rush_hour"] = ((out["is_morning_peak"] == 1) | (out["is_evening_peak"] == 1)).astype(int)
        out["time_since_midnight"] = ts.dt.hour * 60 + ts.dt.minute
        out["hour_sin"] = np.sin(2 * np.pi * out["hour_of_day"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour_of_day"] / 24)
        out["day_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
        out["day_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
        return out

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        grouped = out.groupby("road_id", sort=False)["current_speed"]
        for lag in self.lag_steps:
            out[f"lag_{lag}"] = grouped.shift(lag)
        return out

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        grouped = out.groupby("road_id", sort=False)["current_speed"]
        for window in self.rolling_windows:
            shifted = grouped.shift(1)
            rolling = shifted.groupby(out["road_id"], sort=False).rolling(window=window, min_periods=1)
            out[f"rolling_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
            out[f"rolling_std_{window}"] = rolling.std().reset_index(level=0, drop=True)
            out[f"rolling_min_{window}"] = rolling.min().reset_index(level=0, drop=True)
            out[f"rolling_max_{window}"] = rolling.max().reset_index(level=0, drop=True)
        return out

    def add_traffic_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["speed_delta"] = out.groupby("road_id", sort=False)["current_speed"].diff()
        out["speed_acceleration"] = out.groupby("road_id", sort=False)["speed_delta"].diff()
        out["congestion_transition_indicator"] = (out["speed_delta"] < -10).astype(int)
        shifted = out.groupby("road_id", sort=False)["current_speed"].shift(1)
        out["speed_volatility"] = (
            shifted.groupby(out["road_id"], sort=False)
            .rolling(window=4, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )
        return out

    def add_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not self.neighbor_mapping:
            out["neighbor_speed_mean"] = out["current_speed"]
            out["neighbor_speed_std"] = 0.0
            out["neighbor_speed_min"] = out["current_speed"]
            out["neighbor_speed_max"] = out["current_speed"]
            return out

        out = out.reset_index(drop=True)
        out["_row_id"] = np.arange(len(out))
        mapping_rows = [
            {"road_id": road_id, "neighbor_road_id": neighbor_id}
            for road_id, neighbors in self.neighbor_mapping.items()
            for neighbor_id in neighbors
        ]
        if not mapping_rows:
            out = out.drop(columns=["_row_id"])
            out["neighbor_speed_mean"] = out["current_speed"]
            out["neighbor_speed_std"] = 0.0
            out["neighbor_speed_min"] = out["current_speed"]
            out["neighbor_speed_max"] = out["current_speed"]
            return out

        mapping = pd.DataFrame(mapping_rows)
        expanded = out[["_row_id", "road_id", "collected_at_wib", "current_speed"]].merge(
            mapping,
            on="road_id",
            how="left",
        )
        neighbor_speeds = out[["collected_at_wib", "road_id", "current_speed"]].rename(
            columns={
                "road_id": "neighbor_road_id",
                "current_speed": "neighbor_speed",
            }
        )
        expanded = expanded.merge(
            neighbor_speeds,
            on=["collected_at_wib", "neighbor_road_id"],
            how="left",
        )
        aggregated = expanded.groupby("_row_id")["neighbor_speed"].agg(["mean", "std", "min", "max"])
        out["neighbor_speed_mean"] = aggregated["mean"].reindex(out["_row_id"]).to_numpy()
        out["neighbor_speed_std"] = aggregated["std"].reindex(out["_row_id"]).to_numpy()
        out["neighbor_speed_min"] = aggregated["min"].reindex(out["_row_id"]).to_numpy()
        out["neighbor_speed_max"] = aggregated["max"].reindex(out["_row_id"]).to_numpy()

        out["neighbor_speed_mean"] = out["neighbor_speed_mean"].fillna(out["current_speed"])
        out["neighbor_speed_std"] = out["neighbor_speed_std"].fillna(0.0)
        out["neighbor_speed_min"] = out["neighbor_speed_min"].fillna(out["current_speed"])
        out["neighbor_speed_max"] = out["neighbor_speed_max"].fillna(out["current_speed"])
        return out.drop(columns=["_row_id"])

    def fill_feature_edges(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        feature_columns = [column for column in out.columns if column not in {"road_id", "collected_at_wib"}]
        out[feature_columns] = (
            out.groupby("road_id", sort=False)[feature_columns]
            .transform(lambda group: group.ffill().bfill())
            .fillna(0)
        )
        return out
