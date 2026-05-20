from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    lat1_rad, lon1_rad = np.radians([lat1, lon1])
    lat2_rad, lon2_rad = np.radians([lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def build_neighbor_mapping(roads: pd.DataFrame, neighbor_count: int = 4) -> dict[str, list[str]]:
    required = {"road_id", "mid_lat", "mid_lon"}
    missing = required - set(roads.columns)
    if missing:
        raise ValueError(f"roads metadata missing columns for neighbor mapping: {sorted(missing)}")

    mapping: dict[str, list[str]] = {}
    records = roads[["road_id", "mid_lat", "mid_lon"]].dropna().to_dict("records")
    for road in records:
        distances = []
        for candidate in records:
            if candidate["road_id"] == road["road_id"]:
                continue
            distances.append(
                (
                    candidate["road_id"],
                    haversine_km(
                        float(road["mid_lat"]),
                        float(road["mid_lon"]),
                        float(candidate["mid_lat"]),
                        float(candidate["mid_lon"]),
                    ),
                )
            )
        distances.sort(key=lambda item: item[1])
        mapping[str(road["road_id"])] = [road_id for road_id, _ in distances[:neighbor_count]]
    return mapping

