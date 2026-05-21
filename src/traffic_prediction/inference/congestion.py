from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CongestionClassification:
    level: str
    speed_ratio: float
    description: str


def classify_congestion(speed: float, free_flow_speed: float) -> str:
    return congestion_details(speed, free_flow_speed).level


def congestion_details(speed: float, free_flow_speed: float) -> CongestionClassification:
    ratio = max(float(speed), 0.0) / max(float(free_flow_speed), 1e-8)
    if ratio >= 0.75:
        return CongestionClassification("free_flow", ratio, "Traffic speed is close to free-flow conditions.")
    if ratio >= 0.50:
        return CongestionClassification("moderate", ratio, "Traffic is slower than free flow but still moving steadily.")
    if ratio >= 0.30:
        return CongestionClassification("congested", ratio, "Traffic is materially slower than normal conditions.")
    return CongestionClassification("severe", ratio, "Traffic is severely constrained relative to free flow.")
