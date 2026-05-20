class TrafficPredictionError(Exception):
    """Base exception for the traffic prediction system."""


class ValidationError(TrafficPredictionError):
    """Raised when input data violates a required schema or value rule."""


class DataLeakageError(TrafficPredictionError):
    """Raised when a pipeline step may use future or evaluation data improperly."""

