"""Optional persistence backends for lightweight deployments."""

from traffic_prediction.persistence.sqlite import SQLitePersistence
from traffic_prediction.persistence.postgresql import PostgreSQLPersistence

__all__ = ["SQLitePersistence", "PostgreSQLPersistence"]

