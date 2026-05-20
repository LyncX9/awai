from __future__ import annotations

from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path

import pandas as pd
import pytest

from traffic_prediction.config.settings import load_config
from traffic_prediction.ingestion.tomtom_client import TomTomTrafficObservation
from traffic_prediction.ingestion.tomtom_mapping import TomTomMappingError, TomTomRoadMapper


def test_tomtom_mapper_loads_versioned_csv_and_builds_queries() -> None:
    scratch = Path("artifacts/test_runs/tomtom_mapping")
    scratch.mkdir(parents=True, exist_ok=True)
    mapping_path = scratch / "mapping.csv"
    mapping_path.write_text(
        "\n".join(
            [
                "tomtom_segment_id,road_id,latitude,longitude,version",
                "tt-001,SBM_BHY_01,-6.9177,106.9164,mapping-v1",
                "tt-002,SBM_BHY_02,-6.9133,106.9312,mapping-v1",
            ]
        ),
        encoding="utf-8",
    )

    mapper = TomTomRoadMapper.from_csv(mapping_path)
    queries = mapper.to_queries()

    assert mapper.road_id_for_segment("tt-001") == "SBM_BHY_01"
    assert [query.tomtom_segment_id for query in queries] == ["tt-001", "tt-002"]
    assert queries[0].latitude == -6.9177
    assert queries[0].longitude == 106.9164


def test_tomtom_mapper_converts_observation_to_live_record_in_jakarta_time() -> None:
    mapper = TomTomRoadMapper.from_csv(_write_single_mapping())
    observation = TomTomTrafficObservation(
        tomtom_segment_id="tt-001",
        current_speed=24.5,
        confidence=0.88,
        timestamp_utc=datetime(2026, 5, 17, 23, 45, tzinfo=UTC),
    )

    record = mapper.to_live_record(observation, received_at=datetime(2026, 5, 18, 0, 0, tzinfo=UTC))

    assert record.road_id == "SBM_BHY_01"
    assert record.current_speed == 24.5
    assert record.confidence == 0.88
    assert record.timestamp.isoformat() == "2026-05-18T06:45:00+07:00"
    assert record.freshness_indicator is not None
    assert record.freshness_indicator.total_seconds() == 900


def test_tomtom_mapper_reports_missing_and_duplicate_mappings() -> None:
    mapper = TomTomRoadMapper.from_csv(_write_single_mapping())

    with pytest.raises(TomTomMappingError, match="Missing road mapping"):
        mapper.road_id_for_segment("unknown-segment")

    with pytest.raises(TomTomMappingError, match="Duplicate TomTom segment id"):
        TomTomRoadMapper.from_csv(_write_duplicate_mapping())


def test_tomtom_mapper_can_fallback_to_roads_master_midpoints() -> None:
    config = load_config(project_root=".")
    roads = pd.read_csv(config.paths.roads_csv).head(2)

    mapper = TomTomRoadMapper.from_roads_master(roads)
    queries = mapper.to_queries()

    assert len(queries) == 2
    assert queries[0].tomtom_segment_id == roads.iloc[0]["road_id"]
    assert mapper.road_id_for_segment(roads.iloc[0]["road_id"]) == roads.iloc[0]["road_id"]


def test_tomtom_mapping_path_is_configured() -> None:
    config = load_config(project_root=".")

    assert config.paths.tomtom_mapping_csv.name == "tomtom_road_mapping.csv"


def _write_single_mapping() -> Path:
    scratch = Path("artifacts/test_runs/tomtom_mapping")
    scratch.mkdir(parents=True, exist_ok=True)
    mapping_path = scratch / "single_mapping.csv"
    mapping_path.write_text(
        "\n".join(
            [
                "tomtom_segment_id,road_id,latitude,longitude,version",
                "tt-001,SBM_BHY_01,-6.9177,106.9164,mapping-v1",
            ]
        ),
        encoding="utf-8",
    )
    return mapping_path


def _write_duplicate_mapping() -> Path:
    scratch = Path("artifacts/test_runs/tomtom_mapping")
    scratch.mkdir(parents=True, exist_ok=True)
    mapping_path = scratch / "duplicate_mapping.csv"
    mapping_path.write_text(
        "\n".join(
            [
                "tomtom_segment_id,road_id,latitude,longitude,version",
                "tt-dup,SBM_BHY_01,-6.9177,106.9164,mapping-v1",
                "tt-dup,SBM_BHY_02,-6.9133,106.9312,mapping-v1",
            ]
        ),
        encoding="utf-8",
    )
    return mapping_path
