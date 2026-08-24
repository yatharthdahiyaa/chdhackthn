"""
tests/test_ingest.py
Unit tests for the ingestion + normalization layer.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CDR_FILE, IPDR_FILE, TX_FILE, SOCIAL_FILE,
    ENTITIES_FILE, EVENTS_FILE, OUTPUT_DIR, DATA_DIR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_data():
    """Run the full pipeline once per test module if outputs don't exist."""
    # Check if data files exist; if not, generate them
    if not all(os.path.exists(f) for f in [CDR_FILE, IPDR_FILE, TX_FILE, SOCIAL_FILE]):
        import generate_synthetic_data
        generate_synthetic_data.main()

    # Run ingestion
    if not all(os.path.exists(f) for f in [ENTITIES_FILE, EVENTS_FILE]):
        import ingest
        ingest.run()

    entities_df = pd.read_parquet(ENTITIES_FILE)
    events_df   = pd.read_parquet(EVENTS_FILE)
    return entities_df, events_df


# ---------------------------------------------------------------------------
# Schema correctness tests
# ---------------------------------------------------------------------------

class TestEntitySchema:
    def test_entities_not_empty(self, pipeline_data):
        entities_df, _ = pipeline_data
        assert len(entities_df) > 0, "Entities dataframe must not be empty"

    def test_required_columns_present(self, pipeline_data):
        entities_df, _ = pipeline_data
        required = ["entity_id", "name", "entity_type", "source", "metadata_json"]
        for col in required:
            assert col in entities_df.columns, f"Missing column: {col}"

    def test_entity_id_not_null(self, pipeline_data):
        entities_df, _ = pipeline_data
        assert entities_df["entity_id"].notna().all(), "entity_id must not be null"

    def test_entity_id_unique(self, pipeline_data):
        entities_df, _ = pipeline_data
        assert entities_df["entity_id"].is_unique, "entity_id must be unique"

    def test_entity_type_values(self, pipeline_data):
        entities_df, _ = pipeline_data
        valid_types = {"PERSON", "DEVICE", "ACCOUNT", "HANDLE"}
        types_found = set(entities_df["entity_type"].dropna().unique())
        assert types_found.issubset(valid_types | {""}), \
            f"Unexpected entity types: {types_found - valid_types}"


class TestEventSchema:
    def test_events_not_empty(self, pipeline_data):
        _, events_df = pipeline_data
        assert len(events_df) > 10_000, \
            f"Expected >10,000 events, got {len(events_df)}"

    def test_required_columns_present(self, pipeline_data):
        _, events_df = pipeline_data
        required = ["event_id", "raw_id", "entity_id", "source",
                    "timestamp_utc", "event_type", "attributes_json"]
        for col in required:
            assert col in events_df.columns, f"Missing column: {col}"

    def test_event_id_not_null(self, pipeline_data):
        _, events_df = pipeline_data
        assert events_df["event_id"].notna().all(), "event_id must not be null"

    def test_event_id_unique(self, pipeline_data):
        _, events_df = pipeline_data
        assert events_df["event_id"].is_unique, "event_id must be unique (deduplication failed)"

    def test_sources_present(self, pipeline_data):
        _, events_df = pipeline_data
        expected_sources = {"CDR", "IPDR", "TRANSACTIONS", "SOCIAL"}
        found_sources    = set(events_df["source"].unique())
        assert expected_sources == found_sources, \
            f"Expected sources {expected_sources}, got {found_sources}"

    def test_timestamps_utc(self, pipeline_data):
        _, events_df = pipeline_data
        ts = pd.to_datetime(events_df["timestamp_utc"], utc=True)
        assert ts.notna().all(), "All timestamps must be non-null"
        # All timestamps should be in UTC (no tzinfo mismatch)
        assert ts.dt.tz is not None, "Timestamps must have timezone info"

    def test_timestamps_monotonic_after_sort(self, pipeline_data):
        _, events_df = pipeline_data
        ts = pd.to_datetime(events_df["timestamp_utc"], utc=True)
        sorted_ts = ts.sort_values()
        # SIM_START = 2025-01-01 IST → 2024-12-31 18:30 UTC, so 2024 is valid
        assert sorted_ts.iloc[0].year >= 2024, "Events should not be before 2024"
        assert sorted_ts.iloc[-1].year <= 2026, "Events should not be after 2026"


    def test_attributes_valid_json(self, pipeline_data):
        _, events_df = pipeline_data
        bad = 0
        for val in events_df["attributes_json"].head(1000):
            try:
                json.loads(val)
            except Exception:
                bad += 1
        assert bad == 0, f"Found {bad} rows with invalid attributes_json"

    def test_event_type_not_null(self, pipeline_data):
        _, events_df = pipeline_data
        assert events_df["event_type"].notna().all(), "event_type must not be null"


class TestDeduplication:
    def test_no_duplicate_event_ids(self, pipeline_data):
        _, events_df = pipeline_data
        dupes = events_df["event_id"].duplicated().sum()
        assert dupes == 0, f"Found {dupes} duplicate event_ids after dedup"

    def test_no_duplicate_entity_ids(self, pipeline_data):
        entities_df, _ = pipeline_data
        dupes = entities_df["entity_id"].duplicated().sum()
        assert dupes == 0, f"Found {dupes} duplicate entity_ids"


class TestDataVolumes:
    def test_cdr_event_count(self, pipeline_data):
        _, events_df = pipeline_data
        cdr_count = (events_df["source"] == "CDR").sum()
        assert cdr_count >= 5_000, f"Expected ≥5000 CDR events, got {cdr_count}"

    def test_ipdr_event_count(self, pipeline_data):
        _, events_df = pipeline_data
        ipdr_count = (events_df["source"] == "IPDR").sum()
        assert ipdr_count >= 8_000, f"Expected ≥8000 IPDR events, got {ipdr_count}"

    def test_tx_event_count(self, pipeline_data):
        _, events_df = pipeline_data
        tx_count = (events_df["source"] == "TRANSACTIONS").sum()
        assert tx_count >= 3_000, f"Expected ≥3000 TX events, got {tx_count}"

    def test_social_event_count(self, pipeline_data):
        _, events_df = pipeline_data
        soc_count = (events_df["source"] == "SOCIAL").sum()
        assert soc_count >= 1_000, f"Expected ≥1000 social events, got {soc_count}"
