"""
ingest.py
---------
Ingestion and normalization layer for the Unified Analytics Platform MVP.

Reads the four synthetic data files, normalizes them into the common schema:
  - Entity(entity_id, name, entity_type, source, metadata_json)
  - Event(event_id, raw_id, entity_id, source, timestamp_utc, event_type, attributes_json)

Uses vectorized pandas operations for speed on ~17K events.

Outputs:
  output/entities.parquet
  output/events.parquet

Usage:
    python ingest.py
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

from config import (
    CDR_FILE, IPDR_FILE, TX_FILE, SOCIAL_FILE,
    ENTITIES_FILE, EVENTS_FILE, OUTPUT_DIR, GROUND_TRUTH_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_ts_series(ts_series: pd.Series) -> pd.Series:
    """Parse a Series of IST timestamp strings -> UTC datetime (vectorized)."""
    parsed = pd.to_datetime(ts_series, utc=False, errors="coerce", format="mixed")
    # If timezone-naive, assume IST
    try:
        parsed = parsed.dt.tz_localize(IST, ambiguous="NaT", nonexistent="NaT")
    except TypeError:
        # Already tz-aware (mixed): convert each
        parsed = parsed.apply(
            lambda x: x.replace(tzinfo=IST) if x is not pd.NaT and x.tzinfo is None else x
        )
    return parsed.dt.tz_convert(UTC)


def make_event_id(source: str, raw_id: pd.Series) -> pd.Series:
    """Vectorized event_id generation."""
    prefix = f"EVT_{source.upper()}_"
    return prefix + raw_id.astype(str).apply(
        lambda x: hashlib.md5(f"{source}:{x}".encode()).hexdigest()[:12]
    )


def _safe_json(obj) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return "{}"


def build_attributes_json(df: pd.DataFrame, col_names: list[str]) -> pd.Series:
    """Vectorized: turn selected columns into a JSON string per row."""
    sub = df[col_names].copy()
    return sub.apply(lambda row: _safe_json(row.to_dict()), axis=1)


# ---------------------------------------------------------------------------
# CDR ingestion
# ---------------------------------------------------------------------------

def ingest_cdr(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info(f"Loading CDR from {path}")
    df = pd.read_csv(path, dtype=str)

    required = ["cdr_id", "timestamp", "calling_number", "called_number",
                 "duration_sec", "call_type", "roaming_flag",
                 "tower_id", "tower_lat", "tower_lon", "imei", "imsi", "entity_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CDR missing columns: {missing}")

    # Entities
    ent_rows = df[["entity_id"]].drop_duplicates().copy()
    ent_rows["name"]          = ""
    ent_rows["entity_type"]   = "PERSON"
    ent_rows["source"]        = "CDR"
    ent_rows["metadata_json"] = "{}"

    # Parse timestamps (vectorized)
    ts_utc = parse_ts_series(df["timestamp"])
    valid  = ts_utc.notna()
    df     = df[valid].copy()
    ts_utc = ts_utc[valid]

    # Numeric coercions
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0).astype(int)
    df["tower_lat"]    = pd.to_numeric(df["tower_lat"], errors="coerce")
    df["tower_lon"]    = pd.to_numeric(df["tower_lon"], errors="coerce")
    df["roaming_flag"] = df["roaming_flag"].str.lower().isin(["true", "1"])

    # Build attributes_json as a column
    attr_cols = ["calling_number", "called_number", "duration_sec", "call_type",
                 "roaming_flag", "tower_id", "tower_lat", "tower_lon", "imei", "imsi"]
    df["attributes_json"] = build_attributes_json(df, attr_cols)

    evt = pd.DataFrame({
        "event_id":        make_event_id("CDR", df["cdr_id"]),
        "raw_id":          df["cdr_id"].values,
        "entity_id":       df["entity_id"].values,
        "source":          "CDR",
        "timestamp_utc":   ts_utc.values,
        "event_type":      ("CDR_" + df["call_type"]).values,
        "attributes_json": df["attributes_json"].values,
    })

    log.info(f"  CDR: {len(ent_rows)} entities, {len(evt)} events")
    return ent_rows.reset_index(drop=True), evt.reset_index(drop=True)


# ---------------------------------------------------------------------------
# IPDR ingestion
# ---------------------------------------------------------------------------

def ingest_ipdr(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info(f"Loading IPDR from {path}")
    df = pd.read_csv(path, dtype=str)

    required = ["ipdr_id", "timestamp", "entity_id", "imei", "imsi",
                 "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
                 "session_duration_sec", "bytes_up", "bytes_down",
                 "service_category", "app_fingerprint"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"IPDR missing columns: {missing}")

    ent_rows = df[["entity_id"]].drop_duplicates().copy()
    ent_rows["name"]          = ""
    ent_rows["entity_type"]   = "PERSON"
    ent_rows["source"]        = "IPDR"
    ent_rows["metadata_json"] = "{}"

    ts_utc = parse_ts_series(df["timestamp"])
    valid  = ts_utc.notna()
    df     = df[valid].copy()
    ts_utc = ts_utc[valid]

    for col in ["src_port", "dst_port", "session_duration_sec", "bytes_up", "bytes_down"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    attr_cols = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol",
                 "session_duration_sec", "bytes_up", "bytes_down",
                 "service_category", "app_fingerprint", "imei", "imsi"]
    df["attributes_json"] = build_attributes_json(df, attr_cols)

    evt = pd.DataFrame({
        "event_id":        make_event_id("IPDR", df["ipdr_id"]),
        "raw_id":          df["ipdr_id"].values,
        "entity_id":       df["entity_id"].values,
        "source":          "IPDR",
        "timestamp_utc":   ts_utc.values,
        "event_type":      "IPDR_SESSION",
        "attributes_json": df["attributes_json"].values,
    })

    log.info(f"  IPDR: {len(ent_rows)} entities, {len(evt)} events")
    return ent_rows.reset_index(drop=True), evt.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Transaction ingestion
# ---------------------------------------------------------------------------

def ingest_transactions(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info(f"Loading transactions from {path}")
    df = pd.read_csv(path, dtype=str)

    required = ["tx_id", "timestamp", "entity_id", "account_id",
                 "counterparty_account", "counterparty_entity_id",
                 "amount_inr", "tx_type", "channel", "reference_note"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transactions missing columns: {missing}")

    ent_rows = df[["entity_id"]].drop_duplicates().copy()
    ent_rows["name"]          = ""
    ent_rows["entity_type"]   = "PERSON"
    ent_rows["source"]        = "TRANSACTIONS"
    ent_rows["metadata_json"] = "{}"

    ts_utc = parse_ts_series(df["timestamp"])
    valid  = ts_utc.notna()
    df     = df[valid].copy()
    ts_utc = ts_utc[valid]

    df["amount_inr"] = pd.to_numeric(df["amount_inr"], errors="coerce").fillna(0.0)

    attr_cols = ["account_id", "counterparty_account", "counterparty_entity_id",
                 "amount_inr", "tx_type", "channel", "reference_note"]
    df["attributes_json"] = build_attributes_json(df, attr_cols)

    evt = pd.DataFrame({
        "event_id":        make_event_id("TX", df["tx_id"]),
        "raw_id":          df["tx_id"].values,
        "entity_id":       df["entity_id"].values,
        "source":          "TRANSACTIONS",
        "timestamp_utc":   ts_utc.values,
        "event_type":      ("TX_" + df["tx_type"]).values,
        "attributes_json": df["attributes_json"].values,
    })

    log.info(f"  Transactions: {len(ent_rows)} entities, {len(evt)} events")
    return ent_rows.reset_index(drop=True), evt.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Social / OSINT ingestion
# ---------------------------------------------------------------------------

def ingest_social(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info(f"Loading social posts from {path}")
    with open(path, "r") as f:
        posts = json.load(f)

    if not posts:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(posts)

    # Normalise list/dict columns -> JSON strings for attributes_json
    for col in ["mentions", "hashtags", "geotag"]:
        if col not in df.columns:
            df[col] = None

    ent_ids = df["entity_id"].unique()
    ent_rows = pd.DataFrame([{
        "entity_id":    eid,
        "name":         "",
        "entity_type":  "PERSON",
        "source":       "SOCIAL",
        "metadata_json": "{}",
    } for eid in ent_ids])

    ts_utc = parse_ts_series(df["timestamp"])
    valid  = ts_utc.notna()
    df     = df[valid].copy()
    ts_utc = ts_utc[valid]

    # Build attributes_json
    def make_social_attrs(row):
        return _safe_json({
            "handle":   row.get("handle", ""),
            "text":     row.get("text", ""),
            "mentions": row.get("mentions") or [],
            "hashtags": row.get("hashtags") or [],
            "geotag":   row.get("geotag"),
        })

    df["attributes_json"] = df.apply(make_social_attrs, axis=1)

    evt = pd.DataFrame({
        "event_id":        make_event_id("SOCIAL", df["post_id"]),
        "raw_id":          df["post_id"].values,
        "entity_id":       df["entity_id"].values,
        "source":          "SOCIAL",
        "timestamp_utc":   ts_utc.values,
        "event_type":      "SOCIAL_POST",
        "attributes_json": df["attributes_json"].values,
    })

    log.info(f"  Social: {len(ent_rows)} entities, {len(evt)} events")
    return ent_rows.reset_index(drop=True), evt.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Patch entity names from ground truth
# ---------------------------------------------------------------------------

def patch_names(entities_df: pd.DataFrame, ground_truth_path: str) -> pd.DataFrame:
    if not os.path.exists(ground_truth_path):
        log.warning("Ground truth not found — names will be empty.")
        return entities_df
    with open(ground_truth_path) as f:
        gt = json.load(f)
    name_map = {e["entity_id"]: e["name"] for e in gt["entities"]}
    meta_map = {
        e["entity_id"]: _safe_json({
            k: v for k, v in e.items()
            if k not in ("entity_id", "name")
        })
        for e in gt["entities"]
    }
    entities_df = entities_df.copy()
    entities_df["name"]          = entities_df["entity_id"].map(name_map).fillna("")
    entities_df["metadata_json"] = entities_df["entity_id"].map(meta_map).fillna("{}")
    return entities_df


# ---------------------------------------------------------------------------
# Deduplicate events
# ---------------------------------------------------------------------------

def deduplicate_events(events_df: pd.DataFrame) -> pd.DataFrame:
    before = len(events_df)
    events_df = events_df.drop_duplicates(subset=["event_id"])
    after = len(events_df)
    if before != after:
        log.info(f"  Deduplication: dropped {before - after} duplicate events")
    return events_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_entities = []
    all_events   = []

    for ingest_fn, path in [
        (ingest_cdr,          CDR_FILE),
        (ingest_ipdr,         IPDR_FILE),
        (ingest_transactions, TX_FILE),
        (ingest_social,       SOCIAL_FILE),
    ]:
        ent_df, evt_df = ingest_fn(path)
        all_entities.append(ent_df)
        all_events.append(evt_df)

    # Combine and deduplicate entities
    entities_df = pd.concat(all_entities, ignore_index=True)
    entities_df = entities_df.drop_duplicates(subset=["entity_id"])
    entities_df = patch_names(entities_df, GROUND_TRUTH_FILE)
    entities_df = entities_df.reset_index(drop=True)

    # Combine and deduplicate events
    events_df = pd.concat(all_events, ignore_index=True)
    events_df = deduplicate_events(events_df)
    events_df["timestamp_utc"] = pd.to_datetime(events_df["timestamp_utc"], utc=True)
    events_df = events_df.sort_values("timestamp_utc").reset_index(drop=True)

    # Save
    entities_df.to_parquet(ENTITIES_FILE, index=False)
    events_df.to_parquet(EVENTS_FILE, index=False)

    log.info(f"OK Entities saved -> {ENTITIES_FILE}  ({len(entities_df)} rows)")
    log.info(f"OK Events saved   -> {EVENTS_FILE}  ({len(events_df):,} rows)")
    log.info(f"  Sources present: {events_df['source'].unique().tolist()}")
    log.info(f"  Date range: {events_df['timestamp_utc'].min()} -> {events_df['timestamp_utc'].max()}")

    return entities_df, events_df


if __name__ == "__main__":
    run()
