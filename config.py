"""
config.py — Central configuration for the Unified Analytics Platform MVP.
All parameters for dataset generation, scoring weights, and file paths live here.
"""

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Dataset sizes
# ---------------------------------------------------------------------------
NUM_ENTITIES = 50          # synthetic persons
NUM_SUSPECT_ENTITIES = 5   # planted anomaly cluster (subset of NUM_ENTITIES)
NUM_CDR_ROWS = 5_000
NUM_IPDR_ROWS = 8_000
NUM_TX_ROWS = 3_000
NUM_SOCIAL_POSTS = 1_000

# ---------------------------------------------------------------------------
# Geographic bounding box (India — Punjab/Chandigarh region)
# ---------------------------------------------------------------------------
GEO_LAT_MIN = 30.0
GEO_LAT_MAX = 31.5
GEO_LON_MIN = 76.0
GEO_LON_MAX = 77.5

# Simulated ATM cluster lat/lon pairs (used for CR-1 rule)
ATM_LOCATIONS = [
    (30.733, 76.779),  # Chandigarh sector 17
    (30.900, 75.857),  # Ludhiana
    (31.101, 77.167),  # Shimla foothills
]

# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------
SIM_START = "2025-01-01"
SIM_END   = "2025-06-30"

# ---------------------------------------------------------------------------
# Planted anomaly configuration
# ---------------------------------------------------------------------------
PLANTED = {
    # Entity indices (0-based) that form the suspect cluster
    "suspect_indices": list(range(NUM_SUSPECT_ENTITIES)),

    # Shared IMEI: entities 0 and 1 both use this device
    "shared_imei": "IMEI-SUSPECT-SHARED-001",

    # Circular transaction chain: entity_0 → entity_1 → entity_2 → entity_0
    "circular_tx_chain": [0, 1, 2],
    "circular_tx_amount_min": 50_000,
    "circular_tx_amount_max": 200_000,

    # Impossible travel: entity_3 appears at two towers > 500 km apart within 30 min
    "impossible_travel_entity": 3,
    "impossible_travel_gap_minutes": 25,

    # Tor usage: entity_4 has Tor sessions aligned with large outbound transfers
    "tor_entity": 4,
    "tor_tx_threshold": 75_000,
}

# ---------------------------------------------------------------------------
# Entity resolution thresholds
# ---------------------------------------------------------------------------
FUZZY_NAME_THRESHOLD = 85   # rapidfuzz WRatio score (0-100)

# ---------------------------------------------------------------------------
# Cross-domain correlation rule parameters
# ---------------------------------------------------------------------------
RULES = {
    "CR1_distance_km": 0.5,       # CDR tower within 0.5 km of ATM
    "CR1_time_window_min": 15,    # ±15 min
    "CR2_tor_tx_amount": 50_000,  # INR
    "CR2_time_window_hr": 1,
    "CR3_social_burst_z": 2.5,    # Z-score threshold for social burst
    "CR3_time_window_hr": 24,
    "CR4_travel_km": 200,         # impossible travel threshold km
    "CR4_time_window_min": 60,
}

# ---------------------------------------------------------------------------
# Composite risk score weights (must sum to 1.0)
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "cdr":        0.25,
    "ipdr":       0.20,
    "tx":         0.30,
    "social":     0.10,
    "graph":      0.15,
}

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CDR_FILE         = os.path.join(DATA_DIR, "cdr.csv")
IPDR_FILE        = os.path.join(DATA_DIR, "ipdr.csv")
TX_FILE          = os.path.join(DATA_DIR, "transactions.csv")
SOCIAL_FILE      = os.path.join(DATA_DIR, "social.json")

ENTITIES_FILE    = os.path.join(OUTPUT_DIR, "entities.parquet")
EVENTS_FILE      = os.path.join(OUTPUT_DIR, "events.parquet")
GRAPH_FILE       = os.path.join(OUTPUT_DIR, "entity_graph.pkl")
RISK_FILE        = os.path.join(OUTPUT_DIR, "risk_scores.parquet")
GROUND_TRUTH_FILE = os.path.join(OUTPUT_DIR, "ground_truth.json")

# ---------------------------------------------------------------------------
# spaCy model
# ---------------------------------------------------------------------------
SPACY_MODEL = "en_core_web_sm"
