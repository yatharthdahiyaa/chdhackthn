"""
generate_synthetic_data.py
--------------------------
Generates all four synthetic data sources for the Unified Analytics Platform MVP.

Sources:
  1. CDR  (Call Detail Records)          -> data/cdr.csv
  2. IPDR (Internet Protocol Detail Rec) -> data/ipdr.csv
  3. Bank Transactions                   -> data/transactions.csv
  4. Social / OSINT                      -> data/social.json

ALL data is completely synthetic / fictional. No real person data,
real phone numbers, real IMEIs, or real accounts are used.
See README.md for legal/ethical framing.

Usage:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --seed 99 --entities 100
"""

import os
import json
import random
import argparse
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from faker import Faker

from config import (
    RANDOM_SEED, NUM_ENTITIES, NUM_SUSPECT_ENTITIES,
    NUM_CDR_ROWS, NUM_IPDR_ROWS, NUM_TX_ROWS, NUM_SOCIAL_POSTS,
    GEO_LAT_MIN, GEO_LAT_MAX, GEO_LON_MIN, GEO_LON_MAX,
    ATM_LOCATIONS, SIM_START, SIM_END, PLANTED,
    DATA_DIR, OUTPUT_DIR, GROUND_TRUTH_FILE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
IST = timezone(timedelta(hours=5, minutes=30))

fake = Faker("en_IN")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)


def rand_ts(start: str, end: str) -> datetime:
    """Return a random datetime between start and end (inclusive), in IST."""
    t0 = datetime.fromisoformat(start).replace(tzinfo=IST)
    t1 = datetime.fromisoformat(end).replace(tzinfo=IST)
    delta = (t1 - t0).total_seconds()
    return t0 + timedelta(seconds=random.uniform(0, delta))


def rand_ts_near(base: datetime, window_seconds: int) -> datetime:
    """Return a timestamp within ±window_seconds of base."""
    return base + timedelta(seconds=random.uniform(-window_seconds, window_seconds))


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S+05:30")


# ---------------------------------------------------------------------------
# Entity registry
# ---------------------------------------------------------------------------

def build_entity_registry(n: int, n_suspect: int, seed: int):
    """
    Build a list of synthetic entity dictionaries.
    Each entity has:
      - entity_id, name, dob, gender
      - phone (E.164-style, fictional +91 prefix)
      - imei, imsi (15-digit fictional)
      - account_id (fictional bank account)
      - handle (social media username)
      - is_suspect flag
    """
    seed_all(seed)
    entities = []
    used_phones = set()
    used_handles = set()

    for i in range(n):
        is_suspect = i < n_suspect

        # Phone: +91 followed by 10 random digits (not a real number)
        while True:
            phone = "+91" + "".join([str(random.randint(0, 9)) for _ in range(10)])
            if phone not in used_phones:
                used_phones.add(phone)
                break

        # IMEI: 15 digits (fictional — starts with 99 to signal synthetic)
        imei = "99" + "".join([str(random.randint(0, 9)) for _ in range(13)])

        # Planted: entities 0 and 1 share the same IMEI
        if is_suspect and i == 1:
            imei = PLANTED["shared_imei"]
        elif is_suspect and i == 0:
            imei = PLANTED["shared_imei"]

        # IMSI: 15 digits (fictional MCC=404 India, MNC=99 synthetic)
        imsi = "40499" + "".join([str(random.randint(0, 9)) for _ in range(10)])

        # Account ID
        account_id = "ACC" + "".join([str(random.randint(0, 9)) for _ in range(10)])

        # Social handle
        while True:
            handle = fake.user_name() + str(random.randint(10, 999))
            if handle not in used_handles:
                used_handles.add(handle)
                break

        name = fake.name()

        entities.append({
            "entity_id":  f"E{i:03d}",
            "name":       name,
            "dob":        fake.date_of_birth(minimum_age=20, maximum_age=65).isoformat(),
            "gender":     random.choice(["M", "F", "Other"]),
            "phone":      phone,
            "imei":       imei,
            "imsi":       imsi,
            "account_id": account_id,
            "handle":     handle,
            "is_suspect": is_suspect,
        })

    return entities


# ---------------------------------------------------------------------------
# Cell tower registry
# ---------------------------------------------------------------------------

def build_tower_registry(n_towers: int = 80):
    towers = []
    for i in range(n_towers):
        towers.append({
            "tower_id":  f"TWR{i:04d}",
            "lat":       round(random.uniform(GEO_LAT_MIN, GEO_LAT_MAX), 6),
            "lon":       round(random.uniform(GEO_LON_MIN, GEO_LON_MAX), 6),
            "cell_id":   f"CELL{random.randint(1000,9999)}",
        })
    return towers


# ---------------------------------------------------------------------------
# Layer 1a — CDR generator
# ---------------------------------------------------------------------------

def generate_cdr(entities, towers, n_rows: int) -> pd.DataFrame:
    """
    CDR schema:
      cdr_id, timestamp, calling_number, called_number,
      duration_sec, call_type, roaming_flag,
      tower_id, tower_lat, tower_lon, imei, imsi, entity_id
    """
    rows = []
    call_types = ["VOICE", "SMS", "MMS"]
    call_type_weights = [0.65, 0.30, 0.05]

    # Build lookup: phone -> entity
    phone_map = {e["phone"]: e for e in entities}
    phone_list = [e["phone"] for e in entities]

    # ---- Normal traffic ----
    for i in range(n_rows):
        caller_e = random.choice(entities)
        called_phone = random.choice(phone_list)
        while called_phone == caller_e["phone"]:
            called_phone = random.choice(phone_list)
        called_e = phone_map[called_phone]

        ts = rand_ts(SIM_START, SIM_END)
        tower = random.choice(towers)
        ctype = random.choices(call_types, weights=call_type_weights)[0]
        duration = 0 if ctype != "VOICE" else random.randint(5, 1800)
        roaming = random.random() < 0.05

        rows.append({
            "cdr_id":         f"CDR{i:07d}",
            "timestamp":      fmt_ts(ts),
            "calling_number": caller_e["phone"],
            "called_number":  called_phone,
            "duration_sec":   duration,
            "call_type":      ctype,
            "roaming_flag":   roaming,
            "tower_id":       tower["tower_id"],
            "tower_lat":      tower["lat"],
            "tower_lon":      tower["lon"],
            "imei":           caller_e["imei"],
            "imsi":           caller_e["imsi"],
            "entity_id":      caller_e["entity_id"],
        })

    # ---- Planted anomaly: impossible travel for entity_3 ----
    travel_e = entities[PLANTED["impossible_travel_entity"]]
    n_travel = 20  # 20 pairs of impossible travel events
    for j in range(n_travel):
        ts_base = rand_ts(SIM_START, SIM_END)
        # Tower A — near Chandigarh
        tower_a = {"tower_id": "TWR_ANOM_A",
                   "lat": 30.733, "lon": 76.779}
        # Tower B — near Delhi (≈250 km away)
        tower_b = {"tower_id": "TWR_ANOM_B",
                   "lat": 28.613, "lon": 77.209}
        ts_b = ts_base + timedelta(minutes=PLANTED["impossible_travel_gap_minutes"])

        for (twr, ts_use) in [(tower_a, ts_base), (tower_b, ts_b)]:
            rows.append({
                "cdr_id":         f"CDR_TRAVEL_{j}_{twr['tower_id']}",
                "timestamp":      fmt_ts(ts_use),
                "calling_number": travel_e["phone"],
                "called_number":  random.choice(phone_list),
                "duration_sec":   random.randint(10, 300),
                "call_type":      "VOICE",
                "roaming_flag":   False,
                "tower_id":       twr["tower_id"],
                "tower_lat":      twr["lat"],
                "tower_lon":      twr["lon"],
                "imei":           travel_e["imei"],
                "imsi":           travel_e["imsi"],
                "entity_id":      travel_e["entity_id"],
            })

    # ---- Planted anomaly: entity_0 near ATM before transaction ----
    atm_e = entities[PLANTED["circular_tx_chain"][0]]
    for k, atm_loc in enumerate(ATM_LOCATIONS):
        ts_atm = rand_ts(SIM_START, SIM_END)
        # Place the call tower 200m from ATM
        near_lat = atm_loc[0] + random.uniform(-0.002, 0.002)
        near_lon = atm_loc[1] + random.uniform(-0.002, 0.002)
        rows.append({
            "cdr_id":         f"CDR_ATM_{k}",
            "timestamp":      fmt_ts(ts_atm),
            "calling_number": atm_e["phone"],
            "called_number":  random.choice(phone_list),
            "duration_sec":   random.randint(10, 120),
            "call_type":      "VOICE",
            "roaming_flag":   False,
            "tower_id":       f"TWR_ATM_{k}",
            "tower_lat":      near_lat,
            "tower_lon":      near_lon,
            "imei":           atm_e["imei"],
            "imsi":           atm_e["imsi"],
            "entity_id":      atm_e["entity_id"],
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Layer 1b — IPDR generator
# ---------------------------------------------------------------------------

def generate_ipdr(entities, n_rows: int) -> pd.DataFrame:
    """
    IPDR schema (privacy-preserving — no payload):
      ipdr_id, timestamp, entity_id, imei, imsi,
      src_ip, dst_ip, src_port, dst_port, protocol,
      session_duration_sec, bytes_up, bytes_down,
      service_category, app_fingerprint
    """
    rows = []
    protocols = ["TCP", "UDP", "QUIC"]
    proto_weights = [0.6, 0.3, 0.1]
    service_categories = [
        "WhatsApp", "YouTube", "Web-Browse", "Email",
        "Cloud-Backup", "VoIP", "Social-Media", "Streaming",
    ]
    svc_weights = [0.20, 0.15, 0.25, 0.10, 0.05, 0.10, 0.10, 0.05]

    def rand_ip():
        # RFC 5737 documentation ranges — fictional
        prefix = random.choice(["192.0.2", "198.51.100", "203.0.113"])
        return f"{prefix}.{random.randint(1, 254)}"

    for i in range(n_rows):
        ent = random.choice(entities)
        ts  = rand_ts(SIM_START, SIM_END)
        proto = random.choices(protocols, weights=proto_weights)[0]
        svc   = random.choices(service_categories, weights=svc_weights)[0]
        duration = random.randint(10, 7200)
        bytes_up   = random.randint(100, 5_000_000)
        bytes_down = random.randint(1_000, 50_000_000)

        rows.append({
            "ipdr_id":              f"IPDR{i:08d}",
            "timestamp":            fmt_ts(ts),
            "entity_id":            ent["entity_id"],
            "imei":                 ent["imei"],
            "imsi":                 ent["imsi"],
            "src_ip":               rand_ip(),
            "dst_ip":               rand_ip(),
            "src_port":             random.randint(1024, 65535),
            "dst_port":             random.choice([80, 443, 8080, 5060, 5228]),
            "protocol":             proto,
            "session_duration_sec": duration,
            "bytes_up":             bytes_up,
            "bytes_down":           bytes_down,
            "service_category":     svc,
            "app_fingerprint":      svc.lower().replace("-", "_"),
        })

    # ---- Planted anomaly: entity_4 (tor_entity) uses Tor-like sessions ----
    tor_e = entities[PLANTED["tor_entity"]]
    for j in range(150):
        ts = rand_ts("2025-02-01", "2025-04-30")
        rows.append({
            "ipdr_id":              f"IPDR_TOR_{j:04d}",
            "timestamp":            fmt_ts(ts),
            "entity_id":            tor_e["entity_id"],
            "imei":                 tor_e["imei"],
            "imsi":                 tor_e["imsi"],
            "src_ip":               rand_ip(),
            "dst_ip":               rand_ip(),
            "src_port":             9050,    # Tor SOCKS port
            "dst_port":             9001,    # Tor OR port
            "protocol":             "TCP",
            "session_duration_sec": random.randint(600, 14400),
            "bytes_up":             random.randint(10_000_000, 50_000_000),
            "bytes_down":           random.randint(50_000_000, 200_000_000),
            "service_category":     "Anonymizer-Tor",
            "app_fingerprint":      "tor_onion",
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Layer 1c — Transaction generator
# ---------------------------------------------------------------------------

def generate_transactions(entities, n_rows: int) -> pd.DataFrame:
    """
    Transaction schema:
      tx_id, timestamp, entity_id, account_id,
      counterparty_account, counterparty_entity_id,
      amount_inr, tx_type, channel, reference_note
    """
    rows = []
    tx_types = ["NEFT", "IMPS", "UPI", "Cash-Deposit", "Cash-Withdraw", "RTGS"]
    tx_weights = [0.20, 0.25, 0.35, 0.08, 0.07, 0.05]
    channels   = ["Mobile-App", "Branch", "ATM", "Net-Banking", "POS"]
    chan_weights = [0.45, 0.15, 0.20, 0.15, 0.05]

    acc_to_entity = {e["account_id"]: e for e in entities}
    acc_list = [e["account_id"] for e in entities]

    # ---- Normal transactions ----
    for i in range(n_rows):
        sender = random.choice(entities)
        recv_acc = random.choice(acc_list)
        while recv_acc == sender["account_id"]:
            recv_acc = random.choice(acc_list)
        recv_e = acc_to_entity[recv_acc]

        ts = rand_ts(SIM_START, SIM_END)
        amount = round(np.random.lognormal(mean=9, sigma=2), 2)  # INR, log-normal
        amount = max(10.0, min(amount, 2_000_000.0))
        ttype = random.choices(tx_types, weights=tx_weights)[0]
        channel = random.choices(channels, weights=chan_weights)[0]

        rows.append({
            "tx_id":                  f"TX{i:08d}",
            "timestamp":              fmt_ts(ts),
            "entity_id":              sender["entity_id"],
            "account_id":             sender["account_id"],
            "counterparty_account":   recv_acc,
            "counterparty_entity_id": recv_e["entity_id"],
            "amount_inr":             amount,
            "tx_type":                ttype,
            "channel":                channel,
            "reference_note":         fake.bs()[:60],
        })

    # ---- Planted anomaly: circular flow E000 -> E001 -> E002 -> E000 ----
    chain = PLANTED["circular_tx_chain"]
    ent_chain = [entities[i] for i in chain]
    n_cycles = 30  # 30 round-trips
    for c in range(n_cycles):
        ts_base = rand_ts("2025-03-01", "2025-05-31")
        for step in range(len(ent_chain)):
            sender_e = ent_chain[step]
            receiver_e = ent_chain[(step + 1) % len(ent_chain)]
            amount = random.randint(
                PLANTED["circular_tx_amount_min"],
                PLANTED["circular_tx_amount_max"]
            )
            ts_step = ts_base + timedelta(hours=step * random.randint(1, 6))
            rows.append({
                "tx_id":                  f"TX_CIRC_{c}_{step}",
                "timestamp":              fmt_ts(ts_step),
                "entity_id":              sender_e["entity_id"],
                "account_id":             sender_e["account_id"],
                "counterparty_account":   receiver_e["account_id"],
                "counterparty_entity_id": receiver_e["entity_id"],
                "amount_inr":             float(amount),
                "tx_type":                "IMPS",
                "channel":                "Mobile-App",
                "reference_note":         "business settlement",
            })

    # ---- Planted anomaly: entity_4 (tor) large outbound during Tor sessions ----
    tor_e = entities[PLANTED["tor_entity"]]
    normal_e = [e for e in entities if not e["is_suspect"]]
    for j in range(40):
        ts = rand_ts("2025-02-01", "2025-04-30")
        recv = random.choice(normal_e)
        rows.append({
            "tx_id":                  f"TX_TOR_{j:04d}",
            "timestamp":              fmt_ts(ts),
            "entity_id":              tor_e["entity_id"],
            "account_id":             tor_e["account_id"],
            "counterparty_account":   recv["account_id"],
            "counterparty_entity_id": recv["entity_id"],
            "amount_inr":             float(random.randint(
                PLANTED["tor_tx_threshold"],
                500_000
            )),
            "tx_type":                "NEFT",
            "channel":                "Net-Banking",
            "reference_note":         "consulting fees",
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Layer 1d — Social / OSINT generator
# ---------------------------------------------------------------------------

# Sample templates for synthetic social posts
_TEMPLATES = [
    "Just arrived at {place}. Great weather!",
    "Meeting with {person} at {place} tomorrow.",
    "Anyone else following #{tag}? Big news!",
    "Transferred funds to {person} — settle later.",
    "New phone who dis — dropped my old {device}.",
    "#{tag} is trending for a reason. #news",
    "Caught up with {person} and {person2} in {place}.",
    "Working late on #{tag} project. Send help.",
    "The {org} deal closed today. Big milestone.",
    "Shoutout to {person} for the introduction!",
    "Heading to {place} for the weekend. #{tag}",
    "Just completed a transfer — #{tag} #fintech",
    "Network is down at {place}. Using mobile data.",
    "Interesting times at {org}. Watch this space.",
    "Back in {place} after a long trip. Exhausted.",
]

_PLACES = [
    "Chandigarh", "Mohali", "Panchkula", "Ludhiana",
    "Amritsar", "Jalandhar", "Shimla", "Delhi",
]
_ORGS = [
    "TechVentures", "FinEdge", "DataSync", "NexusGroup",
    "AlphaCorp", "BlueStar", "PrimeLink",
]
_TAGS = [
    "crypto", "fintech", "privacy", "startup", "india",
    "investing", "tech", "anonymous", "trading", "news",
]


def _make_post(entity, entities) -> dict:
    template = random.choice(_TEMPLATES)
    other_e = random.choice([e for e in entities if e["entity_id"] != entity["entity_id"]])
    other_e2 = random.choice([e for e in entities if e["entity_id"] != entity["entity_id"]])

    text = template.format(
        place=random.choice(_PLACES),
        person=other_e["name"].split()[0],
        person2=other_e2["name"].split()[0],
        org=random.choice(_ORGS),
        tag=random.choice(_TAGS),
        device=f"Synth-Phone-{random.randint(1000,9999)}",
    )

    # Extract mentions from template substitution
    mentions = []
    if "{person}" in template:
        mentions.append(f"@{other_e['handle']}")
    if "{person2}" in template:
        mentions.append(f"@{other_e2['handle']}")

    hashtags = []
    if "{tag}" in template:
        hashtags.append(random.choice(_TAGS))

    geotag = None
    if random.random() < 0.15:
        geotag = {
            "lat": round(random.uniform(GEO_LAT_MIN, GEO_LAT_MAX), 6),
            "lon": round(random.uniform(GEO_LON_MIN, GEO_LON_MAX), 6),
        }

    return {
        "post_id":   f"POST_{entity['entity_id']}_{random.randint(0, 999999):06d}",
        "entity_id": entity["entity_id"],
        "handle":    entity["handle"],
        "timestamp": fmt_ts(rand_ts(SIM_START, SIM_END)),
        "text":      text,
        "mentions":  mentions,
        "hashtags":  hashtags,
        "geotag":    geotag,
    }


def generate_social(entities, n_posts: int) -> list:
    posts = []

    for i in range(n_posts):
        ent = random.choice(entities)
        posts.append(_make_post(ent, entities))

    # ---- Planted anomaly: suspect cluster burst (high frequency, coordinated) ----
    suspect_entities = [e for e in entities if e["is_suspect"]]
    burst_ts_base = datetime(2025, 4, 15, 10, 0, 0, tzinfo=IST)
    burst_tag = "anonymous"

    for se in suspect_entities:
        for b in range(30):  # 30 posts in tight window
            ts_burst = burst_ts_base + timedelta(minutes=random.randint(0, 120))
            posts.append({
                "post_id":   f"POST_BURST_{se['entity_id']}_{b:03d}",
                "entity_id": se["entity_id"],
                "handle":    se["handle"],
                "timestamp": fmt_ts(ts_burst),
                "text":      f"#{burst_tag} — spread the word. This is important. #{random.choice(_TAGS)}",
                "mentions":  [f"@{random.choice(suspect_entities)['handle']}"],
                "hashtags":  [burst_tag, random.choice(_TAGS)],
                "geotag":    None,
            })

    random.shuffle(posts)
    return posts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = RANDOM_SEED,
         n_entities: int = NUM_ENTITIES,
         n_suspect: int = NUM_SUSPECT_ENTITIES):

    print(f"[generate] seed={seed}  entities={n_entities}  suspects={n_suspect}")
    seed_all(seed)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build registries
    print("[generate] Building entity registry...")
    entities = build_entity_registry(n_entities, n_suspect, seed)
    towers   = build_tower_registry(80)

    # Save entity registry as ground truth for validation
    ground_truth = {
        "entities": entities,
        "planted_anomalies": {
            "shared_imei": {
                "imei": PLANTED["shared_imei"],
                "entity_ids": [entities[0]["entity_id"], entities[1]["entity_id"]],
            },
            "circular_tx_chain": {
                "entity_ids": [entities[i]["entity_id"] for i in PLANTED["circular_tx_chain"]],
            },
            "impossible_travel": {
                "entity_id": entities[PLANTED["impossible_travel_entity"]]["entity_id"],
            },
            "tor_entity": {
                "entity_id": entities[PLANTED["tor_entity"]]["entity_id"],
            },
        },
    }
    with open(GROUND_TRUTH_FILE, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"[generate] Ground truth -> {GROUND_TRUTH_FILE}")

    # CDR
    print("[generate] Generating CDR...")
    cdr_df = generate_cdr(entities, towers, NUM_CDR_ROWS)
    cdr_df.to_csv(
        "data/cdr.csv" if not os.path.isabs("data/cdr.csv")
        else "data/cdr.csv",
        index=False,
    )
    from config import CDR_FILE
    cdr_df.to_csv(CDR_FILE, index=False)
    print(f"[generate] CDR -> {CDR_FILE}  ({len(cdr_df):,} rows)")

    # IPDR
    print("[generate] Generating IPDR...")
    ipdr_df = generate_ipdr(entities, NUM_IPDR_ROWS)
    from config import IPDR_FILE
    ipdr_df.to_csv(IPDR_FILE, index=False)
    print(f"[generate] IPDR -> {IPDR_FILE}  ({len(ipdr_df):,} rows)")

    # Transactions
    print("[generate] Generating transactions...")
    tx_df = generate_transactions(entities, NUM_TX_ROWS)
    from config import TX_FILE
    tx_df.to_csv(TX_FILE, index=False)
    print(f"[generate] Transactions -> {TX_FILE}  ({len(tx_df):,} rows)")

    # Social
    print("[generate] Generating social posts...")
    posts = generate_social(entities, NUM_SOCIAL_POSTS)
    from config import SOCIAL_FILE
    with open(SOCIAL_FILE, "w") as f:
        json.dump(posts, f, indent=2)
    print(f"[generate] Social -> {SOCIAL_FILE}  ({len(posts):,} posts)")

    print("\n[generate] [OK] All synthetic data generated successfully.")
    print(f"  CDR rows      : {len(cdr_df):,}")
    print(f"  IPDR rows     : {len(ipdr_df):,}")
    print(f"  TX rows       : {len(tx_df):,}")
    print(f"  Social posts  : {len(posts):,}")
    print(f"  Entities      : {len(entities)}")
    print(f"  Suspects      : {n_suspect} (indices 0-{n_suspect-1})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for the analytics platform.")
    parser.add_argument("--seed",     type=int, default=RANDOM_SEED)
    parser.add_argument("--entities", type=int, default=NUM_ENTITIES)
    parser.add_argument("--suspects", type=int, default=NUM_SUSPECT_ENTITIES)
    args = parser.parse_args()
    main(seed=args.seed, n_entities=args.entities, n_suspect=args.suspects)
