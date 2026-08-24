"""
analytics.py
------------
Analytics layer for the Unified Analytics Platform MVP.

Per-source anomaly detection:
  - CDR:  IsolationForest on call features + device-sharing detection + impossible travel
  - IPDR: Volume Z-score + Tor/proxy flag
  - TX:   IsolationForest on transaction features + circular flow detection
  - Social: spaCy NER extraction + post-burst frequency detection

Fused / cross-domain:
  - Louvain community detection
  - PageRank broker score
  - Cross-domain correlation rules (CR-1 through CR-4), vectorized

Output:
  output/risk_scores.parquet

Usage:
    python analytics.py
"""

import os
import json
import math
import pickle
import logging
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

try:
    import community as community_louvain
except Exception:
    community_louvain = None

from config import (
    EVENTS_FILE, ENTITIES_FILE, GRAPH_FILE, RISK_FILE,
    OUTPUT_DIR, RULES, SCORE_WEIGHTS, ATM_LOCATIONS, SPACY_MODEL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analytics")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def haversine_vec(lat1_arr, lon1_arr, lat2, lon2) -> np.ndarray:
    """Vectorized haversine: lat/lon arrays vs a single point."""
    R = 6371.0
    phi1 = np.radians(lat1_arr)
    phi2 = math.radians(lat2)
    dphi = phi1 - phi2
    dlam = np.radians(lon1_arr) - math.radians(lon2)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*math.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def iso_forest_score(features_df: pd.DataFrame, contamination=0.1) -> np.ndarray:
    if features_df.empty:
        return np.array([])
    clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    clf.fit(features_df)
    raw = clf.score_samples(features_df)
    return MinMaxScaler().fit_transform(-raw.reshape(-1, 1)).flatten()


def load_events_attrs(events_df: pd.DataFrame, source: str) -> pd.DataFrame:
    sub = events_df[events_df["source"] == source].copy()
    if sub.empty:
        return sub
    attrs = sub["attributes_json"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else {}
    )
    attrs_df = pd.json_normalize(attrs)
    return pd.concat([sub.reset_index(drop=True), attrs_df.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# CDR Analytics
# ---------------------------------------------------------------------------

def analyze_cdr(events_df: pd.DataFrame, entities_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Analyzing CDR...")
    cdr = load_events_attrs(events_df, "CDR")
    if cdr.empty:
        return pd.DataFrame(columns=["entity_id", "cdr_score", "cdr_flags",
                                     "cdr_detail", "call_count", "night_ratio", "unique_peers"])

    cdr["timestamp_utc"] = pd.to_datetime(cdr["timestamp_utc"], utc=True)
    cdr["hour"] = cdr["timestamp_utc"].dt.hour
    cdr["duration_sec"] = pd.to_numeric(cdr.get("duration_sec", 0), errors="coerce").fillna(0)
    cdr["tower_lat"] = pd.to_numeric(cdr.get("tower_lat", np.nan), errors="coerce")
    cdr["tower_lon"] = pd.to_numeric(cdr.get("tower_lon", np.nan), errors="coerce")

    feats = cdr.groupby("entity_id").agg(
        call_count     = ("event_id", "count"),
        unique_peers   = ("called_number", "nunique") if "called_number" in cdr.columns else ("event_id", "count"),
        night_calls    = ("hour", lambda h: ((h < 6) | (h >= 22)).sum()),
        total_duration = ("duration_sec", "sum"),
        roaming_count  = ("roaming_flag", lambda x: (x == True).sum()),
        imei_count     = ("imei", "nunique") if "imei" in cdr.columns else ("event_id", "count"),
    ).reset_index()

    feats["night_ratio"]   = feats["night_calls"] / (feats["call_count"] + 1)
    feats["roaming_ratio"] = feats["roaming_count"] / (feats["call_count"] + 1)

    # IMEI sharing
    if "imei" in cdr.columns:
        imei_by_entity = cdr.groupby("imei")["entity_id"].nunique()
        shared_imeis   = set(imei_by_entity[imei_by_entity > 1].index)
        entity_imeis   = cdr.groupby("entity_id")["imei"].apply(set).reset_index()
        entity_imeis.columns = ["entity_id", "imeis"]
        entity_imeis["has_shared_imei"] = entity_imeis["imeis"].apply(
            lambda s: any(x in shared_imeis for x in s)
        )
    else:
        entity_imeis = pd.DataFrame({"entity_id": feats["entity_id"], "has_shared_imei": False})
        entity_imeis["imeis"] = [set() for _ in range(len(entity_imeis))]

    # Impossible travel (vectorized per entity)
    travel_flags = set()
    for eid, grp in cdr.groupby("entity_id"):
        grp = grp.sort_values("timestamp_utc")
        lats = grp["tower_lat"].values
        lons = grp["tower_lon"].values
        tss  = grp["timestamp_utc"].values
        for i in range(len(grp) - 1):
            if np.isnan(lats[i]) or np.isnan(lats[i+1]):
                continue
            dt_min = (tss[i+1] - tss[i]) / np.timedelta64(1, "m")
            if dt_min <= 0:
                continue
            dist = haversine_km(lats[i], lons[i], lats[i+1], lons[i+1])
            if dist > RULES["CR4_travel_km"] and dt_min < RULES["CR4_time_window_min"]:
                travel_flags.add(eid)
                break

    # Isolation Forest
    feat_cols = ["call_count", "unique_peers", "night_ratio", "roaming_ratio", "total_duration"]
    iso_scores = iso_forest_score(feats[feat_cols].fillna(0))
    feats["iso_score"] = iso_scores

    feats = feats.merge(entity_imeis[["entity_id", "has_shared_imei"]], on="entity_id", how="left")
    feats["has_shared_imei"]  = feats["has_shared_imei"].fillna(False)
    feats["impossible_travel"] = feats["entity_id"].isin(travel_flags)

    feats["cdr_score"] = (
        feats["iso_score"] * 0.5 +
        feats["has_shared_imei"].astype(float) * 0.3 +
        feats["impossible_travel"].astype(float) * 0.2
    )

    def build_flags(row):
        flags, parts = [], []
        if row["has_shared_imei"]:
            flags.append("SHARED_IMEI")
            parts.append("IMEI device shared across multiple identities")
        if row["impossible_travel"]:
            flags.append("IMPOSSIBLE_TRAVEL")
            parts.append(
                f"Appeared at locations >{RULES['CR4_travel_km']} km apart "
                f"within {RULES['CR4_time_window_min']} min"
            )
        if row["iso_score"] > 0.7:
            flags.append("CDR_OUTLIER")
            parts.append(
                f"Calling pattern outlier "
                f"(night_ratio={row['night_ratio']:.2f}, "
                f"unique_peers={int(row['unique_peers'])})"
            )
        return json.dumps(flags), "; ".join(parts) if parts else "No CDR anomalies"

    r = feats.apply(build_flags, axis=1)
    feats["cdr_flags"]  = [x[0] for x in r]
    feats["cdr_detail"] = [x[1] for x in r]

    log.info(f"  CDR: {len(feats)} entity scores, "
             f"{feats['has_shared_imei'].sum()} shared-IMEI, "
             f"{feats['impossible_travel'].sum()} impossible-travel")

    return feats[["entity_id", "cdr_score", "cdr_flags", "cdr_detail",
                  "call_count", "night_ratio", "unique_peers"]].copy()


# ---------------------------------------------------------------------------
# IPDR Analytics
# ---------------------------------------------------------------------------

def analyze_ipdr(events_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Analyzing IPDR...")
    ipdr = load_events_attrs(events_df, "IPDR")
    if ipdr.empty:
        return pd.DataFrame(columns=["entity_id", "ipdr_score", "ipdr_flags",
                                     "ipdr_detail", "tor_sessions", "mb_up", "mb_down"])

    ipdr["timestamp_utc"] = pd.to_datetime(ipdr["timestamp_utc"], utc=True)
    for col in ["bytes_up", "bytes_down", "session_duration_sec"]:
        ipdr[col] = pd.to_numeric(ipdr.get(col, 0), errors="coerce").fillna(0)

    feats = ipdr.groupby("entity_id").agg(
        session_count    = ("event_id", "count"),
        total_bytes_up   = ("bytes_up", "sum"),
        total_bytes_down = ("bytes_down", "sum"),
        tor_sessions     = ("service_category",
                            lambda x: (x == "Anonymizer-Tor").sum()
                            if "service_category" in ipdr.columns else 0),
        unique_dst_ips   = ("dst_ip", "nunique") if "dst_ip" in ipdr.columns else ("event_id", "count"),
        avg_session_dur  = ("session_duration_sec", "mean"),
    ).reset_index()

    feats["tor_ratio"] = feats["tor_sessions"] / (feats["session_count"] + 1)
    feats["mb_up"]     = feats["total_bytes_up"] / 1e6
    feats["mb_down"]   = feats["total_bytes_down"] / 1e6
    total_mb = feats["mb_up"] + feats["mb_down"]
    feats["volume_zscore"] = (total_mb - total_mb.mean()).abs() / (total_mb.std() + 1e-9)
    feats["volume_spike"]  = (feats["volume_zscore"] > 2.5).astype(float)
    feats["tor_flag"]      = (feats["tor_sessions"] > 0).astype(float)

    feat_cols = ["session_count", "mb_up", "mb_down", "tor_ratio", "avg_session_dur"]
    feats["iso_score"] = iso_forest_score(feats[feat_cols].fillna(0))

    feats["ipdr_score"] = (
        feats["iso_score"]    * 0.4 +
        feats["volume_spike"] * 0.3 +
        feats["tor_flag"]     * 0.3
    )

    def build_flags(row):
        flags, parts = [], []
        if row["tor_flag"]:
            flags.append("TOR_USAGE")
            parts.append(f"Tor/anonymizer detected ({int(row['tor_sessions'])} sessions)")
        if row["volume_spike"]:
            flags.append("VOLUME_SPIKE")
            parts.append(
                f"Data volume spike (Z={row['volume_zscore']:.1f}): "
                f"{row['mb_up']:.0f} MB up / {row['mb_down']:.0f} MB down"
            )
        if row["iso_score"] > 0.7:
            flags.append("IPDR_OUTLIER")
            parts.append("IPDR session pattern is an outlier")
        return json.dumps(flags), "; ".join(parts) if parts else "No IPDR anomalies"

    r = feats.apply(build_flags, axis=1)
    feats["ipdr_flags"]  = [x[0] for x in r]
    feats["ipdr_detail"] = [x[1] for x in r]

    log.info(f"  IPDR: {len(feats)} entity scores, "
             f"{int(feats['tor_flag'].sum())} Tor users, "
             f"{int(feats['volume_spike'].sum())} volume spikes")

    return feats[["entity_id", "ipdr_score", "ipdr_flags", "ipdr_detail",
                  "session_count", "tor_sessions", "mb_up", "mb_down"]].copy()


# ---------------------------------------------------------------------------
# Transaction Analytics
# ---------------------------------------------------------------------------

def analyze_transactions(events_df: pd.DataFrame) -> tuple:
    log.info("Analyzing transactions...")
    tx = load_events_attrs(events_df, "TRANSACTIONS")
    if tx.empty:
        return pd.DataFrame(columns=["entity_id", "tx_score", "tx_flags",
                                     "tx_detail", "tx_count", "total_sent", "avg_amount"]), set()

    tx["timestamp_utc"] = pd.to_datetime(tx["timestamp_utc"], utc=True)
    tx["amount_inr"] = pd.to_numeric(tx.get("amount_inr", 0), errors="coerce").fillna(0)

    feats = tx.groupby("entity_id").agg(
        tx_count              = ("event_id", "count"),
        total_sent            = ("amount_inr", "sum"),
        avg_amount            = ("amount_inr", "mean"),
        max_amount            = ("amount_inr", "max"),
        unique_counterparties = ("counterparty_entity_id", "nunique")
                                 if "counterparty_entity_id" in tx.columns else ("event_id", "count"),
        cash_tx               = ("tx_type",
                                 lambda x: x.isin(["Cash-Deposit", "Cash-Withdraw"]).sum()
                                 if "tx_type" in tx.columns else 0),
    ).reset_index()
    feats["cash_ratio"] = feats["cash_tx"] / (feats["tx_count"] + 1)

    # Circular flow detection
    tx_graph = nx.DiGraph()
    if "counterparty_entity_id" in tx.columns:
        for src, dst, amt in zip(tx["entity_id"], tx["counterparty_entity_id"], tx["amount_inr"]):
            if dst and src != dst:
                if tx_graph.has_edge(src, dst):
                    tx_graph[src][dst]["weight"] += amt
                else:
                    tx_graph.add_edge(src, dst, weight=float(amt))

    # Fast cycle detection: any node in a non-trivial SCC (size > 1) is in a cycle
    # This is O(V+E) via Tarjan's algorithm, vs exponential for simple_cycles
    circular_entities: set = set()
    try:
        for scc in nx.strongly_connected_components(tx_graph):
            if len(scc) >= 2:
                circular_entities.update(scc)
    except Exception:
        pass


    feats["in_circular_flow"] = feats["entity_id"].isin(circular_entities).astype(float)

    feat_cols = ["tx_count", "avg_amount", "max_amount", "unique_counterparties", "cash_ratio"]
    feats["iso_score"] = iso_forest_score(feats[feat_cols].fillna(0))
    feats["tx_score"]  = feats["iso_score"] * 0.5 + feats["in_circular_flow"] * 0.5

    def build_flags(row):
        flags, parts = [], []
        if row["in_circular_flow"]:
            flags.append("CIRCULAR_FLOW")
            parts.append("Involved in circular transaction flow (possible layering)")
        if row["iso_score"] > 0.7:
            flags.append("TX_OUTLIER")
            parts.append(
                f"Transaction pattern outlier: avg Rs.{row['avg_amount']:,.0f}, "
                f"max Rs.{row['max_amount']:,.0f}, {int(row['unique_counterparties'])} counterparties"
            )
        if row["cash_ratio"] > 0.4:
            flags.append("HIGH_CASH_RATIO")
            parts.append(f"High cash transaction ratio ({row['cash_ratio']:.0%})")
        return json.dumps(flags), "; ".join(parts) if parts else "No transaction anomalies"

    r = feats.apply(build_flags, axis=1)
    feats["tx_flags"]  = [x[0] for x in r]
    feats["tx_detail"] = [x[1] for x in r]

    log.info(f"  TX: {len(feats)} entity scores, "
             f"{len(circular_entities)} in circular flow, "
             f"{int((feats['iso_score'] > 0.7).sum())} outliers")

    return feats[["entity_id", "tx_score", "tx_flags", "tx_detail",
                  "tx_count", "total_sent", "avg_amount"]].copy(), circular_entities


# ---------------------------------------------------------------------------
# Social / OSINT Analytics
# ---------------------------------------------------------------------------

def analyze_social(events_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Analyzing social posts...")
    social = load_events_attrs(events_df, "SOCIAL")
    if social.empty:
        return pd.DataFrame(columns=["entity_id", "social_score", "social_flags",
                                     "social_detail", "post_count", "ner_entities"])

    social["timestamp_utc"] = pd.to_datetime(social["timestamp_utc"], utc=True)

    # NER extraction (spaCy if available, fast regex fallback)
    ner_results: dict = {}
    if "text" in social.columns:
        spacy_ok = False
        try:
            import spacy
            nlp = spacy.load(SPACY_MODEL)
            spacy_ok = True
        except Exception:
            pass

        if spacy_ok:
            eid_texts = social.groupby("entity_id")["text"].apply(list)
            eid_list  = list(eid_texts.index)
            text_list = [" ".join(str(t) for t in texts[:30]) for texts in eid_texts]
            for eid, doc in zip(eid_list, nlp.pipe(text_list, batch_size=10)):
                ner_results[eid] = [
                    {"text": ent.text, "label": ent.label_}
                    for ent in doc.ents
                    if ent.label_ in ("PERSON", "ORG", "GPE", "LOC")
                ]
        else:
            import re
            for eid, group in social.groupby("entity_id"):
                ents = []
                for txt in group["text"].dropna():
                    matches = re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?\b', str(txt))
                    for m in matches[:5]:
                        ents.append({"text": m, "label": "ENTITY"})
                ner_results[eid] = ents[:10]

    feats = social.groupby("entity_id").agg(
        post_count     = ("event_id", "count"),
        unique_hashtags= ("hashtags",
                          lambda x: len({h for hs in x for h in (hs if isinstance(hs, list) else [])})),
        mention_count  = ("mentions",
                          lambda x: sum(len(m) if isinstance(m, list) else 0 for m in x)),
    ).reset_index()

    feats["post_zscore"] = (
        (feats["post_count"] - feats["post_count"].mean()) /
        (feats["post_count"].std() + 1e-9)
    ).abs()
    feats["burst_flag"]   = (feats["post_zscore"] > 2.5).astype(float)
    feats["social_score"] = feats["burst_flag"] * 0.6 + (feats["post_zscore"].clip(0, 3) / 3) * 0.4
    feats["ner_entities"] = feats["entity_id"].map(
        lambda eid: json.dumps(ner_results.get(eid, []))
    )

    def build_flags(row):
        flags, parts = [], []
        if row["burst_flag"]:
            flags.append("SOCIAL_BURST")
            parts.append(
                f"Abnormally high post frequency (Z={row['post_zscore']:.1f}, "
                f"count={int(row['post_count'])})"
            )
        return json.dumps(flags), "; ".join(parts) if parts else "No social anomalies"

    r = feats.apply(build_flags, axis=1)
    feats["social_flags"]  = [x[0] for x in r]
    feats["social_detail"] = [x[1] for x in r]

    log.info(f"  Social: {len(feats)} entity scores, "
             f"{int(feats['burst_flag'].sum())} burst flagged")

    return feats[["entity_id", "social_score", "social_flags", "social_detail",
                  "post_count", "ner_entities"]].copy()


# ---------------------------------------------------------------------------
# Graph Analytics (fused)
# ---------------------------------------------------------------------------

def analyze_graph(G: nx.MultiDiGraph) -> pd.DataFrame:
    log.info("Analyzing entity graph (Louvain + PageRank)...")

    UG = nx.Graph()
    for u, v in G.edges():
        if u != v:
            if UG.has_edge(u, v):
                UG[u][v]["weight"] += 1
            else:
                UG.add_edge(u, v, weight=1)

    communities = {}
    if community_louvain is not None:
        try:
            communities = community_louvain.best_partition(UG, random_state=42)
        except Exception:
            communities = {}

    if not communities:
        try:
            comms = list(nx.community.louvain_communities(UG, seed=42))
            for cid, c_nodes in enumerate(comms):
                for n in c_nodes:
                    communities[n] = cid
        except Exception:
            communities = {n: 0 for n in UG.nodes}

    SG = nx.DiGraph()
    for u, v in G.edges():
        if u != v:
            if SG.has_edge(u, v):
                SG[u][v]["weight"] += 1
            else:
                SG.add_edge(u, v, weight=1)

    try:
        pagerank = nx.pagerank(SG, alpha=0.85, weight="weight", max_iter=200)
    except Exception:
        pagerank = {n: 1/max(len(SG.nodes), 1) for n in SG.nodes}

    try:
        betweenness = nx.betweenness_centrality(SG, normalized=True, weight="weight")
    except Exception:
        betweenness = {n: 0.0 for n in SG.nodes}

    from collections import Counter
    comm_sizes = Counter(communities.values())

    rows = []
    for node in G.nodes:
        comm_id   = communities.get(node, -1)
        pr        = pagerank.get(node, 0.0)
        bw        = betweenness.get(node, 0.0)
        rows.append({
            "entity_id":      node,
            "community_id":   int(comm_id),
            "community_size": int(comm_sizes.get(comm_id, 1)),
            "pagerank":       float(pr),
            "betweenness":    float(bw),
            "degree_in":      int(G.in_degree(node)),
            "degree_out":     int(G.out_degree(node)),
        })

    df = pd.DataFrame(rows)
    pr_bw = df[["pagerank", "betweenness"]].fillna(0)
    if pr_bw.std().sum() > 0:
        df["graph_score"] = MinMaxScaler().fit_transform(pr_bw).mean(axis=1)
    else:
        df["graph_score"] = 0.0

    log.info(f"  Graph: {len(df)} nodes scored, "
             f"{df['community_id'].nunique()} communities detected")
    return df


# ---------------------------------------------------------------------------
# Cross-Domain Correlation Rules (vectorized)
# ---------------------------------------------------------------------------

def apply_cross_domain_rules(events_df: pd.DataFrame,
                              entities_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized cross-domain correlation rules.
    Returns DataFrame: entity_id, rule_id, rule_description, triggered, details
    """
    log.info("Applying cross-domain correlation rules (vectorized)...")
    entity_ids = entities_df["entity_id"].unique().tolist()

    cdr_evts  = load_events_attrs(events_df, "CDR")
    tx_evts   = load_events_attrs(events_df, "TRANSACTIONS")
    ipdr_evts = load_events_attrs(events_df, "IPDR")
    soc_evts  = load_events_attrs(events_df, "SOCIAL")

    for df_ in [cdr_evts, tx_evts, ipdr_evts, soc_evts]:
        if not df_.empty and "timestamp_utc" in df_.columns:
            df_["timestamp_utc"] = pd.to_datetime(df_["timestamp_utc"], utc=True)

    if not tx_evts.empty and "amount_inr" in tx_evts.columns:
        tx_evts["amount_inr"] = pd.to_numeric(tx_evts["amount_inr"], errors="coerce").fillna(0)

    # ---- CR-1: CDR near ATM + transaction within ±15 min (vectorized haversine) ----
    cr1_hits: dict[str, str] = {}
    if not cdr_evts.empty and "tower_lat" in cdr_evts.columns:
        cdr_geo = cdr_evts[["entity_id", "timestamp_utc", "tower_lat", "tower_lon"]].copy()
        cdr_geo["tower_lat"] = pd.to_numeric(cdr_geo["tower_lat"], errors="coerce")
        cdr_geo["tower_lon"] = pd.to_numeric(cdr_geo["tower_lon"], errors="coerce")
        cdr_geo = cdr_geo.dropna(subset=["tower_lat", "tower_lon"])

        for atm_lat, atm_lon in ATM_LOCATIONS:
            dists = haversine_vec(
                cdr_geo["tower_lat"].values,
                cdr_geo["tower_lon"].values,
                atm_lat, atm_lon
            )
            near_mask = dists <= RULES["CR1_distance_km"]
            near_cdr  = cdr_geo[near_mask].copy()

            if near_cdr.empty or tx_evts.empty:
                continue

            tw = timedelta(minutes=RULES["CR1_time_window_min"])
            for _, crow in near_cdr.iterrows():
                eid = crow["entity_id"]
                if eid in cr1_hits:
                    continue
                ts   = crow["timestamp_utc"]
                e_tx = tx_evts[tx_evts["entity_id"] == eid]
                hit  = e_tx[
                    (e_tx["timestamp_utc"] >= ts - tw) &
                    (e_tx["timestamp_utc"] <= ts + tw)
                ]
                if not hit.empty:
                    amt = hit["amount_inr"].iloc[0] if "amount_inr" in hit.columns else 0
                    cr1_hits[eid] = f"CDR near ATM ({atm_lat:.3f},{atm_lon:.3f}) + Rs.{amt:,.0f} tx"

    # ---- CR-2: Tor session + large transfer within 1h ----
    cr2_hits: dict[str, str] = {}
    if not ipdr_evts.empty and "service_category" in ipdr_evts.columns:
        tor = ipdr_evts[ipdr_evts["service_category"] == "Anonymizer-Tor"][
            ["entity_id", "timestamp_utc"]
        ].copy()
        if not tor.empty and not tx_evts.empty:
            tx_large = tx_evts[tx_evts.get("amount_inr", pd.Series(0)) >= RULES["CR2_tor_tx_amount"]]
            tw = timedelta(hours=RULES["CR2_time_window_hr"])
            for _, tr in tor.iterrows():
                eid = tr["entity_id"]
                if eid in cr2_hits:
                    continue
                ts = tr["timestamp_utc"]
                hit = tx_large[
                    (tx_large["entity_id"] == eid) &
                    (tx_large["timestamp_utc"] >= ts - tw) &
                    (tx_large["timestamp_utc"] <= ts + tw)
                ]
                if not hit.empty:
                    amt = hit["amount_inr"].iloc[0] if "amount_inr" in hit.columns else 0
                    cr2_hits[eid] = f"Tor session + Rs.{amt:,.0f} transfer within 1h"

    # ---- CR-3: Social burst + tx cluster within 24h ----
    cr3_hits: dict[str, str] = {}
    if not soc_evts.empty and not tx_evts.empty:
        soc_daily = (
            soc_evts.set_index("timestamp_utc")
            .groupby("entity_id")
            .resample("D")
            .size()
            .reset_index(name="n_posts")
        )
        stats = soc_daily.groupby("entity_id")["n_posts"].agg(["mean", "std"]).reset_index()
        soc_daily = soc_daily.merge(stats, on="entity_id", how="left")
        soc_daily["z"] = (soc_daily["n_posts"] - soc_daily["mean"]) / (soc_daily["std"] + 1e-9)
        bursts = soc_daily[soc_daily["z"] > RULES["CR3_social_burst_z"]]

        tw = timedelta(hours=RULES["CR3_time_window_hr"])
        for _, bd in bursts.iterrows():
            eid    = bd["entity_id"]
            if eid in cr3_hits:
                continue
            day_ts = pd.Timestamp(bd["timestamp_utc"])
            if day_ts.tzinfo is None:
                day_ts = day_ts.tz_localize("UTC")
            hit = tx_evts[
                (tx_evts["entity_id"] == eid) &
                (tx_evts["timestamp_utc"] >= day_ts) &
                (tx_evts["timestamp_utc"] <= day_ts + tw)
            ]
            if not hit.empty:
                cr3_hits[eid] = f"Social burst ({int(bd['n_posts'])} posts) + {len(hit)} txs on {day_ts.date()}"

    # ---- CR-4: Populated from CDR impossible-travel flags in build_risk_scores ----

    # Assemble all rows
    all_rows = []
    for eid in entity_ids:
        all_rows.extend([
            {
                "entity_id":        eid,
                "rule_id":          "CR-1",
                "rule_description": "CDR active near ATM during transaction window",
                "triggered":        eid in cr1_hits,
                "details":          cr1_hits.get(eid, ""),
            },
            {
                "entity_id":        eid,
                "rule_id":          "CR-2",
                "rule_description": "Tor/anonymizer session concurrent with large transfer (>Rs.50K)",
                "triggered":        eid in cr2_hits,
                "details":          cr2_hits.get(eid, ""),
            },
            {
                "entity_id":        eid,
                "rule_id":          "CR-3",
                "rule_description": "Social post burst coincides with transaction cluster (24h window)",
                "triggered":        eid in cr3_hits,
                "details":          cr3_hits.get(eid, ""),
            },
            {
                "entity_id":        eid,
                "rule_id":          "CR-4",
                "rule_description": "Impossible travel detected in CDR (>200 km in <60 min)",
                "triggered":        False,   # merged from CDR flags
                "details":          "",
            },
        ])

    rules_df = pd.DataFrame(all_rows)
    log.info(f"  Rules triggered: {rules_df[rules_df['triggered']].groupby('rule_id').size().to_dict()}")
    return rules_df


# ---------------------------------------------------------------------------
# Composite Risk Score
# ---------------------------------------------------------------------------

def build_risk_scores(entities_df, cdr_df, ipdr_df, tx_df, social_df,
                       graph_df, rules_df) -> tuple:
    log.info("Building composite risk scores...")

    all_eids = entities_df["entity_id"].unique()
    base = pd.DataFrame({"entity_id": all_eids})

    base = base.merge(
        cdr_df[["entity_id", "cdr_score", "cdr_flags", "cdr_detail",
                "call_count", "night_ratio"]].rename(columns={"call_count": "cdr_call_count"}),
        on="entity_id", how="left"
    )
    base = base.merge(
        ipdr_df[["entity_id", "ipdr_score", "ipdr_flags", "ipdr_detail",
                 "tor_sessions", "mb_up", "mb_down"]],
        on="entity_id", how="left"
    )
    base = base.merge(
        tx_df[["entity_id", "tx_score", "tx_flags", "tx_detail",
               "tx_count", "total_sent"]],
        on="entity_id", how="left"
    )
    base = base.merge(
        social_df[["entity_id", "social_score", "social_flags",
                   "social_detail", "post_count", "ner_entities"]],
        on="entity_id", how="left"
    )
    base = base.merge(
        graph_df[["entity_id", "graph_score", "community_id", "community_size",
                  "pagerank", "betweenness", "degree_in", "degree_out"]],
        on="entity_id", how="left"
    )

    score_cols = ["cdr_score", "ipdr_score", "tx_score", "social_score", "graph_score"]
    base[score_cols] = base[score_cols].fillna(0.0)

    W = SCORE_WEIGHTS
    base["risk_score"] = (
        base["cdr_score"]    * W["cdr"] +
        base["ipdr_score"]   * W["ipdr"] +
        base["tx_score"]     * W["tx"] +
        base["social_score"] * W["social"] +
        base["graph_score"]  * W["graph"]
    )

    # Merge CR-4 from CDR impossible-travel flag
    def has_impossible_travel(cdr_flags_str):
        try:
            return "IMPOSSIBLE_TRAVEL" in json.loads(cdr_flags_str or "[]")
        except Exception:
            return False

    base["cr4_triggered"] = base["cdr_flags"].apply(has_impossible_travel)
    rules_df_out = rules_df.copy()
    for eid, val in zip(base["entity_id"], base["cr4_triggered"]):
        mask = (rules_df_out["entity_id"] == eid) & (rules_df_out["rule_id"] == "CR-4")
        rules_df_out.loc[mask, "triggered"] = val

    # Rule boost
    triggered_per_eid = (
        rules_df_out[rules_df_out["triggered"]]
        .groupby("entity_id")["rule_id"]
        .apply(list)
    )
    base["triggered_rules"] = base["entity_id"].map(triggered_per_eid).apply(
        lambda x: json.dumps(x) if isinstance(x, list) else "[]"
    )
    n_rules = base["triggered_rules"].apply(lambda x: len(json.loads(x)))
    base["rule_boost"] = (n_rules * 0.05).clip(0, 0.20)
    base["risk_score"] = (base["risk_score"] + base["rule_boost"]).clip(0, 1.0)

    # Plain-English explanation
    def build_explanation(row) -> str:
        parts = []
        rules = json.loads(row.get("triggered_rules", "[]"))
        if rules:
            parts.append(f"Triggered cross-domain rules: {', '.join(rules)}")
        for flag_col, detail_col in [
            ("cdr_flags", "cdr_detail"),
            ("ipdr_flags", "ipdr_detail"),
            ("tx_flags", "tx_detail"),
            ("social_flags", "social_detail"),
        ]:
            try:
                flags = json.loads(row.get(flag_col, "[]") or "[]")
            except Exception:
                flags = []
            detail = row.get(detail_col, "")
            if flags and detail:
                parts.append(detail)
        return " | ".join(parts) if parts else "No significant anomalies detected."

    base["explanation"] = base.apply(build_explanation, axis=1)
    base["risk_tier"]   = base["risk_score"].apply(
        lambda s: "HIGH" if s >= 0.7 else "MEDIUM" if s >= 0.4 else "LOW"
    )

    name_map = dict(zip(entities_df["entity_id"], entities_df["name"]))
    base["name"] = base["entity_id"].map(name_map).fillna("")
    base = base.sort_values("risk_score", ascending=False).reset_index(drop=True)

    log.info(f"  Risk tiers: HIGH={(base['risk_tier']=='HIGH').sum()}, "
             f"MEDIUM={(base['risk_tier']=='MEDIUM').sum()}, "
             f"LOW={(base['risk_tier']=='LOW').sum()}")

    return base, rules_df_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    entities_df = pd.read_parquet(ENTITIES_FILE)
    events_df   = pd.read_parquet(EVENTS_FILE)

    with open(GRAPH_FILE, "rb") as f:
        graph_data = pickle.load(f)
    G = graph_data["graph"]

    cdr_df    = analyze_cdr(events_df, entities_df)
    ipdr_df   = analyze_ipdr(events_df)
    tx_df, _  = analyze_transactions(events_df)
    social_df = analyze_social(events_df)
    graph_df  = analyze_graph(G)
    rules_df  = apply_cross_domain_rules(events_df, entities_df)

    risk_df, rules_df = build_risk_scores(
        entities_df, cdr_df, ipdr_df, tx_df, social_df, graph_df, rules_df
    )

    # Sync risk scores back into graph nodes
    for _, row in risk_df.iterrows():
        eid = row["entity_id"]
        if eid in G.nodes:
            G.nodes[eid]["risk_score"] = float(row["risk_score"])
            G.nodes[eid]["risk_tier"]  = row["risk_tier"]

    risk_df.to_parquet(RISK_FILE, index=False)
    with open(GRAPH_FILE, "wb") as f:
        pickle.dump({
            "graph":      G,
            "root_map":   graph_data.get("root_map", {}),
            "validation": graph_data.get("validation", {}),
            "rules":      rules_df.to_dict("records"),
        }, f)

    log.info(f"\nOK Risk scores saved -> {RISK_FILE}  ({len(risk_df)} entities)")
    log.info(f"OK Graph updated    -> {GRAPH_FILE}")
    log.info("\nTop 10 flagged entities:")
    cols = ["entity_id", "name", "risk_score", "risk_tier", "explanation"]
    print(risk_df[cols].head(10).to_string(index=False))

    return risk_df, G


if __name__ == "__main__":
    run()
