"""
entity_resolution.py
---------------------
Entity Resolution layer for the Unified Analytics Platform MVP.

Builds a unified entity graph (NetworkX MultiDiGraph) by:
  1. Deterministic joins: shared phone / IMEI / account_id / handle
  2. Fuzzy name matching via rapidfuzz (WRatio ≥ FUZZY_NAME_THRESHOLD)
  3. Merges duplicate entity nodes into canonical entities
  4. Adds relationship edges from events (called, transacted_with, mentioned)

Outputs:
  output/entity_graph.pkl   (NetworkX graph, pickled)
  Also returns the graph for use by analytics.py

Usage:
    python entity_resolution.py
"""

import os
import json
import pickle
import logging
from collections import defaultdict

import pandas as pd
import networkx as nx
from rapidfuzz import fuzz

from config import (
    ENTITIES_FILE, EVENTS_FILE, GRAPH_FILE, GROUND_TRUTH_FILE,
    OUTPUT_DIR, FUZZY_NAME_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("entity_resolution")


# ---------------------------------------------------------------------------
# Step 1 — Load entities + events
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading entities and events from parquet...")
    entities_df = pd.read_parquet(ENTITIES_FILE)
    events_df   = pd.read_parquet(EVENTS_FILE)
    log.info(f"  {len(entities_df)} entities, {len(events_df):,} events")
    return entities_df, events_df


# ---------------------------------------------------------------------------
# Step 2 — Build identifier index from ground-truth metadata
# ---------------------------------------------------------------------------

def build_identifier_index(entities_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Returns a mapping: identifier_value -> [entity_id, ...]
    Identifiers extracted: phone, imei, imsi, account_id, handle
    """
    index: dict[str, list[str]] = defaultdict(list)

    for _, row in entities_df.iterrows():
        eid = row["entity_id"]
        try:
            meta = json.loads(row.get("metadata_json", "{}") or "{}")
        except Exception:
            meta = {}

        for key in ("phone", "imei", "imsi", "account_id", "handle"):
            val = meta.get(key)
            if val and str(val).strip():
                index[str(val).strip()].append(eid)

    # Also extract identifiers from event attributes (for IMEI/IMSI in CDR/IPDR)
    return dict(index)


# ---------------------------------------------------------------------------
# Step 3 — Union-Find for merging
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank   = {x: 0 for x in items}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def components(self) -> dict[str, set]:
        groups: dict[str, set] = defaultdict(set)
        for x in self.parent:
            groups[self.find(x)].add(x)
        return dict(groups)


# ---------------------------------------------------------------------------
# Step 4 — Deterministic join
# ---------------------------------------------------------------------------

def deterministic_join(entities_df: pd.DataFrame,
                        id_index: dict[str, list[str]],
                        uf: UnionFind):
    """Union any entities sharing a common identifier value."""
    merged = 0
    for id_val, eids in id_index.items():
        if len(eids) > 1:
            for i in range(1, len(eids)):
                if uf.find(eids[0]) != uf.find(eids[i]):
                    log.debug(f"  Det join: {eids[0]} ↔ {eids[i]}  via {id_val}")
                    uf.union(eids[0], eids[i])
                    merged += 1
    log.info(f"  Deterministic joins: {merged} merges")


# ---------------------------------------------------------------------------
# Step 5 — Fuzzy name join
# ---------------------------------------------------------------------------

def fuzzy_name_join(entities_df: pd.DataFrame, uf: UnionFind, threshold: int):
    """Union entities whose names fuzzy-match above threshold."""
    names = entities_df[["entity_id", "name"]].dropna().values.tolist()
    names = [(eid, n.strip()) for eid, n in names if isinstance(n, str) and len(n) > 3]

    fuzzy_merged = 0
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            eid_a, name_a = names[i]
            eid_b, name_b = names[j]
            if uf.find(eid_a) == uf.find(eid_b):
                continue  # already merged
            score = fuzz.WRatio(name_a, name_b)
            if score >= threshold:
                log.debug(f"  Fuzzy join: '{name_a}' ↔ '{name_b}' score={score}")
                uf.union(eid_a, eid_b)
                fuzzy_merged += 1
    log.info(f"  Fuzzy name joins: {fuzzy_merged} merges (threshold={threshold})")


# ---------------------------------------------------------------------------
# Step 6 — Build the NetworkX graph
# ---------------------------------------------------------------------------

def build_graph(entities_df: pd.DataFrame,
                events_df: pd.DataFrame,
                uf: UnionFind) -> nx.MultiDiGraph:
    """
    Nodes: one per canonical entity (post-resolution root of UnionFind)
    Node attributes: entity_id, canonical_ids, name, is_suspect, ...
    Edge types: CALLED, TRANSACTED, MENTIONED, CO_LOCATION
    """
    G = nx.MultiDiGraph()

    # Build canonical entity metadata
    components = uf.components()
    # Map every entity_id -> canonical root
    root_map = {}
    for root, members in components.items():
        for m in members:
            root_map[m] = root

    # Build per-root metadata
    meta_map = {}
    for _, row in entities_df.iterrows():
        eid = row["entity_id"]
        try:
            meta = json.loads(row.get("metadata_json", "{}") or "{}")
        except Exception:
            meta = {}
        meta_map[eid] = {
            "name":       row.get("name", ""),
            "is_suspect": meta.get("is_suspect", False),
            "phone":      meta.get("phone", ""),
            "imei":       meta.get("imei", ""),
            "account_id": meta.get("account_id", ""),
            "handle":     meta.get("handle", ""),
        }

    # Add nodes
    for root, members in components.items():
        # Pick name from first member that has one
        name = ""
        is_suspect = False
        phones, imeis, accounts, handles = [], [], [], []
        for m in members:
            mm = meta_map.get(m, {})
            if not name and mm.get("name"):
                name = mm["name"]
            if mm.get("is_suspect"):
                is_suspect = True
            if mm.get("phone"):   phones.append(mm["phone"])
            if mm.get("imei"):    imeis.append(mm["imei"])
            if mm.get("account_id"): accounts.append(mm["account_id"])
            if mm.get("handle"):  handles.append(mm["handle"])

        G.add_node(root, **{
            "canonical_id":  root,
            "member_ids":    list(members),
            "name":          name,
            "is_suspect":    is_suspect,
            "phones":        list(set(phones)),
            "imeis":         list(set(imeis)),
            "accounts":      list(set(accounts)),
            "handles":       list(set(handles)),
            "event_count":   0,
            "risk_score":    0.0,
        })

    # Add edges from events
    edge_count = {"CALLED": 0, "TRANSACTED": 0, "MENTIONED": 0}

    for _, evt in events_df.iterrows():
        src_root = root_map.get(evt["entity_id"], evt["entity_id"])
        if src_root not in G:
            G.add_node(src_root, canonical_id=src_root, name="", is_suspect=False,
                       member_ids=[src_root], phones=[], imeis=[], accounts=[],
                       handles=[], event_count=0, risk_score=0.0)
        G.nodes[src_root]["event_count"] += 1

        try:
            attrs = json.loads(evt["attributes_json"])
        except Exception:
            attrs = {}

        src = evt["source"]
        ts  = str(evt["timestamp_utc"])

        if src == "CDR":
            # Edge: caller -> called
            called_phone = attrs.get("called_number", "")
            # Find entity with this phone (via id_index lookup — we'll do a simple scan)
            # This will be resolved properly via the id_index passed in; here we store phone
            G.add_edge(src_root, src_root,  # self-edges resolved post-hoc
                       key=f"CDR_{evt['event_id']}",
                       edge_type="CALLED",
                       timestamp=ts,
                       called_phone=called_phone,
                       call_type=attrs.get("call_type", ""),
                       tower_lat=attrs.get("tower_lat"),
                       tower_lon=attrs.get("tower_lon"),
                       tower_id=attrs.get("tower_id", ""),
            )
            edge_count["CALLED"] += 1

        elif src == "TRANSACTIONS":
            cp_eid = attrs.get("counterparty_entity_id", "")
            if cp_eid:
                dst_root = root_map.get(cp_eid, cp_eid)
                if dst_root not in G:
                    G.add_node(dst_root, canonical_id=dst_root, name="",
                               is_suspect=False, member_ids=[dst_root],
                               phones=[], imeis=[], accounts=[], handles=[],
                               event_count=0, risk_score=0.0)
                G.add_edge(src_root, dst_root,
                           key=f"TX_{evt['event_id']}",
                           edge_type="TRANSACTED",
                           timestamp=ts,
                           amount_inr=attrs.get("amount_inr", 0),
                           tx_type=attrs.get("tx_type", ""),
                           channel=attrs.get("channel", ""),
                )
                edge_count["TRANSACTED"] += 1

        elif src == "SOCIAL":
            for mention in attrs.get("mentions", []):
                # Mention is "@handle" — find entity by handle
                handle = mention.lstrip("@")
                # Store as a MENTIONED edge (resolved by handle later)
                G.add_edge(src_root, src_root,
                           key=f"SOCIAL_{evt['event_id']}_{handle}",
                           edge_type="MENTIONED",
                           timestamp=ts,
                           mentioned_handle=handle,
                )
                edge_count["MENTIONED"] += 1

    log.info(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    log.info(f"  Edge types: {edge_count}")
    return G, root_map


# ---------------------------------------------------------------------------
# Step 7 — Resolve CDR phone edges to actual entity nodes
# ---------------------------------------------------------------------------

def resolve_phone_edges(G: nx.MultiDiGraph,
                        entities_df: pd.DataFrame,
                        id_index: dict[str, list[str]],
                        root_map: dict[str, str]):
    """Replace self-loop CDR edges with real caller->called edges."""
    to_add = []
    to_remove = []

    for u, v, key, data in list(G.edges(keys=True, data=True)):
        if data.get("edge_type") == "CALLED":
            called_phone = data.get("called_phone", "")
            if called_phone and called_phone in id_index:
                called_eids = id_index[called_phone]
                if called_eids:
                    dst_root = root_map.get(called_eids[0], called_eids[0])
                    if dst_root != u:  # avoid true self-loops
                        to_add.append((u, dst_root, key, data))
                        to_remove.append((u, v, key))

    removed = 0
    for (u, v, k) in to_remove:
        if G.has_edge(u, v, key=k):
            G.remove_edge(u, v, key=k)
            removed += 1

    for (u, dst, k, d) in to_add:
        G.add_edge(u, dst, key=k, **d)

    log.info(f"  Resolved {len(to_add)} CDR phone edges -> real targets "
             f"(removed {removed} self-loops)")


# ---------------------------------------------------------------------------
# Step 8 — Resolve social mention edges
# ---------------------------------------------------------------------------

def resolve_mention_edges(G: nx.MultiDiGraph,
                           id_index: dict[str, list[str]],
                           root_map: dict[str, str]):
    to_add = []
    to_remove = []

    for u, v, key, data in list(G.edges(keys=True, data=True)):
        if data.get("edge_type") == "MENTIONED":
            handle = data.get("mentioned_handle", "")
            if handle and handle in id_index:
                dst_eids = id_index[handle]
                if dst_eids:
                    dst_root = root_map.get(dst_eids[0], dst_eids[0])
                    if dst_root != u:
                        to_add.append((u, dst_root, key, data))
                        to_remove.append((u, v, key))

    for (u, v, k) in to_remove:
        if G.has_edge(u, v, key=k):
            G.remove_edge(u, v, key=k)

    for (u, dst, k, d) in to_add:
        G.add_edge(u, dst, key=k, **d)

    log.info(f"  Resolved {len(to_add)} social mention edges -> real targets")


# ---------------------------------------------------------------------------
# Step 9 — Ground-truth validation
# ---------------------------------------------------------------------------

def validate_against_ground_truth(G: nx.MultiDiGraph,
                                   root_map: dict[str, str],
                                   gt_path: str) -> dict:
    """
    Check that planted linked entities (shared IMEI, circular chain)
    actually end up in the same connected component in the undirected view.
    """
    if not os.path.exists(gt_path):
        log.warning("Ground truth file not found — skipping validation.")
        return {}

    with open(gt_path) as f:
        gt = json.load(f)

    results = {}
    UG = G.to_undirected()
    components = list(nx.connected_components(UG))

    def same_component(eid_a, eid_b):
        ra = root_map.get(eid_a, eid_a)
        rb = root_map.get(eid_b, eid_b)
        for comp in components:
            if ra in comp and rb in comp:
                return True
        return False

    # Check shared IMEI pair
    planted = gt.get("planted_anomalies", {})
    shared = planted.get("shared_imei", {})
    eids = shared.get("entity_ids", [])
    if len(eids) >= 2:
        ok = same_component(eids[0], eids[1])
        results["shared_imei"] = {
            "pass": ok,
            "entities": eids,
            "detail": f"{eids[0]} and {eids[1]} {'ARE' if ok else 'ARE NOT'} in same component",
        }
        log.info(f"  [GT] Shared IMEI: {'OK PASS' if ok else 'FAIL FAIL'} — {results['shared_imei']['detail']}")

    # Check circular chain all connected
    chain_eids = planted.get("circular_tx_chain", {}).get("entity_ids", [])
    if len(chain_eids) >= 2:
        pairs_ok = all(same_component(chain_eids[i], chain_eids[i+1])
                       for i in range(len(chain_eids)-1))
        results["circular_tx_chain"] = {
            "pass": pairs_ok,
            "entities": chain_eids,
            "detail": f"Chain {chain_eids} {'connected' if pairs_ok else 'NOT all connected'}",
        }
        log.info(f"  [GT] Circular chain: {'OK PASS' if pairs_ok else 'FAIL FAIL'}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    entities_df, events_df = load_data()

    # Build identifier index
    id_index = build_identifier_index(entities_df)
    log.info(f"  Identifier index: {len(id_index)} unique identifier values")

    # Union-Find over all entity_ids
    all_eids = entities_df["entity_id"].tolist()
    uf = UnionFind(all_eids)

    # Deterministic joins (shared phone/IMEI/account/handle)
    deterministic_join(entities_df, id_index, uf)

    # Fuzzy name joins
    fuzzy_name_join(entities_df, uf, FUZZY_NAME_THRESHOLD)

    # Build the graph
    G, root_map = build_graph(entities_df, events_df, uf)

    # Resolve edge targets
    resolve_phone_edges(G, entities_df, id_index, root_map)
    resolve_mention_edges(G, id_index, root_map)

    # Ground-truth validation
    val_results = validate_against_ground_truth(G, root_map, GROUND_TRUTH_FILE)

    # Save
    with open(GRAPH_FILE, "wb") as f:
        pickle.dump({"graph": G, "root_map": root_map, "validation": val_results}, f)

    log.info(f"\nOK Entity graph saved -> {GRAPH_FILE}")
    log.info(f"  Nodes: {G.number_of_nodes()}")
    log.info(f"  Edges: {G.number_of_edges()}")
    log.info(f"  Connected components (undirected): "
             f"{nx.number_connected_components(G.to_undirected())}")

    return G, root_map, val_results


if __name__ == "__main__":
    run()
