"""
tests/test_entity_resolution.py
Unit tests for the entity resolution layer.
Validates that planted linked entities are correctly merged.
"""

import os
import sys
import json
import pickle
import pytest
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    GRAPH_FILE, GROUND_TRUTH_FILE, ENTITIES_FILE, EVENTS_FILE,
    OUTPUT_DIR, CDR_FILE, IPDR_FILE, TX_FILE, SOCIAL_FILE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graph_data():
    """Run full pipeline (gen → ingest → entity_res) if needed."""
    # Generate data
    if not all(os.path.exists(f) for f in [CDR_FILE, IPDR_FILE, TX_FILE, SOCIAL_FILE]):
        import generate_synthetic_data
        generate_synthetic_data.main()

    # Ingest
    if not all(os.path.exists(f) for f in [ENTITIES_FILE, EVENTS_FILE]):
        import ingest
        ingest.run()

    # Entity resolution
    if not os.path.exists(GRAPH_FILE):
        import entity_resolution
        entity_resolution.run()

    with open(GRAPH_FILE, "rb") as f:
        data = pickle.load(f)

    return data


@pytest.fixture(scope="module")
def ground_truth():
    if not os.path.exists(GROUND_TRUTH_FILE):
        pytest.skip("Ground truth file not found")
    with open(GROUND_TRUTH_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Graph structure tests
# ---------------------------------------------------------------------------

class TestGraphStructure:
    def test_graph_not_empty(self, graph_data):
        G = graph_data["graph"]
        assert G.number_of_nodes() > 0, "Graph must have nodes"

    def test_graph_has_edges(self, graph_data):
        G = graph_data["graph"]
        assert G.number_of_edges() > 0, "Graph must have edges"

    def test_graph_type(self, graph_data):
        G = graph_data["graph"]
        assert isinstance(G, nx.MultiDiGraph), "Graph must be a NetworkX MultiDiGraph"

    def test_node_attributes(self, graph_data):
        G = graph_data["graph"]
        for node in list(G.nodes)[:10]:
            attrs = G.nodes[node]
            assert "canonical_id" in attrs, f"Node {node} missing 'canonical_id'"
            assert "name" in attrs, f"Node {node} missing 'name'"
            assert "is_suspect" in attrs, f"Node {node} missing 'is_suspect'"

    def test_edge_types_present(self, graph_data):
        G = graph_data["graph"]
        edge_types = {d.get("edge_type") for _, _, d in G.edges(data=True)}
        # At least one real edge type should be present
        known_types = {"CALLED", "TRANSACTED", "MENTIONED", "RELATED"}
        assert len(edge_types & known_types) > 0, \
            f"No known edge types found. Got: {edge_types}"

    def test_no_isolated_nodes_for_suspects(self, graph_data, ground_truth):
        """Suspect entities should have at least some edges."""
        G = graph_data["graph"]
        root_map = graph_data.get("root_map", {})
        planted = ground_truth.get("planted_anomalies", {})
        chain_eids = planted.get("circular_tx_chain", {}).get("entity_ids", [])
        for eid in chain_eids:
            root = root_map.get(eid, eid)
            if root in G:
                degree = G.degree(root)
                assert degree > 0, \
                    f"Suspect entity {eid} (root: {root}) has no edges in the graph"


# ---------------------------------------------------------------------------
# Ground-truth validation tests
# ---------------------------------------------------------------------------

class TestGroundTruthValidation:
    def _same_component(self, G, root_map, eid_a, eid_b) -> bool:
        ra = root_map.get(eid_a, eid_a)
        rb = root_map.get(eid_b, eid_b)
        UG = G.to_undirected()
        try:
            return nx.has_path(UG, ra, rb)
        except nx.NetworkXError:
            return False

    def test_shared_imei_entities_connected(self, graph_data, ground_truth):
        """
        Entities 0 and 1 share an IMEI. After entity resolution,
        they should be reachable from each other in the undirected graph.
        """
        G = graph_data["graph"]
        root_map = graph_data.get("root_map", {})
        planted  = ground_truth.get("planted_anomalies", {})
        eids     = planted.get("shared_imei", {}).get("entity_ids", [])

        assert len(eids) >= 2, "Ground truth must have at least 2 shared-IMEI entities"

        connected = self._same_component(G, root_map, eids[0], eids[1])
        assert connected, (
            f"Shared-IMEI entities {eids[0]} and {eids[1]} are NOT in the same "
            f"connected component — entity resolution failed to merge them."
        )

    def test_circular_chain_entities_connected(self, graph_data, ground_truth):
        """
        Entities in the circular transaction chain (E000→E001→E002→E000)
        should all be reachable from each other via the transaction edges.
        """
        G = graph_data["graph"]
        root_map = graph_data.get("root_map", {})
        planted  = ground_truth.get("planted_anomalies", {})
        eids     = planted.get("circular_tx_chain", {}).get("entity_ids", [])

        assert len(eids) >= 2, "Ground truth must have circular chain entities"

        for i in range(len(eids) - 1):
            connected = self._same_component(G, root_map, eids[i], eids[i+1])
            assert connected, (
                f"Circular chain entities {eids[i]} and {eids[i+1]} "
                f"are NOT connected — transaction edges missing."
            )

    def test_impossible_travel_entity_present(self, graph_data, ground_truth):
        """Entity_3 (impossible travel) should be in the graph."""
        G = graph_data["graph"]
        root_map = graph_data.get("root_map", {})
        planted  = ground_truth.get("planted_anomalies", {})
        eid      = planted.get("impossible_travel", {}).get("entity_id", "")

        assert eid, "Impossible travel entity not in ground truth"
        root = root_map.get(eid, eid)
        assert root in G, \
            f"Impossible travel entity {eid} (root: {root}) not found in graph"

    def test_tor_entity_present(self, graph_data, ground_truth):
        """Entity_4 (Tor user) should be in the graph."""
        G = graph_data["graph"]
        root_map = graph_data.get("root_map", {})
        planted  = ground_truth.get("planted_anomalies", {})
        eid      = planted.get("tor_entity", {}).get("entity_id", "")

        assert eid, "Tor entity not in ground truth"
        root = root_map.get(eid, eid)
        assert root in G, \
            f"Tor entity {eid} (root: {root}) not found in graph"

    def test_validation_results_recorded(self, graph_data):
        """Check that entity_resolution.py recorded its own validation results."""
        validation = graph_data.get("validation", {})
        assert validation, "No validation results recorded in graph pickle"

    def test_validation_shared_imei_pass(self, graph_data):
        """The shared-IMEI validation recorded in the graph must PASS."""
        validation = graph_data.get("validation", {})
        result = validation.get("shared_imei", {})
        assert result.get("pass", False), \
            f"Shared-IMEI ground truth validation FAILED: {result.get('detail','')}"
