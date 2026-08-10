"""Invariants of the graph layer.

Each test here pins something the original implementation got wrong silently.
"""

import numpy as np
import pytest

from graph import (
    build_graph,
    cosine_similarity,
    edge_weight_spread,
    serialize_graph_data,
    strongest_path,
    top_similarities,
)


def unit(*components) -> np.ndarray:
    vector = np.array(components, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class TestSimilarity:
    def test_identical_vectors_score_one(self):
        v = unit(1, 2, 3)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_similarity(unit(1, 0), unit(0, 1)) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_does_not_produce_nan(self):
        assert cosine_similarity(np.zeros(3, dtype=np.float32), unit(1, 1, 1)) == 0.0

    def test_only_earlier_indices_are_candidates(self):
        vectors = np.array([unit(1, 0), unit(0, 1), unit(1, 0.01)], dtype=np.float32)
        result = top_similarities(vectors, 2, top_k=2)
        assert [index for index, _ in result] == [0, 1]

    def test_first_node_has_no_candidates(self):
        vectors = np.array([unit(1, 0), unit(0, 1)], dtype=np.float32)
        assert top_similarities(vectors, 0) == []


class TestBuildGraph:
    """The node-id to embedding-index mapping is the invariant that used to break."""

    def test_node_ids_and_vectors_stay_aligned(self):
        ids = [f"Step{i}" for i in range(1, 5)]
        labels = [f"Step {i}" for i in range(1, 5)]
        vectors = np.array([unit(1, 0), unit(0, 1), unit(1, 0.02), unit(0, 1.02)], dtype=np.float32)

        graph = build_graph(ids, labels, vectors, top_k=1)

        # Step3 is nearly identical to Step1, Step4 to Step2. If ids and indices ever
        # drift apart, these edges land on the wrong nodes.
        by_target = {e["to"]: e["from"] for e in graph["edges"]}
        assert by_target["Step3"] == "Step1"
        assert by_target["Step4"] == "Step2"

    def test_edges_only_ever_point_at_existing_nodes(self):
        ids = [f"Step{i}" for i in range(1, 6)]
        vectors = np.random.default_rng(0).standard_normal((5, 8)).astype(np.float32)
        graph = build_graph(ids, ids, vectors, top_k=2)

        known = {node["id"] for node in graph["nodes"]}
        for edge in graph["edges"]:
            assert edge["from"] in known and edge["to"] in known

    def test_mismatched_input_lengths_raise_instead_of_dropping_edges(self):
        with pytest.raises(ValueError):
            build_graph(["a", "b"], ["a"], np.zeros((2, 4), dtype=np.float32))
        with pytest.raises(ValueError):
            build_graph(["a", "b"], ["a", "b"], np.zeros((3, 4), dtype=np.float32))

    def test_isolated_node_gets_default_size(self):
        graph = build_graph(["Step1"], ["Step 1"], np.array([unit(1, 0)]))
        assert graph["nodes"][0]["value"] == 20.0


class TestSerialization:
    def test_length_survives_serialization(self):
        """The spring length was computed and then dropped before reaching vis.js."""
        graph = {
            "nodes": [{"id": "Step1", "label": "a"}, {"id": "Step2", "label": "b"}],
            "edges": [{"from": "Step1", "to": "Step2", "value": 0.5, "length": 150.0}],
        }
        edge = serialize_graph_data(graph)["edges"][0]
        assert edge["length"] == pytest.approx(150.0)
        assert edge["label"] == "0.50"

    def test_length_is_derived_when_absent(self):
        graph = {"nodes": [], "edges": [{"from": "a", "to": "b", "value": 0.25}]}
        assert serialize_graph_data(graph)["edges"][0]["length"] == pytest.approx(225.0)

    def test_numpy_floats_become_json_safe_floats(self):
        graph = {"nodes": [], "edges": [{"from": "a", "to": "b", "value": np.float32(0.5)}]}
        assert type(serialize_graph_data(graph)["edges"][0]["value"]) is float


class TestStrongestPath:
    def test_picks_the_strongest_route_not_the_first_one_found(self):
        """A greedy search over negated weights returns whichever path it reaches first."""
        graph = {
            "nodes": [{"id": n} for n in ("Step1", "Step2", "Step3", "Step4")],
            "edges": [
                # Direct but weak.
                {"from": "Step1", "to": "Step4", "value": 0.10},
                # Longer but far stronger: 0.9 * 0.9 * 0.9 = 0.729 > 0.10
                {"from": "Step1", "to": "Step2", "value": 0.90},
                {"from": "Step2", "to": "Step3", "value": 0.90},
                {"from": "Step3", "to": "Step4", "value": 0.90},
            ],
        }
        path, weights, mean = strongest_path(graph, "Step1", "Step4")
        assert path == ["Step1", "Step2", "Step3", "Step4"]
        assert weights == pytest.approx([0.9, 0.9, 0.9])
        assert mean == pytest.approx(0.9)

    def test_missing_start_node_returns_none_instead_of_raising(self):
        """A graph without Step1 used to raise NetworkXError mid-stream."""
        graph = {"nodes": [{"id": "Step7"}, {"id": "Step8"}],
                 "edges": [{"from": "Step7", "to": "Step8", "value": 0.5}]}
        assert strongest_path(graph, "Step1", "Step8") == (None, None, None)

    def test_defaults_span_first_to_last_node(self):
        graph = {"nodes": [{"id": "Step7"}, {"id": "Step8"}],
                 "edges": [{"from": "Step7", "to": "Step8", "value": 0.5}]}
        path, weights, mean = strongest_path(graph)
        assert path == ["Step7", "Step8"]
        assert mean == pytest.approx(0.5)

    def test_disconnected_graph_returns_none(self):
        graph = {"nodes": [{"id": "Step1"}, {"id": "Step2"}], "edges": []}
        assert strongest_path(graph) == (None, None, None)

    def test_single_node_graph(self):
        assert strongest_path({"nodes": [{"id": "Step1"}], "edges": []}) == (["Step1"], [], 1.0)

    def test_non_positive_similarities_are_not_traversable(self):
        """Cosine can go negative; a negative edge is not a strong link."""
        graph = {
            "nodes": [{"id": "Step1"}, {"id": "Step2"}],
            "edges": [{"from": "Step1", "to": "Step2", "value": -0.4}],
        }
        assert strongest_path(graph) == (None, None, None)

    def test_empty_graph(self):
        assert strongest_path({"nodes": [], "edges": []}) == (None, None, None)


class TestEdgeSpread:
    def test_reports_coefficient_of_variation(self):
        graph = {"edges": [{"value": 0.8}, {"value": 0.8}, {"value": 0.8}]}
        spread = edge_weight_spread(graph)
        assert spread["n"] == 3
        assert spread["cv"] == pytest.approx(0.0)

    def test_empty_graph_is_not_a_division_by_zero(self):
        assert edge_weight_spread({"edges": []})["n"] == 0
