"""Graph construction and the strongest-path search.

Pure functions over plain dicts and numpy arrays. No I/O, no Flask, no model. This
is the part the render tests pin, so it must stay free of side effects.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

# Similarities at or below this are treated as "no usable link". Cosine can go
# negative, and the log transform below is only defined on positive weights.
MIN_USABLE_SIMILARITY = 1e-6


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, safe on zero vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Full pairwise cosine similarity for a stack of row vectors."""
    if len(vectors) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalised = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return normalised @ normalised.T


def top_similarities(vectors, current_index: int, top_k: int = 2) -> list[tuple[int, float]]:
    """The ``top_k`` earlier vectors most similar to ``vectors[current_index]``.

    Returns ``(index, similarity)`` pairs, strongest first. Only indices strictly
    before ``current_index`` are considered, because an edge always points forward
    in the reasoning chain.
    """
    stack = np.asarray(vectors, dtype=np.float32)
    if current_index <= 0 or current_index >= len(stack):
        return []
    current = stack[current_index]
    scored = [(i, cosine_similarity(current, stack[i])) for i in range(current_index)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def build_graph(node_ids: list[str], labels: list[str], vectors, top_k: int = 2) -> dict:
    """Build the whole graph from scratch out of the current steps.

    Rebuilding rather than patching is what keeps node ids and embedding indices in
    lockstep: ``node_ids[i]`` owns ``vectors[i]``, by construction, for every i.
    The original code derived a node id from a separate running counter that could
    skip, so an edge could point at the wrong step -- and the guard for that case
    silently dropped the edge instead of reporting the mismatch.
    """
    if len(node_ids) != len(labels):
        raise ValueError(f"{len(node_ids)} node ids but {len(labels)} labels")
    stack = np.asarray(vectors, dtype=np.float32) if len(vectors) else np.zeros((0, 0), dtype=np.float32)
    if len(stack) != len(node_ids):
        raise ValueError(f"{len(node_ids)} node ids but {len(stack)} vectors")

    edges: list[dict] = []
    for index in range(1, len(node_ids)):
        for previous, similarity in top_similarities(stack, index, top_k=top_k):
            edges.append(
                {
                    "from": node_ids[previous],
                    "to": node_ids[index],
                    "value": float(similarity),
                    "length": float(300 * (1 - similarity)),
                }
            )

    nodes = []
    for index, node_id in enumerate(node_ids):
        connected = [e["value"] for e in edges if e["from"] == node_id or e["to"] == node_id]
        value = (sum(connected) / len(connected)) * 30 + 10 if connected else 20.0
        nodes.append({"id": node_id, "label": labels[index], "value": float(value)})

    return {"nodes": nodes, "edges": edges}


def serialize_graph_data(graph_data: dict) -> dict:
    """Convert internal graph state into what vis.js consumes.

    ``length`` is carried through deliberately. The original serializer dropped it,
    so the similarity-proportional spring length was computed on every edge and then
    thrown away before it could reach the browser.
    """
    return {
        "nodes": [dict(node) for node in graph_data.get("nodes", [])],
        "edges": [
            {
                "from": edge["from"],
                "to": edge["to"],
                "value": float(edge["value"]),
                "length": float(edge.get("length", 300 * (1 - float(edge["value"])))),
                "label": f"{float(edge['value']):.2f}",
                "font": {"size": 10},
            }
            for edge in graph_data.get("edges", [])
        ],
    }


def strongest_path(graph_data: dict, start_node: str = "", end_node: str = "") -> tuple:
    """Find the path whose similarities multiply to the largest value.

    Maximising a product of similarities is the same as minimising a sum of
    ``-log(similarity)``, and those costs are non-negative, so Dijkstra applies and
    the answer is exactly optimal.

    The original implementation accumulated ``cost - weight``, which makes every
    effective weight negative. Dijkstra's greedy choice is invalid on negative
    weights, so it returned the first path it happened to reach rather than the
    strongest one, and reported that as a "weighted average".

    Returns ``(path, weights, mean_similarity)``, or ``(None, None, None)`` when no
    path exists. ``mean_similarity`` is the geometric mean of the edges on the path,
    which is the quantity actually being maximised.
    """
    nodes = [node["id"] for node in graph_data.get("nodes", [])]
    if not nodes:
        return None, None, None

    start = start_node or nodes[0]
    end = end_node or nodes[-1]
    if start not in nodes or end not in nodes:
        return None, None, None
    if start == end:
        return [start], [], 1.0

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for edge in graph_data.get("edges", []):
        similarity = float(edge["value"])
        if similarity <= MIN_USABLE_SIMILARITY:
            continue
        cost = -math.log(min(similarity, 1.0))
        # Keep the strongest edge when a pair appears twice.
        existing = graph.get_edge_data(edge["from"], edge["to"])
        if existing is None or cost < existing["cost"]:
            graph.add_edge(edge["from"], edge["to"], cost=cost, similarity=similarity)

    try:
        path = nx.dijkstra_path(graph, start, end, weight="cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, None, None

    weights = [graph[u][v]["similarity"] for u, v in zip(path[:-1], path[1:], strict=True)]
    if not weights:
        return path, [], 1.0
    mean = float(math.exp(sum(math.log(w) for w in weights) / len(weights)))
    return path, weights, mean


def edge_weight_spread(graph_data: dict) -> dict:
    """Descriptive statistics for the edge weights actually drawn.

    Cosine similarities in a high-dimensional space concentrate: they cluster in a
    narrow band, so every edge looks equally strong and the picture stops carrying
    information. This reports the spread so the claim can be checked rather than
    assumed.
    """
    values = [float(edge["value"]) for edge in graph_data.get("edges", [])]
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "cv": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std())
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }
