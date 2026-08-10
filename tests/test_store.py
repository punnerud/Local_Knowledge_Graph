"""The embedding store: exactness, growth across questions, and mixed models.

Exactness is the point of these. The index this project used to depend on returns
the wrong neighbour on a current numpy (see scripts/bench_search.py and the
README), so "the search is exact" is now a property with a test behind it rather
than an assumption.
"""

import numpy as np
import pytest
from conftest import normal_script, read_events

from mpe_lkg.store import EmbeddingStore


def unit(rng, dim=32):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store(tmp_path):
    s = EmbeddingStore(str(tmp_path / "s.db"))
    yield s
    s.close()


class TestExactness:
    def test_a_vector_is_its_own_nearest_neighbour(self, store):
        """The failure mode that made the old index useless."""
        rng = np.random.default_rng(0)
        vectors = [unit(rng) for _ in range(50)]
        for i, v in enumerate(vectors):
            store.add(f"text {i}", v, model="m")

        for i in (0, 7, 23, 49):
            hits = store.find_similar(vectors[i], top_k=1, model="m")
            assert hits[0]["text"] == f"text {i}"
            assert hits[0]["similarity"] == pytest.approx(1.0, abs=1e-5)

    def test_ranking_matches_a_full_brute_force_sort(self, store):
        rng = np.random.default_rng(1)
        vectors = [unit(rng) for _ in range(200)]
        for i, v in enumerate(vectors):
            store.add(f"text {i}", v, model="m")

        query = unit(rng)
        matrix = np.vstack(vectors)
        expected = np.argsort(-(matrix @ query))[:5]

        hits = store.find_similar(query, top_k=5, model="m")
        assert [h["text"] for h in hits] == [f"text {i}" for i in expected]
        # Scores must be monotonically decreasing.
        assert all(a["similarity"] >= b["similarity"] for a, b in zip(hits, hits[1:], strict=False))

    def test_requesting_more_than_exists_is_not_an_error(self, store):
        rng = np.random.default_rng(2)
        store.add("only one", unit(rng), model="m")
        assert len(store.find_similar(unit(rng), top_k=10, model="m")) == 1

    def test_excluded_ids_never_appear(self, store):
        rng = np.random.default_rng(3)
        target = unit(rng)
        row_id = store.add("the query itself", target, model="m")
        for i in range(20):
            store.add(f"other {i}", unit(rng), model="m")

        hits = store.find_similar(target, top_k=5, model="m", exclude_ids={row_id})
        assert all(h["id"] != row_id for h in hits)
        assert len(hits) == 5

    def test_excluding_everything_returns_nothing(self, store):
        rng = np.random.default_rng(4)
        ids = {store.add(f"t{i}", unit(rng), model="m") for i in range(3)}
        assert store.find_similar(unit(rng), model="m", exclude_ids=ids) == []

    def test_empty_store(self, store):
        assert store.find_similar(np.ones(32, dtype=np.float32), model="m") == []


class TestCacheInvalidation:
    def test_a_new_row_is_visible_to_the_next_search(self, store):
        """The in-memory matrix must not outlive a write."""
        rng = np.random.default_rng(5)
        store.add("first", unit(rng), model="m")
        target = unit(rng)
        store.find_similar(target, model="m")  # warms the cache

        store.add("second", target, model="m")
        hits = store.find_similar(target, top_k=1, model="m")
        assert hits[0]["text"] == "second"

    def test_clear_empties_the_cache_too(self, store):
        rng = np.random.default_rng(6)
        v = unit(rng)
        store.add("gone", v, model="m")
        store.find_similar(v, model="m")
        store.clear()
        assert store.find_similar(v, model="m") == []
        assert store.count() == 0


class TestMixedModels:
    def test_dimensions_never_mix(self, store):
        rng = np.random.default_rng(7)
        small = unit(rng, 32)
        store.add("384-ish row", small, model="small")
        store.add("legacy 4096 row", np.ones(4096, dtype=np.float32), model="legacy")

        hits = store.find_similar(small, model="small")
        assert [h["text"] for h in hits] == ["384-ish row"]

    def test_same_dimension_different_model_is_kept_apart(self, store):
        rng = np.random.default_rng(8)
        v = unit(rng, 32)
        store.add("from model A", v, model="A")
        store.add("from model B", v, model="B")

        assert [h["text"] for h in store.find_similar(v, model="A")] == ["from model A"]
        assert [h["text"] for h in store.find_similar(v, model="B")] == ["from model B"]


class TestGrowthAcrossQuestions:
    """The store accumulates now; it used to be wiped at the start of each request."""

    def test_a_later_question_can_see_an_earlier_one(self, flask_client):
        client, _ = flask_client(normal_script() + normal_script())

        read_events(client.get("/query?query=What+is+the+capital+of+France"))
        events = read_events(client.get("/query?query=Tell+me+about+French+cities"))

        similar = [e for e in events if e["type"] == "similar"][0]
        assert similar["store_size"] > 6, "the second run must still see the first run's rows"
        assert similar["items"]

    def test_store_size_grows_monotonically(self, flask_client):
        client, _ = flask_client(normal_script() * 3)

        sizes = []
        for i in range(3):
            events = read_events(client.get(f"/query?query=question+{i}"))
            sizes.append([e for e in events if e["type"] == "similar"][0]["store_size"])

        assert sizes == sorted(sizes) and len(set(sizes)) == 3

    def test_hundreds_of_rows_stay_fast_and_exact(self, store):
        """The scale actually in question: hundreds of rows across many questions."""
        rng = np.random.default_rng(9)
        vectors = [unit(rng, 384) for _ in range(500)]
        for i, v in enumerate(vectors):
            store.add(f"row {i}", v, model="m")

        assert store.count() == 500
        hits = store.find_similar(vectors[321], top_k=3, model="m")
        assert hits[0]["text"] == "row 321"
        assert hits[0]["similarity"] == pytest.approx(1.0, abs=1e-5)
