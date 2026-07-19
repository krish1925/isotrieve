"""Tests for aecp.serve (WS-3 query-adapter serving mode)."""

from __future__ import annotations

import numpy as np
import pytest

from aecp.mapping.linear import RidgeMapping
from aecp.serve import QueryAdapter, csls_scores, merge_results


def _make_bidirectional_mapping(d: int = 16, k: int = 200):
    """Create a well-fitted mapping with inverse."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(k, d))
    W = rng.normal(size=(d, d))
    Y = X @ W + 0.01 * rng.normal(size=(k, d))
    m = RidgeMapping(alpha=1.0, seed=0)
    m.fit(X, Y)
    return m, X, Y


def test_query_adapter_map_single():
    m, X, Y = _make_bidirectional_mapping()
    qa = QueryAdapter(m)

    vec = X[0]
    mapped = qa.map_query(vec)

    assert mapped.shape == (16,)
    assert np.isclose(np.linalg.norm(mapped), 1.0, atol=1e-6)


def test_query_adapter_map_batch():
    m, X, Y = _make_bidirectional_mapping()
    qa = QueryAdapter(m)

    vecs = X[:5]
    mapped = qa.map_queries(vecs)

    assert mapped.shape == (5, 16)
    norms = np.linalg.norm(mapped, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_query_adapter_properties():
    m, X, Y = _make_bidirectional_mapping()
    qa = QueryAdapter(m)

    assert qa.d_new == 16
    assert qa.d_legacy == 16


def test_query_adapter_dimension_mismatch():
    m, X, Y = _make_bidirectional_mapping()
    qa = QueryAdapter(m)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        qa.map_query(np.zeros(32))


def test_csls_basic():
    rng = np.random.default_rng(42)
    q = rng.normal(size=(10, 16))
    c = rng.normal(size=(100, 16))

    scores = csls_scores(q, c, k=5)

    assert scores.shape == (10, 100)
    assert np.all(np.isfinite(scores))


def test_merge_results_dedup():
    legacy = [{"id": "a", "score": 0.8}, {"id": "b", "score": 0.6}]
    native = [{"id": "a", "score": 0.9}, {"id": "c", "score": 0.7}]

    merged = merge_results(legacy, native, migrated_ids=set())

    ids = [r["id"] for r in merged]
    assert "a" in ids
    assert "b" in ids
    assert "c" in ids
    assert len(ids) == 3  # deduplicated


def test_merge_results_weights():
    legacy = [{"id": "a", "score": 0.5}]
    native = [{"id": "a", "score": 0.3}]

    merged = merge_results(legacy, native, migrated_ids=set(), legacy_weight=2.0)

    # a gets max(0.5*2.0, 0.3*1.0) = 1.0
    a_score = next(r["score"] for r in merged if r["id"] == "a")
    assert a_score == 1.0
