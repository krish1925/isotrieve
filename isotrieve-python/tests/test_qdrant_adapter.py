"""Qdrant adapter integration tests — in-memory mode, no server needed."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from isotrieve.mapping.linear import RidgeMapping

qdrant_client = pytest.importorskip(
    "qdrant_client", reason="qdrant-client not installed"
)
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

D_SRC = 8
D_TGT = 12


def _make_mapping(d_src=D_SRC, d_tgt=D_TGT, k=120):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(k, d_src))
    W = rng.normal(size=(d_src, d_tgt))
    Y = X @ W
    m = RidgeMapping(alpha="auto", seed=0).fit(X, Y)
    return m


def _populate_collection(client: QdrantClient, name: str, n: int, dim: int):
    """Insert n random vectors into a collection."""
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    rng = np.random.default_rng(99)
    points = [
        PointStruct(id=i, vector=rng.normal(size=dim).tolist(), payload={"idx": i})
        for i in range(n)
    ]
    client.upsert(collection_name=name, points=points)


def _make_adapter(mapping, client: QdrantClient, collection: str):
    """Create a QdrantAdapter with an in-memory client, bypassing __init__ URL parsing."""
    from isotrieve.adapters.qdrant import QdrantAdapter

    fake_factory = lambda url, api_key=None: client  # noqa: E731
    with patch(
        "isotrieve.adapters.qdrant._require_qdrant",
        return_value=(fake_factory, PointStruct),
    ):
        adapter = QdrantAdapter(mapping, url=":memory:", collection=collection)
    return adapter


class TestQdrantInMemory:
    """Integration tests using QdrantClient(':memory:').

    Source collection stores d_src vectors (old model).
    Serve mode: queries come in d_tgt (new model), inverse-transformed to d_src.
    Migrate mode: source d_src vectors transformed to d_tgt in new collection.
    """

    def test_query_serve_mode(self):
        client = QdrantClient(":memory:")
        _populate_collection(client, "docs", n=50, dim=D_SRC)

        m = _make_mapping()
        adapter = _make_adapter(m, client, "docs")

        query = np.random.default_rng(7).normal(size=(3, D_TGT))
        results = adapter.query(query, k=5)

        assert len(results) == 3
        for hits in results:
            assert len(hits) <= 5
            for hit in hits:
                assert "id" in hit
                assert "score" in hit

    def test_query_single_vector(self):
        client = QdrantClient(":memory:")
        _populate_collection(client, "docs", n=10, dim=D_SRC)

        m = _make_mapping()
        adapter = _make_adapter(m, client, "docs")

        query = np.random.default_rng(7).normal(size=D_TGT)
        results = adapter.query(query.reshape(1, -1), k=3)
        assert len(results) == 1
        assert len(results[0]) <= 3

    def test_migrate_creates_target(self):
        client = QdrantClient(":memory:")
        _populate_collection(client, "src", n=20, dim=D_SRC)

        m = _make_mapping()
        adapter = _make_adapter(m, client, "src")

        report = adapter.migrate(new_collection="dst")

        assert report.rows_processed == 20
        assert report.target_collection == "dst"

        info = client.get_collection("dst")
        assert info.points_count == 20

    def test_migrate_dry_run(self):
        client = QdrantClient(":memory:")
        _populate_collection(client, "src", n=15, dim=D_SRC)

        m = _make_mapping()
        adapter = _make_adapter(m, client, "src")

        report = adapter.migrate(dry_run=True)
        assert report.rows_processed == 15

        info = client.get_collection("src")
        assert info.points_count == 15

    def test_migrate_preserves_payload(self):
        client = QdrantClient(":memory:")
        _populate_collection(client, "src", n=10, dim=D_SRC)

        m = _make_mapping()
        adapter = _make_adapter(m, client, "src")

        adapter.migrate(new_collection="dst")

        points, _ = client.scroll(collection_name="dst", limit=10, with_payload=True)
        for pt in points:
            assert pt.payload is not None
            assert "idx" in pt.payload
            assert "isotrieve_mapping_id" in pt.payload

    def test_migrate_default_target_name(self):
        client = QdrantClient(":memory:")
        _populate_collection(client, "myvecs", n=5, dim=D_SRC)

        m = _make_mapping()
        adapter = _make_adapter(m, client, "myvecs")

        report = adapter.migrate()
        assert report.target_collection == "myvecs_migrated"

        info = client.get_collection("myvecs_migrated")
        assert info.points_count == 5

    def test_empty_collection(self):
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name="empty",
            vectors_config=VectorParams(size=D_SRC, distance=Distance.COSINE),
        )

        m = _make_mapping()
        adapter = _make_adapter(m, client, "empty")

        report = adapter.migrate(new_collection="empty_out")
        assert report.rows_processed == 0

    def test_double_migration_detection(self):
        client = QdrantClient(":memory:")
        _populate_collection(client, "src", n=5, dim=D_SRC)

        m = _make_mapping()
        adapter = _make_adapter(m, client, "src")

        report = adapter.migrate(new_collection="dst")
        assert report.idempotent

    def test_query_k_limit(self):
        client = QdrantClient(":memory:")
        _populate_collection(client, "docs", n=30, dim=D_SRC)

        m = _make_mapping()
        adapter = _make_adapter(m, client, "docs")

        query = np.random.default_rng(7).normal(size=(1, D_TGT))
        results = adapter.query(query, k=2)
        assert len(results) == 1
        assert len(results[0]) <= 2


class TestMigrationReport:
    def test_to_dict(self):
        from isotrieve.adapters.base import MigrationReport

        r = MigrationReport(
            rows_processed=100,
            elapsed_seconds=1.5,
            sampled_recall_at_10=0.95,
            mapping_checksum="abc123",
            source_collection="src",
            target_collection="dst",
            errors=[],
            idempotent=True,
        )
        d = r.to_dict()
        assert d["rows_processed"] == 100
        assert d["sampled_recall_at_10"] == 0.95
        assert d["idempotent"] is True
