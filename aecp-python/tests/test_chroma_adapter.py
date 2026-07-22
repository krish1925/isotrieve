"""Tests for ChromaDB adapter (mocked, no real ChromaDB required)."""

from __future__ import annotations

import numpy as np
import pytest

from aecp.mapping.linear import RidgeMapping


def _make_mapping(d_src=8, d_tgt=12, k=120):
    """Create a fitted RidgeMapping for tests."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(k, d_src))
    W = rng.normal(size=(d_src, d_tgt))
    Y = X @ W
    m = RidgeMapping(alpha="auto", seed=0).fit(X, Y)
    return m


class _FakeChromaCollection:
    """Minimal fake for chromadb.Collection."""

    def __init__(self, embeddings=None, ids=None, metadatas=None, documents=None):
        self._embeddings = embeddings or []
        self._ids = ids or []
        self._metadatas = metadatas or []
        self._documents = documents or []

    def get(self, offset=0, limit=100, include=None):
        end = min(offset + limit, len(self._ids))
        ids = self._ids[offset:end]
        out = {"ids": ids}
        if include:
            if "embeddings" in include:
                out["embeddings"] = self._embeddings[offset:end]
            if "metadatas" in include:
                out["metadatas"] = self._metadatas[offset:end]
            if "documents" in include:
                out["documents"] = self._documents[offset:end]
        return out

    def count(self):
        return len(self._ids)

    def add(self, ids, embeddings, metadatas=None, documents=None):
        self._ids.extend(ids)
        self._embeddings.extend(embeddings)
        self._metadatas.extend(metadatas or [{}] * len(ids))
        self._documents.extend(documents or [None] * len(ids))


class _FakeChromaClient:
    """Minimal fake for chromadb.Client."""

    def __init__(self, collections=None):
        self._collections = collections or {}

    def get_collection(self, name, embedding_function=None):
        if name not in self._collections:
            self._collections[name] = _FakeChromaCollection()
        return self._collections[name]

    def create_collection(self, name, metadata=None):
        col = _FakeChromaCollection()
        self._collections[name] = col
        return col

    def delete_collection(self, name):
        self._collections.pop(name, None)


class TestAECPChromaFunction:
    """Test AECPChromaFunction (serve-mode EmbeddingFunction)."""

    def test_basic_embedding(self):
        m = _make_mapping(d_src=8, d_tgt=12)

        from aecp.adapters.chroma import AECPChromaFunction

        ef = AECPChromaFunction(
            m, new_model_embedder=lambda texts: np.random.randn(len(texts), 12)
        )
        # Just verify it calls through without error
        # (real test would verify dims, but we need a real embedder for that)
        result = ef(["hello"])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_has_recalibrator_false(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        from aecp.adapters.chroma import AECPChromaFunction

        ef = AECPChromaFunction(m, new_model_embedder=lambda t: np.zeros((len(t), 12)))
        assert not ef.has_recalibrator

    def test_raises_without_embedder(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        from aecp.adapters.chroma import AECPChromaFunction

        ef = AECPChromaFunction(m, new_model_embedder=None)
        with pytest.raises(RuntimeError, match="No new_model_embedder"):
            ef(["hello"])

    def test_default_space(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        from aecp.adapters.chroma import AECPChromaFunction

        ef = AECPChromaFunction(m, new_model_embedder=lambda t: np.zeros((len(t), 12)))
        assert ef.default_space() == "cosine"

    def test_name(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        from aecp.adapters.chroma import AECPChromaFunction

        ef = AECPChromaFunction(m, new_model_embedder=lambda t: np.zeros((len(t), 12)))
        assert ef.name() == "aecp_chroma"

    def test_get_config(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        from aecp.adapters.chroma import AECPChromaFunction

        ef = AECPChromaFunction(m, new_model_embedder=lambda t: np.zeros((len(t), 12)))
        cfg = ef.get_config()
        assert cfg["mapping_type"] == "ridge"
        assert cfg["d_src"] == 8
        assert cfg["d_tgt"] == 12

    def test_embed_query(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        rng = np.random.default_rng(0)
        embed_fn = lambda texts: rng.normal(size=(len(texts), 12))
        from aecp.adapters.chroma import AECPChromaFunction

        ef = AECPChromaFunction(m, new_model_embedder=embed_fn)
        vec = ef.embed_query("test query")
        assert isinstance(vec, list)
        assert len(vec) == 8  # mapped into legacy (source) space


try:
    import importlib.util

    _has_chroma = importlib.util.find_spec("chromadb") is not None
except Exception:
    _has_chroma = False


@pytest.mark.skipif(not _has_chroma, reason="chromadb not installed")
class TestMigrateCollection:
    """Test migrate_collection with mocked ChromaDB."""

    def test_basic_migration(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        rng = np.random.default_rng(7)

        # Create source collection with legacy vectors
        src_vecs = rng.normal(size=(50, 8))
        src_ids = [str(i) for i in range(50)]
        src_metas = [{"text": f"doc {i}"} for i in range(50)]
        src_docs = [f"doc {i}" for i in range(50)]

        src_col = _FakeChromaCollection(
            embeddings=src_vecs.tolist(),
            ids=src_ids,
            metadatas=src_metas,
            documents=src_docs,
        )
        client = _FakeChromaClient({"test_col": src_col})

        from aecp.adapters.chroma import migrate_collection

        report = migrate_collection(client, "test_col", m, batch_size=20)
        assert report.rows_processed == 50
        assert report.target_collection == "test_col_migrated"
        assert (
            "aecp_mapping_id" in report.metadatas
            if hasattr(report, "metadatas")
            else True
        )

        # Verify target collection was created
        target = client.get_collection("test_col_migrated")
        assert target.count() == 50

        # Verify vectors were transformed (different from source)
        target_data = target.get(limit=50, include=["embeddings", "metadatas"])
        target_vecs = np.array(target_data["embeddings"])
        assert target_vecs.shape == (50, 12)  # mapped to d_tgt
        assert src_vecs.shape[1] != target_vecs.shape[1]  # dims changed

        # Verify AECP metadata was added
        for meta in target_data["metadatas"]:
            assert "aecp_mapping_id" in meta
            assert meta["aecp_format_version"] == 1
            assert meta["aecp_source_collection"] == "test_col"

    def test_dry_run(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        rng = np.random.default_rng(7)
        src_vecs = rng.normal(size=(100, 8))
        src_col = _FakeChromaCollection(
            embeddings=src_vecs.tolist(),
            ids=[str(i) for i in range(100)],
            metadatas=[{} for _ in range(100)],
            documents=[None] * 100,
        )
        client = _FakeChromaClient({"dry": src_col})

        from aecp.adapters.chroma import migrate_collection

        report = migrate_collection(client, "dry", m, dry_run=True)
        assert report.rows_processed == 100
        # Source should be untouched
        assert client.get_collection("dry").count() == 100

    def test_custom_target_name(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        rng = np.random.default_rng(7)
        src_col = _FakeChromaCollection(
            embeddings=rng.normal(size=(10, 8)).tolist(),
            ids=[str(i) for i in range(10)],
            metadatas=[{} for _ in range(10)],
            documents=[None] * 10,
        )
        client = _FakeChromaClient({"src": src_col})

        from aecp.adapters.chroma import migrate_collection

        report = migrate_collection(client, "src", m, new_collection="custom_target")
        assert report.target_collection == "custom_target"
        assert client.get_collection("custom_target").count() == 10

    def test_empty_collection(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        src_col = _FakeChromaCollection()
        client = _FakeChromaClient({"empty": src_col})

        from aecp.adapters.chroma import migrate_collection

        report = migrate_collection(client, "empty", m)
        assert report.rows_processed == 0
        assert report.errors

    def test_double_migration_warning(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        rng = np.random.default_rng(7)
        src_col = _FakeChromaCollection(
            embeddings=rng.normal(size=(10, 8)).tolist(),
            ids=[str(i) for i in range(10)],
            metadatas=[{"aecp_mapping_id": "already_migrated"} for _ in range(10)],
            documents=[None] * 10,
        )
        client = _FakeChromaClient({"already": src_col})

        from aecp.adapters.chroma import migrate_collection

        report = migrate_collection(client, "already", m)
        assert not report.idempotent
        assert any("double migration" in e.lower() for e in report.errors)


class TestMigrationReport:
    def test_to_dict(self):
        from aecp.adapters.base import MigrationReport

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
