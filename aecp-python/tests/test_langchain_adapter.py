"""Tests for LangChain adapter (mocked, no real LangChain required)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("langchain_core", reason="langchain_core not installed")

from aecp.mapping.linear import RidgeMapping


def _make_mapping(d_src=8, d_tgt=12, k=120):
    """Create a fitted RidgeMapping for tests."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(k, d_src))
    W = rng.normal(size=(d_src, d_tgt))
    Y = X @ W
    m = RidgeMapping(alpha="auto", seed=0).fit(X, Y)
    return m


class _FakeBaseEmbeddings:
    """Mock LangChain Embeddings base class."""

    def __init__(self, dim=12):
        self._dim = dim

    def embed_documents(self, texts):
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(texts), self._dim)).tolist()

    def embed_query(self, text):
        rng = np.random.default_rng(1)
        return rng.normal(size=(self._dim,)).tolist()


class TestAECPEmbeddings:
    """Test AECPEmbeddings shim."""

    def test_embed_documents(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        base = _FakeBaseEmbeddings(dim=12)

        from aecp.adapters.langchain import AECPEmbeddings

        ae = AECPEmbeddings(m, base)
        result = ae.embed_documents(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 8  # mapped to source dim

    def test_embed_query(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        base = _FakeBaseEmbeddings(dim=12)

        from aecp.adapters.langchain import AECPEmbeddings

        ae = AECPEmbeddings(m, base)
        vec = ae.embed_query("test")
        assert isinstance(vec, list)
        assert len(vec) == 8

    def test_embed_documents_consistency(self):
        """Same input -> same output (deterministic mapping)."""
        m = _make_mapping(d_src=8, d_tgt=12)

        class DeterministicEmbed:
            def embed_documents(self, texts):
                return np.ones((len(texts), 12)).tolist()
            def embed_query(self, text):
                return np.ones(12).tolist()

        from aecp.adapters.langchain import AECPEmbeddings

        ae = AECPEmbeddings(m, DeterministicEmbed())
        r1 = ae.embed_documents(["test"])
        r2 = ae.embed_documents(["test"])
        assert np.allclose(r1, r2)

    def test_has_recalibrator_false(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        base = _FakeBaseEmbeddings(dim=12)

        from aecp.adapters.langchain import AECPEmbeddings

        ae = AECPEmbeddings(m, base)
        assert not ae.has_recalibrator

    def test_repr(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        base = _FakeBaseEmbeddings(dim=12)

        from aecp.adapters.langchain import AECPEmbeddings

        ae = AECPEmbeddings(m, base)
        r = repr(ae)
        assert "AECPEmbeddings" in r
        assert "ridge" in r
        assert "d_src=8" in r
        assert "d_tgt=12" in r

    def test_rejects_bad_base(self):
        m = _make_mapping(d_src=8, d_tgt=12)

        from aecp.adapters.langchain import AECPEmbeddings

        with pytest.raises(TypeError, match="embed_documents"):
            AECPEmbeddings(m, "not an embeddings object")

    def test_async_embed_documents(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        base = _FakeBaseEmbeddings(dim=12)

        from aecp.adapters.langchain import AECPEmbeddings

        ae = AECPEmbeddings(m, base)
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            ae.aembed_documents(["hello"])
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == 8

    def test_async_embed_query(self):
        m = _make_mapping(d_src=8, d_tgt=12)
        base = _FakeBaseEmbeddings(dim=12)

        from aecp.adapters.langchain import AECPEmbeddings

        ae = AECPEmbeddings(m, base)
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            ae.aembed_query("hello")
        )
        assert isinstance(result, list)
        assert len(result) == 8
