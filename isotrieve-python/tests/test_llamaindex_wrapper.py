"""Tests for LlamaIndex wrapper (mocked, no real LlamaIndex required)."""

from __future__ import annotations

import asyncio

import pytest
from tests.fakes import FakeLlamaIndexEmbedding, make_mapping, save_mapping

from isotrieve.wrappers import IsotrieveWrapperUsageError


class TestIsotrieveEmbedding:
    """Test LlamaIndex query-time wrapper."""

    def test_query_embedding(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12)
        path = save_mapping(m, tmp_path)
        embedder = FakeLlamaIndexEmbedding(dim=12)

        from isotrieve.wrappers.llamaindex import IsotrieveEmbedding

        wrapper = IsotrieveEmbedding(embedder, path)
        vec = wrapper._get_query_embedding("test query")
        assert isinstance(vec, list)
        assert len(vec) == 8  # mapped to legacy (source) space

    def test_async_query_embedding(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12)
        path = save_mapping(m, tmp_path)
        embedder = FakeLlamaIndexEmbedding(dim=12)

        from isotrieve.wrappers.llamaindex import IsotrieveEmbedding

        wrapper = IsotrieveEmbedding(embedder, path)
        result = asyncio.run(wrapper._aget_query_embedding("hello"))
        assert isinstance(result, list)
        assert len(result) == 8

    def test_text_embedding_raises(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12)
        path = save_mapping(m, tmp_path)
        embedder = FakeLlamaIndexEmbedding(dim=12)

        from isotrieve.wrappers.llamaindex import IsotrieveEmbedding

        wrapper = IsotrieveEmbedding(embedder, path)
        with pytest.raises(IsotrieveWrapperUsageError, match="query-time only"):
            wrapper._get_text_embedding("doc text")

    def test_text_embeddings_raises(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12)
        path = save_mapping(m, tmp_path)
        embedder = FakeLlamaIndexEmbedding(dim=12)

        from isotrieve.wrappers.llamaindex import IsotrieveEmbedding

        wrapper = IsotrieveEmbedding(embedder, path)
        with pytest.raises(IsotrieveWrapperUsageError, match="query-time only"):
            wrapper._get_text_embeddings(["doc1", "doc2"])

    def test_async_text_embedding_raises(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12)
        path = save_mapping(m, tmp_path)
        embedder = FakeLlamaIndexEmbedding(dim=12)

        from isotrieve.wrappers.llamaindex import IsotrieveEmbedding

        wrapper = IsotrieveEmbedding(embedder, path)
        with pytest.raises(IsotrieveWrapperUsageError, match="query-time only"):
            asyncio.run(wrapper._aget_text_embedding("doc"))

    def test_dimension_mismatch_raises(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12)
        path = save_mapping(m, tmp_path)
        embedder = FakeLlamaIndexEmbedding(dim=99)  # wrong dim

        from isotrieve.wrappers.llamaindex import IsotrieveEmbedding

        with pytest.raises(ValueError, match="dim"):
            IsotrieveEmbedding(embedder, path)

    def test_repr(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12)
        path = save_mapping(m, tmp_path)
        embedder = FakeLlamaIndexEmbedding(dim=12)

        from isotrieve.wrappers.llamaindex import IsotrieveEmbedding

        wrapper = IsotrieveEmbedding(embedder, path)
        r = repr(wrapper)
        assert "IsotrieveEmbedding" in r
        assert "d_src=8" in r
        assert "d_tgt=12" in r

    def test_properties(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12)
        path = save_mapping(m, tmp_path)
        embedder = FakeLlamaIndexEmbedding(dim=12)

        from isotrieve.wrappers.llamaindex import IsotrieveEmbedding

        wrapper = IsotrieveEmbedding(embedder, path)
        assert wrapper.d_src == 8
        assert wrapper.d_tgt == 12
        assert not wrapper.has_recalibrator
