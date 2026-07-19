"""Offline unit tests for CachedEmbedder (Fix F3 verification)."""

from __future__ import annotations

import numpy as np
import pytest

from aecp.providers.base import Embedder
from aecp.providers.cached import CachedEmbedder, with_disk_cache


class DummyEmbedder(Embedder):
    """Trivial embedder for testing: deterministic hash-based vectors."""

    def __init__(self, model_id: str = "dummy", dims: int = 8) -> None:
        self._model_id = model_id
        self._dims = dims
        self.call_count = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dims(self) -> int:
        return self._dims

    def embed(self, texts):
        self.call_count += len(texts)
        rng = np.random.default_rng(42)
        return rng.normal(size=(len(texts), self._dims))


def test_cache_hit_and_miss(tmp_path):
    inner = DummyEmbedder()
    cached = CachedEmbedder(inner, tmp_path)

    texts = ["hello", "world", "test"]
    v1 = cached.embed(texts)
    assert inner.call_count == 3

    v2 = cached.embed(texts)
    assert inner.call_count == 3  # no new calls
    assert cached.hits == 3
    assert cached.misses == 3
    np.testing.assert_allclose(v1, v2)


def test_partial_cache_hit(tmp_path):
    inner = DummyEmbedder()
    cached = CachedEmbedder(inner, tmp_path)

    cached.embed(["a", "b"])
    assert inner.call_count == 2

    cached.embed(["b", "c"])
    assert inner.call_count == 3  # only "c" is new
    assert cached.hits == 1
    assert cached.misses == 3


def test_stats_tracking(tmp_path):
    inner = DummyEmbedder()
    cached = CachedEmbedder(inner, tmp_path)

    cached.embed(["x"])
    stats = cached.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1
    assert stats["api_calls"] == 1
    assert stats["index_size"] == 1


def test_empty_input(tmp_path):
    inner = DummyEmbedder()
    cached = CachedEmbedder(inner, tmp_path)

    result = cached.embed([])
    assert result.shape == (0, 8)
    assert inner.call_count == 0


def test_idempotent_wrapping(tmp_path):
    inner = DummyEmbedder()
    cached1 = with_disk_cache(inner, tmp_path)
    cached2 = with_disk_cache(cached1, tmp_path)
    assert cached1 is cached2  # idempotent


def test_model_id_passthrough(tmp_path):
    inner = DummyEmbedder(model_id="my-model-v2")
    cached = CachedEmbedder(inner, tmp_path)
    assert cached.model_id == "my-model-v2"
    assert cached.dims == 8
