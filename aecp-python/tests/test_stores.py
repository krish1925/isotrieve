"""Tests for NumpyFileStore."""

from __future__ import annotations

import numpy as np

from aecp.mapping.linear import RidgeMapping
from aecp.stores.base import VectorRecord
from aecp.stores.numpy_files import NumpyFileStore


def test_numpy_store_roundtrip(tmp_path):
    vecs = np.random.randn(100, 16)
    texts = [f"doc {i}" for i in range(100)]
    store = NumpyFileStore.from_arrays(tmp_path / "src", vecs, texts=texts)
    assert store.count() == 100
    batches = list(store.iter_vectors(batch_size=32))
    assert sum(len(b) for b in batches) == 100
    assert batches[0][0].text == "doc 0"


def test_migrate_numpy_store(tmp_path):
    rng = np.random.default_rng(0)
    d_src, d_tgt = 8, 12
    k = 10 * d_src
    X = rng.normal(size=(k, d_src))
    W = rng.normal(size=(d_src, d_tgt))
    Y = X @ W
    m = RidgeMapping(alpha=0.1, seed=0).fit(X, Y)

    corpus = rng.normal(size=(500, d_src))
    src = NumpyFileStore.from_arrays(tmp_path / "src", corpus)
    dst = NumpyFileStore(tmp_path / "dst", create=True)

    def gen():
        for batch in src.iter_vectors(batch_size=128):
            V = np.stack([r.vector for r in batch])
            Z = m.transform(V)
            yield [
                VectorRecord(id=batch[i].id, vector=Z[i], text=batch[i].text)
                for i in range(len(batch))
            ]

    n = dst.write_vectors(gen())
    assert n == 500
    assert dst.count() == 500
    out = np.load(dst.vectors_path)
    assert out.shape == (500, d_tgt)
