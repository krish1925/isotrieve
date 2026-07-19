"""
Example 01 — migrate a numpy file store with a fitted RidgeMapping.

Runs fully offline on synthetic paired embeddings (no API keys).
"""

from pathlib import Path

import numpy as np

from aecp import RidgeMapping
from aecp.stores import NumpyFileStore, VectorRecord


def main() -> None:
    rng = np.random.default_rng(0)
    d_src, d_tgt = 32, 48
    k = 10 * min(d_src, d_tgt)

    # Stand-in for: embed same K texts with source and target models
    X = rng.normal(size=(k, d_src))
    W_true = rng.normal(size=(d_src, d_tgt))
    Y = X @ W_true + 0.02 * rng.normal(size=(k, d_tgt))

    mapping = RidgeMapping(alpha="auto", seed=0)
    mapping.fit(X, Y)
    print("Validation:", mapping.validation_report())

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    map_path = out / "synthetic.aecp"
    mapping.save(map_path)
    print(f"Saved {map_path}")

    # Fake "old" corpus in source space
    corpus = rng.normal(size=(5_000, d_src))
    src_dir = out / "corpus_src"
    dst_dir = out / "corpus_tgt"
    NumpyFileStore.from_arrays(src_dir, corpus)

    src = NumpyFileStore(src_dir)
    dst = NumpyFileStore(dst_dir, create=True)

    def batches():
        for batch in src.iter_vectors(batch_size=1024):
            V = np.stack([r.vector for r in batch])
            Z = mapping.transform(V)
            yield [
                VectorRecord(id=batch[i].id, vector=Z[i])
                for i in range(len(batch))
            ]

    n = dst.write_vectors(batches())
    print(f"Migrated {n} vectors → {dst_dir} (old store kept at {src_dir})")


if __name__ == "__main__":
    main()
