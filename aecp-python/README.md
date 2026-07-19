# aecp

Migrate a vector database to a new embedding model without re-embedding the corpus. Fits a linear map from ~2K calibration texts; 87-91% retrieval retention measured on BEIR.

## Install

```bash
pip install aecp
```

Python >= 3.10. Core deps: numpy, scikit-learn, typer, rich.

Optional: `pip install aecp[sentence-transformers]` for local models, `pip install aecp[qdrant]` for Qdrant store.

## Quickstart

```python
import numpy as np
from aecp import RidgeMapping

# Paired calibration embeddings (K texts embedded with both models)
rng = np.random.default_rng(0)
K, d_src, d_tgt = 500, 32, 48
X = rng.normal(size=(K, d_src))
Y = X @ rng.normal(size=(d_src, d_tgt))

m = RidgeMapping(alpha="auto", seed=0)
m.fit(X, Y)
m.save("mapping.aecp")

Z = m.transform(X[:10])  # (10, d_tgt), L2-normalized
```

## CLI

```bash
aecp plan --source-model text-embedding-ada-002 \
          --target-model text-embedding-3-large \
          --corpus-size 1000000

aecp calibrate --source-vectors X.npy --target-vectors Y.npy -o map.aecp
aecp transform --mapping map.aecp --source-dir ./old_store --target-dir ./new_store
aecp inspect map.aecp
```

## Serve mode (zero corpus writes)

Map new-model queries into legacy space. No re-embedding, instant rollback:

```python
from aecp.serve import QueryAdapter

qa = QueryAdapter.load("mapping.aecp")
legacy_vec = qa.map_query(new_model_embed(query))
results = qdrant.search(collection="docs", vector=legacy_vec)
```

## Results

All numbers from `benchmarks/results/`, verified by `benchmarks/audit_configs.py`.

### Adapter comparison (SciFact, MiniLM->bge-large, K=4000, 3 seeds)

| Adapter | nDCG@10 retention | Notes |
|---------|------------------|-------|
| Ridge | 0.866 +/- 0.008 | Default. Fast, stable. |
| LowRank | 0.857 +/- 0.009 | Compressed matrix. ~1% worse. |
| MLP | 0.719 +/- 0.008 | No tuning. Linear wins. |

### K-sweep (all adapters averaged, SciFact, 3 seeds)

| K | nDCG@10 retention | Gate |
|---|------------------|------|
| 500 | 0.667 +/- 0.039 | WARN |
| 1000 | 0.732 +/- 0.056 | WARN |
| 2000 | 0.788 +/- 0.054 | PASS |
| 4000 | 0.814 +/- 0.068 | PASS |

### Same-dim pair (bge-large->e5-large, 1024->1024)

| Metric | Value |
|--------|-------|
| Floor (raw cross-space) | 0.0 |
| AECP (mapped) | 0.656 |
| Ceiling (full re-embed) | 0.722 |
| Retention | 0.908 |

Same dimension != same space. e5 models require "query: "/"passage: " prefixes; without them ceiling drops to 0.36.

## When NOT to use AECP

- Maximum retrieval quality matters more than cost -> re-embed
- Calibration domain mismatches corpus (e.g., code index calibrated on prose)
- Quality gate returns FAIL -> do not migrate; re-embed
- You need unsupervised migration (AECP requires paired calibration)
- K < 2000 (quality degrades significantly below this)

## Anti-patterns

- Do not mix vectors from different models in one collection
- Do not assume same dimensionality means compatibility
- Do not skip the quality gate
- Do not use MLP adapter (0.719 vs 0.866 for Ridge, same cost)

## How it works

1. Embed K texts with source and target models -> matrices X, Y
2. Fit ridge map Y = [X | 1] W (handles unequal dims)
3. Hold out 10% to estimate quality
4. Transform corpus: V' = normalize(V @ W) (streaming batches)
5. Write to new collection; keep old as rollback

## Prior art

Engineering, not research. Built on:
- vec2vec (Jha et al., 2025)
- Drift-Adapter (EMNLP 2025)
- Platonic Representation Hypothesis (Huh et al., 2024)

## Security

Embedding translation enables inversion-style attacks. Treat mapped vectors with same sensitivity as source text.

## License

Apache-2.0
