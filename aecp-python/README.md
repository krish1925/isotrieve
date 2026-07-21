# aecp

Migrate a vector database to a new embedding model without re-embedding the corpus. Fits a ridge regression map from ~2K calibration texts; 87-91% retrieval retention measured on BEIR.

Use cases: ada-002 to text-embedding-3 migration, changing embedding models in Qdrant/pgvector/ChromaDB, replacing a deprecated embedding provider, or switching from a hosted API to local models.

## Install

```bash
pip install aecp
```

Python >= 3.10. Core deps: numpy, scikit-learn, typer, rich.

Optional extras:
- `pip install aecp[chroma]` — ChromaDB adapter
- `pip install aecp[langchain]` — LangChain embeddings shim
- `pip install aecp[sentence-transformers]` — local model support
- `pip install aecp[qdrant]` — Qdrant store adapter
- `pip install aecp[all]` — everything above

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

For vector database migration without touching stored data: map new-model queries into legacy space. No re-embedding, instant rollback:

```python
from aecp.serve import QueryAdapter

qa = QueryAdapter.load("mapping.aecp")
legacy_vec = qa.map_query(new_model_embed(query))
results = qdrant.search(collection="docs", vector=legacy_vec)
```

## Vector DB adapters (v0.2)

### ChromaDB

**Serve mode** — drop-in `EmbeddingFunction`:

```python
from aecp.adapters.chroma import AECPChromaFunction
from aecp.mapping.base import Mapping

mapping = Mapping.load("ada002_to_te3.aecp")
ef = AECPChromaFunction(mapping, new_model_embedder=my_embed_fn)
col = client.get_collection("docs", embedding_function=ef)
results = col.query(query_texts=["..."], n_results=10)
```

**Offline migration** — transform stored vectors:

```python
from aecp.adapters.chroma import migrate_collection

report = migrate_collection(
    client, "docs", mapping,
    new_collection="docs_v2",
    batch_size=1000,
)
print(f"Migrated {report.rows_processed} rows, recall@10={report.sampled_recall_at_10:.3f}")
```

### LangChain

Drop-in `Embeddings` shim:

```python
from aecp.adapters.langchain import AECPEmbeddings
from langchain_openai import OpenAIEmbeddings

mapping = Mapping.load("ada002_to_te3.aecp")
base = OpenAIEmbeddings(model="text-embedding-3-small")
ae = AECPEmbeddings(mapping, base)

# Works with any LangChain vector store
from langchain_chroma import Chroma
db = Chroma.from_documents(docs, embedding=ae)
results = db.similarity_search("query", k=10)
```

## Score recalibration (v0.2)

Isotonic regression maps cross-space scores to ceiling-equivalent scores. Built into the mapping file; no extra steps:

```python
m = RidgeMapping(alpha="auto", seed=0).fit(X, Y)
m.fit_recalibrator(X_heldout, Y_heldout)  # optional
m.save("mapping.aecp")  # recalibrator saved alongside mapping

# At serve time
qa = QueryAdapter.load("mapping.aecp")
calibrated_scores = qa.recalibrate_scores(raw_scores)
```

## Confidence scoring (v0.2)

Per-query confidence flags with adaptive percentile-based margins:

```python
from aecp.reranking import ConfidenceScorer

scorer = ConfidenceScorer(margin_high=0.955, margin_low=0.637)
result = scorer.score(query_vector, top_scores)
print(result.flag)  # "high", "medium", or "low"
```

## Results

All numbers from `benchmarks/results/`, verified by `benchmarks/audit_configs.py`.

### Score recalibration agreement (bge-large→e5-large, same-dim)

| Threshold | Raw recall | + Recalibration | Δ |
|-----------|-----------|-----------------|---|
| τ ≤ 0.75 | 100% | 100% | 0 |
| τ = 0.80 | 12% | 17% | +4.7% |

### Score recalibration agreement (MiniLM→bge-large, rectangular)

| Threshold | Raw recall | + Recalibration | Δ |
|-----------|-----------|-----------------|---|
| τ = 0.60 | 78% | 100% | +22% |
| τ = 0.70 | 27% | 67% | +40% |
| τ = 0.80 | 8% | 19% | +11% |

### Confidence flags (predictive across both pairs)

| Pair | High-conf R@10 | Low-conf R@10 | Gap |
|------|---------------|---------------|-----|
| bge→e5 | 0.955 | 0.637 | 0.318 |
| MiniLM→bge | 0.875 | 0.651 | 0.224 |

### Adapter comparison (SciFact, MiniLM→bge-large, K=3840, 3 seeds)

| Adapter | nDCG@10 retention | Notes |
|---------|------------------|-------|
| Ridge | 0.871 +/- 0.006 | Default. Fast, stable. |
| LowRank | 0.862 +/- 0.010 | Compressed matrix. ~1% worse. |
| MLP | 0.729 +/- 0.008 | No tuning. Linear wins. |

### K-sweep (all adapters averaged, SciFact, 3 seeds)

| K | nDCG@10 retention | Gate |
|---|------------------|------|
| 500 | 0.667 +/- 0.039 | WARN |
| 1000 | 0.732 +/- 0.056 | WARN |
| 2000 | 0.788 +/- 0.054 | PASS |
| 4000 | 0.817 +/- 0.064 | PASS |

### Same-dim pair (bge-large→e5-large, 1024→1024)

| Metric | Value |
|--------|-------|
| Floor (raw cross-space) | 0.0 |
| AECP (mapped) | 0.656 |
| Ceiling (full re-embed) | 0.722 |
| Retention | 0.908 |

Same dimension != same space. e5 models require "query: "/"passage: " prefixes; without them ceiling drops to 0.36.

## When NOT to use AECP

- Maximum retrieval quality matters more than cost → re-embed
- Calibration domain mismatches corpus (e.g., code index calibrated on prose)
- Quality gate returns FAIL → do not migrate; re-embed
- You need unsupervised migration (AECP requires paired calibration)
- K < 2000 (quality degrades significantly below this)

## Anti-patterns

- Do not mix vectors from different models in one collection
- Do not assume same dimensionality means compatibility
- Do not skip the quality gate
- Do not use MLP adapter (0.729 vs 0.871 for Ridge, same cost)

## How it works

AECP solves embedding migration (changing your vector database's embedding model) with ridge regression:

1. Embed K texts with source and target models → matrices X, Y
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
