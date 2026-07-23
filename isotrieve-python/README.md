# isotrieve

Embedding providers deprecate models constantly — ada-002 is gone, text-embedding-3 is next. When that happens, you either re-embed your entire corpus (expensive, slow, risky) or get stuck on a dead model. Isotrieve lets you switch without re-embedding: fit a lightweight linear transform from ~2K calibration texts, apply it to stored vectors, and gate the migration on measured retrieval retention. 87-91% retention on BEIR benchmarks.

## Install

```bash
pip install isotrieve
```

Python >= 3.10. Core deps: numpy, scikit-learn, typer, rich.

Optional extras:
- `pip install isotrieve[chroma]` — ChromaDB adapter
- `pip install isotrieve[langchain]` — LangChain embeddings shim
- `pip install isotrieve[llamaindex]` — LlamaIndex query wrapper
- `pip install isotrieve[sentence-transformers]` — local model support
- `pip install isotrieve[qdrant]` — Qdrant store adapter
- `pip install isotrieve[openai]` — OpenAI client shim
- `pip install isotrieve[all]` — everything above

## 5-minute trial: query-time wrapper

Zero writes to your vector store. Map new-model queries into legacy space on-the-fly. Fully reversible.

### LlamaIndex

```python
from isotrieve.wrappers.llamaindex import IsotrieveEmbedding
from isotrieve.mapping.registry import load_mapping

mapping = load_mapping("mapping.isotrieve")
wrapper = IsotrieveEmbedding(
    new_model_embedder=your_llamaindex_embedder,
    transform_artifact_path="mapping.isotrieve",
)
# Use wrapper anywhere LlamaIndex expects a BaseEmbedding
# Queries are mapped; document embeddings raise IsotrieveWrapperUsageError
```

### OpenAI client

```python
import openai
from isotrieve.wrappers.openai_shim import IsotrieveOpenAI

client = openai.OpenAI()
shim = IsotrieveOpenAI(client, "mapping.isotrieve")
response = shim.embeddings.create(input=["query text"], model="text-embedding-3-small")
# response.data[0].embedding is now in legacy-model space
```

### LangChain

```python
from isotrieve.adapters.langchain import IsotrieveEmbeddings
from langchain_openai import OpenAIEmbeddings

mapping = Mapping.load("mapping.isotrieve")
base = OpenAIEmbeddings(model="text-embedding-3-small")
ae = IsotrieveEmbeddings(mapping, base)

from langchain_chroma import Chroma
db = Chroma.from_documents(docs, embedding=ae)
results = db.similarity_search("query", k=10)
```

## Quality gate

Before migrating anything, verify the transform preserves retrieval quality:

```bash
isotrieve gate --mapping mapping.isotrieve \
          --source-vectors X_sample.npy \
          --target-vectors Y_sample.npy
```

Output: retention table (Recall@1/5/10, MRR), bootstrap confidence intervals, per-metric pass/fail, and a one-line verdict. Exit code 0 for PASS, 1 for WARN/FAIL — use it in CI.

## Full migration

```bash
# 1. Plan cost
isotrieve plan --source-model ada-002 --target-model te3-large --corpus-size 1000000

# 2. Calibrate
isotrieve calibrate --source-vectors X.npy --target-vectors Y.npy -o mapping.isotrieve

# 3. Gate
isotrieve gate --mapping mapping.isotrieve --source-vectors X.npy --target-vectors Y.npy

# 4. Migrate
isotrieve transform --mapping mapping.isotrieve --source-dir ./old_store --target-dir ./new_store
```

### Serve mode (zero corpus writes)

Map queries on-the-fly without touching stored data:

```python
from isotrieve.serve import QueryAdapter

qa = QueryAdapter.load("mapping.isotrieve")
legacy_vec = qa.map_query(new_model_embed(query))
```

## Adapter status

| Store | Serve mode | Offline migration | Status |
|-------|-----------|-------------------|--------|
| ChromaDB | `IsotrieveChromaFunction` | `migrate_collection()` | Supported |
| LangChain | `IsotrieveEmbeddings` | via store adapter | Supported |
| LlamaIndex | `IsotrieveEmbedding` wrapper | via store adapter | Query wrapper |
| OpenAI | `IsotrieveOpenAI` shim | N/A | Query shim |
| Qdrant | `QdrantStore` | checkpointed in-place | Supported |
| Pinecone | — | shadow-namespace | Planned |

## Claims policy

Every quantitative claim in this README or docs references a committed artifact in `benchmarks/results/` and a row in `isotrieve-python/CLAIMS.md`. No exceptions. If a number isn't in CLAIMS.md, it isn't a claim.

### Adapter comparison (SciFact, MiniLM→bge-large, K=4000, 3 seeds)

| Adapter | nDCG@10 retention | Notes |
|---------|------------------|-------|
| Ridge | 0.871 ± 0.006 | Default. Fast, stable. |
| LowRank | 0.857 ± 0.009 | Compressed matrix. ~1% worse. |
| MLP | 0.727 ± 0.007 | No tuning. Linear wins. |

### K-sweep (all adapters averaged, SciFact, 3 seeds)

| K | nDCG@10 retention | Gate |
|---|------------------|------|
| 500 | 0.671 ± 0.041 | WARN |
| 1000 | 0.735 ± 0.058 | WARN |
| 2000 | 0.785 ± 0.052 | PASS |
| 4000 | 0.832 ± 0.061 | PASS |

### Same-dim pair (bge-large→e5-large, 1024→1024)

| Metric | Value |
|--------|-------|
| Floor (raw cross-space) | 0.0 |
| Isotrieve (mapped) | 0.667 |
| Ceiling (full re-embed) | 0.722 |
| Retention | 0.923 ± 0.010 |

Same dimension ≠ same space. e5 models require "query: "/"passage: " prefixes; without them ceiling drops to 0.36.

### Confidence flags (predictive across both pairs)

| Pair | High-conf R@10 | Low-conf R@10 | Gap |
|------|---------------|---------------|-----|
| bge→e5 | 0.955 | 0.637 | 0.318 |
| MiniLM→bge | 0.875 | 0.651 | 0.224 |

### Score recalibration (MiniLM→bge, rectangular)

| Threshold | Raw recall | + Recalibration | Δ |
|-----------|-----------|-----------------|---|
| τ = 0.60 | 78% | 100% | +22% |
| τ = 0.70 | 27% | 67% | +40% |

## When NOT to use Isotrieve

- Maximum retrieval quality matters more than cost → re-embed
- Calibration domain mismatches corpus (e.g., code index calibrated on prose)
- Quality gate returns FAIL → do not migrate; re-embed
- You need unsupervised migration (Isotrieve requires paired calibration)
- K < 2000 (quality degrades significantly below this)

## Anti-patterns

- Do not mix vectors from different models in one collection
- Do not assume same dimensionality means compatibility
- Do not skip the quality gate
- Do not use MLP adapter (0.727 vs 0.871 for Ridge, same cost)

## How it works

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
