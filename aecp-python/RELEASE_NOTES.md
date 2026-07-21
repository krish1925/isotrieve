# aecp 0.2.0

Embedding migration with vector DB adapters, score recalibration, and confidence scoring.

## What's new in v0.2

- **ChromaDB adapter** — `AECPChromaFunction` (serve-mode `EmbeddingFunction`) + `migrate_collection()` (offline migration)
- **LangChain adapter** — `AECPEmbeddings` (drop-in `Embeddings` shim)
- **Score recalibration** — isotonic regression maps cross-space scores to ceiling-equivalent scores
- **Confidence flags** — per-query high/medium/low with adaptive P33/P67 margins
- **Independent inverse α** — separate regularization for forward and inverse directions (+2.2pts)
- **Core abstractions** — `EmbeddingAdapter`, `VectorStoreAdapter` ABCs

## What was tested and rejected

- Cross-encoder reranking: −10.7pts (MS MARCO domain-mismatched for sci-text)
- TSVD shrinkage: −0.33pt at rank=512, not worth complexity
- Procrustes centering: −55pt on unit vectors (breaks serve-mode)

## What was validated

- Full threshold agreement tables on both pairs (bge→e5, MiniLM→bge)
- Confidence flags predictive across both pairs (high=0.955, low=0.637 on bge→e5)
- Rectangular pair re-validation: 86% retention, margin compression 0.85x

## Install

```bash
pip install aecp
pip install aecp[chroma]      # ChromaDB
pip install aecp[langchain]   # LangChain
pip install aecp[qdrant]      # Qdrant
pip install aecp[all]         # Everything
```

## Highlights

- **RidgeMapping** with auto alpha selection and independent inverse α
- **QueryAdapter serve mode** — map queries into legacy space, zero corpus writes
- **QualityGate v2** — data-driven PASS/WARN/FAIL from trained model
- **ChromaDB + LangChain adapters** — drop-in wrappers for popular vector DBs
- **Score recalibration** — isotonic regression for threshold reliability
- **Confidence scoring** — per-query high/medium/low flags

## Benchmarks

| Adapter | nDCG@10 retention (SciFact, K=4000, 3 seeds) |
|---------|----------------------------------------------|
| Ridge | 0.866 ± 0.008 |
| LowRank | 0.857 ± 0.009 |
| MLP | 0.719 ± 0.008 |

Same-dim pair (bge-large→e5-large): 90.8% retention.

Score recalibration: essential for rectangular pairs (+22% at τ=0.60).
Confidence flags: high-conf R@10=0.955, low-conf R@10=0.637.

All numbers from `benchmarks/results/`, verified by `benchmarks/audit_configs.py`.

## What's next

- API model pair benchmarks (ada-002→te3-large)
- pgvector adapter
- LlamaIndex adapter
- MCP wrapper for agent frameworks

## License

Apache-2.0
