# isotrieve 0.3.0

Compatibility update: modern Qdrant/ChromaDB support, hardened CLI, CI pipeline, and demo notebooks.

## What's new in v0.3

- **Qdrant v1.18+ compatibility** — migrated from removed `search()` to `query_points()` API; minimum `qdrant-client>=1.12`
- **ChromaDB v1.0+ compatibility** — updated empty-collection check and `include=` parameter; minimum `chromadb>=1.0`
- **MLP device selection** — `ResidualMLPMapping(device=)` parameter for MPS/CUDA auto-detection; ~3-5x training speedup on Apple Silicon
- **MLP rectangular dims** — fixed residual connection for non-square source/target dimensions
- **MLP load fix** — registry now peeks header before full load, preventing numpy parse errors on `.pt` state dicts
- **Zero raw tracebacks** — all CLI commands (`calibrate`, `transform`, `inspect`, `gate`) wrapped in try/except with user-friendly error messages
- **6 demo notebooks** — quickstart, ChromaDB migration, Qdrant migration, adapter intermediary, cross-architecture dims, recalibration
- **CI pipeline** — GitHub Actions: ruff lint, mypy typecheck, pytest matrix (3.10-3.12), claims linter
- **Claims linter** — `scripts/lint_claims.py` validates CLAIMS.md artifact references against `benchmarks/results/`
- **Visual assets** — pipeline flow diagram, benchmark recall chart (SVG + PNG)
- **Qdrant integration tests** — 8 in-memory tests (serve, migrate, dry run, empty, double-migrate, k-limit, MigrationReport)
- **CLI gate tests** — `gate` command with `--format json`, `calibrate --queries-only` mode

## Migration notes

- `qdrant-client` minimum bumped from `>=1.7` to `>=1.12` (required for `query_points()`)
- `chromadb` minimum bumped from `>=0.4` to `>=1.0` (required for updated API)
- MLP `save()` now includes `"device"` and `"matrix_shape"` in header metadata (backward-compatible)

---

# isotrieve 0.2.0

Embedding migration with vector DB adapters, score recalibration, and confidence scoring.

## What's new in v0.2

- **ChromaDB adapter** — `IsotrieveChromaFunction` (serve-mode `EmbeddingFunction`) + `migrate_collection()` (offline migration)
- **LangChain adapter** — `IsotrieveEmbeddings` (drop-in `Embeddings` shim)
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
pip install isotrieve
pip install isotrieve[chroma]      # ChromaDB
pip install isotrieve[langchain]   # LangChain
pip install isotrieve[qdrant]      # Qdrant
pip install isotrieve[all]         # Everything
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

- API model pair benchmarks (ada-002->te3-large)
- pgvector adapter (planned, no adapter yet)
- MCP wrapper for agent frameworks

## License

Apache-2.0
