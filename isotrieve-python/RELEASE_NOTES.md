# isotrieve 0.3.0

**Migration CI for vector stores.** Compatibility update: modern Qdrant/ChromaDB support, hardened CLI, CI pipeline, and demo notebooks.

## What's new

### Adapter compatibility
- **Qdrant v1.18+** — migrated from removed `search()` to `query_points()` API
- **ChromaDB v1.0+** — updated empty-collection check (`src.count()`) and removed `"ids"` from `include=`
- `qdrant-client` floor bumped `>=1.7` → `>=1.12`; `chromadb` floor bumped `>=0.4` → `>=1.0`

### MLP mapping
- **Device selection** — `ResidualMLPMapping(device=)` for MPS/CUDA auto-detection; ~3-5x training speedup on Apple Silicon
- **Rectangular dims** — fixed residual connection for `d_in != d_out` (was forcing `x + net(x)` which crashes on dimension mismatch)
- **Load fix** — registry peeks header before full load, preventing numpy parse errors on `.pt` state dicts
- Dummy `_W`/`_W_inv` set on load so `_require_fitted()` passes

### CLI hardening
- Zero raw tracebacks — all commands (`calibrate`, `transform`, `inspect`, `gate`) wrapped in try/except
- `_load_npy()` helper validates file existence before `np.load()`, catches corrupt `.npy`
- Dimension mismatch, NaN/Inf, and empty-collection errors show friendly messages with hints
- `calibrate --queries-only` mode for query-side calibration

### CI & tooling
- GitHub Actions: ruff lint, mypy typecheck, pytest matrix (3.10–3.12), claims linter
- `scripts/lint_claims.py` validates CLAIMS.md artifact references against `benchmarks/results/`
- Pip caching on all CI jobs

### Tests (158 total, +35 since v0.2.1)
- 8 in-memory Qdrant integration tests (serve, migrate, dry run, empty, double-migrate, k-limit, MigrationReport)
- 15 MLP tests (fit/transform, rectangular, inverse, save/load roundtrip, device, determinism, Ridge comparison)
- 2 calibrate `--queries-only` tests
- Version consistency tests made dynamic (no hardcoded version strings)

### Notebooks (6 self-contained demos)
1. Quickstart — RidgeMapping fit → transform → evaluate → gate
2. ChromaDB migration — end-to-end collection migration
3. Qdrant migration — in-memory Qdrant adapter demo
4. Adapter as intermediary — same mapping used 3 ways (transform, serve, adapter)
5. Cross-architecture dimension change — 384→1536, Ridge vs Procrustes
6. Recalibration and drift — ScoreRecalibrator before/after

### Other
- Visual assets — pipeline flow diagram, benchmark recall chart (SVG + PNG)
- `write_vectors()` accepts `list[dict]` in addition to `VectorRecord`
- `.gitignore` added for isotrieve-python/
- README rewritten: "Migration CI for vector stores" framing, verified adapter table, PyPI/CI/license badges

## Migration notes

- `qdrant-client` minimum `>=1.7` → `>=1.12` (required for `query_points()`)
- `chromadb` minimum `>=0.4` → `>=1.0` (required for updated API)
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
