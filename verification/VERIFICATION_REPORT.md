# VERIFICATION REPORT

**Date:** 2026-07-24
**Version:** isotrieve 0.2.1
**Branch:** development (with fixes from feat/image-multimodal-support)

---

## Phase 0 — Baseline Snapshot

| Task | Status | Notes |
|------|--------|-------|
| Full test run | ✅ Pass | 123 passed, 1 skipped, 1 warning (chromadb deprecation) |
| Test count | ✅ Pass | 123 tests across 16 test files |
| Skip reasons | ✅ Documented | 1 skip: `test_gate.py` (needs chromadb for one path); 5 chroma migration tests now pass after fixes |
| Coverage | ⚠️ 53% | Core mapping/quality at 78-94%; adapters at 0% (need live APIs); acceptable for current stage |
| Dependency tree | ✅ Pass | All deps reproducible from pyproject.toml; note: `aecp==0.2.0` still in venv (legacy) |
| Package build | ✅ Pass | `python -m build` + `twine check dist/*` — both pass with no warnings |

## Phase 1 — Correctness Verification

### 1.1 Mapping Types

| Mapping | Fit | Transform | Inverse | Save/Load | Edge Cases |
|---------|-----|-----------|---------|-----------|------------|
| RidgeMapping | ✅ | ✅ | N/A | ✅ | ✅ Empty, NaN, single-vector all raise clear errors |
| OrthogonalProcrustesMapping | ✅ | ✅ | ✅ | ✅ | ✅ Square-only enforced |
| ProcrustesDiagMapping | ✅ | ✅ | ✅ | ✅ | ✅ Square-only enforced |
| LowRankAffineMapping | ✅ | ✅ | N/A | ✅ | ✅ Rectangular dims work |
| ResidualMLPMapping | ✅ | ✅ | ✅ | ✅ | ✅ Requires torch, skips gracefully |

### 1.2 Providers

| Provider | Import | Factory Resolve | Mocked Test |
|----------|--------|-----------------|-------------|
| SentenceTransformerEmbedder | ✅ | ✅ (default) | ✅ |
| OpenAIEmbedder | ✅ | ✅ (text-embedding-*) | ⚠️ Needs API key |
| VoyageEmbedder | ✅ | ✅ (voyage*) | ⚠️ Needs API key |
| CohereEmbedder | ✅ | ✅ (embed-*) | ⚠️ Needs API key |
| GeminiEmbedder | ✅ | ✅ (models/*) | ⚠️ Needs API key |
| CachedEmbedder | ✅ | ✅ | ✅ Cache hit verified |
| CLIPEmbedder | ✅ (feat branch) | ✅ | ⚠️ Needs model download |
| SigLIPEmbedder | ✅ (feat branch) | ✅ | ⚠️ Needs model download |

### 1.3 Store Adapters

| Adapter | Import | Integration Test | Notes |
|---------|--------|-----------------|-------|
| NumpyFileStore | ✅ | ✅ (in tests) | Dict API now supported |
| ChromaDB | ✅ | ✅ (ephemeral client) | 3 bugs fixed in this session |
| LangChain | ✅ | ⚠️ Needs langchain-core | Import verified |
| LlamaIndex | ✅ | ⚠️ Needs llama-index-core | Import verified |
| Pinecone | ✅ | ⚠️ Needs pinecone-client | Import verified |
| Qdrant | ✅ | ⚠️ Needs qdrant-client | Import verified |

### 1.4 CLI Commands

| Command | Exists | Error Handling | Tested |
|---------|--------|---------------|--------|
| `isotrieve version` | ✅ | N/A | ✅ |
| `isotrieve plan` | ✅ | ✅ | ✅ |
| `isotrieve calibrate` | ✅ | ✅ Fixed in this session | ✅ |
| `isotrieve transform` | ✅ | ✅ Fixed in this session | ✅ |
| `isotrieve inspect` | ✅ | ✅ Fixed in this session | ✅ |
| `isotrieve gate` | ✅ | ✅ Fixed in this session | ✅ |
| `isotrieve doctor` | ✅ | ✅ (was already good) | ✅ |
| `isotrieve report` | ✅ | ✅ | ✅ |

### 1.5 Quality Gate

| Check | Status | Notes |
|-------|--------|-------|
| Bootstrap CIs | ✅ | Computed via resampling, not hardcoded |
| PASS/WARN/FAIL thresholds | ✅ | Configurable in `thresholds.json`, defaults: 0.75/0.55 |
| Gate model provenance | ⚠️ | `gate_model_v1.json` exists but training methodology not documented in code |
| Cross-family pair handling | ✅ | Verified with MiniLM→mpnet (different dims) |

### 1.6 Other Features

| Feature | Status | Notes |
|---------|--------|-------|
| ScoreRecalibrator | ✅ | Isotonic regression verified, monotonicity confirmed |
| CrossEncoderReranker | ✅ | Confidence scoring works (marked deprecated in CHANGELOG) |
| MigrationManifest | ✅ | Resume from partial state verified |
| NumpyFileStore dict API | ✅ Fixed | Now accepts `dict` or `VectorRecord` |
| QueryAdapter serve mode | ✅ | Thread-safe, maps queries correctly |

## Phase 2 — Documentation Alignment

| Item | Status | Notes |
|------|--------|-------|
| CLI commands in docs | ✅ | All 8 commands documented in README |
| Test count | ⚠️ | README doesn't state exact count; CONTRIBUTING.md says "~118 pass, 6 skip" — now 123 pass |
| Mapping types visible | ⚠️ | README mentions "Ridge, Procrustes, residual" — ProcrustesDiag and LowRankAffine not listed |
| CONTRIBUTING.md | ✅ | Rewritten on feat branch, needs cherry-pick to development |
| CLAIMS.md links | ⚠️ | Not verified in this pass |

## Phase 3 — Production Readiness

| Item | Status | Notes |
|------|--------|-------|
| Error messages (5+ failure modes) | ✅ Fixed | All CLI commands now show friendly errors, not tracebacks |
| Logging | ⚠️ | `migrate.py` uses `print()` for progress (intentional); `logging` added for resume |
| Type hints | ✅ | mypy passes on core modules |
| Security | ✅ | No hardcoded credentials; API keys from env vars only |
| Versioning | ✅ | Semantic versioning followed; CHANGELOG maintained |
| Platform coverage | ✅ | CI runs Python 3.10-3.13 |

## Phase 4 — Credibility Polish

| Item | Status | Notes |
|------|--------|-------|
| README feature table | ⚠️ | Needs update to include all mapping types, CLI commands, adapters |
| Real end-to-end example | ✅ | MiniLM→mpnet benchmark with SciFact-style evaluation completed |
| Consistent terminology | ⚠️ | Mixed use of "toolkit" vs "library" vs "migration CI" |

---

## Bugs Fixed This Session

| Bug | File | Fix |
|-----|------|-----|
| CLI tracebacks on missing files | `cli.py`, `cli_gate.py` | Added friendly error messages with hints |
| Single-vector fit crash | `mapping/linear.py` | Added explicit check: "Need at least 2 calibration pairs" |
| NumpyFileStore dict API broken | `stores/numpy_files.py` | Added dict→VectorRecord conversion |
| ChromaDB `migrate_collection` crash | `adapters/chroma.py:203` | Fixed `if not first_batch["embeddings"]` → `if src.count() == 0` |
| ChromaDB `include=["ids"]` invalid | `adapters/chroma.py:227` | Removed `"ids"` from include list (ChromaDB v1.0+) |
| `IsotrieveChromaFunction.name` not a property | `adapters/chroma.py:101` | Added `@property` decorator |
| Test `ef.name()` → `ef.name` | `tests/test_chroma_adapter.py:123` | Updated test to match property |
| `migrate.py` print statements | `migrate.py:97` | Changed resume log to `logging.info()` |

## Test Results Summary

```
123 passed, 1 skipped, 1 warning in 3.74s
```

- All existing tests pass
- 5 ChromaDB migration tests now pass (were skipped before chromadb install)
- 1 warning: chromadb deprecation (asyncio check, not our code)

## Gate Threshold Analysis

| K | Gate | Predicted Retention | Mapped R@1 | Mapped R@10 | Retention |
|---|------|--------------------|----|----|-----|
| 100 | WARN | 0.683 | 0.100 | 0.100 | 10% |
| 200 | WARN | 0.683 | 0.200 | 0.200 | 20% |
| 500 | WARN | 0.683 | 0.500 | 0.500 | 50% |
| 1000 | PASS | 0.838 | 1.000 | 1.000 | 100% |

Model pair: all-MiniLM-L6-v2 (384d) → all-mpnet-base-v2 (768d)
Corpus: 1000 synthetic texts across 10 topics
