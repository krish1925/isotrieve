# Changelog

All notable changes to Isotrieve will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Cross-package naming-convention regression tests (`tests/test_naming_conventions.py`, 13 tests): pin the v0.2.1 `AECP → Isotrieve` rename across both packages — LICENSE attribution, no pre-rename name in live source/docs/packaging metadata (Python src, notebooks, npm package src/README/package.json, website), notebook Colab URLs, npm directory/lockfile layout, and resolvable `@isotrieve/core` workspace ranges
- Jest test suite for `@isotrieve/demo-cli` (6 tests) with the results table extracted into a pure `buildResultsTable()` helper — `npm test` at the monorepo root previously failed because the package had no test script
- LlamaIndex query-time wrapper (`isotrieve.wrappers.llamaindex`)
- OpenAI client shim (`isotrieve.wrappers.openai_shim`)
- Shared test fakes (`tests/fakes.py`)
- Deprecation playbooks (`docs/playbooks/`)
- `isotrieve doctor` CLI command
- `isotrieve gate` CLI command with retention table, bootstrap CIs, and exit codes
- Self-contained HTML gate report
- GitHub Actions composite action for gate
- `isotrieve.toml` configuration file for gate thresholds
- Pinecone adapter (shadow-namespace strategy)
- Qdrant adapter promoted to `VectorStoreAdapter`
- LlamaIndex storage-context migration path
- `docs/when-reembedding-is-impossible.md`
- Wrapper telemetry (opt-in, local-only JSONL)
- `isotrieve report --from-wrapper` command
- First-class manifest lifecycle CLI: `isotrieve manifest list|show`, `isotrieve rollback` (with `--dry-run`), and `isotrieve resume` (closes #24)
- `isotrieve verify` post-migration drift & gate revalidation against the recorded gate result (closes #23)
- `MigrationManifest` extended backward-compatibly with `id`, `mapping_path`, `transform_invertible`, `rollback_strategy`, `gate_result`, and store specs for resume/rollback (closes #23, #24)
- `Mapping.has_inverse` / `Mapping.meta` accessors so manifests can record invertibility and model ids
- `docs/migration-lifecycle.md` with a Mermaid state diagram
- Domain-matrix benchmark (issue #37): dataset→domain routing (scifact/nfcorpus=medical, fiqa=general) plus offline synthetic `code`/`legal` corpora; `probe_retention` (same-index identifier top-1/top-10) in every run result
- `benchmarks/domain_probes.py` — deterministic, offline identifier-probe definitions (ICD-10 codes, case citations, file paths, error codes, UUIDs, dates, version strings)
- Gate report `domain_regime` field (general/legal/medical/code) inferred from sampled corpus text via `isotrieve.quality.domain.infer_domain`; `isotrieve gate --corpus-texts` wires it in
- `isotrieve doctor` now infers the store's domain regime and suggests the most relevant published domain benchmark

### Changed
- README restructured: problem → wrapper quickstart → gate → migration → adapters → claims → prior art
- **Binary format v2**: 8-byte header padding, CRC32 checksum, 1MB header cap (cross-compatible with TS @isotrieve/core)
- CI: add Python 3.13 to the test matrix; run mypy with `--strict` (closes #13)
- CLAIMS linter: warn on stale `verified` dates (>180 days) and cross-check docs numbers against CLAIMS.md (closes #14; linter stays internal/informational)
- Qdrant adapter test suite to Chroma parity: 10k seed → kill mid-run → resume, rollback via target drop, scroll batch-boundary edge cases (closes #17)
- **pgvector adapter** (`isotrieve.adapters.pgvector.PgvectorAdapter`): transactional in-place migration via shadow-column (`embedding_isotrieve_new`) + atomic column swap, idempotent resume, drop-column rollback, HNSW/IVFFlat index-rebuild guidance in `docs/pgvector.md` (closes #22)
- `NumpyFileStore.write_vectors` now appends to existing vectors so multi-batch migrations and `resume` accumulate correctly instead of clobbering earlier batches
- Removed the last `AECP` naming remnants from the v0.2.1 rename: LICENSE copyright lines in both packages, `isotrieve-npm/packages/aecp-demo-cli/` → `isotrieve-demo-cli/` (directory now matches `package-lock.json`, which already resolved `packages/isotrieve-demo-cli`), demo-cli internal variable names, stale `aecp-python/` `.gitignore` entry, and the root README project tree now points at `SKILLS.md` instead of the nonexistent `AGENTS.md` (closes #82)
- npm workspace: `@isotrieve/core` dependency ranges in all consumer packages (`adapters-*`, `demo-cli`) changed `^1.0.0` → `*`; core was version-reset to 0.1.0 (DECISIONS.md) and never published, so `^1.0.0` could resolve neither to the workspace nor the registry — `npm install` failed for the whole monorepo
- `@isotrieve/demo-cli`: description/keywords updated from the old "agent communication" framing to embedding migration; README no longer expands Isotrieve as "Agent Embedding Communication Protocol"

### Fixed
- Notebook Colab badge URLs pointed at the pre-rename repo `krish1925/AECP` — now `krish1925/isotrieve` (closes #82)

## [0.3.0] - 2026-07-25

### Added
- Qdrant v1.18+ support: migrated from removed `search()` to `query_points()` API
- ChromaDB v1.0+ support: updated empty-collection check and `include=` parameter
- `ResidualMLPMapping(device=)` for MPS/CUDA auto-detection (~3-5x training speedup on Apple Silicon)
- Rectangular dimension support for MLP (`d_in != d_out`)
- Registry header peek before full model load (prevents numpy parse errors on `.pt` state dicts)
- `_load_npy()` helper for safe `.npy` file loading with validation
- `calibrate --queries-only` mode for query-side calibration
- 8 in-memory Qdrant integration tests
- 15 MLP mapping tests (fit/transform, rectangular, inverse, save/load, device, determinism)
- 2 calibrate `--queries-only` tests
- 6 self-contained demo notebooks (Quickstart, ChromaDB, Qdrant, Adapter-as-Intermediary, Cross-Architecture, Recalibration)
- GitHub Actions CI workflow: ruff lint, mypy typecheck, pytest matrix (3.10–3.12), claims linter
- `scripts/lint_claims.py` validates CLAIMS.md artifact references
- Pipeline flow diagram and benchmark recall chart (SVG + PNG)
- `.gitignore` for isotrieve-python/

### Changed
- `qdrant-client` minimum bumped `>=1.7` → `>=1.12`
- `chromadb` minimum bumped `>=0.4` → `>=1.0`
- MLP `save()` includes `"device"` and `"matrix_shape"` in header metadata (backward-compatible)
- README rewritten: "Migration CI for vector stores" framing, verified adapter table, PyPI/CI/license badges
- All CLI commands (`calibrate`, `transform`, `inspect`, `gate`) wrapped in try/except with friendly error messages
- Version tests made dynamic (no hardcoded version strings)

### Fixed
- MLP residual connection crash on dimension mismatch (`x + net(x)` when `d_in != d_out`)
- Registry loading `.pt` state dicts without peeking header first
- CLI traceback leaks — all commands now catch errors and display actionable hints
- Notebook `pass_realloc` undefined variable bug
- Lambda and unused import lint violations

## [0.2.1] - 2026-07-22

### Added
- PyPI publish workflow via trusted publishing (`release.yml` on `v*` tags)
- Release validation tests (`tests/test_release.py`)
- GitHub Pages deployment workflow for the website
- Expansion program: LangChain/LlamaIndex wrappers, adapters, gate telemetry, deprecation playbooks
- Sitemap, robots.txt, and SEO meta tags for the website
- Mermaid flowcharts in README

### Changed
- Renamed `aecp` → `isotrieve` across the entire codebase
- Repo URLs updated to `krish1925/isotrieve`
- Benchmark K values and retention numbers corrected to match actual results
- GitHub Pages deploy trigger moved to `main` (was `development`)

### Fixed
- Gate margin compression always returned `None` (issue #8)
- Qdrant `write_vectors` iterator materialization + `list[VectorRecord]` handling (issue #9)
- Chroma metadata enrichment dropped AECP metadata when row metadata was `None` (issue #10)
- Optional-dep test failures converted to skips instead of failures (issue #11)
- CLAIMS.md drift from fresh SciFact runs (issue #12)
- Gate HTML report output
- Version test made dynamic instead of hardcoded
- CI lint, typecheck, and format failures

## [0.2.0] - 2026-07-19

### Added
- Score recalibration (isotonic regression) for reliable similarity thresholds
- Confidence scoring with adaptive P33/P67 margins
- ChromaDB adapter: serve-mode `IsotrieveChromaFunction` and offline `migrate_collection()`
- LangChain adapter: `IsotrieveEmbeddings` drop-in wrapper
- Qdrant vector store (`QdrantStore`)
- Provider implementations: OpenAI, Voyage, Cohere, Gemini, Sentence Transformers
- Cached embedding provider (content-addressed disk cache)
- CLI commands: `plan`, `calibrate`, `transform`, `inspect`
- `MigrationManifest` for resumable migrations
- `QueryAdapter` serve mode (zero-corpus-write migration)
- `csls_scores` hubness correction
- `merge_results` for progressive migration
- Built-in calibration corpus (`isotrieve-calib-v1`)
- Binary `.isotrieve` format with versioned header

### Changed
- MLP adapter is non-default; Ridge is the recommended adapter (0.866 vs 0.719 retention)

### Deprecated
- Cross-encoder reranking (NULL RESULT: -10.7pts due to domain mismatch)

## [0.1.0] - 2026-02-04

### Added
- Initial release
- Ridge, Procrustes, LowRank, ResidualMLP mapping algorithms
- Quality gate with isotonic regression model
- BEIR SciFact benchmark results
