# Changelog

All notable changes to Isotrieve will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Changed
- README restructured: problem → wrapper quickstart → gate → migration → adapters → claims → prior art
- **Binary format v2**: 8-byte header padding, CRC32 checksum, 1MB header cap (cross-compatible with TS @isotrieve/core)

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
