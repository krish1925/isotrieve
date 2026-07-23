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
- `isotrieve calibrate --queries-only` mode
- Pinecone adapter (shadow-namespace strategy)
- Qdrant adapter promoted to `VectorStoreAdapter`
- LlamaIndex storage-context migration path
- `docs/when-reembedding-is-impossible.md`
- Wrapper telemetry (opt-in, local-only JSONL)
- `isotrieve report --from-wrapper` command

### Changed
- README restructured: problem → wrapper quickstart → gate → migration → adapters → claims → prior art

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
