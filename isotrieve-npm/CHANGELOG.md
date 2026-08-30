# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-30

### Added
- Jest test suite for `@isotrieve/demo-cli` (6 tests): results-table rendering extracted into a pure `buildResultsTable()` helper in `src/results.ts`; `npm test` at the monorepo root now passes for every workspace (previously failed: demo-cli had no test script)

### Fixed
- `npm install` failed for the whole monorepo: `@isotrieve/core` ranges in `adapters-cohere`, `adapters-huggingface`, `adapters-openai`, `adapters-voyage`, and `demo-cli` were `^1.0.0`, which can resolve neither to the local workspace (core is 0.1.0 after the version reset) nor to the registry (package never published). All workspace deps on core now use `*`
- demo-cli README described Isotrieve as "Agent Embedding Communication Protocol" (pre-rename expansion) — now uses the embedding-migration framing; package description/keywords updated to match
- Directory `packages/aecp-demo-cli` → `packages/isotrieve-demo-cli` and LICENSE attribution `AECP Contributors` → `Isotrieve Contributors` (landed in #83; pinned by Python-side naming regression tests)

### Known issues

- #85 — `.isotrieve` files written by this package are rejected by the strict Python reader (header key casing) — TS→PY handoff broken until fixed

## [0.1.0] - 2026-07-28

### Added

#### Binary format v2
- **8-byte header padding**: JSON header padded to 8-byte boundary for zero-copy Float64Array views
- **CRC32 checksum**: CRC32 over entire file (excluding checksum field) for corruption detection
- **Strict reader**: unknown header keys throw, missing required keys throw (prevents silent empty matrices)
- **Header length cap**: headerLen validated against 1MB ceiling before allocation (DoS mitigation)
- **Payload size assertion**: rows × cols × 8 must equal remaining bytes exactly (catches truncation at format layer)
- **Endianness assertion**: module-init check that platform is LE (documented as format requirement)
- **Prototype pollution fix**: normalizeHeaderKeys uses Object.create(null) to prevent __proto__ injection

#### Core Package (@isotrieve/core)
- **Mapping types**: RidgeMapping, OrthogonalProcrustesMapping, ProcrustesDiagMapping, LowRankAffineMapping, ExternalMapping
- **Binary format**: `.isotrieve` file read/write, cross-compatible with Python's isotrieve v2 format
- **Quality gate**: `QualityGate` with isotonic regression model (gate_model_v1.json shared with Python)
- **Score recalibration**: `ScoreRecalibrator` with PAVA isotonic regression
- **Calibration planning**: `planCalibration`, `recommendK`
- **Reranking**: `ConfidenceScorer`, `confidenceSummary`
- **Migration**: `migrateStore` with batch processing, manifest, resumability
- **Serve**: `QueryAdapter`, `cslsScores`, `mergeResults`
- **Math**: SVD (one-sided Jacobi with m<n transpose fix), ridge regression with GCV alpha, matrix operations, cosine similarity, L2 normalization
- **Tests**: 268 tests across 8 suites (linalg, normalize/metrics, errors, mappings, quality, correctness, cross-compat, stress)

### Notes
- This is the first release of the new embedding-migration architecture for TypeScript
- The old protocol code (`Isotrieve`, `IsotrieveNegotiator`, `protocol.ts`, `negotiation.ts`) has been removed from `@isotrieve/core`
- Adapter packages (`adapters-openai`, `adapters-voyage`, `adapters-cohere`, `adapters-huggingface`) still reference the old API and will be updated in a future release
- This package is pre-1.0. APIs may change between minor versions.
- v2 format is not backward-compatible with v1 readers; v1 files can still be loaded
