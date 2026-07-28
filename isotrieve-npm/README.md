# Isotrieve — NPM Packages

> **Pre-1.0 / Beta** — TypeScript port of [isotrieve](https://pypi.org/project/isotrieve/). The Python package is the mature, benchmark-validated implementation.

Embedding migration without re-embedding. Learn a linear mapping between source and target embedding spaces from a small calibration sample (~2K texts), then transform stored vectors in-place or map queries on-the-fly. 87–91% retrieval retention on BEIR.

## Packages

| Package | Description | Status |
|---|---|---|
| [`@isotrieve/core`](./packages/core/) | Mapping, quality gate, migration, recalibration, serve | Beta |
| `@isotrieve/adapters-openai` | OpenAI embedding provider | Legacy (old protocol API) |
| `@isotrieve/adapters-voyage` | Voyage AI | Legacy |
| `@isotrieve/adapters-cohere` | Cohere | Legacy |
| `@isotrieve/adapters-huggingface` | HuggingFace (local) | Legacy |

**Only `@isotrieve/core` is current.** The adapter packages use the old `Isotrieve` class / `IsotrieveNegotiator` protocol and will be updated or removed before 1.0.

## Installation

```bash
npm install @isotrieve/core
```

Requires Node.js >= 18.

## Quick Start

```typescript
import { RidgeMapping, QualityGate } from '@isotrieve/core';

// Calibration pairs: X (old model), Y (new model)
const mapping = new RidgeMapping({ alpha: 'auto', seed: 0 });
mapping.fit(X_cal, Y_cal);

// Validate
mapping.getValidationReport(); // holdout cosine mean, top1 retention, etc.

// Save (cross-compatible with Python .isotrieve format)
mapping.save('mapping.isotrieve');

// Load
const loaded = mapping.constructor.load('mapping.isotrieve');
const Z = loaded.transform(corpus) as Float64Array[];

// Gate before committing
const gate = new QualityGate();
const report = gate.evaluate(loaded, X_test, Y_test);
console.log(report.verdict); // 'PASS' | 'WARN' | 'FAIL'
```

## Mapping Classes

All mappings share the same interface: `fit(X, Y)`, `transform(V)`, `inverseTransform(V)`, `save(path)`, `load(path)`.

| Mapping | Cross-dim | Inverse | Bias | Best for |
|---|---|---|---|---|
| `RidgeMapping` | Yes | Yes (fitted) | Default on | Default. Noise-robust, GCV alpha. |
| `OrthogonalProcrustesMapping` | No (d_src=d_tgt) | Yes (transpose) | No | Equal dims, orthogonal constraint. |
| `ProcrustesDiagMapping` | No (d_src=d_tgt) | Yes | No | Axis-aligned scaling + rotation. |
| `LowRankAffineMapping` | Yes | Yes (fitted) | Default on | Limited cal data, rank constraint. |
| `ExternalMapping` | Any | Optional | No | Wrap a custom transform fn. |

### RidgeMapping

```typescript
import { RidgeMapping } from '@isotrieve/core';

const m = new RidgeMapping({
  alpha: 'auto',           // 'auto' for GCV, or a number (default: 'auto')
  bias: true,              // include bias column (default: true)
  seed: 0,                 // for train/test split reproducibility
  normalizeOutput: true,   // L2-normalize transformed vectors (default: true)
  holdoutFraction: 0.1,    // fraction held out for validation (default: 0.1)
});

m.fit(X, Y);
console.log(m.dSrc, m.dTarget);    // dimensions
console.log(m.isFitted);            // true
console.log(m.getValidationReport().holdoutCosineMean);

const Z = m.transform(X) as Float64Array[];         // forward
const XBack = m.inverseTransform(Z) as Float64Array[]; // inverse
```

### OrthogonalProcrustesMapping

```typescript
import { OrthogonalProcrustesMapping } from '@isotrieve/core';

const m = new OrthogonalProcrustesMapping({
  seed: 0,
  normalizeOutput: true,
  holdoutFraction: 0.1,
});
m.fit(X, Y); // d_src === d_tgt required
// Inverse is exact: W^T
```

### ProcrustesDiagMapping

```typescript
import { ProcrustesDiagMapping } from '@isotrieve/core';

const m = new ProcrustesDiagMapping({
  seed: 0,
  normalizeOutput: true,
  holdoutFraction: 0.1,
});
m.fit(X, Y); // d_src === d_tgt required
// Diagonal scaling factors available via m.diag
```

### LowRankAffineMapping

```typescript
import { LowRankAffineMapping } from '@isotrieve/core';

const m = new LowRankAffineMapping({
  alpha: 'auto',
  rank: 8,         // truncate mapping to rank 8
  bias: true,
  seed: 0,
  normalizeOutput: true,
  holdoutFraction: 0.1,
});
m.fit(X, Y);
```

### ExternalMapping

```typescript
import { ExternalMapping } from '@isotrieve/core';

const m = new ExternalMapping({
  forwardFn: (vecs: Float64Array[]) => vecs.map(v => transform(v)),
  inverseFn: (vecs: Float64Array[]) => vecs.map(v => inverseTransform(v)),
  dSrc: 384,
  dTarget: 1536,
});
```

## Quality Gate

Data-driven PASS / WARN / FAIL verdict based on isotonic regression from calibrated gate model.

```typescript
import { QualityGate } from '@isotrieve/core';

const gate = new QualityGate();
const report = gate.evaluate(mapping, X_sample, Y_sample, {
  holdoutTop1: mapping.getValidationReport()?.top1Retention ?? null,
});

console.log(report.verdict);              // 'PASS' | 'WARN' | 'FAIL'
console.log(report.predictedRetention);    // estimated retention
console.log(report.predictionInterval);    // 80% CI [lower, upper]
console.log(report.marginCompression);     // score margin ratio
console.log(report.scoreRecomRecommendation); // recalibration advice
```

Thresholds (configurable in `thresholds.json`):
| Verdict | Predicted Retention |
|---|---|
| PASS | ≥ 0.75 |
| WARN | 0.55 – 0.75 |
| FAIL | < 0.55 |

## Score Recalibration

Maps compressed post-migration scores back to ceiling-equivalent values using PAVA isotonic regression.

```typescript
import { ScoreRecalibrator } from '@isotrieve/core';

// Fit on holdout (mapped_score, ceiling_score) pairs
const recal = new ScoreRecalibrator();
recal.fit(mappedScores, ceilingScores);

const adjusted = recal.transform(rawScores);
console.log(recal.report?.meanShift); // how much scores shifted
```

## Confidence Scoring

Per-query confidence flags (high / medium / low) based on top-1 vs top-2 margin.

```typescript
import { ConfidenceScorer, confidenceSummary } from '@isotrieve/core';

const scorer = new ConfidenceScorer();
const reports = scorer.scoreQueries(queryIds, similarities, topK);
const summary = confidenceSummary(reports);
```

## Serve Mode (QueryAdapter)

Map new-model queries into legacy space without writing to the corpus. Requires a mapping with fitted inverse.

```typescript
import { QueryAdapter, cslsScores } from '@isotrieve/core';

const qa = QueryAdapter.load('mapping.isotrieve');
const legacyVec = qa.mapQuery(newQueryVec);

// Batch
const legacyVecs = qa.mapQueries(batchQueryVecs);

// Recalibrate scores if recalibrator was fitted
const calibrated = qa.recalibrateScores(rawScores);

// CSLS scores (hubness correction)
const scores = cslsScores(queryVecs, candidateVecs, k=10);
```

## Migration

```typescript
import { migrateStore } from '@isotrieve/core';

const manifest = migrateStore(sourceStore, targetStore, mapping, {
  batchSize: 1024,
  manifestPath: './migration_manifest.json',
  resume: true,
});
```

## Binary Format (.isotrieve)

Cross-compatible with Python `isotrieve`. Layout:

```
[4 bytes] Magic "ISTR"
[4 bytes] Header length (LE uint32)
[4 bytes] CRC32 checksum
[N bytes] JSON header (padded to 8-byte boundary)
[...]     Float64 matrix payloads (row-major)
```

Read/write utilities:

```typescript
import { readIsotrieveHeader, readIsotrievePayload, writeIsotrieveFile } from '@isotrieve/core';

const header = readIsotrieveHeader('mapping.isotrieve');
const { header, matrices } = readIsotrievePayload('mapping.isotrieve');
```

## Notebooks

Interactive Python notebooks demonstrating end-to-end usage:

| Notebook | Description |
|---|---|
| [01 Quickstart: Ridge Mapping](https://github.com/krish1925/Isotrieve/blob/main/notebooks/01_quickstart_ridge_mapping.ipynb) | Fit, transform, and evaluate |
| [02 ChromaDB Migration](https://github.com/krish1925/Isotrieve/blob/main/notebooks/02_chromadb_migration.ipynb) | Migrate a ChromaDB collection |
| [03 Qdrant Migration](https://github.com/krish1925/Isotrieve/blob/main/notebooks/03_qdrant_migration.ipynb) | Migrate a Qdrant collection |
| [04 Adapter as Intermediary](https://github.com/krish1925/Isotrieve/blob/main/notebooks/04_adapter_as_intermediary.ipynb) | LangChain / LlamaIndex integration |
| [05 Cross-Architecture Dim Change](https://github.com/krish1925/Isotrieve/blob/main/notebooks/05_cross_architecture_dimension_change.ipynb) | Non-square transforms |
| [06 Recalibration and Drift](https://github.com/krish1925/Isotrieve/blob/main/notebooks/06_recalibration_and_drift.ipynb) | Score recalibration |
| [Reproduction](https://github.com/krish1925/Isotrieve/blob/main/reproduce.ipynb) | BEIR benchmark reproduction |

## Which package to use?

| | Python (`pip install isotrieve`) | TypeScript (`npm install @isotrieve/core`) |
|---|---|---|
| Status | Stable, benchmark-validated | Beta |
| CLI | Full (`isotrieve calibrate/gate/migrate`) | Library only |
| Store adapters | ChromaDB, Qdrant, Pinecone, LlamaIndex | In-memory reference store |
| Benchmarks | Committed in `benchmarks/results/` | Not yet independent |
| Binary format | v2 (CRC32 + padding) | v2 (cross-compatible) |

Use Python for production today. Use TypeScript for programmatic mapping / format interop in Node.js.

## Development

```bash
npm install
npm run build    # tsc
npm test         # jest
```

## License

Apache-2.0. See [LICENSE](https://github.com/krish1925/Isotrieve/blob/main/LICENSE).
