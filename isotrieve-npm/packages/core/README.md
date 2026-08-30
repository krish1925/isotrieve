# @isotrieve/core

> **Pre-1.0 / Beta** — This TypeScript port of [isotrieve](https://pypi.org/project/isotrieve/) is under active development. The Python package is the mature, benchmark-validated implementation. Until Phases 0–4 of the production readiness plan are closed, treat this as a preview.

Embedding migration without re-embedding. Learn a linear mapping between source and target embedding spaces from a small calibration sample, then transform stored vectors in place — and gate the migration on measured retrieval retention before you commit.

This is the TypeScript port of the Python [`isotrieve`](https://pypi.org/project/isotrieve/) package. The `.isotrieve` binary format is cross-compatible between both runtimes.

## Installation

```bash
npm install @isotrieve/core
```

Requires Node.js >= 18.

## Quick start

```typescript
import {
  RidgeMapping,
  QualityGate,
  trainTestSplit,
} from '@isotrieve/core';

// Calibration pairs: vectors embedded with the old model (X) and new model (Y)
const { X, Y } = trainTestSplit(sourceVectors, targetVectors, { testSize: 0.2, seed: 42 });

// Fit a ridge mapping (supports cross-dimension: 384 → 1536 works)
const mapping = new RidgeMapping({ alpha: 1.0 });
mapping.fit(X_train, Y_train);

// Validate on held-out pairs
const validation = mapping.validate(X_test, Y_test);
console.log(`Holdout cosine mean: ${validation.holdoutCosineMean.toFixed(3)}`);

// Save the mapping (cross-compatible with Python's isotrieve format)
mapping.save('migration.isotrieve');

// Gate the migration before committing
const gate = new QualityGate();
const report = gate.evaluate(mapping, X_test, Y_test, { holdoutTop1: validation.top1Retention });
console.log(report.verdict); // 'PASS' | 'WARN' | 'FAIL'
```

## Mapping types

| Type | Use case | Inverse | Cross-dim |
|------|----------|---------|-----------|
| `RidgeMapping` | Default. Noise-robust, unequal dims. | No | Yes |
| `OrthogonalProcrustesMapping` | Similar spaces, square dims. | Yes | No |
| `ProcrustesDiagMapping` | Axis-aligned transform. | Yes | No |
| `LowRankAffineMapping` | Limited calibration data. | No | Yes |
| `ExternalMapping` | Wraps a user-provided function. | No | Any |

## Quality gate

```typescript
import { QualityGate } from '@isotrieve/core';

const gate = new QualityGate();
const report = gate.evaluate(mapping, X_test, Y_test, { holdoutTop1: 0.95 });

report.verdict;              // 'PASS' | 'WARN' | 'FAIL'
report.predictedRetention;   // isotonic regression prediction from gate model v1
report.predictionInterval;   // [lower, upper] 80% CI
report.marginCompression;    // compressed ↔ 1.0
```

The gate model (`gate_model_v1.json`) is shared with the Python package and produces identical verdicts on the same inputs.

## Score recalibration

```typescript
import { ScoreRecalibrator } from '@isotrieve/core';

const recal = new ScoreRecalibrator();
recal.fit(mappedScores, ceilingScores);  // PAVA isotonic regression
const adjusted = recal.transform(newScores);
```

## Migration

```typescript
import { migrateStore } from '@isotrieve/core';

const manifest = await migrateStore({
  source: sourceVectorStore,
  target: targetVectorStore,
  mapping,
  batchSize: 1000,
  onBatchComplete: (batch) => console.log(`Migrated batch ${batch.batchNum}`),
});
```

## Binary format

`.isotrieve` files are cross-compatible with Python's `isotrieve` package. Format:

- Magic bytes: `ISTR` (4 bytes)
- Header length: uint32 LE
- Header: JSON (format version, mapping type, dimensions, validation report, etc.)
- Payload: Float64 matrices in row-major order

## Which package should I use?

| | Python (`isotrieve`) | TypeScript (`@isotrieve/core`) |
|---|---|---|
| Status | Stable, benchmark-validated | Beta, under active development |
| CLI | Full (`isotrieve calibrate/gate/migrate/...`) | Library only (no CLI yet) |
| Store adapters | ChromaDB, Qdrant, Pinecone, LlamaIndex | In-memory reference store (more planned) |
| Benchmarks | Committed results in `benchmarks/results/` | Not yet benchmarked independently |
| Binary format | v1 | v1 (cross-compatible) |

Use the Python package for production migrations today. Use this package if you need a TypeScript library for programmatic mapping, gating, or format interop.

## Status

Pre-1.0. APIs may change between minor versions. See the root [README.md](../../README.md) for the full production readiness plan.

## License

Apache-2.0. See [LICENSE](../../isotrieve-python/LICENSE).
