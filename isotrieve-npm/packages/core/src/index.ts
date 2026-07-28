/**
 * @isotrieve/core
 *
 * Embedding migration without re-embedding.
 * Learn a linear (or optional shallow non-linear) mapping between source and
 * target embedding spaces from a small calibration sample, then transform
 * stored vectors in-place instead of re-embedding an entire corpus.
 */

// ── Mapping ──────────────────────────────────────────────────────
export {
  Mapping,
  RidgeMapping,
  OrthogonalProcrustesMapping,
  ProcrustesDiagMapping,
  LowRankAffineMapping,
  ExternalMapping,
  registerMapping,
  getMappingClass,
  loadMapping,
  writeIsotrieveFile,
  readIsotrieveHeader,
  readIsotrievePayload,
  ISOTRIEVE_MAGIC,
  FORMAT_VERSION,
} from './mapping';
export type {
  ApplyMappingOptions,
  RidgeMappingOptions,
  LowRankAffineMappingOptions,
  ExternalTransformFn,
} from './mapping';

// ── Math ─────────────────────────────────────────────────────────
export {
  cosineSimilarity,
  matrixMultiply,
  vectorMatrixMultiply,
  transpose,
  svd,
  solve,
  leastSquares,
  ridgeCv,
  zeros,
  eye,
  shape,
  matClone,
  matrixTrace,
  frobeniusNorm,
  vecNorm,
} from './math/linalg';
export {
  l2Normalize,
  checkFinite,
  augmentBias,
} from './math/normalize';
export {
  pairwiseCosineStats,
  topkRetention,
  spearmanRho,
  holdoutRankCorrelation,
  mrrDelta,
  retrievalRetentionReport,
} from './math/metrics';
export {
  createRng,
  shuffle,
  trainTestSplit,
} from './math/random';

// ── Quality Gate ─────────────────────────────────────────────────
export { QualityGate } from './quality';

// ── Recalibration ────────────────────────────────────────────────
export { ScoreRecalibrator } from './recalibration';

// ── Calibration ──────────────────────────────────────────────────
export { planCalibration, recommendK } from './calibration';

// ── Reranking ────────────────────────────────────────────────────
export { ConfidenceScorer, confidenceSummary } from './reranking';

// ── Migration ────────────────────────────────────────────────────
export { migrateStore } from './migrate';

// ── Serve ────────────────────────────────────────────────────────
export { QueryAdapter, cslsScores, mergeResults } from './serve';

// ── Types ────────────────────────────────────────────────────────
export type {
  MappingType,
  ValidationReport,
  IsotrieveFileHeader,
  GateVerdict,
  GateReport,
  VectorRecord,
  VectorStore,
  Embedder,
  CalibrationPlan,
  CalibrationPlanOptions,
  RecalibrationReport,
  ScoreRecalibratorData,
  MigrationManifest,
  MigrationBatch,
  ConfidenceReport,
  ConfidenceSummary,
  GateModel,
  GateThresholds,
} from './types';

// ── Version ──────────────────────────────────────────────────────
export const VERSION = '0.1.0';
