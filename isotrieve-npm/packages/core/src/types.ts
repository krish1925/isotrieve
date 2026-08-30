/**
 * Core type definitions for Isotrieve embedding migration.
 */

// ── Mapping Types ────────────────────────────────────────────────

export type MappingType =
  | 'ridge'
  | 'orthogonal_procrustes'
  | 'procrustes_diag'
  | 'lowrank_affine'
  | 'residual_mlp'
  | 'external';

export interface ValidationReport {
  holdoutCosineMean: number;
  holdoutCosineMedian: number;
  holdoutCosineP5: number;
  top1Retention: number;
  top10Retention: number;
  nTrain: number;
  nHoldout: number;
  seed: number;
  alpha: number | null;
  notes: string[];
}

export interface IsotrieveFileHeader {
  formatVersion: number;
  isotrieveVersion: string;
  mappingType: MappingType;
  dSrc: number;
  dTarget: number;
  bias: boolean;
  seed: number;
  fitDate: string;
  expiresHint: string | null;
  validation: ValidationReport | null;
  meta: Record<string, unknown>;
  matrixShape: [number, number];
  hasInverse: boolean;
  inverseMatrixShape?: [number, number];
  extraMatrices?: Record<string, [number, number]>;
  scoreRecalV1?: ScoreRecalibratorData;
}

// ── Quality Types ────────────────────────────────────────────────

export type GateVerdict = 'PASS' | 'WARN' | 'FAIL';

export interface GateReport {
  verdict: GateVerdict;
  predictedRetention: number;
  predictionInterval: [number, number];
  cosineMean: number;
  cosineMedian: number;
  cosineP5: number;
  top1Retention: number;
  top10Retention: number;
  holdoutRankCorr: number;
  nSample: number;
  rationale: string;
  holdoutTop1: number | null;
  optimismGap: number | null;
  provisionalThresholds: boolean;
  gateModelUsed: boolean;
  gateModelScope: string | null;
  lopoError: number | null;
  thresholdsUsed: Record<string, number>;
  marginCompression: number | null;
  scoreRecomRecommendation: string | null;
}

// ── Store Types ──────────────────────────────────────────────────

export interface VectorRecord {
  id: string;
  vector: number[];
  text?: string;
  payload?: Record<string, unknown>;
}

export interface VectorStore {
  count(): number;
  iterVectors(batchSize?: number): VectorRecord[][] | IterableIterator<VectorRecord[]>;
  writeVectors(
    records: VectorRecord[][] | VectorRecord[],
    options?: { batchSize?: number },
  ): number;
}

// ── Provider Types ───────────────────────────────────────────────

export interface Embedder {
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
  getDimensions(): number;
  getModelId(): string;
}

// ── Calibration Types ────────────────────────────────────────────

export interface CalibrationPlan {
  corpusSize: number;
  recommendedK: number;
  sourceModel: string;
  targetModel: string;
  estCalibrationCalls: number;
  estReembedCalls: number;
  estCalibrationUsd: number;
  estReembedUsd: number;
  notes: string[];
}

export interface CalibrationPlanOptions {
  corpusSize: number;
  sourceModel: string;
  targetModel: string;
  dSrc?: number;
  dTarget?: number;
  tokensPerText?: number;
}

// ── Recalibration Types ──────────────────────────────────────────

export interface RecalibrationReport {
  nPairs: number;
  meanMappedScore: number;
  meanCeilingScore: number;
  meanShift: number;
  preMarginMean: number;
  postMarginMean: number;
  marginRatio: number;
  thresholdAgreement: Record<number, number>;
}

export interface ScoreRecalibratorData {
  thresholds: number[];
  values: number[];
  report: RecalibrationReport | null;
}

// ── Migration Types ──────────────────────────────────────────────

export interface MigrationBatch {
  batchNum: number;
  startIdx: number;
  count: number;
  hash: string;
}

export interface MigrationManifest {
  sourceCollection: string;
  targetCollection: string;
  sourceModel: string;
  targetModel: string;
  totalVectors: number;
  migratedVectors: number;
  batchStart: number;
  batchEnd: number;
  lastBatchHash: string;
  startedAt: string;
  completedAt: string;
  batches: MigrationBatch[];
}

// ── Reranking Types ──────────────────────────────────────────────

export interface ConfidenceReport {
  queryId: string;
  top1Margin: number;
  top1Score: number;
  confidence: 'high' | 'medium' | 'low';
  nCandidates: number;
}

export interface ConfidenceSummary {
  n: number;
  nHigh: number;
  nMedium: number;
  nLow: number;
  pctHigh: number;
  pctLow: number;
  meanMargin: number;
  meanTop1Score: number;
}

// ── Gate Model Types ─────────────────────────────────────────────

export interface GateModel {
  XThresholds: number[];
  yThresholds: number[];
  scope?: string;
  lipo?: {
    mae?: number;
    intervalHalfWidth80?: number;
  };
}

export interface GateThresholds {
  passRetention: number;
  warnRetention: number;
  maxOptimismGap: number;
  provisional: boolean;
}
