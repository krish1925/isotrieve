/**
 * Query-adapter serving mode: map new-model queries into legacy space
 * for zero-corpus-write migration.
 */

import { loadMapping } from './mapping/registry';
import type { Mapping } from './mapping/base';
import { l2Normalize } from './math/normalize';
import type { ConfidenceReport, ConfidenceSummary } from './types';
import { ConfidenceScorer, confidenceSummary } from './reranking';

export class QueryAdapter {
  private _mapping: Mapping;
  private _dSrc: number;
  private _dTarget: number;

  constructor(mapping: Mapping) {
    this._mapping = mapping;
    this._dSrc = mapping.dSrc;
    this._dTarget = mapping.dTarget;
  }

  /**
   * Load a mapping file (must have inverse direction fitted).
   */
  static load(path: string): QueryAdapter {
    const mapping = loadMapping(path);
    if (!mapping.hasInverse) {
      throw new Error(
        `Mapping at ${path} has no inverse. ` +
        'Fit the inverse direction before using serve mode.',
      );
    }
    return new QueryAdapter(mapping);
  }

  /** Dimension of new-model vectors (target space). */
  get dNew(): number {
    return this._dTarget;
  }

  /** Dimension of legacy-model vectors (source space). */
  get dLegacy(): number {
    return this._dSrc;
  }

  /**
   * Map a single query vector from new-model space to legacy space.
   */
  mapQuery(vec: Float64Array): Float64Array {
    if (vec.length !== this._dTarget) {
      throw new Error(
        `Dimension mismatch: expected ${this._dTarget} (new model dim), got ${vec.length}. ` +
        'Query vectors must be from the new embedding model.',
      );
    }
    const out = this._mapping.inverseTransform(vec) as Float64Array;
    const normalized = l2Normalize(out) as Float64Array;
    return normalized;
  }

  /**
   * Batch map query vectors from new-model space to legacy space.
   */
  mapQueries(vecs: Float64Array[]): Float64Array[] {
    if (vecs.length === 0) return [];
    if (vecs[0].length !== this._dTarget) {
      throw new Error(
        `Dimension mismatch: expected ${this._dTarget} (new model dim), got ${vecs[0].length}. ` +
        'Query vectors must be from the new embedding model.',
      );
    }
    const out = this._mapping.inverseTransform(vecs) as Float64Array[];
    return l2Normalize(out) as Float64Array[];
  }

  /** Whether the underlying mapping has a fitted score recalibrator. */
  get hasRecalibrator(): boolean {
    return this._mapping.hasRecalibrator;
  }

  /**
   * Map post-migration scores to ceiling-equivalent scores.
   * Returns scores unchanged if no recalibrator is fitted.
   */
  recalibrateScores(scores: Float64Array): Float64Array {
    return this._mapping.recalibrateScores(scores);
  }

  /**
   * Score per-query confidence from a similarity matrix.
   */
  scoreConfidence(
    queryIds: string[],
    similarities: Float64Array[],
    topK: number = 10,
  ): { reports: ConfidenceReport[]; summary: ConfidenceSummary } {
    const scorer = new ConfidenceScorer();
    const reports = scorer.scoreQueries(queryIds, similarities, topK);
    return {
      reports,
      summary: confidenceSummary(reports),
    };
  }
}

// ── CSLS Scores ─────────────────────────────────────────────────

/**
 * Compute CSLS scores for query-candidate pairs.
 *
 * CSLS (Cross-Domain Similarity Local Scaling) corrects for hubness
 * in cross-lingual retrieval.
 * Reference: Joulin et al., "Loss in Translation" (ACL 2016).
 */
export function cslsScores(
  queryVecs: Float64Array[],
  candidateVecs: Float64Array[],
  k: number = 10,
): Float64Array[] {
  const qN = l2Normalize(queryVecs) as Float64Array[];
  const cN = l2Normalize(candidateVecs) as Float64Array[];

  const nQ = qN.length;
  const nC = cN.length;
  const d = qN[0].length;
  const kEff = Math.min(k, nQ);

  // Cosine similarity matrix: (nQ, nC)
  const S: Float64Array[] = [];
  for (let qi = 0; qi < nQ; qi++) {
    const row = new Float64Array(nC);
    for (let ci = 0; ci < nC; ci++) {
      let dot = 0;
      for (let j = 0; j < d; j++) {
        dot += qN[qi][j] * cN[ci][j];
      }
      row[ci] = dot;
    }
    S.push(row);
  }

  // For each candidate, find k nearest queries and compute mean
  const rT = new Float64Array(nC);
  for (let ci = 0; ci < nC; ci++) {
    // S transposed row for this candidate
    const col = new Float64Array(nQ);
    for (let qi = 0; qi < nQ; qi++) {
      col[qi] = S[qi][ci];
    }
    // Get top-k indices
    const indices = argSortDesc(col).slice(0, kEff);
    let sum = 0;
    for (const qi of indices) {
      sum += col[qi];
    }
    rT[ci] = sum / kEff;
  }

  // CSLS: 2 * cos(q, x) - r_T(x)
  const result: Float64Array[] = [];
  for (let qi = 0; qi < nQ; qi++) {
    const row = new Float64Array(nC);
    for (let ci = 0; ci < nC; ci++) {
      row[ci] = 2 * S[qi][ci] - rT[ci];
    }
    result.push(row);
  }

  return result;
}

/**
 * Merge results from legacy (mapped) and native (new-model) collections.
 * For progressive migration: some docs are in legacy space, some in native.
 */
export function mergeResults(
  legacyResults: Array<{ id: string; score: number }>,
  nativeResults: Array<{ id: string; score: number }>,
  migratedIds: Set<string>,
  legacyWeight: number = 1.0,
  nativeWeight: number = 1.0,
): Array<{ id: string; score: number }> {
  const scores = new Map<string, number>();

  for (const r of legacyResults) {
    const score = r.score * legacyWeight;
    const existing = scores.get(r.id) ?? 0;
    scores.set(r.id, Math.max(existing, score));
  }

  for (const r of nativeResults) {
    const score = r.score * nativeWeight;
    const existing = scores.get(r.id) ?? 0;
    scores.set(r.id, Math.max(existing, score));
  }

  const merged = Array.from(scores.entries())
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score);

  return merged;
}

// ── Helpers ─────────────────────────────────────────────────────

function argSortDesc(arr: Float64Array): number[] {
  const indices = Array.from({ length: arr.length }, (_, i) => i);
  indices.sort((a, b) => arr[b] - arr[a]);
  return indices;
}
