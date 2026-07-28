/**
 * Cross-encoder reranking and per-query confidence scoring for Isotrieve.
 */

import type { ConfidenceReport, ConfidenceSummary } from './types';

// ── ConfidenceScorer ────────────────────────────────────────────

export class ConfidenceScorer {
  private _marginLow: number;
  private _marginHigh: number;
  private _scoreLow: number;
  private _adaptive: boolean;

  constructor(options?: {
    marginLow?: number;
    marginHigh?: number;
    scoreLow?: number;
    adaptive?: boolean;
  }) {
    this._marginLow = options?.marginLow ?? 0.005;
    this._marginHigh = options?.marginHigh ?? 0.025;
    this._scoreLow = options?.scoreLow ?? 0.8;
    this._adaptive = options?.adaptive ?? true;
  }

  /**
   * Score confidence for each query.
   *
   * @param queryIds - List of query IDs
   * @param similarities - (n_queries, n_docs) similarity matrix
   * @param topK - Number of top results to consider for margin computation
   */
  scoreQueries(
    queryIds: string[],
    similarities: Float64Array[],
    topK: number = 10,
  ): ConfidenceReport[] {
    const nQueries = queryIds.length;

    // Compute margins and top-1 scores for all queries
    const allMargins = new Float64Array(nQueries);
    const allTop1 = new Float64Array(nQueries);

    for (let qi = 0; qi < nQueries; qi++) {
      const sims = similarities[qi];
      const kEff = Math.min(topK, sims.length);
      const sorted = Float64Array.from(sims).sort();
      const top1 = sorted[sims.length - 1];
      const top2 = sorted[sims.length - 2] ?? top1;
      allMargins[qi] = top1 - top2;
      allTop1[qi] = top1;
    }

    // Determine thresholds
    let marginLow = this._marginLow;
    let marginHigh = this._marginHigh;

    if (this._adaptive && nQueries > 10) {
      marginLow = percentile(allMargins, 33);
      marginHigh = percentile(allMargins, 67);
    }

    // Build reports
    const reports: ConfidenceReport[] = [];
    for (let qi = 0; qi < nQueries; qi++) {
      const margin = allMargins[qi];
      const top1 = allTop1[qi];
      let confidence: 'high' | 'medium' | 'low';

      if (margin >= marginHigh) {
        confidence = 'high';
      } else if (margin <= marginLow) {
        confidence = 'low';
      } else {
        confidence = 'medium';
      }

      reports.push({
        queryId: queryIds[qi],
        top1Margin: margin,
        top1Score: top1,
        confidence,
        nCandidates: topK,
      });
    }

    return reports;
  }
}

/**
 * Aggregate confidence reports into a summary.
 */
export function confidenceSummary(reports: ConfidenceReport[]): ConfidenceSummary {
  const n = reports.length;
  if (n === 0) {
    return {
      n: 0,
      nHigh: 0,
      nMedium: 0,
      nLow: 0,
      pctHigh: 0,
      pctLow: 0,
      meanMargin: 0,
      meanTop1Score: 0,
    };
  }

  let nHigh = 0;
  let nMedium = 0;
  let nLow = 0;
  let sumMargin = 0;
  let sumTop1 = 0;

  for (const r of reports) {
    if (r.confidence === 'high') nHigh++;
    else if (r.confidence === 'medium') nMedium++;
    else nLow++;
    sumMargin += r.top1Margin;
    sumTop1 += r.top1Score;
  }

  return {
    n,
    nHigh,
    nMedium,
    nLow,
    pctHigh: nHigh / n,
    pctLow: nLow / n,
    meanMargin: sumMargin / n,
    meanTop1Score: sumTop1 / n,
  };
}

// ── Helpers ─────────────────────────────────────────────────────

function percentile(arr: Float64Array, p: number): number {
  const sorted = Float64Array.from(arr).sort();
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const t = idx - lo;
  return sorted[lo] * (1 - t) + sorted[hi] * t;
}
