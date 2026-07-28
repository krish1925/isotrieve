/**
 * Score recalibration: map post-migration scores to ceiling-equivalent scores.
 *
 * After embedding-space mapping, similarity scores are systematically compressed.
 * ScoreRecalibrator fits an isotonic regression on holdout
 * (mapped_score, ceiling_score) pairs and provides a monotone map that restores
 * score interpretability.
 */

import type { RecalibrationReport, ScoreRecalibratorData } from './types';
import { l2Normalize } from './math/normalize';

export class ScoreRecalibrator {
  private _thresholds: Float64Array | null = null;
  private _values: Float64Array | null = null;
  private _report: RecalibrationReport | null = null;

  get isFitted(): boolean {
    return this._thresholds !== null;
  }

  get report(): RecalibrationReport | null {
    return this._report;
  }

  /**
   * Fit isotonic regression on (mapped_score -> ceiling_score) pairs.
   *
   * Isotonic regression: find the monotone non-decreasing function f that
   * minimizes sum of squared residuals. Uses the Pool Adjacent Violators
   * algorithm (PAVA).
   */
  fit(
    mappedScores: Float64Array | number[],
    ceilingScores: Float64Array | number[],
  ): this {
    const m = Float64Array.from(mappedScores);
    const c = Float64Array.from(ceilingScores);

    if (m.length < 10) {
      throw new Error(`Need >=10 score pairs, got ${m.length}`);
    }

    // Sort by mapped score
    const indices = Array.from({ length: m.length }, (_, i) => i);
    indices.sort((a, b) => m[a] - m[b]);
    const sortedM = new Float64Array(indices.map((i) => m[i]));
    const sortedC = new Float64Array(indices.map((i) => c[i]));

    // PAVA (Pool Adjacent Violators) for isotonic regression
    const n = sortedM.length;
    const fitted = new Float64Array(n);
    const blockStart = new Float64Array(n);
    const blockSize = new Float64Array(n);
    let nBlocks = 1;

    blockStart[0] = 0;
    blockSize[0] = 1;
    fitted[0] = sortedC[0];

    for (let i = 1; i < n; i++) {
      // Add new block
      blockStart[nBlocks] = i;
      blockSize[nBlocks] = 1;
      fitted[nBlocks] = sortedC[i];
      nBlocks++;

      // Merge blocks that violate monotonicity
      while (nBlocks > 1 && fitted[nBlocks - 2] > fitted[nBlocks - 1]) {
        const b2 = nBlocks - 1;
        const b1 = nBlocks - 2;
        const size1 = blockSize[b1];
        const size2 = blockSize[b2];
        const totalSize = size1 + size2;
        const mergedVal = (fitted[b1] * size1 + fitted[b2] * size2) / totalSize;

        fitted[b1] = mergedVal;
        blockSize[b1] = totalSize;
        nBlocks--;
      }
    }

    // Extract breakpoints (unique fitted values)
    const breakpoints: number[] = [];
    const breakValues: number[] = [];
    let prevVal = fitted[0] - 1;
    for (let i = 0; i < nBlocks; i++) {
      if (Math.abs(fitted[i] - prevVal) > 1e-15) {
        breakpoints.push(sortedM[Math.round(blockStart[i])]);
        breakValues.push(fitted[i]);
        prevVal = fitted[i];
      }
    }

    this._thresholds = new Float64Array(breakpoints);
    this._values = new Float64Array(breakValues);

    // Compute report
    const pre = m;
    const post = this.transform(pre);

    // Margin computation (top-20 sorted differences)
    const preSorted = Float64Array.from(pre).sort();
    const postSorted = Float64Array.from(post).sort();
    const marginSize = Math.min(20, preSorted.length);
    let preMargin = 0;
    let postMargin = 0;
    for (let i = preSorted.length - marginSize; i < preSorted.length - 1; i++) {
      preMargin += preSorted[i + 1] - preSorted[i];
    }
    preMargin /= marginSize - 1;
    for (let i = postSorted.length - marginSize; i < postSorted.length - 1; i++) {
      postMargin += postSorted[i + 1] - postSorted[i];
    }
    postMargin /= marginSize - 1;

    // Threshold agreement
    const thresholds = [0.5, 0.6, 0.7, 0.8, 0.9];
    const agreement: Record<number, number> = {};
    for (const tau of thresholds) {
      let agree = 0;
      for (let i = 0; i < post.length; i++) {
        const postDec = post[i] >= tau ? 1 : 0;
        const ceilDec = c[i] >= tau ? 1 : 0;
        if (postDec === ceilDec) agree++;
      }
      agreement[tau] = agree / post.length;
    }

    // Compute means
    let meanPre = 0;
    let meanPost = 0;
    let meanCeil = 0;
    for (let i = 0; i < m.length; i++) {
      meanPre += m[i];
      meanPost += post[i];
      meanCeil += c[i];
    }
    meanPre /= m.length;
    meanPost /= m.length;
    meanCeil /= c.length;

    this._report = {
      nPairs: m.length,
      meanMappedScore: meanPre,
      meanCeilingScore: meanCeil,
      meanShift: meanPost - meanPre,
      preMarginMean: preMargin,
      postMarginMean: postMargin,
      marginRatio: preMargin > 1e-12 ? postMargin / preMargin : 1.0,
      thresholdAgreement: agreement,
    };

    return this;
  }

  /**
   * Recalibrate scores: mapped -> ceiling-equivalent.
   */
  transform(scores: Float64Array | number[]): Float64Array {
    if (!this.isFitted || this._thresholds === null || this._values === null) {
      throw new Error('ScoreRecalibrator not fitted; call fit() first');
    }
    const input = Float64Array.from(scores);
    const result = new Float64Array(input.length);
    for (let i = 0; i < input.length; i++) {
      result[i] = interp(input[i], this._thresholds, this._values);
    }
    return result;
  }

  /**
   * Build recalibrator from holdout data.
   */
  static fitFromHoldout(
    docSrc: Float64Array[],
    docTgt: Float64Array[],
    qryTgt: Float64Array[],
    docIds: string[],
    queryIds: string[],
    qrels: Map<string, Set<string>>,
    mappedDocs: Float64Array[],
    options?: { nRandomPairs?: number; seed?: number },
  ): ScoreRecalibrator {
    const nRandomPairs = options?.nRandomPairs ?? 50_000;
    const seed = options?.seed ?? 0;
    const rng = createRng(seed);

    const nDocs = docIds.length;
    const nQueries = queryIds.length;

    const qtN = l2Normalize(qryTgt) as Float64Array[];
    const dtN = l2Normalize(docTgt) as Float64Array[];
    const dmN = l2Normalize(mappedDocs) as Float64Array[];

    const d = qtN[0].length;
    const mappedPairs: number[] = [];
    const ceilingPairs: number[] = [];

    // Sample random pairs
    const nRandom = Math.min(nRandomPairs, nQueries * nDocs);
    for (let i = 0; i < nRandom; i++) {
      const qi = Math.floor(rng() * nQueries);
      const di = Math.floor(rng() * nDocs);
      let mapDot = 0;
      let ceilDot = 0;
      for (let j = 0; j < d; j++) {
        mapDot += qtN[qi][j] * dmN[di][j];
        ceilDot += qtN[qi][j] * dtN[di][j];
      }
      mappedPairs.push(mapDot);
      ceilingPairs.push(ceilDot);
    }

    // Add top-20 neighbor pairs (cover the high-score region)
    const topK = Math.min(20, nDocs);
    for (let qi = 0; qi < nQueries; qi++) {
      // Compute all similarities for this query
      const sims = new Float64Array(nDocs);
      for (let di = 0; di < nDocs; di++) {
        let dot = 0;
        for (let j = 0; j < d; j++) {
          dot += qtN[qi][j] * dmN[di][j];
        }
        sims[di] = dot;
      }
      // Get top-K indices
      const indices = argSortDesc(sims).slice(0, topK);
      for (const di of indices) {
        let mapDot = 0;
        let ceilDot = 0;
        for (let j = 0; j < d; j++) {
          mapDot += qtN[qi][j] * dmN[di][j];
          ceilDot += qtN[qi][j] * dtN[di][j];
        }
        mappedPairs.push(mapDot);
        ceilingPairs.push(ceilDot);
      }
    }

    const recal = new ScoreRecalibrator();
    recal.fit(mappedPairs, ceilingPairs);
    return recal;
  }

  /**
   * Serialize to ScoreRecalibratorData for .isotrieve header storage.
   */
  toJSON(): ScoreRecalibratorData {
    if (!this.isFitted || this._thresholds === null || this._values === null) {
      throw new Error('Cannot serialize unfitted recalibrator');
    }
    return {
      thresholds: Array.from(this._thresholds),
      values: Array.from(this._values),
      report: this._report,
    };
  }

  /**
   * Load from serialized data.
   */
  static fromJSON(data: ScoreRecalibratorData): ScoreRecalibrator {
    const recal = new ScoreRecalibrator();
    recal._thresholds = new Float64Array(data.thresholds);
    recal._values = new Float64Array(data.values);
    recal._report = data.report ?? null;
    return recal;
  }
}

// ── Helpers ─────────────────────────────────────────────────────

function interp(x: number, xp: Float64Array, yp: Float64Array): number {
  const n = xp.length;
  if (n === 0) return 0;
  if (x <= xp[0]) return yp[0];
  if (x >= xp[n - 1]) return yp[n - 1];
  for (let i = 0; i < n - 1; i++) {
    if (x >= xp[i] && x <= xp[i + 1]) {
      const t = (x - xp[i]) / (xp[i + 1] - xp[i]);
      return yp[i] + t * (yp[i + 1] - yp[i]);
    }
  }
  return yp[n - 1];
}

function argSortDesc(arr: Float64Array): number[] {
  const indices = Array.from({ length: arr.length }, (_, i) => i);
  indices.sort((a, b) => arr[b] - arr[a]);
  return indices;
}

// Re-export createRng from math for internal use
function createRng(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
