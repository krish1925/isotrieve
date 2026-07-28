/**
 * Quality gate: evaluate a fitted mapping before full corpus migration.
 *
 * Uses a fresh sample disjoint from calibration.
 * Predicts retention from holdout proxies using an isotonic regression model.
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import type { GateReport, GateVerdict, GateModel, GateThresholds } from '../types';
import type { Mapping } from '../mapping/base';
import {
  pairwiseCosineStats,
  topkRetention,
  holdoutRankCorrelation,
} from '../math/metrics';
import { l2Normalize } from '../math/normalize';
import { createRng } from '../math/random';

// ── Defaults ────────────────────────────────────────────────────

const DEFAULT_THRESHOLDS: GateThresholds = {
  passRetention: 0.75,
  warnRetention: 0.55,
  maxOptimismGap: 0.20,
  provisional: true,
};

function loadThresholds(): GateThresholds {
  try {
    const raw = readFileSync(
      join(__dirname, 'thresholds.json'),
      'utf-8',
    );
    return { ...DEFAULT_THRESHOLDS, ...JSON.parse(raw) };
  } catch {
    return { ...DEFAULT_THRESHOLDS };
  }
}

function loadGateModel(): GateModel | null {
  try {
    const raw = readFileSync(
      join(__dirname, 'gate_model_v1.json'),
      'utf-8',
    );
    const json = JSON.parse(raw) as Record<string, unknown>;
    // Map snake_case from Python-compatible JSON to camelCase TS interface
    const lipoRaw = json.lipo as Record<string, unknown> | undefined;
    const lipo = lipoRaw
      ? {
          mae: lipoRaw.mae as number | undefined,
          intervalHalfWidth80:
            (lipoRaw.interval_half_width_80 ?? lipoRaw.intervalHalfWidth80) as number | undefined,
        }
      : undefined;
    return {
      XThresholds: (json.X_thresholds ?? json.XThresholds) as number[],
      yThresholds: (json.y_thresholds ?? json.yThresholds) as number[],
      scope: json.scope as string | undefined,
      lipo,
    };
  } catch {
    return null;
  }
}

// ── QualityGate ─────────────────────────────────────────────────

export class QualityGate {
  readonly thresholds: GateThresholds;
  readonly gateModel: GateModel | null;

  constructor(overrides?: Partial<GateThresholds>) {
    this.thresholds = { ...loadThresholds(), ...overrides };
    this.gateModel = loadGateModel();
  }

  /**
   * Predict retention from top1_retention using isotonic interpolation.
   */
  private predictRetention(
    top1Retention: number,
    marginCompression: number | null,
  ): [number, [number, number]] {
    if (this.gateModel === null) {
      return [
        top1Retention,
        [Math.max(0, top1Retention - 0.1), Math.min(1, top1Retention + 0.1)],
      ];
    }

    const X = new Float64Array(this.gateModel.XThresholds);
    const y = new Float64Array(this.gateModel.yThresholds);

    // Isotonic interpolation: np.interp equivalent
    const predicted = interp(top1Retention, X, y);

    // Prediction interval from LOPO stats
    let intervalHW = this.gateModel.lipo?.intervalHalfWidth80 ?? 0.1;

    // Gate v3: widen interval when margin compression is severe
    if (marginCompression !== null && marginCompression < 0.85) {
      const compressionPenalty = (0.85 - marginCompression) * 0.5;
      intervalHW += compressionPenalty;
    }

    const lo = Math.max(0, predicted - intervalHW);
    const hi = Math.min(1, predicted + intervalHW);
    return [predicted, [lo, hi]];
  }

  /**
   * Estimate margin compression from mapped vs target cosine distributions.
   * Returns ratio of mapped margin to target margin (< 1 = compression).
   */
  static computeMarginCompression(
    mapped: Float64Array[],
    target: Float64Array[],
  ): number | null {
    const mN = l2Normalize(mapped) as Float64Array[];
    const tN = l2Normalize(target) as Float64Array[];
    const k = mN.length;

    // Paired cosine: each mapped vector vs its corresponding target
    let pairedSum = 0;
    let pairedSumSq = 0;
    for (let i = 0; i < k; i++) {
      let dot = 0;
      for (let j = 0; j < mN[i].length; j++) {
        dot += mN[i][j] * tN[i][j];
      }
      pairedSum += dot;
      pairedSumSq += dot * dot;
    }
    const pairedVar = pairedSumSq / k - (pairedSum / k) ** 2;

    // Reference: variance of cosine similarities between random target pairs
    const rng = createRng(0);
    const idxA = Array.from({ length: k }, () => Math.floor(rng() * k));
    const idxB = Array.from({ length: k }, () => Math.floor(rng() * k));
    let refSum = 0;
    let refSumSq = 0;
    for (let i = 0; i < k; i++) {
      let dot = 0;
      for (let j = 0; j < tN[i].length; j++) {
        dot += tN[idxA[i]][j] * tN[idxB[i]][j];
      }
      refSum += dot;
      refSumSq += dot * dot;
    }
    const refVar = refSumSq / k - (refSum / k) ** 2;

    if (refVar < 1e-12) return null;
    return pairedVar / refVar;
  }

  static scoreRecomRecommendation(marginRatio: number | null): string | null {
    if (marginRatio === null) return null;
    if (marginRatio < 0.8) {
      return (
        `Score margins are compressed (ratio=${marginRatio.toFixed(2)}). ` +
        'If your application uses absolute score thresholds, ' +
        'enable score recalibration or re-tune thresholds.'
      );
    }
    return null;
  }

  /**
   * Run gate on paired source/target embeddings of the same texts.
   * X_sample / Y_sample must NOT overlap the calibration fit set.
   */
  evaluate(
    mapping: Mapping,
    XSample: Float64Array[],
    YSample: Float64Array[],
    options?: { holdoutTop1?: number | null },
  ): GateReport {
    const mapped = mapping.transform(XSample) as Float64Array[];
    const cos = pairwiseCosineStats(mapped, YSample);
    const t1 = topkRetention(mapped, YSample, 1);
    const t10 = topkRetention(mapped, YSample, Math.min(10, YSample.length));
    const rankCorr = holdoutRankCorrelation(mapped, YSample);

    let optimismGap: number | null = null;
    if (options?.holdoutTop1 !== undefined && options.holdoutTop1 !== null) {
      optimismGap = options.holdoutTop1 - t1;
    }

    const mc = QualityGate.computeMarginCompression(mapped, YSample);
    const [predicted, [lower, upper]] = this.predictRetention(t1, mc);

    const passR = this.thresholds.passRetention;
    const warnR = this.thresholds.warnRetention;

    let verdict: GateVerdict;
    let rationale: string;

    if (lower >= passR) {
      verdict = 'PASS';
      rationale =
        `PASS: predicted retention=${predicted.toFixed(3)} ` +
        `(80% CI: [${lower.toFixed(3)}, ${upper.toFixed(3)}]). ` +
        'Holdout proxies indicate good mapping quality.';
    } else if (upper < warnR) {
      verdict = 'FAIL';
      rationale =
        `FAIL: predicted retention=${predicted.toFixed(3)} ` +
        `(80% CI: [${lower.toFixed(3)}, ${upper.toFixed(3)}]). ` +
        'Mapping quality too low for migration. Recommend full re-embedding.';
    } else {
      verdict = 'WARN';
      rationale =
        `WARN: predicted retention=${predicted.toFixed(3)} ` +
        `(80% CI: [${lower.toFixed(3)}, ${upper.toFixed(3)}]). ` +
        'Usable for recall-tolerant workloads only. ' +
        'Consider more in-domain calibration.';
    }

    return {
      verdict,
      predictedRetention: predicted,
      predictionInterval: [lower, upper],
      cosineMean: cos.mean,
      cosineMedian: cos.median,
      cosineP5: cos.p5,
      top1Retention: t1,
      top10Retention: t10,
      holdoutRankCorr: rankCorr,
      nSample: XSample.length,
      rationale,
      holdoutTop1: options?.holdoutTop1 ?? null,
      optimismGap,
      provisionalThresholds: this.thresholds.provisional,
      gateModelUsed: this.gateModel !== null,
      gateModelScope: this.gateModel?.scope ?? null,
      lopoError: this.gateModel?.lipo?.mae ?? null,
      thresholdsUsed: {
        passRetention: passR,
        warnRetention: warnR,
      },
      marginCompression: mc,
      scoreRecomRecommendation: QualityGate.scoreRecomRecommendation(mc),
    };
  }
}

// ── Helpers ─────────────────────────────────────────────────────

function interp(x: number, xp: Float64Array, yp: Float64Array): number {
  const n = xp.length;
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
