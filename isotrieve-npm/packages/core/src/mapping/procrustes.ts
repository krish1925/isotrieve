/**
 * Orthogonal Procrustes and Procrustes+Diagonal mappings.
 *
 * Procrustes finds the best orthogonal rotation R that aligns source to target:
 *   min ||X R - Y||_F  subject to  R^T R = I
 *
 * ProcrustesDiag adds a per-dimension diagonal scaling:
 *   Y ≈ diag(s) · X · R
 */

import { Mapping } from './base';
import { transpose, matrixMultiply, svd, vectorMatrixMultiply, zeros, shape } from '../math/linalg';
import { l2Normalize, checkFinite } from '../math/normalize';
import { pairwiseCosineStats, topkRetention } from '../math/metrics';
import { trainTestSplit } from '../math/random';
import type { MappingType, ValidationReport } from '../types';

// ── Shared holdout metrics helper ────────────────────────────────

function computeHoldoutMetrics(
  mapping: Mapping,
  XTest: Float64Array[],
  YTest: Float64Array[],
): {
  cosineMean: number;
  cosineMedian: number;
  cosineP5: number;
  top1Retention: number;
  top10Retention: number;
} {
  const mapped = mapping.transform(XTest) as Float64Array[];
  const normMapped = l2Normalize(mapped) as Float64Array[];
  const normTarget = l2Normalize(YTest) as Float64Array[];
  const cosine = pairwiseCosineStats(normMapped, normTarget);
  return {
    cosineMean: cosine.mean,
    cosineMedian: cosine.median,
    cosineP5: cosine.p5,
    top1Retention: topkRetention(normMapped, normTarget, 1),
    top10Retention: topkRetention(normMapped, normTarget, Math.min(10, normTarget.length)),
  };
}

// ── Validate paired input ────────────────────────────────────────

function validateXY(X: Float64Array[], Y: Float64Array[]): void {
  if (X.length !== Y.length) {
    throw new Error(`Sample counts must match: X has ${X.length}, Y has ${Y.length}`);
  }
  if (X.length < 2) {
    throw new Error(`Need at least 2 calibration pairs, got ${X.length}.`);
  }
  if (X[0].length !== Y[0].length) {
    throw new Error(
      `Procrustes requires d_src == d_target; got ${X[0].length} vs ${Y[0].length}. ` +
      'Use RidgeMapping for rectangular mappings.',
    );
  }
  checkFinite('X', X);
  checkFinite('Y', Y);
}

// ── OrthogonalProcrustesMapping ──────────────────────────────────

/**
 * Orthogonal Procrustes mapping (square dims only).
 *
 * Preserves pairwise geometry exactly under the orthogonal constraint.
 * Only available when d_src == d_target.
 */
export class OrthogonalProcrustesMapping extends Mapping {
  static readonly mappingType: MappingType = 'orthogonal_procrustes';

  private _normalizeOutput: boolean;
  private _holdoutFraction: number;

  constructor(options: { seed?: number; normalizeOutput?: boolean; holdoutFraction?: number } = {}) {
    super({ bias: false, seed: options.seed ?? 0 });
    this._normalizeOutput = options.normalizeOutput ?? true;
    this._holdoutFraction = options.holdoutFraction ?? 0.1;
  }

  /**
   * Fit the orthogonal Procrustes mapping.
   *
   * Solves min ||X R - Y||_F subject to R^T R = I via SVD of X^T Y.
   */
  fit(X: Float64Array[], Y: Float64Array[]): this {
    validateXY(X, Y);
    this._dSrc = X[0].length;
    this._dTarget = Y[0].length;

    const { XTrain, XTest, YTrain, YTest } = trainTestSplit(X, Y, this._holdoutFraction, this._seed);

    // Solve Procrustes: M = X^T Y, SVD(M) = U Σ V^T, R = U V^T
    const M = matrixMultiply(transpose(XTrain), YTrain);
    const { U, Vt } = svd(M);
    const R = matrixMultiply(U, Vt);

    this._W = R;
    this._WInv = transpose(R);
    this._fitted = true;

    const metrics = computeHoldoutMetrics(this, XTest, YTest);
    this._validationReport = {
      holdoutCosineMean: metrics.cosineMean,
      holdoutCosineMedian: metrics.cosineMedian,
      holdoutCosineP5: metrics.cosineP5,
      top1Retention: metrics.top1Retention,
      top10Retention: metrics.top10Retention,
      nTrain: XTrain.length,
      nHoldout: XTest.length,
      seed: this._seed,
      alpha: null,
      notes: [],
    };
    return this;
  }

  transform(V: Float64Array | Float64Array[]): Float64Array | Float64Array[] {
    this.requireFitted();
    return this.applyMapping(V, this._W!, this._dSrc!, {
      direction: 'forward',
      bias: false,
      normalize: this._normalizeOutput,
    });
  }

  inverseTransform(V: Float64Array | Float64Array[]): Float64Array | Float64Array[] {
    this.requireFitted();
    return this.applyMapping(V, this._WInv!, this._dTarget!, {
      direction: 'inverse',
      bias: false,
      normalize: this._normalizeOutput,
    });
  }
}

// ── ProcrustesDiagMapping ────────────────────────────────────────

/**
 * Orthogonal Procrustes + Diagonal Scaling (square dims only).
 *
 * First fits an orthogonal rotation via Procrustes, then learns a
 * per-dimension diagonal scaling to absorb magnitude differences.
 *
 * Y ≈ diag(s) · X · R  where R is orthogonal.
 */
export class ProcrustesDiagMapping extends Mapping {
  static readonly mappingType: MappingType = 'procrustes_diag';

  private _normalizeOutput: boolean;
  private _holdoutFraction: number;
  private _scales: Float64Array | null = null;
  private _scalesInv: Float64Array | null = null;

  constructor(options: { seed?: number; normalizeOutput?: boolean; holdoutFraction?: number } = {}) {
    super({ bias: false, seed: options.seed ?? 0 });
    this._normalizeOutput = options.normalizeOutput ?? true;
    this._holdoutFraction = options.holdoutFraction ?? 0.1;
  }

  /**
   * Fit the Procrustes + diagonal scaling mapping.
   *
   * Step 1: Procrustes rotation R = U V^T
   * Step 2: Diagonal scaling s = (XR · Y) / (XR · XR) per dimension
   * Combined: W = R * diag(s), W_inv = diag(s)^{-1} * R^T
   */
  fit(X: Float64Array[], Y: Float64Array[]): this {
    validateXY(X, Y);
    this._dSrc = X[0].length;
    this._dTarget = Y[0].length;

    const { XTrain, XTest, YTrain, YTest } = trainTestSplit(X, Y, this._holdoutFraction, this._seed);

    // Step 1: Procrustes rotation
    const M = matrixMultiply(transpose(XTrain), YTrain);
    const { U, Vt } = svd(M);
    const R = matrixMultiply(U, Vt);

    // Step 2: Diagonal scaling after rotation
    const XR = matrixMultiply(XTrain, R); // (n, d)
    const d = XR[0].length;

    const s = new Float64Array(d);
    const sInv = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      let denom = 0;
      let numer = 0;
      for (let i = 0; i < XR.length; i++) {
        denom += XR[i][j] * XR[i][j];
        numer += XR[i][j] * YTrain[i][j];
      }
      s[j] = numer / Math.max(denom, 1e-8);
      sInv[j] = Math.abs(s[j]) > 1e-8 ? 1.0 / s[j] : 0.0;
    }

    // Combined: W = R * diag(s)  =>  v_out = v_in @ R @ diag(s)
    // W[i][j] = R[i][j] * s[j]
    const W: Float64Array[] = new Array(d);
    for (let i = 0; i < d; i++) {
      W[i] = new Float64Array(d);
      for (let j = 0; j < d; j++) {
        W[i][j] = R[i][j] * s[j];
      }
    }

    // Inverse: diag(s)^{-1} * R^T
    const Rt = transpose(R);
    const WInv: Float64Array[] = new Array(d);
    for (let i = 0; i < d; i++) {
      WInv[i] = new Float64Array(d);
      for (let j = 0; j < d; j++) {
        WInv[i][j] = Rt[i][j] * sInv[j];
      }
    }

    this._W = W;
    this._WInv = WInv;
    this._scales = s;
    this._scalesInv = sInv;
    this._fitted = true;

    const metrics = computeHoldoutMetrics(this, XTest, YTest);
    this._validationReport = {
      holdoutCosineMean: metrics.cosineMean,
      holdoutCosineMedian: metrics.cosineMedian,
      holdoutCosineP5: metrics.cosineP5,
      top1Retention: metrics.top1Retention,
      top10Retention: metrics.top10Retention,
      nTrain: XTrain.length,
      nHoldout: XTest.length,
      seed: this._seed,
      alpha: null,
      notes: [],
    };
    return this;
  }

  transform(V: Float64Array | Float64Array[]): Float64Array | Float64Array[] {
    this.requireFitted();
    return this.applyMapping(V, this._W!, this._dSrc!, {
      direction: 'forward',
      bias: false,
      normalize: this._normalizeOutput,
    });
  }

  inverseTransform(V: Float64Array | Float64Array[]): Float64Array | Float64Array[] {
    this.requireFitted();
    return this.applyMapping(V, this._WInv!, this._dTarget!, {
      direction: 'inverse',
      bias: false,
      normalize: this._normalizeOutput,
    });
  }
}
