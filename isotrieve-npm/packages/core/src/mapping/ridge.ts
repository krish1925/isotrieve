/**
 * Ridge regression embedding-space mapping.
 *
 * Fits a ridge-regularized linear map between source and target embeddings.
 * When `alpha` is "auto", uses generalized cross-validation (GCV) to select
 * the optimal regularization strength. Supports optional dimensionality
 * reduction via truncated SVD and holdout validation.
 */

import { Mapping } from './base';
import {
  transpose,
  matrixMultiply,
  vectorMatrixMultiply,
  svd,
  ridgeCv,
  leastSquares,
  zeros,
  shape,
} from '../math/linalg';
import { augmentBias, checkFinite, l2Normalize } from '../math/normalize';
import {
  pairwiseCosineStats,
  topkRetention,
  holdoutRankCorrelation,
} from '../math/metrics';
import { trainTestSplit } from '../math/random';
import type { MappingType, ValidationReport } from '../types';

/**
 * Configuration options for RidgeMapping.
 */
export interface RidgeMappingOptions {
  /** Regularization strength. "auto" uses GCV. Default: "auto". */
  alpha?: 'auto' | number;
  /** Whether to include a bias (intercept) term. Default: true. */
  bias?: boolean;
  /** Random seed for train/test split. Default: 0. */
  seed?: number;
  /** Whether to L2-normalize output vectors. Default: true. */
  normalizeOutput?: boolean;
  /** Fraction of data to hold out for validation. Default: 0.1. */
  holdoutFraction?: number;
  /** If set, truncate to this many singular components. null = no truncation. */
  rank?: number | null;
}

/**
 * Ridge regression mapping between embedding spaces.
 *
 * The mapping solves: W = argmin ||Y - X W||² + α||W||²
 *
 * When alpha is "auto", the optimal α is selected via generalized
 * cross-validation over a log-spaced grid.
 */
export class RidgeMapping extends Mapping {
  static readonly mappingType: MappingType = 'ridge';

  private _alpha: 'auto' | number;
  private _normalizeOutput: boolean;
  private _holdoutFraction: number;
  private _rank: number | null;
  private _selectedAlpha: number | null = null;

  constructor(options: RidgeMappingOptions = {}) {
    super({ bias: options.bias ?? true, seed: options.seed ?? 0 });
    this._alpha = options.alpha ?? 'auto';
    this._normalizeOutput = options.normalizeOutput ?? true;
    this._holdoutFraction = options.holdoutFraction ?? 0.1;
    this._rank = options.rank ?? null;
  }

  // ── Fit ────────────────────────────────────────────────────────

  /**
   * Fit the ridge regression mapping.
   *
   * Steps:
   * 1. Validate input dimensions and finiteness.
   * 2. Train/test split for holdout validation.
   * 3. Augment bias column if enabled.
   * 4. Fit forward ridge (GCV if auto, else fixed alpha).
   * 5. Fit inverse ridge the same way.
   * 6. Optionally compress via truncated SVD.
   * 7. Compute holdout validation metrics.
   *
   * @param X - Source embedding vectors.
   * @param Y - Target embedding vectors.
   * @returns this (for chaining).
   */
  fit(X: Float64Array[], Y: Float64Array[]): this {
    if (X.length < 2) {
      throw new Error('Need at least 2 samples to fit a mapping');
    }

    // Set dimensions
    this._dSrc = X[0].length;
    this._dTarget = Y[0].length;

    // Validate finiteness
    checkFinite('X', X);
    checkFinite('Y', Y);

    // Warn if K < 10 × min_dim
    const minDim = Math.min(this._dSrc, this._dTarget);
    if (X.length < 10 * minDim) {
      console.warn(
        `Warning: K=${X.length} < 10×min_dim=${10 * minDim}. ` +
        'Results may be unreliable; consider collecting more calibration pairs.',
      );
    }

    // Train/test split
    const { XTrain, XTest, YTrain, YTest } = trainTestSplit(
      X, Y, this._holdoutFraction, this._seed,
    );

    // Augment bias for training (only augment X, NOT Y)
    const XTrainAug = this._bias ? augmentBias(XTrain) : XTrain;

    // Fit forward ridge
    let W: Float64Array[];
    let selectedAlpha: number;

    if (this._alpha === 'auto') {
      const result = ridgeCv(XTrainAug, YTrain);
      W = result.W;
      selectedAlpha = result.bestAlpha;
    } else {
      W = leastSquares(XTrainAug, YTrain, this._alpha);
      selectedAlpha = this._alpha;
    }

    this._selectedAlpha = selectedAlpha;

    // Truncated SVD compression for forward mapping
    if (this._rank !== null && this._rank > 0) {
      const truncated = this._truncateSvd(W, this._rank);
      W = truncated;
    }

    this._W = W;

    // Fit inverse ridge (Y→X, only augment the input side)
    const XInvAug = this._bias ? augmentBias(YTrain) : YTrain;

    let WInv: Float64Array[];
    if (this._alpha === 'auto') {
      const result = ridgeCv(XInvAug, XTrain);
      WInv = result.W;
    } else {
      WInv = leastSquares(XInvAug, XTrain, this._alpha);
    }

    if (this._rank !== null && this._rank > 0) {
      WInv = this._truncateSvd(WInv, this._rank);
    }

    this._WInv = WInv;

    // Holdout validation
    const holdoutMetrics = this._computeHoldoutMetrics(XTest, YTest, W);

    this._validationReport = {
      holdoutCosineMean: holdoutMetrics.cosineMean,
      holdoutCosineMedian: holdoutMetrics.cosineMedian,
      holdoutCosineP5: holdoutMetrics.cosineP5,
      top1Retention: holdoutMetrics.top1Retention,
      top10Retention: holdoutMetrics.top10Retention,
      nTrain: XTrain.length,
      nHoldout: XTest.length,
      seed: this._seed,
      alpha: selectedAlpha,
      notes: [],
    };

    this._fitted = true;
    return this;
  }

  // ── Transform ──────────────────────────────────────────────────

  /**
   * Map source-space vectors to target space via the learned ridge mapping.
   *
   * @param V - Single vector or batch of source-space vectors.
   * @returns Mapped target-space vector(s), L2-normalized if normalizeOutput is true.
   */
  transform(V: Float64Array | Float64Array[]): Float64Array | Float64Array[] {
    this.requireFitted();
    return this.applyMapping(V, this._W!, this._dSrc!, {
      direction: 'forward',
      bias: this._bias,
      normalize: this._normalizeOutput,
    });
  }

  // ── Inverse Transform ──────────────────────────────────────────

  /**
   * Map target-space vectors back to source space via the inverse ridge mapping.
   *
   * @param V - Single vector or batch of target-space vectors.
   * @returns Reconstructed source-space vector(s), L2-normalized if normalizeOutput is true.
   */
  inverseTransform(V: Float64Array | Float64Array[]): Float64Array | Float64Array[] {
    this.requireFitted();
    if (this._WInv === null) {
      throw new Error('Inverse mapping not available for this RidgeMapping.');
    }
    return this.applyMapping(V, this._WInv, this._dTarget!, {
      direction: 'inverse',
      bias: this._bias,
      normalize: this._normalizeOutput,
    });
  }

  // ── Private helpers ────────────────────────────────────────────

  /**
   * Compute holdout validation metrics.
   */
  private _computeHoldoutMetrics(
    XTest: Float64Array[],
    YTest: Float64Array[],
    W: Float64Array[],
  ): {
    cosineMean: number;
    cosineMedian: number;
    cosineP5: number;
    top1Retention: number;
    top10Retention: number;
  } {
    // Apply forward mapping to holdout source vectors
    let mapped: Float64Array[];
    if (this._bias) {
      const XTestAug = augmentBias(XTest);
      mapped = XTestAug.map((v) => vectorMatrixMultiply(v, W));
    } else {
      mapped = XTest.map((v) => vectorMatrixMultiply(v, W));
    }

    // Normalize for cosine computation
    const normMapped = l2Normalize(mapped) as Float64Array[];
    const normTarget = l2Normalize(YTest) as Float64Array[];

    const cosine = pairwiseCosineStats(normMapped, normTarget);
    const top1 = topkRetention(normMapped, normTarget, 1);
    const top10 = topkRetention(normMapped, normTarget, Math.min(10, normTarget.length));

    return {
      cosineMean: cosine.mean,
      cosineMedian: cosine.median,
      cosineP5: cosine.p5,
      top1Retention: top1,
      top10Retention: top10,
    };
  }

  /**
   * Truncate a weight matrix to `rank` components via SVD.
   *
   * Given W of shape (d_src, d_target), computes SVD and keeps only the
   * top `rank` singular vectors: W_truncated = U[:, :r] @ S[:r] @ Vt[:r, :]
   *
   * This is done on the transpose to get meaningful components.
   */
  private _truncateSvd(W: Float64Array[], rank: number): Float64Array[] {
    const [rows, cols] = shape(W);

    // SVD of W itself
    const { U, S, Vt } = svd(W);

    const actualRank = Math.min(rank, S.length);

    // Reconstruct: U[:, :r] @ diag(S[:r]) @ Vt[:r, :]
    const Ur = U.map((row) => row.subarray(0, actualRank));
    const Vtr = Vt.slice(0, actualRank);

    // Scale rows of Vtr by singular values
    const scaledVtr: Float64Array[] = new Array(actualRank);
    for (let i = 0; i < actualRank; i++) {
      const s = S[i];
      scaledVtr[i] = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        scaledVtr[i][j] = Vtr[i][j] * s;
      }
    }

    return matrixMultiply(Ur, scaledVtr);
  }
}
