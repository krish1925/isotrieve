/**
 * Low-rank affine mapping via truncated SVD on the ridge residual.
 *
 * Fits a ridge mapping, then compresses the learned weight matrix to rank `r`
 * via truncated SVD. Useful when d_src/d_target are large and you want a more
 * compact mapping (smaller .isotrieve file, faster transform).
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
import { pairwiseCosineStats, topkRetention } from '../math/metrics';
import { trainTestSplit } from '../math/random';
import type { MappingType, ValidationReport } from '../types';

export interface LowRankAffineMappingOptions {
  alpha?: 'auto' | number;
  rank?: number | null;
  bias?: boolean;
  seed?: number;
  normalizeOutput?: boolean;
  holdoutFraction?: number;
}

/**
 * Low-rank affine mapping via truncated SVD on ridge.
 *
 * When rank >= min(d_src, d_target), behaves identically to RidgeMapping.
 */
export class LowRankAffineMapping extends Mapping {
  static readonly mappingType: MappingType = 'lowrank_affine';

  private _alpha: 'auto' | number;
  private _rank: number | null;
  private _normalizeOutput: boolean;
  private _holdoutFraction: number;
  private _selectedAlpha: number | null = null;

  constructor(options: LowRankAffineMappingOptions = {}) {
    super({ bias: options.bias ?? true, seed: options.seed ?? 0 });
    this._alpha = options.alpha ?? 'auto';
    this._rank = options.rank ?? null;
    this._normalizeOutput = options.normalizeOutput ?? true;
    this._holdoutFraction = options.holdoutFraction ?? 0.1;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    if (X.length < 2) {
      throw new Error('Need at least 2 samples to fit a mapping');
    }

    this._dSrc = X[0].length;
    this._dTarget = Y[0].length;

    checkFinite('X', X);
    checkFinite('Y', Y);

    const minDim = Math.min(this._dSrc, this._dTarget);
    if (X.length < 10 * minDim) {
      console.warn(
        `Warning: K=${X.length} < 10×min_dim=${10 * minDim}. ` +
        'Results may be unreliable; consider collecting more calibration pairs.',
      );
    }

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

    // Truncated SVD for low-rank approximation
    const rank = this._rank || Math.min(...shape(W));
    if (rank < Math.min(...shape(W))) {
      const { U, S, Vt } = svd(W);
      const actualRank = Math.min(rank, S.length);

      // Reconstruct: U[:, :r] @ diag(sigma[:r]) @ Vt[:r, :]
      const Ur = U.map((row) => row.subarray(0, actualRank));
      const Vtr = Vt.slice(0, actualRank);

      const scaledVtr: Float64Array[] = new Array(actualRank);
      for (let i = 0; i < actualRank; i++) {
        scaledVtr[i] = new Float64Array(W[0].length);
        for (let j = 0; j < W[0].length; j++) {
          scaledVtr[i][j] = Vtr[i][j] * S[i];
        }
      }
      W = matrixMultiply(Ur, scaledVtr);
    }

    this._W = W;

    // Fit inverse via ridge Y → X (only augment the input side)
    const XInvAug = this._bias ? augmentBias(YTrain) : YTrain;

    let WInv: Float64Array[];
    if (this._alpha === 'auto') {
      const result = ridgeCv(XInvAug, XTrain);
      WInv = result.W;
    } else {
      WInv = leastSquares(XInvAug, XTrain, this._alpha);
    }

    if (rank < Math.min(...shape(WInv))) {
      const { U, S, Vt } = svd(WInv);
      const actualRank = Math.min(rank, S.length);
      const Ur = U.map((row) => row.subarray(0, actualRank));
      const Vtr = Vt.slice(0, actualRank);
      const scaledVtr: Float64Array[] = new Array(actualRank);
      for (let i = 0; i < actualRank; i++) {
        scaledVtr[i] = new Float64Array(WInv[0].length);
        for (let j = 0; j < WInv[0].length; j++) {
          scaledVtr[i][j] = Vtr[i][j] * S[i];
        }
      }
      WInv = matrixMultiply(Ur, scaledVtr);
    }

    this._WInv = WInv;
    this._fitted = true;

    // Holdout validation
    let mapped: Float64Array[];
    if (this._bias) {
      const XTestAug = augmentBias(XTest);
      mapped = XTestAug.map((v) => vectorMatrixMultiply(v, W));
    } else {
      mapped = XTest.map((v) => vectorMatrixMultiply(v, W));
    }
    const normMapped = l2Normalize(mapped) as Float64Array[];
    const normTarget = l2Normalize(YTest) as Float64Array[];
    const cosine = pairwiseCosineStats(normMapped, normTarget);

    this._validationReport = {
      holdoutCosineMean: cosine.mean,
      holdoutCosineMedian: cosine.median,
      holdoutCosineP5: cosine.p5,
      top1Retention: topkRetention(normMapped, normTarget, 1),
      top10Retention: topkRetention(normMapped, normTarget, Math.min(10, normTarget.length)),
      nTrain: XTrain.length,
      nHoldout: XTest.length,
      seed: this._seed,
      alpha: selectedAlpha,
      notes: [],
    };

    return this;
  }

  transform(V: Float64Array | Float64Array[]): Float64Array | Float64Array[] {
    this.requireFitted();
    return this.applyMapping(V, this._W!, this._dSrc!, {
      direction: 'forward',
      bias: this._bias,
      normalize: this._normalizeOutput,
    });
  }

  inverseTransform(V: Float64Array | Float64Array[]): Float64Array | Float64Array[] {
    this.requireFitted();
    if (this._WInv === null) {
      throw new Error('Inverse mapping not available for this LowRankAffineMapping.');
    }
    return this.applyMapping(V, this._WInv, this._dTarget!, {
      direction: 'inverse',
      bias: this._bias,
      normalize: this._normalizeOutput,
    });
  }
}
