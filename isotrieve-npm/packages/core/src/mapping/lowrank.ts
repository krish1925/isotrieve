import { Mapping } from './base';
import {
  transpose,
  matrixMultiply,
  svd,
  ridgeCv,
  leastSquares,
  TypedMatrix,
  toTypedMatrix,
  SingleOrBatch,
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

  fit(X: Float64Array[] | Float32Array[] | TypedMatrix, Y: Float64Array[] | Float32Array[] | TypedMatrix): this {
    const Xm = toTypedMatrix(X);
    const Ym = toTypedMatrix(Y);

    if (Xm.rows < 2) {
      throw new Error('Need at least 2 samples to fit a mapping');
    }

    this._dSrc = Xm.cols;
    this._dTarget = Ym.cols;

    checkFinite('X', Xm);
    checkFinite('Y', Ym);

    const minDim = Math.min(this._dSrc, this._dTarget);
    if (Xm.rows < 10 * minDim) {
      console.warn(
        `Warning: K=${Xm.rows} < 10×min_dim=${10 * minDim}. ` +
        'Results may be unreliable; consider collecting more calibration pairs.',
      );
    }

    const XRows = Xm.toFloat64Arrays();
    const YRows = Ym.toFloat64Arrays();
    const { XTrain, XTest, YTrain, YTest } = trainTestSplit(
      XRows, YRows, this._holdoutFraction, this._seed,
    );

    const XTrainM = TypedMatrix.fromRows(XTrain);
    const YTrainM = TypedMatrix.fromRows(YTrain);

    let XTrainAugM: TypedMatrix;
    if (this._bias) {
      const augRows = augmentBias(XTrain);
      XTrainAugM = TypedMatrix.fromRows(augRows);
    } else {
      XTrainAugM = XTrainM;
    }

    let W: TypedMatrix;
    let selectedAlpha: number;
    if (this._alpha === 'auto') {
      const result = ridgeCv(XTrainAugM, YTrainM);
      W = result.W;
      selectedAlpha = result.bestAlpha;
    } else {
      W = leastSquares(XTrainAugM, YTrainM, this._alpha);
      selectedAlpha = this._alpha;
    }

    this._selectedAlpha = selectedAlpha;

    const effectiveRank = this._rank || Math.min(W.rows, W.cols);
    if (effectiveRank < Math.min(W.rows, W.cols)) {
      W = lowRankTruncate(W, effectiveRank);
    }

    this._W = W;

    const XInvAugRows = this._bias ? augmentBias(YTrain) : YRows;
    const XInvAugM = TypedMatrix.fromRows(XInvAugRows);
    const XTrainForInv = TypedMatrix.fromRows(XTrain);

    let WInv: TypedMatrix;
    if (this._alpha === 'auto') {
      const result = ridgeCv(XInvAugM, XTrainForInv);
      WInv = result.W;
    } else {
      WInv = leastSquares(XInvAugM, XTrainForInv, this._alpha);
    }

    if (effectiveRank < Math.min(WInv.rows, WInv.cols)) {
      WInv = lowRankTruncate(WInv, effectiveRank);
    }

    this._WInv = WInv;
    this._fitted = true;

    const Wcols = W.cols;
    const result = new TypedMatrix(new Float64Array(XTest.length * Wcols), XTest.length, Wcols);
    const rd = result.data;
    const XTM = TypedMatrix.fromRows(XTest);
    const xd = XTM.data;
    const wd = W.data;
    for (let i = 0; i < XTest.length; i++) {
      const xOff = i * XTM.cols;
      const rOff = i * Wcols;
      for (let k = 0; k < XTM.cols; k++) {
        const xik = xd[xOff + k];
        const wOff = k * Wcols;
        for (let j = 0; j < Wcols; j++) {
          rd[rOff + j] += xik * wd[wOff + j];
        }
      }
    }
    const mapped = result.toFloat64Arrays();

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

  transform(V: SingleOrBatch): Float64Array | Float64Array[] {
    this.requireFitted();
    return this.applyMapping(V, this._W!, this._dSrc!, {
      direction: 'forward',
      bias: this._bias,
      normalize: this._normalizeOutput,
    });
  }

  inverseTransform(V: SingleOrBatch): Float64Array | Float64Array[] {
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

function lowRankTruncate(W: TypedMatrix, effectiveRank: number): TypedMatrix {
  const { U, S, Vt } = svd(W);
  const r = Math.min(effectiveRank, S.length);

  const scaledU = TypedMatrix.zeros(U.rows, r);
  const sud = scaledU.data;
  const ud = U.data;
  for (let i = 0; i < U.rows; i++) {
    for (let j = 0; j < r; j++) {
      sud[i * r + j] = ud[i * U.cols + j] * S[j];
    }
  }

  const Vtr = TypedMatrix.zeros(r, Vt.cols);
  const vtrd = Vtr.data;
  const vtd = Vt.data;
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < Vt.cols; j++) {
      vtrd[i * Vt.cols + j] = vtd[i * Vt.cols + j];
    }
  }

  return matrixMultiply(scaledU, Vtr);
}
