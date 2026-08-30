import { Mapping, ApplyMappingOptions } from './base';
import {
  transpose,
  matrixMultiply,
  vectorMatrixMultiply,
  svd,
  ridgeCv,
  leastSquares,
  TypedMatrix,
  toTypedMatrix,
  SingleOrBatch,
} from '../math/linalg';
import { augmentBias, checkFinite, l2Normalize } from '../math/normalize';
import {
  pairwiseCosineStats,
  topkRetention,
} from '../math/metrics';
import { trainTestSplit } from '../math/random';
import type { MappingType, ValidationReport } from '../types';

export interface RidgeMappingOptions {
  alpha?: 'auto' | number;
  bias?: boolean;
  seed?: number;
  normalizeOutput?: boolean;
  holdoutFraction?: number;
  rank?: number | null;
}

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

    if (this._rank !== null && this._rank > 0) {
      W = this._truncateSvdFlat(W, this._rank);
    }

    this._W = W;

    const XInvAugRows = this._bias ? augmentBias(YTrain) : YTrain;
    const XInvAugM = TypedMatrix.fromRows(XInvAugRows);
    const XTrainForInv = TypedMatrix.fromRows(XTrain);

    let WInv: TypedMatrix;
    if (this._alpha === 'auto') {
      const result = ridgeCv(XInvAugM, XTrainForInv);
      WInv = result.W;
    } else {
      WInv = leastSquares(XInvAugM, XTrainForInv, this._alpha);
    }

    if (this._rank !== null && this._rank > 0) {
      WInv = this._truncateSvdFlat(WInv, this._rank);
    }

    this._WInv = WInv;

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
      throw new Error('Inverse mapping not available for this RidgeMapping.');
    }
    return this.applyMapping(V, this._WInv, this._dTarget!, {
      direction: 'inverse',
      bias: this._bias,
      normalize: this._normalizeOutput,
    });
  }

  private _computeHoldoutMetrics(
    XTest: Float64Array[],
    YTest: Float64Array[],
    W: TypedMatrix,
  ): {
    cosineMean: number;
    cosineMedian: number;
    cosineP5: number;
    top1Retention: number;
    top10Retention: number;
  } {
    let mapped: Float64Array[];
    if (this._bias) {
      const XTestAug = augmentBias(XTest);
      const XTestAugM = TypedMatrix.fromRows(XTestAug);
      const result = new TypedMatrix(new Float64Array(XTestAugM.rows * W.cols), XTestAugM.rows, W.cols);
      const rd = result.data;
      const ad = XTestAugM.data;
      const md = W.data;
      const aCols = XTestAugM.cols;
      for (let i = 0; i < XTestAugM.rows; i++) {
        const aOff = i * aCols;
        const rOff = i * W.cols;
        for (let k = 0; k < aCols; k++) {
          const aik = ad[aOff + k];
          if (aik === 0) continue;
          const mOff = k * W.cols;
          for (let j = 0; j < W.cols; j++) {
            rd[rOff + j] += aik * md[mOff + j];
          }
        }
      }
      mapped = result.toFloat64Arrays();
    } else {
      const XTestM = TypedMatrix.fromRows(XTest);
      const result = new TypedMatrix(new Float64Array(XTestM.rows * W.cols), XTestM.rows, W.cols);
      const rd = result.data;
      const ad = XTestM.data;
      const md = W.data;
      const aCols = XTestM.cols;
      for (let i = 0; i < XTestM.rows; i++) {
        const aOff = i * aCols;
        const rOff = i * W.cols;
        for (let k = 0; k < aCols; k++) {
          const aik = ad[aOff + k];
          if (aik === 0) continue;
          const mOff = k * W.cols;
          for (let j = 0; j < W.cols; j++) {
            rd[rOff + j] += aik * md[mOff + j];
          }
        }
      }
      mapped = result.toFloat64Arrays();
    }

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

  private _truncateSvdFlat(W: TypedMatrix, rank: number): TypedMatrix {
    const { U, S, Vt } = svd(W);
    const actualRank = Math.min(rank, S.length);

    const Ur = U.sliceRow(0, U.rows);
    const Vtr = Vt.sliceRow(0, actualRank);

    const scaledVtrData = new Float64Array(actualRank * Vtr.cols);
    for (let i = 0; i < actualRank; i++) {
      const s = S[i];
      const off = i * Vtr.cols;
      for (let j = 0; j < Vtr.cols; j++) {
        scaledVtrData[off + j] = Vtr.data[i * Vtr.cols + j] * s;
      }
    }
    const scaledVtr = new TypedMatrix(scaledVtrData, actualRank, Vtr.cols);

    return matrixMultiply(Ur, scaledVtr);
  }
}
