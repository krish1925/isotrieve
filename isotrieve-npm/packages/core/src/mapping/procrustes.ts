import { Mapping } from './base';
import { transpose, matrixMultiply, svd, vectorMatrixMultiply, TypedMatrix, toTypedMatrix, SingleOrBatch } from '../math/linalg';
import { l2Normalize, checkFinite } from '../math/normalize';
import { pairwiseCosineStats, topkRetention } from '../math/metrics';
import { trainTestSplit } from '../math/random';
import type { MappingType, ValidationReport } from '../types';

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

export class OrthogonalProcrustesMapping extends Mapping {
  static readonly mappingType: MappingType = 'orthogonal_procrustes';

  private _normalizeOutput: boolean;
  private _holdoutFraction: number;

  constructor(options: { seed?: number; normalizeOutput?: boolean; holdoutFraction?: number } = {}) {
    super({ bias: false, seed: options.seed ?? 0 });
    this._normalizeOutput = options.normalizeOutput ?? true;
    this._holdoutFraction = options.holdoutFraction ?? 0.1;
  }

  fit(X: Float64Array[] | Float32Array[] | TypedMatrix, Y: Float64Array[] | Float32Array[] | TypedMatrix): this {
    const Xrows = toTypedMatrix(X).toFloat64Arrays();
    const Yrows = toTypedMatrix(Y).toFloat64Arrays();
    validateXY(Xrows, Yrows);
    this._dSrc = Xrows[0].length;
    this._dTarget = Yrows[0].length;

    const { XTrain, XTest, YTrain, YTest } = trainTestSplit(Xrows, Yrows, this._holdoutFraction, this._seed);

    const XTM = TypedMatrix.fromRows(XTrain);
    const YTM = TypedMatrix.fromRows(YTrain);

    const M = matrixMultiply(transpose(XTM), YTM);
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

  transform(V: SingleOrBatch): Float64Array | Float64Array[] {
    this.requireFitted();
    return this.applyMapping(V, this._W!, this._dSrc!, {
      direction: 'forward',
      bias: false,
      normalize: this._normalizeOutput,
    });
  }

  inverseTransform(V: SingleOrBatch): Float64Array | Float64Array[] {
    this.requireFitted();
    return this.applyMapping(V, this._WInv!, this._dTarget!, {
      direction: 'inverse',
      bias: false,
      normalize: this._normalizeOutput,
    });
  }
}

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

  fit(X: Float64Array[] | Float32Array[] | TypedMatrix, Y: Float64Array[] | Float32Array[] | TypedMatrix): this {
    const Xrows = toTypedMatrix(X).toFloat64Arrays();
    const Yrows = toTypedMatrix(Y).toFloat64Arrays();
    validateXY(Xrows, Yrows);
    this._dSrc = Xrows[0].length;
    this._dTarget = Yrows[0].length;

    const { XTrain, XTest, YTrain, YTest } = trainTestSplit(Xrows, Yrows, this._holdoutFraction, this._seed);

    const XTM = TypedMatrix.fromRows(XTrain);
    const YTM = TypedMatrix.fromRows(YTrain);

    const M = matrixMultiply(transpose(XTM), YTM);
    const { U, Vt } = svd(M);
    const R = matrixMultiply(U, Vt);

    const XR = matrixMultiply(XTM, R);
    const d = XR.cols;

    const s = new Float64Array(d);
    const sInv = new Float64Array(d);
    const xrd = XR.data;
    const ytd = YTM.data;
    for (let j = 0; j < d; j++) {
      let denom = 0;
      let numer = 0;
      for (let i = 0; i < XR.rows; i++) {
        denom += xrd[i * d + j] * xrd[i * d + j];
        numer += xrd[i * d + j] * ytd[i * d + j];
      }
      s[j] = numer / Math.max(denom, 1e-8);
      sInv[j] = Math.abs(s[j]) > 1e-8 ? 1.0 / s[j] : 0.0;
    }

    const rd = R.data;
    const Wdata = new Float64Array(d * d);
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        Wdata[i * d + j] = rd[i * d + j] * s[j];
      }
    }

    const Rt = transpose(R);
    const rtd = Rt.data;
    const WInvData = new Float64Array(d * d);
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        WInvData[i * d + j] = rtd[i * d + j] * sInv[j];
      }
    }

    this._W = new TypedMatrix(Wdata, d, d);
    this._WInv = new TypedMatrix(WInvData, d, d);
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

  transform(V: SingleOrBatch): Float64Array | Float64Array[] {
    this.requireFitted();
    return this.applyMapping(V, this._W!, this._dSrc!, {
      direction: 'forward',
      bias: false,
      normalize: this._normalizeOutput,
    });
  }

  inverseTransform(V: SingleOrBatch): Float64Array | Float64Array[] {
    this.requireFitted();
    return this.applyMapping(V, this._WInv!, this._dTarget!, {
      direction: 'inverse',
      bias: false,
      normalize: this._normalizeOutput,
    });
  }
}
