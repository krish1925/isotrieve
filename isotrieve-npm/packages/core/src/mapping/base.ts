import {
  vectorMatrixMultiply,
  TypedMatrix,
  toTypedMatrix,
  isSingleVector,
  SingleOrBatch,
} from '../math/linalg';
import { l2Normalize, checkFinite, augmentBias } from '../math/normalize';
import { shape } from '../math/linalg';
import type {
  MappingType,
  ValidationReport,
  IsotrieveFileHeader,
  ScoreRecalibratorData,
  RecalibrationReport,
} from '../types';
import {
  writeIsotrieveFile,
  readIsotrieveHeader,
  readIsotrievePayload,
  FORMAT_VERSION,
} from './format';

const __VERSION__ = '0.1.0';

export interface ApplyMappingOptions {
  direction: 'forward' | 'inverse';
  bias: boolean;
  normalize: boolean;
}

export abstract class Mapping {
  static readonly mappingType: MappingType;

  protected _fitted = false;
  protected _W: TypedMatrix | null = null;
  protected _WInv: TypedMatrix | null = null;
  protected _dSrc: number | null = null;
  protected _dTarget: number | null = null;
  protected _bias: boolean;
  protected _seed: number;
  protected _validationReport: ValidationReport | null = null;
  protected _meta: Record<string, unknown> = {};
  protected _recalibrator: any = null;

  constructor(options: { bias?: boolean; seed?: number } = {}) {
    this._bias = options.bias ?? true;
    this._seed = options.seed ?? 0;
  }

  abstract fit(X: Float64Array[] | Float32Array[] | TypedMatrix, Y: Float64Array[] | Float32Array[] | TypedMatrix): this;

  abstract transform(V: SingleOrBatch): Float64Array | Float64Array[];
  abstract inverseTransform(V: SingleOrBatch): Float64Array | Float64Array[];

  get isFitted(): boolean {
    return this._fitted;
  }

  get dSrc(): number {
    this.requireFitted();
    return this._dSrc!;
  }

  get dTarget(): number {
    this.requireFitted();
    return this._dTarget!;
  }

  get hasRecalibrator(): boolean {
    return this._recalibrator !== null;
  }

  get hasInverse(): boolean {
    return this._WInv !== null;
  }

  validationReport(): ValidationReport {
    if (this._validationReport === null) {
      throw new Error('No validation report available. Was the mapping fitted?');
    }
    return this._validationReport;
  }

  setMeta(...args: Array<string | unknown>): void {
    if (args.length === 1 && typeof args[0] === 'object' && args[0] !== null) {
      Object.assign(this._meta, args[0]);
    } else {
      for (let i = 0; i < args.length - 1; i += 2) {
        this._meta[String(args[i])] = args[i + 1];
      }
    }
  }

  recalibrateScores(scores: Float64Array): Float64Array {
    if (this._recalibrator === null) {
      throw new Error('No recalibrator attached to this mapping.');
    }
    return this._recalibrator.recalibrate(scores);
  }

  save(path: string): void {
    this.requireFitted();

    const matrixShape: [number, number] = [this._W!.rows, this._W!.cols];
    const hasInverse = this._WInv !== null;
    const inverseMatrixShape = hasInverse ? [this._WInv!.rows, this._WInv!.cols] as [number, number] : undefined;

    const extraMatrices: Record<string, [number, number]> = {};

    const header: IsotrieveFileHeader = {
      formatVersion: FORMAT_VERSION,
      isotrieveVersion: __VERSION__,
      mappingType: (this.constructor as typeof Mapping).mappingType,
      dSrc: this._dSrc!,
      dTarget: this._dTarget!,
      bias: this._bias,
      seed: this._seed,
      fitDate: new Date().toISOString(),
      expiresHint: null,
      validation: this._validationReport,
      meta: this._meta,
      matrixShape,
      hasInverse,
      inverseMatrixShape,
      extraMatrices: Object.keys(extraMatrices).length > 0 ? extraMatrices : undefined,
    };

    if (this._recalibrator !== null) {
      header.scoreRecalV1 = this._recalibrator.toJSON();
    }

    const matrices: Array<{ name: string; mat: TypedMatrix }> = [
      { name: 'forward', mat: this._W! },
    ];
    if (hasInverse) {
      matrices.push({ name: 'inverse', mat: this._WInv! });
    }

    const extras = this._extraMatrices();
    for (const { name, mat } of extras) {
      extraMatrices[name] = [mat.rows, mat.cols];
      matrices.push({ name, mat });
    }

    if (Object.keys(extraMatrices).length > 0) {
      header.extraMatrices = extraMatrices;
    }

    writeIsotrieveFile(path, header as unknown as Record<string, unknown>, matrices);
  }

  static load(path: string): Mapping {
    const { loadMapping } = require('./registry') as typeof import('./registry');
    return loadMapping(path);
  }

  requireFitted(): void {
    if (!this._fitted) {
      throw new Error(
        `Mapping of type "${(this.constructor as typeof Mapping).mappingType}" has not been fitted. ` +
        'Call fit() before transform/inverseTransform.',
      );
    }
  }

  applyMapping(
    V: SingleOrBatch,
    matrix: TypedMatrix,
    expectedDim: number,
    options: ApplyMappingOptions,
  ): Float64Array | Float64Array[] {
    const singleInput = isSingleVector(V);
    let vecs: TypedMatrix;
    if (singleInput) {
      const v = V as Float64Array | Float32Array;
      vecs = new TypedMatrix(
        v instanceof Float64Array ? new Float64Array(v) : new Float64Array(v),
        1,
        v.length,
      );
    } else {
      vecs = toTypedMatrix(V);
    }

    if (vecs.rows > 0 && vecs.cols !== expectedDim) {
      throw new Error(
        `Expected ${expectedDim}-dimensional vectors for ${options.direction} mapping, ` +
        `got ${vecs.cols}`,
      );
    }

    checkFinite(`${options.direction} input`, vecs);

    let augmented: TypedMatrix;
    if (options.bias) {
      const augRows = augmentBias(vecs.toFloat64Arrays());
      augmented = TypedMatrix.fromRows(augRows);
    } else {
      augmented = vecs;
    }

    const n = matrix.cols;
    const result = new TypedMatrix(new Float64Array(augmented.rows * n), augmented.rows, n);
    const rd = result.data;
    const ad = augmented.data;
    const md = matrix.data;
    const aCols = augmented.cols;
    for (let i = 0; i < augmented.rows; i++) {
      const aOff = i * aCols;
      const rOff = i * n;
      for (let k = 0; k < aCols; k++) {
        const aik = ad[aOff + k];
        const mOff = k * n;
        for (let j = 0; j < n; j++) {
          rd[rOff + j] += aik * md[mOff + j];
        }
      }
    }

    let output: Float64Array[];
    if (options.normalize) {
      output = l2Normalize(result) as Float64Array[];
    } else {
      output = result.toFloat64Arrays();
    }

    if (singleInput) {
      return output[0];
    }
    return output;
  }

  protected _extraMatrices(): Array<{ name: string; mat: TypedMatrix }> {
    return [];
  }

  protected _restoreExtraMatrices(_extras: Map<string, TypedMatrix>): void {
    // no-op by default
  }

  protected _setDimsFromExtra(_extras: Map<string, TypedMatrix>): void {
    // no-op by default
  }
}
