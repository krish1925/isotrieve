/**
 * Abstract base class for all isotrieve embedding-space mappings.
 *
 * Provides shared logic for transform/inverseTransform (bias augmentation,
 * dimension checks, finiteness checks, matrix multiply, L2 normalization),
 * serialization via the .isotrieve binary format, and score recalibration.
 */

import {
  vectorMatrixMultiply,
  matrixMultiply,
  shape,
} from '../math/linalg';
import { l2Normalize, checkFinite, augmentBias } from '../math/normalize';
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

/** Package version used in serialized headers. */
const __VERSION__ = '0.1.0';

/**
 * Options for the shared applyMapping helper.
 */
export interface ApplyMappingOptions {
  /** Direction label for error messages. */
  direction: 'forward' | 'inverse';
  /** Whether to augment a bias (ones) column before multiplication. */
  bias: boolean;
  /** Whether to L2-normalize output vectors. */
  normalize: boolean;
}

/**
 * Abstract mapping between two embedding spaces.
 *
 * Concrete subclasses must implement `fit`, `transform`, and `inverseTransform`.
 */
export abstract class Mapping {
  /** Discriminant tag for serialization dispatch. */
  static readonly mappingType: MappingType;

  protected _fitted = false;
  protected _W: Float64Array[] | null = null;
  protected _WInv: Float64Array[] | null = null;
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

  // ── Abstract interface ─────────────────────────────────────────

  /**
   * Fit the mapping to paired source→target vectors.
   * @param X - Source embedding vectors.
   * @param Y - Target embedding vectors.
   */
  abstract fit(X: Float64Array[], Y: Float64Array[]): this;

  /**
   * Map source-space vectors to target space.
   * Accepts a single vector or a batch.
   */
  abstract transform(V: Float64Array | Float64Array[]): Float64Array | Float64Array[];

  /**
   * Map target-space vectors back to source space.
   * Accepts a single vector or a batch.
   */
  abstract inverseTransform(V: Float64Array | Float64Array[]): Float64Array | Float64Array[];

  // ── Getters ────────────────────────────────────────────────────

  /** Whether this mapping has been fitted. */
  get isFitted(): boolean {
    return this._fitted;
  }

  /** Source embedding dimension. Throws if not fitted. */
  get dSrc(): number {
    this.requireFitted();
    return this._dSrc!;
  }

  /** Target embedding dimension. Throws if not fitted. */
  get dTarget(): number {
    this.requireFitted();
    return this._dTarget!;
  }

  /** Whether this mapping has a score recalibrator attached. */
  get hasRecalibrator(): boolean {
    return this._recalibrator !== null;
  }

  /** Whether this mapping has an inverse transformation. */
  get hasInverse(): boolean {
    return this._WInv !== null;
  }

  // ── Validation ─────────────────────────────────────────────────

  /**
   * Return the validation report from fitting.
   * @throws If no validation report exists (mapping not fitted or fit without holdout).
   */
  validationReport(): ValidationReport {
    if (this._validationReport === null) {
      throw new Error('No validation report available. Was the mapping fitted?');
    }
    return this._validationReport;
  }

  // ── Meta ───────────────────────────────────────────────────────

  /**
   * Merge key/value pairs into the metadata dictionary.
   * Metadata is persisted in the .isotrieve header on save.
   */
  setMeta(...args: Array<string | unknown>): void {
    if (args.length === 1 && typeof args[0] === 'object' && args[0] !== null) {
      Object.assign(this._meta, args[0]);
    } else {
      for (let i = 0; i < args.length - 1; i += 2) {
        this._meta[String(args[i])] = args[i + 1];
      }
    }
  }

  // ── Score recalibration ────────────────────────────────────────

  /**
   * Apply score recalibration to raw mapped-vs-target cosine scores.
   * Requires that a recalibrator has been attached.
   *
   * @param scores - Raw cosine similarity scores.
   * @returns Recalibrated scores.
   */
  recalibrateScores(scores: Float64Array): Float64Array {
    if (this._recalibrator === null) {
      throw new Error('No recalibrator attached to this mapping.');
    }
    return this._recalibrator.recalibrate(scores);
  }

  // ── Serialization ──────────────────────────────────────────────

  /**
   * Save the fitted mapping to a .isotrieve binary file.
   *
   * The file contains a JSON header with all parameters, validation,
   * metadata, and recalibrator state, followed by the raw Float64
   * matrix payloads.
   *
   * @param path - Filesystem path to write to.
   */
  save(path: string): void {
    this.requireFitted();

    const matrixShape = shape(this._W!);
    const hasInverse = this._WInv !== null;
    const inverseMatrixShape = hasInverse ? shape(this._WInv!) : undefined;

    // Build extra matrices for Procrustes variants
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

    const matrices: Array<{ name: string; mat: Float64Array[] }> = [
      { name: 'forward', mat: this._W! },
    ];
    if (hasInverse) {
      matrices.push({ name: 'inverse', mat: this._WInv! });
    }

    // Any subclass-specific extra matrices (e.g. mean_X, mean_Y)
    const extras = this._extraMatrices();
    for (const { name, mat } of extras) {
      extraMatrices[name] = shape(mat);
      matrices.push({ name, mat });
    }

    // Re-write header with extra matrix shapes now that we know them
    if (Object.keys(extraMatrices).length > 0) {
      header.extraMatrices = extraMatrices;
    }

    writeIsotrieveFile(path, header as unknown as Record<string, unknown>, matrices);
  }

  /**
   * Load a fitted mapping from a .isotrieve binary file.
   *
   * Reads the header, dispatches to the correct concrete Mapping subclass,
   * reconstructs the matrices, and restores meta/validation/recalibrator state.
   *
   * @param path - Filesystem path to read from.
   * @returns A fully reconstructed, fitted Mapping instance.
   */
  static load(path: string): Mapping {
    // Lazy import to avoid circular deps — resolved at runtime via registry
    const { loadMapping } = require('./registry') as typeof import('./registry');
    return loadMapping(path);
  }

  // ── Internal helpers ───────────────────────────────────────────

  /**
   * Throw if the mapping has not been fitted.
   */
  requireFitted(): void {
    if (!this._fitted) {
      throw new Error(
        `Mapping of type "${(this.constructor as typeof Mapping).mappingType}" has not been fitted. ` +
        'Call fit() before transform/inverseTransform.',
      );
    }
  }

  /**
   * Shared implementation for transform and inverseTransform.
   *
   * Handles:
   * 1. Single-vector reshaping to/from batch
   * 2. Dimension validation against the expected source/target dim
   * 3. Finiteness checks
   * 4. Bias augmentation (appending a ones column)
   * 5. Matrix multiply (v @ W)
   * 6. Optional L2 normalization
   *
   * @param V - Input vector(s).
   * @param matrix - The weight matrix to apply.
   * @param expectedDim - Expected input dimension for validation.
   * @param options - Direction, bias, and normalization flags.
   */
  applyMapping(
    V: Float64Array | Float64Array[],
    matrix: Float64Array[],
    expectedDim: number,
    options: ApplyMappingOptions,
  ): Float64Array | Float64Array[] {
    const singleInput = !(Array.isArray(V) && V.length > 0 && V[0] instanceof Float64Array);
    let vecs: Float64Array[];
    if (singleInput) {
      vecs = [V as Float64Array];
    } else {
      vecs = V as Float64Array[];
    }

    // Dimension check
    if (vecs.length > 0 && vecs[0].length !== expectedDim) {
      throw new Error(
        `Expected ${expectedDim}-dimensional vectors for ${options.direction} mapping, ` +
        `got ${vecs[0].length}`,
      );
    }

    // Finiteness check
    for (let i = 0; i < vecs.length; i++) {
      checkFinite(`${options.direction} input[${i}]`, vecs[i]);
    }

    // Bias augmentation
    let augmented: Float64Array[];
    if (options.bias) {
      augmented = augmentBias(vecs);
    } else {
      augmented = vecs;
    }

    // Matrix multiply: each row vector v becomes v @ matrix
    let result: Float64Array[];
    if (augmented.length === 1) {
      result = [vectorMatrixMultiply(augmented[0], matrix)];
    } else {
      result = augmented.map((v) => vectorMatrixMultiply(v, matrix));
    }

    // Optional L2 normalization
    if (options.normalize) {
      result = l2Normalize(result) as Float64Array[];
    }

    if (singleInput) {
      return result[0];
    }
    return result;
  }

  /**
   * Subclasses override this to provide extra matrices for serialization
   * (e.g. Procrustes mean_X, mean_Y). Default: none.
   */
  protected _extraMatrices(): Array<{ name: string; mat: Float64Array[] }> {
    return [];
  }

  /**
   * Subclasses override this to restore extra matrices after loading.
   */
  protected _restoreExtraMatrices(_extras: Map<string, Float64Array[]>): void {
    // no-op by default
  }

  /**
   * Subclasses override this to set _dSrc/_dTarget from extra matrices.
   */
  protected _setDimsFromExtra(_extras: Map<string, Float64Array[]>): void {
    // no-op by default
  }
}
