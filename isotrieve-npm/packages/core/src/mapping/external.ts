/**
 * External mapping: wrap any pretrained callable for gating.
 *
 * Allows gating pretrained transforms from other libraries
 * (EmbeddingAdapters, sentence-transformers, etc.) without retraining.
 */

import { Mapping } from './base';
import { l2Normalize, checkFinite } from '../math/normalize';
import type { MappingType } from '../types';

export type ExternalTransformFn = (vecs: Float64Array[]) => Float64Array[];

/**
 * Wrap an external callable as an Isotrieve mapping.
 *
 * The callable must accept Float64Array[] and return Float64Array[].
 * Not serializable — use directly in code, not via .isotrieve files.
 */
export class ExternalMapping extends Mapping {
  static readonly mappingType: MappingType = 'external';

  private _fn: ExternalTransformFn;
  private _inverseFn: ExternalTransformFn | null;

  constructor(
    fn: ExternalTransformFn,
    options: { dSrc?: number; dTarget?: number; inverseFn?: ExternalTransformFn | null } = {},
  ) {
    super({ bias: false });
    this._fn = fn;
    this._inverseFn = options.inverseFn ?? null;
    this._dSrc = options.dSrc ?? null;
    this._dTarget = options.dTarget ?? null;
    this._fitted = true;
    this._meta.source = 'external';
  }

  /**
   * No-op fit: external mappings are pre-trained.
   * Dimensions are inferred from calibration pair.
   */
  fit(X: Float64Array[], Y: Float64Array[]): this {
    if (X.length === 0 || Y.length === 0) {
      throw new Error('X and Y must be non-empty');
    }
    this._dSrc = X[0].length;
    this._dTarget = Y[0].length;
    return this;
  }

  transform(V: Float64Array | Float64Array[]): Float64Array | Float64Array[] {
    const single = !Array.isArray(V) || V.length === 0 || !(V[0] instanceof Float64Array);
    const vecs = single ? [V as Float64Array] : V as Float64Array[];

    if (vecs.length > 0 && this._dSrc !== null && vecs[0].length !== this._dSrc) {
      throw new Error(`Expected ${this._dSrc}-D input, got ${vecs[0].length}-D`);
    }
    for (const v of vecs) checkFinite('external input', v);

    const result = this._fn(vecs);
    const normalized = l2Normalize(result) as Float64Array[];

    return single ? normalized[0] : normalized;
  }

  inverseTransform(V: Float64Array | Float64Array[]): Float64Array | Float64Array[] {
    if (this._inverseFn === null) {
      throw new Error('No inverse function provided; cannot inverse-transform.');
    }
    const single = !Array.isArray(V) || V.length === 0 || !(V[0] instanceof Float64Array);
    const vecs = single ? [V as Float64Array] : V as Float64Array[];
    const result = this._inverseFn(vecs);
    const normalized = l2Normalize(result) as Float64Array[];
    return single ? normalized[0] : normalized;
  }

  /** External callables cannot be serialized. */
  save(_path: string): void {
    throw new Error(
      'ExternalMapping wraps a live callable and cannot be saved to .isotrieve format. ' +
      'Use the callable directly.',
    );
  }
}
