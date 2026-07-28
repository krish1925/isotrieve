import { Mapping } from './base';
import { TypedMatrix, toTypedMatrix, SingleOrBatch } from '../math/linalg';

export type ExternalTransformFn = (vecs: Float64Array[]) => Float64Array[];

export class ExternalMapping extends Mapping {
  static readonly mappingType = 'external' as const;

  private _forwardFn: ExternalTransformFn | null = null;
  private _inverseFn: ExternalTransformFn | null = null;

  constructor(options: {
    forwardFn: ExternalTransformFn;
    inverseFn?: ExternalTransformFn;
    dSrc: number;
    dTarget: number;
    bias?: boolean;
    seed?: number;
  }) {
    super({ bias: options.bias ?? false, seed: options.seed ?? 0 });
    this._forwardFn = options.forwardFn;
    this._inverseFn = options.inverseFn ?? null;
    this._dSrc = options.dSrc;
    this._dTarget = options.dTarget;
    this._fitted = true;
  }

  fit(_X: Float64Array[] | Float32Array[] | TypedMatrix, _Y: Float64Array[] | Float32Array[] | TypedMatrix): this {
    throw new Error('ExternalMapping cannot be fitted; it wraps a pre-existing transform.');
  }

  transform(V: SingleOrBatch): Float64Array | Float64Array[] {
    this.requireFitted();
    return this._applyExternal(V, this._forwardFn!);
  }

  inverseTransform(V: SingleOrBatch): Float64Array | Float64Array[] {
    this.requireFitted();
    if (!this._inverseFn) {
      throw new Error('No inverse transform provided for this ExternalMapping.');
    }
    return this._applyExternal(V, this._inverseFn);
  }

  private _applyExternal(
    V: SingleOrBatch,
    fn: ExternalTransformFn,
  ): Float64Array | Float64Array[] {
    const single = !(Array.isArray(V) && V.length > 0 && (V[0] instanceof Float64Array || V[0] instanceof Float32Array));
    const vecs = toTypedMatrix(V).toFloat64Arrays();
    const result = fn(vecs);
    return single ? result[0] : result;
  }
}
