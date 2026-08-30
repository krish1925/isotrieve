import { TypedMatrix } from './matrix';
import { toTypedMatrix, isSingleVector, SingleOrBatch } from './linalg';

function l2NormalizeSingle(v: Float64Array, eps: number): Float64Array {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  if (norm < eps) return new Float64Array(v);
  const result = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) result[i] = v[i] / norm;
  return result;
}

function l2NormalizeBatch(vectors: Float64Array[] | TypedMatrix, eps: number): Float64Array[] {
  if (vectors instanceof TypedMatrix) {
    const out: Float64Array[] = new Array(vectors.rows);
    for (let i = 0; i < vectors.rows; i++) {
      out[i] = l2NormalizeSingle(vectors.row(i), eps);
    }
    return out;
  }
  return vectors.map((v) => l2NormalizeSingle(v, eps));
}

export function l2Normalize(
  vectors: Float64Array | Float32Array | Float64Array[] | Float32Array[] | TypedMatrix,
  eps = 1e-12,
): Float64Array | Float64Array[] {
  if (isSingleVector(vectors)) {
    if (vectors instanceof Float32Array) {
      const d = new Float64Array(vectors.length);
      for (let i = 0; i < vectors.length; i++) d[i] = vectors[i];
      return l2NormalizeSingle(d, eps);
    }
    return l2NormalizeSingle(vectors as Float64Array, eps);
  }
  const tm = toTypedMatrix(vectors);
  return l2NormalizeBatch(tm, eps);
}

function checkFiniteSingle(name: string, arr: Float64Array): void {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) {
      throw new Error(`${name} contains NaN or Inf values`);
    }
  }
}

export function checkFinite(name: string, arr: Float64Array | Float32Array | Float64Array[] | Float32Array[] | TypedMatrix): void {
  if (arr instanceof Float64Array || arr instanceof Float32Array) {
    const d = arr instanceof Float64Array ? arr : new Float64Array(arr);
    checkFiniteSingle(name, d);
    return;
  }
  if (arr instanceof TypedMatrix) {
    for (let i = 0; i < arr.rows; i++) {
      for (let j = 0; j < arr.cols; j++) {
        if (!Number.isFinite(arr.data[i * arr.cols + j])) {
          throw new Error(`${name} contains NaN or Inf values`);
        }
      }
    }
    return;
  }
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    checkFiniteSingle(name, v instanceof Float32Array ? new Float64Array(v) : v);
  }
}

export function augmentBias(X: Float64Array[] | Float32Array[] | TypedMatrix): Float64Array[] {
  let rows: Float64Array[];
  if (X instanceof TypedMatrix) {
    rows = [];
    for (let i = 0; i < X.rows; i++) {
      rows.push(X.row(i));
    }
  } else {
    rows = X instanceof Float64Array ? X as Float64Array[] : (X as Float32Array[]).map(v => new Float64Array(v));
  }
  return rows.map((row) => {
    const r = new Float64Array(row.length + 1);
    r.set(row, 0);
    r[row.length] = 1;
    return r;
  });
}
