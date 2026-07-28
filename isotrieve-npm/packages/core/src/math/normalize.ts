/**
 * Normalization and validation utilities.
 */

/**
 * Row-wise L2 normalize. Zero rows stay zero.
 * Accepts a single vector (Float64Array) or batch (Float64Array[]).
 */
export function l2Normalize(
  vectors: Float64Array | Float64Array[],
  eps = 1e-12,
): Float64Array | Float64Array[] {
  if (isSingleVector(vectors)) {
    return l2NormalizeSingle(vectors, eps);
  }
  return vectors.map((v) => l2NormalizeSingle(v, eps));
}

function l2NormalizeSingle(v: Float64Array, eps: number): Float64Array {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  if (norm < eps) return new Float64Array(v);
  const result = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) result[i] = v[i] / norm;
  return result;
}

function isSingleVector(v: Float64Array | Float64Array[]): v is Float64Array {
  return v instanceof Float64Array;
}

/**
 * Check that array contains no NaN or Inf values.
 */
export function checkFinite(name: string, arr: Float64Array | Float64Array[]): void {
  if (isSingleVector(arr)) {
    for (let i = 0; i < arr.length; i++) {
      if (!Number.isFinite(arr[i])) {
        throw new Error(`${name} contains NaN or Inf values`);
      }
    }
    return;
  }
  for (let i = 0; i < arr.length; i++) {
    for (let j = 0; j < arr[i].length; j++) {
      if (!Number.isFinite(arr[i][j])) {
        throw new Error(`${name} contains NaN or Inf values`);
      }
    }
  }
}

/**
 * Append a column of ones for affine (bias) term.
 */
export function augmentBias(X: Float64Array[]): Float64Array[] {
  return X.map((row) => {
    const r = new Float64Array(row.length + 1);
    r.set(row, 0);
    r[row.length] = 1;
    return r;
  });
}
