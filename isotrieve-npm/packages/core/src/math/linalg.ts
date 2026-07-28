import { TypedMatrix } from './matrix';

export { TypedMatrix } from './matrix';

export type VectorInput = Float64Array | Float32Array;
export type MatrixInput = Float64Array[] | Float32Array[] | TypedMatrix;
export type SingleOrBatch = Float64Array | Float32Array | Float64Array[] | Float32Array[] | TypedMatrix;

export function isSingleVector(v: SingleOrBatch): v is Float64Array | Float32Array {
  return v instanceof Float64Array || v instanceof Float32Array;
}

export function toTypedMatrix(V: SingleOrBatch): TypedMatrix {
  if (V instanceof TypedMatrix) return V;
  if (V instanceof Float64Array) {
    return new TypedMatrix(new Float64Array(V), 1, V.length);
  }
  if (V instanceof Float32Array) {
    const d = new Float64Array(V.length);
    for (let i = 0; i < V.length; i++) d[i] = V[i];
    return new TypedMatrix(d, 1, V.length);
  }
  if (V.length === 0) {
    return new TypedMatrix(new Float64Array(0), 0, 0);
  }
  const first = V[0];
  if (first instanceof Float64Array) {
    return TypedMatrix.fromRows(V as Float64Array[]);
  }
  if (first instanceof Float32Array) {
    return TypedMatrix.fromFloat32Arrays(V as Float32Array[]);
  }
  return TypedMatrix.fromNumberArrays(V as unknown as number[][]);
}

export function toFlatFloat64(V: SingleOrBatch): Float64Array {
  if (V instanceof Float64Array) return V;
  if (V instanceof Float32Array) {
    const d = new Float64Array(V.length);
    for (let i = 0; i < V.length; i++) d[i] = V[i];
    return d;
  }
  if (V instanceof TypedMatrix) return V.data.subarray(0, V.rows * V.cols);
  const m = toTypedMatrix(V);
  return m.data.subarray(0, m.rows * m.cols);
}

export function cosineSimilarity(a: Float64Array, b: Float64Array): number {
  if (a.length !== b.length) {
    throw new Error(`Vectors must have same length: ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  if (normA < 1e-12 || normB < 1e-12) return 0;
  return Math.max(-1, Math.min(1, dot / (normA * normB)));
}

export function vecNorm(v: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  return Math.sqrt(sum);
}

export function zeros(m: number, n: number): TypedMatrix {
  return TypedMatrix.zeros(m, n);
}

export function eye(n: number): TypedMatrix {
  return TypedMatrix.eye(n);
}

export function shape(M: TypedMatrix): [number, number] {
  return [M.rows, M.cols];
}

export function matClone(M: TypedMatrix): TypedMatrix {
  return M.clone();
}

export function transpose(M: TypedMatrix): TypedMatrix {
  const rows = M.rows;
  const cols = M.cols;
  const result = TypedMatrix.zeros(cols, rows);
  const rd = result.data;
  const md = M.data;
  for (let i = 0; i < rows; i++) {
    const mi = i * cols;
    for (let j = 0; j < cols; j++) {
      rd[j * rows + i] = md[mi + j];
    }
  }
  return result;
}

export function matrixMultiply(A: TypedMatrix, B: TypedMatrix): TypedMatrix {
  const m = A.rows;
  const p = A.cols;
  const n = B.cols;
  if (p !== B.rows) {
    throw new Error(`Matrix dimensions incompatible: ${m}×${p} @ ${B.rows}×${n}`);
  }
  const result = TypedMatrix.zeros(m, n);
  const ad = A.data;
  const bd = B.data;
  const rd = result.data;
  for (let i = 0; i < m; i++) {
    const aOff = i * p;
    const rOff = i * n;
    for (let k = 0; k < p; k++) {
      const aik = ad[aOff + k];
      const bOff = k * n;
      for (let j = 0; j < n; j++) {
        rd[rOff + j] += aik * bd[bOff + j];
      }
    }
  }
  return result;
}

export function vectorMatrixMultiply(vec: Float64Array, matrix: TypedMatrix): Float64Array {
  const n = matrix.cols;
  const result = new Float64Array(n);
  const md = matrix.data;
  for (let i = 0; i < vec.length; i++) {
    const vi = vec[i];
    const rowOff = i * n;
    for (let j = 0; j < n; j++) {
      result[j] += vi * md[rowOff + j];
    }
  }
  return result;
}

export function matrixVectorMultiply(M: TypedMatrix, v: Float64Array): Float64Array {
  const m = M.rows;
  const n = M.cols;
  const result = new Float64Array(m);
  const md = M.data;
  for (let i = 0; i < m; i++) {
    let sum = 0;
    const rowOff = i * n;
    for (let j = 0; j < v.length; j++) {
      sum += md[rowOff + j] * v[j];
    }
    result[i] = sum;
  }
  return result;
}

export function matrixTrace(M: TypedMatrix): number {
  const n = Math.min(M.rows, M.cols);
  let sum = 0;
  const md = M.data;
  for (let i = 0; i < n; i++) sum += md[i * M.cols + i];
  return sum;
}

export function frobeniusNorm(M: TypedMatrix): number {
  const len = M.rows * M.cols;
  const md = M.data;
  let sum = 0;
  for (let i = 0; i < len; i++) sum += md[i] * md[i];
  return Math.sqrt(sum);
}

export function svd(
  A: TypedMatrix,
  _fullMatrices = false,
): { U: TypedMatrix; S: Float64Array; Vt: TypedMatrix } {
  const m = A.rows;
  const n = A.cols;

  if (m < n) {
    const At = transpose(A);
    const { U: V, S, Vt: Ut } = svd(At);
    return { U: transpose(Ut), S, Vt: transpose(V) };
  }

  const minmn = n;
  const ad = A.data;
  const aCols = A.cols;

  const B = A.clone();
  const bd = B.data;
  const V = eye(n);
  const vd = V.data;

  const colDots = new TypedMatrix(new Float64Array(n * n), n, n);
  const cd = colDots.data;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let dot = 0;
      for (let k = 0; k < m; k++) dot += bd[k * aCols + i] * bd[k * aCols + j];
      cd[i * n + j] = dot;
    }
  }

  const maxSweeps = 20 * n;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let offDiagNorm = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const v = cd[i * n + j];
        offDiagNorm += v * v;
      }
    }
    if (offDiagNorm < 1e-24) break;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const alpha = cd[i * n + i];
        const gamma = cd[j * n + j];
        const beta = cd[i * n + j];

        if (Math.abs(beta) < 1e-15 * Math.sqrt(Math.abs(alpha * gamma))) continue;

        let c: number, s: number;
        if (Math.abs(alpha - gamma) < 1e-15 * (Math.abs(alpha) + Math.abs(gamma))) {
          c = Math.SQRT1_2;
          s = Math.SQRT1_2;
        } else {
          const tau = (gamma - alpha) / (2 * beta);
          const t = Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
          c = 1 / Math.sqrt(1 + t * t);
          s = t * c;
        }

        for (let k = 0; k < m; k++) {
          const bOff = k * aCols;
          const bi = bd[bOff + i], bj = bd[bOff + j];
          bd[bOff + i] = c * bi - s * bj;
          bd[bOff + j] = s * bi + c * bj;
        }

        for (let k = 0; k < n; k++) {
          const vOff = k * n;
          const vi = vd[vOff + i], vj = vd[vOff + j];
          vd[vOff + i] = c * vi - s * vj;
          vd[vOff + j] = s * vi + c * vj;
        }

        const newII = c * c * alpha + s * s * gamma - 2 * c * s * beta;
        const newJJ = s * s * alpha + c * c * gamma + 2 * c * s * beta;
        const newIJ = (c * c - s * s) * beta + c * s * (alpha - gamma);
        cd[i * n + i] = newII;
        cd[j * n + j] = newJJ;
        cd[i * n + j] = newIJ;
        cd[j * n + i] = newIJ;

        for (let k = 0; k < n; k++) {
          if (k === i || k === j) continue;
          const di = cd[k * n + i];
          const dj = cd[k * n + j];
          cd[k * n + i] = c * di - s * dj;
          cd[i * n + k] = cd[k * n + i];
          cd[k * n + j] = s * di + c * dj;
          cd[j * n + k] = cd[k * n + j];
        }
      }
    }
  }

  const S = new Float64Array(minmn);
  for (let j = 0; j < minmn; j++) {
    S[j] = Math.sqrt(Math.max(0, cd[j * n + j]));
  }

  const U = TypedMatrix.zeros(m, minmn);
  const ud = U.data;
  for (let j = 0; j < minmn; j++) {
    if (S[j] > 1e-15) {
      for (let k = 0; k < m; k++) {
        ud[k * minmn + j] = bd[k * aCols + j] / S[j];
      }
    }
  }

  const indices = Array.from({ length: minmn }, (_, i) => i);
  indices.sort((a, b) => S[b] - S[a]);

  const Ssorted = new Float64Array(minmn);
  const Usorted = TypedMatrix.zeros(m, minmn);
  const VtSorted = TypedMatrix.zeros(minmn, n);

  for (let i = 0; i < minmn; i++) {
    const idx = indices[i];
    Ssorted[i] = S[idx];
    for (let j = 0; j < m; j++) Usorted.data[j * minmn + i] = ud[j * minmn + idx];
    for (let j = 0; j < n; j++) VtSorted.data[i * n + j] = vd[j * n + idx];
  }

  return { U: Usorted, S: Ssorted, Vt: VtSorted };
}

export function solve(A: TypedMatrix, B: TypedMatrix): TypedMatrix {
  const n = A.rows;
  const m = B.cols;
  const ad = A.data;
  const bd = B.data;

  const augData = new Float64Array(n * (n + m));
  for (let i = 0; i < n; i++) {
    const rowOff = i * (n + m);
    for (let j = 0; j < n; j++) augData[rowOff + j] = ad[i * n + j];
    for (let j = 0; j < m; j++) augData[rowOff + n + j] = bd[i * m + j];
  }

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    let maxVal = Math.abs(augData[col * (n + m) + col]);
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(augData[row * (n + m) + col]);
      if (v > maxVal) { maxVal = v; maxRow = row; }
    }
    if (maxRow !== col) {
      const aOff = col * (n + m);
      const bOff = maxRow * (n + m);
      for (let j = 0; j < n + m; j++) {
        const tmp = augData[aOff + j];
        augData[aOff + j] = augData[bOff + j];
        augData[bOff + j] = tmp;
      }
    }

    if (Math.abs(augData[col * (n + m) + col]) < 1e-12) {
      throw new Error('Matrix is singular or nearly singular');
    }

    for (let row = col + 1; row < n; row++) {
      const factor = augData[row * (n + m) + col] / augData[col * (n + m) + col];
      const rOff = row * (n + m);
      const cOff = col * (n + m);
      for (let j = col; j < n + m; j++) {
        augData[rOff + j] -= factor * augData[cOff + j];
      }
    }
  }

  const X = TypedMatrix.zeros(n, m);
  const xd = X.data;
  for (let col = n - 1; col >= 0; col--) {
    const cOff = col * (n + m);
    for (let j = 0; j < m; j++) {
      let val = augData[cOff + n + j];
      for (let k = col + 1; k < n; k++) {
        val -= augData[cOff + k] * xd[k * m + j];
      }
      val /= augData[cOff + col];
      xd[col * m + j] = val;
    }
  }
  return X;
}

export function leastSquares(
  A: TypedMatrix,
  B: TypedMatrix,
  lambda = 1e-4,
): TypedMatrix {
  const AT = transpose(A);
  const ATA = matrixMultiply(AT, A);
  const ATB = matrixMultiply(AT, B);

  const n = ATA.rows;
  const nSamples = A.rows;
  const scaledLambda = lambda * (nSamples / 1000);

  for (let i = 0; i < n; i++) {
    ATA.data[i * n + i] += Math.max(scaledLambda, 1e-10);
  }

  try {
    return solve(ATA, ATB);
  } catch {
    if (lambda < 1.0) {
      return leastSquares(A, B, lambda * 10);
    }
    throw new Error('Ridge regression failed: matrix is singular even with heavy regularization');
  }
}

export function ridgeCv(
  A: TypedMatrix,
  B: TypedMatrix,
  alphaGrid: number[] = [],
): { W: TypedMatrix; bestAlpha: number } {
  if (alphaGrid.length === 0) {
    alphaGrid = [];
    for (let i = 0; i < 25; i++) {
      alphaGrid.push(Math.pow(10, -3 + (6 * i) / 24));
    }
  }

  const n = A.rows;
  const p = A.cols;
  const q = B.cols;

  const { U, S, Vt } = svd(A, false);

  let bestAlpha = alphaGrid[0];
  let bestGcv = Infinity;
  let bestW: TypedMatrix | null = null;

  const AT = transpose(A);
  const ATB = matrixMultiply(AT, B);

  for (const alpha of alphaGrid) {
    const ATA = matrixMultiply(AT, A);
    const ataLen = ATA.rows;
    for (let i = 0; i < ataLen; i++) {
      ATA.data[i * ataLen + i] += alpha;
    }

    try {
      const W = solve(ATA, ATB);
      const fitted = matrixMultiply(A, W);
      let ssRes = 0;
      const fd = fitted.data;
      const bd = B.data;
      for (let i = 0; i < n; i++) {
        const rowOff = i * q;
        for (let j = 0; j < q; j++) {
          const r = bd[rowOff + j] - fd[rowOff + j];
          ssRes += r * r;
        }
      }

      let effectiveDof = 0;
      for (let i = 0; i < Math.min(n, p); i++) {
        effectiveDof += (S[i] * S[i]) / (S[i] * S[i] + alpha);
      }

      const gcv = ssRes / (n - effectiveDof) ** 2;
      if (gcv < bestGcv) {
        bestGcv = gcv;
        bestAlpha = alpha;
        bestW = W;
      }
    } catch {
      // Skip singular alphas
    }
  }

  if (bestW === null) {
    bestW = leastSquares(A, B, bestAlpha);
  }

  return { W: bestW, bestAlpha };
}
