/**
 * Linear algebra operations for embedding-space mappings.
 * All operations use Float64Array[] (array of row vectors) for consistency.
 */

// ── Vector Operations ────────────────────────────────────────────

/**
 * Cosine similarity between two vectors. Result clamped to [-1, 1].
 */
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

/**
 * Euclidean norm of a vector.
 */
export function vecNorm(v: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  return Math.sqrt(sum);
}

// ── Matrix Utilities ─────────────────────────────────────────────

/**
 * Create an m×n zero matrix.
 */
export function zeros(m: number, n: number): Float64Array[] {
  const mat: Float64Array[] = new Array(m);
  for (let i = 0; i < m; i++) mat[i] = new Float64Array(n);
  return mat;
}

/**
 * Create an n×n identity matrix.
 */
export function eye(n: number): Float64Array[] {
  const mat = zeros(n, n);
  for (let i = 0; i < n; i++) mat[i][i] = 1;
  return mat;
}

/**
 * Matrix dimensions [rows, cols].
 */
export function shape(M: Float64Array[]): [number, number] {
  return [M.length, M.length > 0 ? M[0].length : 0];
}

/**
 * Deep copy a matrix.
 */
export function matClone(M: Float64Array[]): Float64Array[] {
  return M.map((row) => new Float64Array(row));
}

/**
 * Matrix transpose.
 */
export function transpose(M: Float64Array[]): Float64Array[] {
  const rows = M.length;
  const cols = M[0].length;
  const result = zeros(cols, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = M[i][j];
    }
  }
  return result;
}

/**
 * Matrix multiplication: A @ B.
 */
export function matrixMultiply(A: Float64Array[], B: Float64Array[]): Float64Array[] {
  const m = A.length;
  const p = A[0].length;
  const n = B[0].length;
  if (p !== B.length) {
    throw new Error(`Matrix dimensions incompatible: ${m}×${p} @ ${B.length}×${n}`);
  }
  const result = zeros(m, n);
  for (let i = 0; i < m; i++) {
    for (let k = 0; k < p; k++) {
      const aik = A[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < n; j++) {
        result[i][j] += aik * B[k][j];
      }
    }
  }
  return result;
}

/**
 * Vector-matrix multiplication: v @ M (treat v as 1×m row vector).
 */
export function vectorMatrixMultiply(vec: Float64Array, matrix: Float64Array[]): Float64Array {
  const n = matrix[0].length;
  const result = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let sum = 0;
    for (let i = 0; i < vec.length; i++) {
      sum += vec[i] * matrix[i][j];
    }
    result[j] = sum;
  }
  return result;
}

/**
 * Matrix-vector multiplication: M @ v (treat v as column vector).
 */
export function matrixVectorMultiply(M: Float64Array[], v: Float64Array): Float64Array {
  const m = M.length;
  const result = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    let sum = 0;
    for (let j = 0; j < v.length; j++) {
      sum += M[i][j] * v[j];
    }
    result[i] = sum;
  }
  return result;
}

/**
 * Matrix trace.
 */
export function matrixTrace(M: Float64Array[]): number {
  const n = Math.min(M.length, M[0].length);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += M[i][i];
  return sum;
}

/**
 * Frobenius norm of a matrix.
 */
export function frobeniusNorm(M: Float64Array[]): number {
  let sum = 0;
  for (let i = 0; i < M.length; i++) {
    for (let j = 0; j < M[i].length; j++) {
      sum += M[i][j] * M[i][j];
    }
  }
  return Math.sqrt(sum);
}

// ── SVD (One-Sided Jacobi) ──────────────────────────────────────

/**
 * Singular Value Decomposition via one-sided Jacobi rotations.
 * For an m×n matrix A, returns U (m×min(m,n)), S (min(m,n)), Vt (min(m,n)×n).
 * The singular values are sorted in descending order.
 *
 * One-sided Jacobi requires m >= n (columns are orthogonalized in R^m).
 * For wide matrices (m < n), we transpose first, compute SVD of A^T, then swap.
 */
export function svd(
  A: Float64Array[],
  _fullMatrices = false,
): { U: Float64Array[]; S: Float64Array; Vt: Float64Array[] } {
  const m = A.length;
  const n = A[0].length;

  // For wide matrices (m < n), transpose, do Jacobi on A^T (n×m, tall), then swap
  if (m < n) {
    const At = transpose(A);
    const { U: V, S, Vt: Ut } = svd(At);
    // At = V @ diag(S) @ Ut  =>  A = Ut^T @ diag(S) @ V^T
    return { U: transpose(Ut), S, Vt: transpose(V) };
  }

  const minmn = n; // m >= n, so min(m,n) = n

  // Work on a copy
  const B = A.map((r) => new Float64Array(r));

  // V = identity (n×n)
  const V = eye(n);

  // Precompute column dot products for efficient off-diagonal norm tracking
  const colDots: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      let dot = 0;
      for (let k = 0; k < m; k++) dot += B[k][i] * B[k][j];
      row[j] = dot;
    }
    colDots.push(row);
  }

  const maxSweeps = 20 * n;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    // Check convergence: off-diagonal norms
    let offDiagNorm = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        offDiagNorm += colDots[i][j] * colDots[i][j];
      }
    }
    if (offDiagNorm < 1e-24) break;

    // Sweep through all column pairs
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const alpha = colDots[i][i];
        const gamma = colDots[j][j];
        const beta = colDots[i][j];

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

        // Apply rotation to B columns i,j
        for (let k = 0; k < m; k++) {
          const bi = B[k][i], bj = B[k][j];
          B[k][i] = c * bi - s * bj;
          B[k][j] = s * bi + c * bj;
        }

        // Apply rotation to V columns i,j
        for (let k = 0; k < n; k++) {
          const vi = V[k][i], vj = V[k][j];
          V[k][i] = c * vi - s * vj;
          V[k][j] = s * vi + c * vj;
        }

        // Update colDots incrementally
        const newII = c * c * alpha + s * s * gamma - 2 * c * s * beta;
        const newJJ = s * s * alpha + c * c * gamma + 2 * c * s * beta;
        const newIJ = (c * c - s * s) * beta + c * s * (alpha - gamma);
        colDots[i][i] = newII;
        colDots[j][j] = newJJ;
        colDots[i][j] = newIJ;
        colDots[j][i] = newIJ;

        // Update cross-terms with other columns
        for (let k = 0; k < n; k++) {
          if (k === i || k === j) continue;
          const di = colDots[k][i];
          const dj = colDots[k][j];
          colDots[k][i] = c * di - s * dj;
          colDots[i][k] = colDots[k][i];
          colDots[k][j] = s * di + c * dj;
          colDots[j][k] = colDots[k][j];
        }
      }
    }
  }

  // Singular values are column norms of B
  const S = new Float64Array(minmn);
  for (let j = 0; j < minmn; j++) {
    S[j] = Math.sqrt(Math.max(0, colDots[j][j]));
  }

  // U = B * diag(1/S)
  const U = zeros(m, minmn);
  for (let j = 0; j < minmn; j++) {
    if (S[j] > 1e-15) {
      for (let k = 0; k < m; k++) U[k][j] = B[k][j] / S[j];
    }
  }

  // Sort singular values in descending order
  const indices = Array.from({ length: minmn }, (_, i) => i);
  indices.sort((a, b) => S[b] - S[a]);

  const Ssorted = new Float64Array(minmn);
  const Usorted = zeros(m, minmn);
  const VtSorted = zeros(minmn, n);

  for (let i = 0; i < minmn; i++) {
    const idx = indices[i];
    Ssorted[i] = S[idx];
    for (let j = 0; j < m; j++) Usorted[j][i] = U[j][idx];
    for (let j = 0; j < n; j++) VtSorted[i][j] = V[j][idx];
  }

  return { U: Usorted, S: Ssorted, Vt: VtSorted };
}

// ── Linear Solve ─────────────────────────────────────────────────

/**
 * Solve AX = B using Gaussian elimination with partial pivoting.
 */
export function solve(A: Float64Array[], B: Float64Array[]): Float64Array[] {
  const n = A.length;
  const m = B[0].length;

  // Create augmented matrix
  const aug = A.map((row, i) => {
    const r = new Float64Array(n + m);
    r.set(row, 0);
    r.set(B[i], n);
    return r;
  });

  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxRow = col;
    let maxVal = Math.abs(aug[col][col]);
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(aug[row][col]);
      if (v > maxVal) {
        maxVal = v;
        maxRow = row;
      }
    }
    // Swap
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    if (Math.abs(aug[col][col]) < 1e-12) {
      throw new Error('Matrix is singular or nearly singular');
    }

    // Eliminate
    for (let row = col + 1; row < n; row++) {
      const factor = aug[row][col] / aug[col][col];
      for (let j = col; j < n + m; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  // Back substitution
  const X = zeros(n, m);
  for (let col = n - 1; col >= 0; col--) {
    for (let j = 0; j < m; j++) {
      X[col][j] = aug[col][n + j];
      for (let k = col + 1; k < n; k++) {
        X[col][j] -= aug[col][k] * X[k][j];
      }
      X[col][j] /= aug[col][col];
    }
  }
  return X;
}

// ── Ridge Regression ─────────────────────────────────────────────

/**
 * Solve ridge regression: X = (A^T A + λI)^{-1} A^T B.
 * Automatically increases regularization if singular.
 */
export function leastSquares(
  A: Float64Array[],
  B: Float64Array[],
  lambda = 1e-4,
): Float64Array[] {
  const AT = transpose(A);
  const ATA = matrixMultiply(AT, A);
  const ATB = matrixMultiply(AT, B);

  const n = ATA.length;
  const nSamples = A.length;
  const scaledLambda = lambda * (nSamples / 1000);

  // Add ridge regularization
  for (let i = 0; i < n; i++) {
    ATA[i][i] += Math.max(scaledLambda, 1e-10);
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

/**
 * GCV alpha selection for ridge regression.
 * Tries a log-spaced grid and picks the alpha with lowest GCV score.
 */
export function ridgeCv(
  A: Float64Array[],
  B: Float64Array[],
  alphaGrid: number[] = [],
): { W: Float64Array[]; bestAlpha: number } {
  if (alphaGrid.length === 0) {
    // Default: 25 points from 1e-3 to 1e3
    alphaGrid = [];
    for (let i = 0; i < 25; i++) {
      alphaGrid.push(Math.pow(10, -3 + (6 * i) / 24));
    }
  }

  const n = A.length;
  const p = A[0].length;
  const q = B[0].length;

  // Compute SVD of A for efficient GCV
  const { U, S, Vt } = svd(A, false);

  let bestAlpha = alphaGrid[0];
  let bestGcv = Infinity;
  let bestW: Float64Array[] | null = null;

  // Compute A^T B once
  const AT = transpose(A);
  const ATB = matrixMultiply(AT, B);

  for (const alpha of alphaGrid) {
    // Ridge: W = (A^T A + αI)^{-1} A^T B
    const ATA = matrixMultiply(AT, A);
    for (let i = 0; i < ATA.length; i++) {
      ATA[i][i] += alpha;
    }

    try {
      const W = solve(ATA, ATB);

      // Compute residuals
      const fitted = matrixMultiply(A, W);
      let ssRes = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < q; j++) {
          const r = B[i][j] - fitted[i][j];
          ssRes += r * r;
        }
      }

      // Effective degrees of freedom via SVD
      let effectiveDof = 0;
      for (let i = 0; i < Math.min(n, p); i++) {
        effectiveDof += (S[i] * S[i]) / (S[i] * S[i] + alpha);
      }

      // GCV score
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
    // Fallback to simple ridge
    bestW = leastSquares(A, B, bestAlpha);
  }

  return { W: bestW, bestAlpha };
}
