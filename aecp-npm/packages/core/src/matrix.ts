/**
 * Matrix operations for embedding transfer
 */

/**
 * Compute cosine similarity between two vectors
 */
export function cosineSimilarity(vec1: number[], vec2: number[]): number {
  if (vec1.length !== vec2.length) {
    throw new Error('Vectors must have same length');
  }

  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;

  for (let i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }

  norm1 = Math.sqrt(norm1);
  norm2 = Math.sqrt(norm2);

  if (norm1 === 0 || norm2 === 0) {
    return 0;
  }

  return dotProduct / (norm1 * norm2);
}

/**
 * Matrix multiplication: A @ B
 */
export function matrixMultiply(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const n = B[0].length;
  const p = B.length;

  if (A[0].length !== p) {
    throw new Error('Matrix dimensions incompatible for multiplication');
  }

  const result: number[][] = Array(m)
    .fill(0)
    .map(() => Array(n).fill(0));

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < p; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return result;
}

/**
 * Vector-matrix multiplication: v @ M
 */
export function vectorMatrixMultiply(vec: number[], matrix: number[][]): number[] {
  const n = matrix[0].length;
  const result = new Array(n).fill(0);

  for (let j = 0; j < n; j++) {
    for (let i = 0; i < vec.length; i++) {
      result[j] += vec[i] * matrix[i][j];
    }
  }

  return result;
}

/**
 * Matrix transpose
 */
export function transpose(matrix: number[][]): number[][] {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const result: number[][] = Array(cols)
    .fill(0)
    .map(() => Array(rows).fill(0));

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = matrix[i][j];
    }
  }

  return result;
}

/**
 * Solve linear system AX = B using Gaussian elimination with partial pivoting
 */
function solveLinearSystem(A: number[][], B: number[][]): number[][] {
  const n = A.length;
  const m = B[0].length;

  // Create augmented matrix [A | B]
  const augmented = A.map((row, i) => [...row, ...B[i]]);

  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(augmented[row][col]) > Math.abs(augmented[maxRow][col])) {
        maxRow = row;
      }
    }

    // Swap rows
    [augmented[col], augmented[maxRow]] = [augmented[maxRow], augmented[col]];

    // Check for singular matrix
    if (Math.abs(augmented[col][col]) < 1e-10) {
      throw new Error('Matrix is singular or nearly singular');
    }

    // Eliminate column
    for (let row = col + 1; row < n; row++) {
      const factor = augmented[row][col] / augmented[col][col];
      for (let j = col; j < n + m; j++) {
        augmented[row][j] -= factor * augmented[col][j];
      }
    }
  }

  // Back substitution
  const X: number[][] = Array(n)
    .fill(0)
    .map(() => Array(m).fill(0));

  for (let col = n - 1; col >= 0; col--) {
    for (let j = 0; j < m; j++) {
      X[col][j] = augmented[col][n + j];
      for (let k = col + 1; k < n; k++) {
        X[col][j] -= augmented[col][k] * X[k][j];
      }
      X[col][j] /= augmented[col][col];
    }
  }

  return X;
}

/**
 * Solve least squares: find X that minimizes ||AX - B||^2
 * Uses normal equations with ridge regularization: X = (A^T A + λI)^-1 A^T B
 * Ridge regularization (λ > 0) ensures numerical stability and prevents overfitting
 */
export function leastSquares(A: number[][], B: number[][], lambda: number = 1e-4): number[][] {
  const AT = transpose(A);
  const ATA = matrixMultiply(AT, A);
  const ATB = matrixMultiply(AT, B);

  // Add ridge regularization: ATA + λI
  const n = ATA.length;
  const n_samples = A.length;
  const scaledLambda = lambda * (n_samples / 1000);
  
  for (let i = 0; i < n; i++) {
    ATA[i][i] += Math.max(scaledLambda, 1e-10);
  }

  // Solve (ATA + λI) * X = ATB using Gaussian elimination
  try {
    return solveLinearSystem(ATA, ATB);
  } catch (e) {
    // If singular, try with more regularization
    if (lambda < 1.0) {
      return leastSquares(A, B, lambda * 10);
    }
    throw e;
  }
}

/**
 * Compute transfer matrices between two embedding spaces
 */
export function computeTransferMatrices(
  embeddingsSource: number[][],
  embeddingsTarget: number[][]
): { forward: number[][]; backward: number[][] } {
  if (embeddingsSource.length !== embeddingsTarget.length) {
    throw new Error('Source and target must have same number of samples');
  }

  // Forward: source -> target
  const forward = leastSquares(embeddingsSource, embeddingsTarget);

  // Backward: target -> source
  const backward = leastSquares(embeddingsTarget, embeddingsSource);

  return { forward, backward };
}
