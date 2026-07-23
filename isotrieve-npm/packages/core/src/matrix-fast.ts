/**
 * High-Performance Matrix Operations
 * Uses WebAssembly when available, falls back to optimized TypeScript
 */

import * as pureTS from './matrix';

// Try to load WASM module if available
let wasmModule: any = null;
try {
  // This will be bundled separately
  wasmModule = require('@isotrieve/core-wasm');
} catch (e) {
  // WASM not available, will use pure TypeScript
}

/**
 * Check if WASM is available
 */
export function isWasmAvailable(): boolean {
  return wasmModule !== null;
}

/**
 * Cosine similarity (WASM-accelerated when available)
 */
export function cosineSimilarity(vec1: number[], vec2: number[]): number {
  if (wasmModule) {
    return wasmModule.cosine_similarity(vec1, vec2);
  }
  return pureTS.cosineSimilarity(vec1, vec2);
}

/**
 * Vector-matrix multiplication (WASM-accelerated)
 */
export function vectorMatrixMultiply(vec: number[], matrix: number[][]): number[] {
  if (wasmModule) {
    const flat = matrix.flat();
    const rows = matrix.length;
    const cols = matrix[0].length;
    return wasmModule.vector_matrix_multiply(vec, flat, rows, cols);
  }
  return pureTS.vectorMatrixMultiply(vec, matrix);
}

/**
 * Matrix multiplication (WASM-accelerated)
 */
export function matrixMultiply(A: number[][], B: number[][]): number[][] {
  if (wasmModule) {
    const aFlat = A.flat();
    const bFlat = B.flat();
    const aRows = A.length;
    const aCols = A[0].length;
    const bRows = B.length;
    const bCols = B[0].length;
    
    const resultFlat = wasmModule.matrix_multiply(
      aFlat, aRows, aCols,
      bFlat, bRows, bCols
    );
    
    // Reshape to 2D
    const result: number[][] = [];
    for (let i = 0; i < aRows; i++) {
      result.push(resultFlat.slice(i * bCols, (i + 1) * bCols));
    }
    return result;
  }
  return pureTS.matrixMultiply(A, B);
}

/**
 * Least squares solver (WASM-accelerated)
 */
export function leastSquares(A: number[][], B: number[][]): number[][] {
  if (wasmModule) {
    const aFlat = A.flat();
    const bFlat = B.flat();
    const aRows = A.length;
    const aCols = A[0].length;
    const bRows = B.length;
    const bCols = B[0].length;
    
    const resultFlat = wasmModule.least_squares(
      aFlat, aRows, aCols,
      bFlat, bRows, bCols
    );
    
    // Reshape to 2D
    const result: number[][] = [];
    for (let i = 0; i < aCols; i++) {
      result.push(resultFlat.slice(i * bCols, (i + 1) * bCols));
    }
    return result;
  }
  return pureTS.leastSquares(A, B);
}

/**
 * Batch cosine similarity (WASM-accelerated)
 * Computes similarities for multiple vector pairs at once
 */
export function batchCosineSimilarity(vecs1: number[][], vecs2: number[][]): number[] {
  if (wasmModule && vecs1.length > 10) {
    // Only use WASM for larger batches
    const flat1 = vecs1.flat();
    const flat2 = vecs2.flat();
    const n = vecs1.length;
    const dim = vecs1[0].length;
    return wasmModule.batch_cosine_similarity(flat1, flat2, n, dim);
  }
  
  // Fall back to TypeScript
  return vecs1.map((vec1, i) => cosineSimilarity(vec1, vecs2[i]));
}

/**
 * Compute transfer matrices (uses fastest available method)
 */
export function computeTransferMatrices(
  embeddingsSource: number[][],
  embeddingsTarget: number[][]
): { forward: number[][]; backward: number[][] } {
  if (embeddingsSource.length !== embeddingsTarget.length) {
    throw new Error('Source and target must have same number of samples');
  }

  // Use WASM for large matrices
  const forward = leastSquares(embeddingsSource, embeddingsTarget);
  const backward = leastSquares(embeddingsTarget, embeddingsSource);

  return { forward, backward };
}

// Re-export other functions from pure TS
export { transpose } from './matrix';
