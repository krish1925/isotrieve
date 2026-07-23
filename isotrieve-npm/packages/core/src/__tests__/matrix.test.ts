/**
 * Matrix Operations Unit Tests
 * Critical tests for core mathematical operations
 */

import {
  cosineSimilarity,
  vectorMatrixMultiply,
  matrixMultiply,
  transpose,
  leastSquares,
  computeTransferMatrices,
} from '../matrix';

describe('Matrix Operations', () => {
  describe('cosineSimilarity', () => {
    test('identical vectors have similarity 1.0', () => {
      const vec = [1, 2, 3];
      expect(cosineSimilarity(vec, vec)).toBeCloseTo(1.0, 6);
    });

    test('orthogonal vectors have similarity 0.0', () => {
      const vec1 = [1, 0, 0];
      const vec2 = [0, 1, 0];
      expect(cosineSimilarity(vec1, vec2)).toBeCloseTo(0.0, 6);
    });

    test('opposite vectors have similarity -1.0', () => {
      const vec1 = [1, 2, 3];
      const vec2 = [-1, -2, -3];
      expect(cosineSimilarity(vec1, vec2)).toBeCloseTo(-1.0, 6);
    });

    test('normalized vectors produce correct similarity', () => {
      const vec1 = [0.6, 0.8];
      const vec2 = [0.8, 0.6];
      const expected = 0.6 * 0.8 + 0.8 * 0.6; // 0.96
      expect(cosineSimilarity(vec1, vec2)).toBeCloseTo(expected, 6);
    });

    test('handles zero vectors gracefully', () => {
      const vec1 = [0, 0, 0];
      const vec2 = [1, 2, 3];
      expect(cosineSimilarity(vec1, vec2)).toBe(0);
    });

    test('throws on dimension mismatch', () => {
      expect(() => cosineSimilarity([1, 2], [1, 2, 3])).toThrow();
    });
  });

  describe('vectorMatrixMultiply', () => {
    test('multiplies vector by identity matrix', () => {
      const vec = [1, 2, 3];
      const identity = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ];
      const result = vectorMatrixMultiply(vec, identity);
      expect(result).toEqual([1, 2, 3]);
    });

    test('transforms vector to different space', () => {
      const vec = [1, 2];
      const matrix = [
        [2, 0],
        [0, 3],
      ];
      const result = vectorMatrixMultiply(vec, matrix);
      expect(result).toEqual([2, 6]);
    });

    test('handles dimension change', () => {
      const vec = [1, 2];
      const matrix = [
        [1, 0, 0],
        [0, 1, 0],
      ];
      const result = vectorMatrixMultiply(vec, matrix);
      expect(result).toEqual([1, 2, 0]);
    });
  });

  describe('matrixMultiply', () => {
    test('multiplies by identity matrix', () => {
      const A = [
        [1, 2],
        [3, 4],
      ];
      const I = [
        [1, 0],
        [0, 1],
      ];
      const result = matrixMultiply(A, I);
      expect(result).toEqual(A);
    });

    test('computes correct matrix product', () => {
      const A = [
        [1, 2],
        [3, 4],
      ];
      const B = [
        [5, 6],
        [7, 8],
      ];
      const expected = [
        [19, 22],
        [43, 50],
      ];
      const result = matrixMultiply(A, B);
      expect(result).toEqual(expected);
    });

    test('throws on incompatible dimensions', () => {
      const A = [[1, 2]];
      const B = [[1], [2], [3]];
      expect(() => matrixMultiply(A, B)).toThrow();
    });
  });

  describe('transpose', () => {
    test('transposes square matrix', () => {
      const matrix = [
        [1, 2],
        [3, 4],
      ];
      const expected = [
        [1, 3],
        [2, 4],
      ];
      expect(transpose(matrix)).toEqual(expected);
    });

    test('transposes rectangular matrix', () => {
      const matrix = [
        [1, 2, 3],
        [4, 5, 6],
      ];
      const expected = [
        [1, 4],
        [2, 5],
        [3, 6],
      ];
      expect(transpose(matrix)).toEqual(expected);
    });
  });

  describe('leastSquares', () => {
    test('solves simple linear system', () => {
      // Solve: [1 0] @ X = [2 0]
      //        [0 1]       [0 3]
      // Solution: X = [2 0]
      //                [0 3]
      const A = [
        [1, 0],
        [0, 1],
      ];
      const B = [
        [2, 0],
        [0, 3],
      ];
      const X = leastSquares(A, B);
      
      // Verify solution
      const result = matrixMultiply(A, X);
      for (let i = 0; i < result.length; i++) {
        for (let j = 0; j < result[0].length; j++) {
          expect(result[i][j]).toBeCloseTo(B[i][j], 4);
        }
      }
    });

    test('finds best-fit solution for overdetermined system', () => {
      // More equations than unknowns
      const A = [
        [1],
        [2],
        [3],
      ];
      const B = [
        [2],
        [4],
        [6],
      ];
      const X = leastSquares(A, B);
      
      // Should find X ≈ [2] (exact fit)
      expect(X[0][0]).toBeCloseTo(2, 2);
    });
  });

  describe('computeTransferMatrices', () => {
    test('identity transfer for same embeddings', () => {
      const embeddings = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ];
      
      const { forward, backward } = computeTransferMatrices(
        embeddings,
        embeddings
      );
      
      // Should be approximately identity
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          const expected = i === j ? 1 : 0;
          expect(forward[i][j]).toBeCloseTo(expected, 1);
        }
      }
    });

    test('computes transfer for scaled embeddings', () => {
      const embA = [
        [1, 0],
        [0, 1],
      ];
      const embB = [
        [2, 0],
        [0, 2],
      ];
      
      const { forward } = computeTransferMatrices(embA, embB);
      
      // Should be approximately [[2, 0], [0, 2]]
      expect(forward[0][0]).toBeCloseTo(2, 1);
      expect(forward[1][1]).toBeCloseTo(2, 1);
    });

    test('handles dimension mismatch', () => {
      // Use more samples than dimensions to avoid singular matrix
      const embA = [
        [1, 0],
        [0, 1],
        [0.7, 0.7],
        [0.3, 0.9],
      ];
      const embB = [
        [1, 0, 0],
        [0, 1, 0],
        [0.5, 0.5, 0.7],
        [0.2, 0.8, 0.6],
      ];
      
      const { forward } = computeTransferMatrices(embA, embB);
      
      // Should produce 2x3 matrix
      expect(forward.length).toBe(2);
      expect(forward[0].length).toBe(3);
    });

    test('throws on sample count mismatch', () => {
      const embA = [[1, 0]];
      const embB = [[1, 0], [0, 1]];
      
      expect(() => computeTransferMatrices(embA, embB)).toThrow();
    });
  });

  describe('Performance', () => {
    test('matrix multiplication is efficient', () => {
      const size = 100;
      const A = Array(size)
        .fill(0)
        .map(() => Array(size).fill(1));
      const B = Array(size)
        .fill(0)
        .map(() => Array(size).fill(1));
      
      const start = performance.now();
      matrixMultiply(A, B);
      const time = performance.now() - start;
      
      // Should complete in reasonable time (< 100ms for 100x100)
      expect(time).toBeLessThan(100);
    });

    test('least squares is efficient', () => {
      const rows = 1000;
      const cols = 10;
      const A = Array(rows)
        .fill(0)
        .map(() => Array(cols).fill(0).map(() => Math.random()));
      const B = Array(rows)
        .fill(0)
        .map(() => [Math.random()]);
      
      const start = performance.now();
      leastSquares(A, B);
      const time = performance.now() - start;
      
      // Should complete in reasonable time (< 500ms for 1000x10)
      expect(time).toBeLessThan(500);
    });
  });
});
