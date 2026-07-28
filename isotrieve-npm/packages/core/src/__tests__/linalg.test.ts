import {
  cosineSimilarity,
  matrixMultiply,
  vectorMatrixMultiply,
  transpose,
  svd,
  solve,
  leastSquares,
  ridgeCv,
  zeros,
  eye,
  shape,
  matClone,
  matrixTrace,
  frobeniusNorm,
  vecNorm,
  TypedMatrix,
} from '../math/linalg';

function toMat(rows: number[][]): TypedMatrix {
  const data = new Float64Array(rows.length * rows[0].length);
  for (let i = 0; i < rows.length; i++) {
    for (let j = 0; j < rows[i].length; j++) {
      data[i * rows[i].length + j] = rows[i][j];
    }
  }
  return new TypedMatrix(data, rows.length, rows[0].length);
}

describe('cosineSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    const a = new Float64Array([1, 0, 0]);
    expect(cosineSimilarity(a, a)).toBeCloseTo(1.0);
  });

  it('returns 0 for orthogonal vectors', () => {
    const a = new Float64Array([1, 0]);
    const b = new Float64Array([0, 1]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.0);
  });

  it('returns -1 for opposite vectors', () => {
    const a = new Float64Array([1, 0]);
    const b = new Float64Array([-1, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1.0);
  });

  it('returns 0 for zero vector', () => {
    const a = new Float64Array([0, 0]);
    const b = new Float64Array([1, 0]);
    expect(cosineSimilarity(a, b)).toBe(0);
  });

  it('computes correct cosine for known vectors', () => {
    const a = new Float64Array([1, 2, 3]);
    const b = new Float64Array([4, 5, 6]);
    const expected = (1 * 4 + 2 * 5 + 3 * 6) / (Math.sqrt(14) * Math.sqrt(77));
    expect(cosineSimilarity(a, b)).toBeCloseTo(expected, 10);
  });
});

describe('zeros', () => {
  it('creates zero matrix', () => {
    const m = zeros(2, 3);
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(3);
    expect(m.get(0, 0)).toBe(0);
    expect(m.get(1, 2)).toBe(0);
  });
});

describe('eye', () => {
  it('creates identity matrix', () => {
    const I = eye(3);
    expect(I.rows).toBe(3);
    expect(I.cols).toBe(3);
    expect(I.get(0, 0)).toBe(1);
    expect(I.get(1, 1)).toBe(1);
    expect(I.get(2, 2)).toBe(1);
    expect(I.get(0, 1)).toBe(0);
  });
});

describe('shape', () => {
  it('returns correct dimensions', () => {
    const m = zeros(2, 3);
    expect(shape(m)).toEqual([2, 3]);
  });
});

describe('matClone', () => {
  it('creates independent copy', () => {
    const m = toMat([[1, 2], [3, 4]]);
    const c = matClone(m);
    c.data[0] = 99;
    expect(m.get(0, 0)).toBe(1);
  });
});

describe('matrixMultiply', () => {
  it('multiplies identity by itself', () => {
    const I = eye(3);
    const result = matrixMultiply(I, I);
    expect(result.get(0, 0)).toBeCloseTo(1.0);
    expect(result.get(0, 1)).toBeCloseTo(0.0);
  });

  it('multiplies known matrices', () => {
    const A = toMat([[1, 2], [3, 4]]);
    const B = toMat([[5, 6], [7, 8]]);
    const C = matrixMultiply(A, B);
    expect(C.get(0, 0)).toBe(19);
    expect(C.get(0, 1)).toBe(22);
    expect(C.get(1, 0)).toBe(43);
    expect(C.get(1, 1)).toBe(50);
  });
});

describe('vectorMatrixMultiply', () => {
  it('multiplies row vector by matrix', () => {
    const v = new Float64Array([1, 2]);
    const M = toMat([[3, 4], [5, 6]]);
    const result = vectorMatrixMultiply(v, M);
    expect(result[0]).toBe(13);
    expect(result[1]).toBe(16);
  });
});

describe('transpose', () => {
  it('transposes 2x3 matrix', () => {
    const M = toMat([[1, 2, 3], [4, 5, 6]]);
    const T = transpose(M);
    expect(T.rows).toBe(3);
    expect(T.cols).toBe(2);
    expect(T.get(0, 0)).toBe(1);
    expect(T.get(0, 1)).toBe(4);
    expect(T.get(2, 0)).toBe(3);
    expect(T.get(2, 1)).toBe(6);
  });
});

describe('frobeniusNorm', () => {
  it('computes norm of identity', () => {
    const I = eye(3);
    expect(frobeniusNorm(I)).toBeCloseTo(Math.sqrt(3));
  });
});

describe('matrixTrace', () => {
  it('computes trace of identity', () => {
    const I = eye(3);
    expect(matrixTrace(I)).toBeCloseTo(3);
  });
});

describe('vecNorm', () => {
  it('computes L2 norm', () => {
    const v = new Float64Array([3, 4]);
    expect(vecNorm(v)).toBeCloseTo(5);
  });
});

describe('svd', () => {
  it('decomposes identity matrix', () => {
    const I = eye(3);
    const { U, S, Vt } = svd(I);
    expect(S.length).toBe(3);
    for (const s of S) {
      expect(Math.abs(s - 1)).toBeLessThan(1e-6);
    }
  });

  it('decomposes known matrix', () => {
    const A = toMat([[1, 0], [0, 2]]);
    const { U, S, Vt } = svd(A);
    expect(S.length).toBe(2);
    expect(S[0]).toBeCloseTo(2, 5);
    expect(S[1]).toBeCloseTo(1, 5);
  });
});

describe('solve', () => {
  it('solves Ax = b for square system', () => {
    const A = toMat([[2, 1], [1, 3]]);
    const b = toMat([[5], [7]]);
    const X = solve(A, b);
    expect(X.rows).toBe(2);
    expect(X.cols).toBe(1);
    expect(X.get(0, 0)).toBeCloseTo(1.6, 5);
    expect(X.get(1, 0)).toBeCloseTo(1.8, 5);
  });
});

describe('leastSquares', () => {
  it('fits perfect linear relationship', () => {
    const X = toMat([[1], [2], [3], [4]]);
    const Y = toMat([[2], [4], [6], [8]]);
    const W = leastSquares(X, Y, 0);
    expect(W.rows).toBe(1);
    expect(W.cols).toBe(1);
    expect(W.get(0, 0)).toBeCloseTo(2, 5);
  });
});

describe('ridgeCv', () => {
  it('picks a reasonable alpha', () => {
    const X = toMat([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
    const Y = toMat([[2.1], [3.9], [6.2], [7.8], [10.1], [12.2], [13.9], [16.1], [18.0], [20.1]]);
    const { W, bestAlpha } = ridgeCv(X, Y);
    expect(bestAlpha).toBeGreaterThanOrEqual(0);
    expect(W.get(0, 0)).toBeCloseTo(2, 0);
  });
});
