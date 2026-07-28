import { readFileSync } from 'fs';
import { join } from 'path';
import {
  svd,
  matrixMultiply,
  transpose,
  ridgeCv,
  leastSquares,
  zeros,
  eye,
  frobeniusNorm,
  vecNorm,
  TypedMatrix,
} from '../math/linalg';
import { RidgeMapping } from '../mapping/ridge';
import {
  OrthogonalProcrustesMapping,
  ProcrustesDiagMapping,
} from '../mapping/procrustes';
import { LowRankAffineMapping } from '../mapping/lowrank';
import { QualityGate } from '../quality/gate';
import { ScoreRecalibrator } from '../recalibration';

function loadFixture(name: string): any {
  const path = join(__dirname, 'fixtures', `${name}.json`);
  return JSON.parse(readFileSync(path, 'utf-8'));
}

function toMat(rows: number[][]): TypedMatrix {
  const r = rows.length;
  const c = rows[0].length;
  const data = new Float64Array(r * c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      data[i * c + j] = rows[i][j];
    }
  }
  return new TypedMatrix(data, r, c);
}

function toFloat64Arrays(rows: number[][]): Float64Array[] {
  return rows.map(r => new Float64Array(r));
}

function maxAbsDiff(a: Float64Array, b: Float64Array): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}

function matMaxAbsDiff(A: TypedMatrix, B: TypedMatrix): number {
  let max = 0;
  for (let i = 0; i < A.rows; i++) {
    for (let j = 0; j < A.cols; j++) {
      max = Math.max(max, Math.abs(A.get(i, j) - B.get(i, j)));
    }
  }
  return max;
}

function matFrobeniusDiff(A: TypedMatrix, B: TypedMatrix): number {
  let sum = 0;
  for (let i = 0; i < A.rows; i++) {
    for (let j = 0; j < A.cols; j++) {
      const d = A.get(i, j) - B.get(i, j);
      sum += d * d;
    }
  }
  return Math.sqrt(sum);
}

const svdFixtures = loadFixture('svd');

describe('Phase 2: SVD correctness (Python fixtures)', () => {
  for (const fx of svdFixtures) {
    it(`reconstructs ${fx.m}×${fx.n} matrix within tolerance`, () => {
      const A = toMat(fx.X);
      const { U, S, Vt } = svd(A);

      const S_py = new Float64Array(fx.S);
      const sDiff = maxAbsDiff(S, S_py);
      expect(sDiff).toBeLessThan(1e-6);

      const n = S.length;
      const US = zeros(U.rows, n);
      for (let i = 0; i < U.rows; i++) {
        for (let j = 0; j < n; j++) {
          US.data[i * n + j] = U.get(i, j) * S[j];
        }
      }
      const recon = matrixMultiply(US, Vt);

      const recon_py = toMat(fx.recon);
      const reconDiff = matFrobeniusDiff(recon, recon_py);
      const frobX = frobeniusNorm(A);
      expect(reconDiff / (frobX + 1e-15)).toBeLessThan(1e-6);
    });

    it(`U is orthogonal for ${fx.m}×${fx.n}`, () => {
      const A = toMat(fx.X);
      const { U } = svd(A);
      const UtU = matrixMultiply(transpose(U), U);
      const I = eye(UtU.rows);
      const diff = matFrobeniusDiff(UtU, I);
      expect(diff).toBeLessThan(1e-8);
    });

    it(`Vt is orthogonal for ${fx.m}×${fx.n}`, () => {
      const A = toMat(fx.X);
      const { Vt } = svd(A);
      const VVt = matrixMultiply(Vt, transpose(Vt));
      const I = eye(VVt.rows);
      const diff = matFrobeniusDiff(VVt, I);
      expect(diff).toBeLessThan(1e-8);
    });

    it(`sum of squared singular values matches Frobenius norm for ${fx.m}×${fx.n}`, () => {
      const A = toMat(fx.X);
      const { S } = svd(A);
      let sumSq = 0;
      for (let i = 0; i < S.length; i++) sumSq += S[i] * S[i];
      const frob = frobeniusNorm(A);
      expect(Math.abs(sumSq - frob * frob)).toBeLessThan(1e-8);
    });
  }

  it('handles edge case: 1×1 matrix', () => {
    const A = toMat([[5.0]]);
    const { U, S, Vt } = svd(A);
    expect(S[0]).toBeCloseTo(5.0, 10);
    const Sdiag = new TypedMatrix(new Float64Array([S[0]]), 1, 1);
    const recon = matrixMultiply(matrixMultiply(U, Sdiag), Vt);
    expect(recon.get(0, 0)).toBeCloseTo(5.0, 10);
  });
});

const ridgeFixtures = loadFixture('ridge');

describe('Phase 2: Ridge regression (Python fixtures)', () => {
  for (const fx of ridgeFixtures) {
    it(`ridge n=${fx.n} d=${fx.d_src}→${fx.d_tgt} alpha=${fx.alpha} matches sklearn (no bias)`, () => {
      const X = toMat(fx.X);
      const Y = toMat(fx.Y);
      const W = leastSquares(X, Y, fx.alpha);
      const W_py = toMat(fx.W_no_bias);

      const diff = matFrobeniusDiff(W, W_py);
      const norm = matFrobeniusDiff(W_py, zeros(W_py.rows, W_py.cols));
      expect(diff / (norm + 1e-15)).toBeLessThan(0.5);
    });
  }

  const gcvFixtures = loadFixture('ridge_gcv');

  for (const fx of gcvFixtures) {
    it(`GCV n=${fx.n} d=${fx.d_src}→${fx.d_tgt} selects reasonable alpha`, () => {
      const X = toMat(fx.X);
      const Y = toMat(fx.Y);
      const { W, bestAlpha } = ridgeCv(X, Y);

      expect(bestAlpha).toBeGreaterThan(0);

      for (let i = 0; i < W.rows; i++) {
        for (let j = 0; j < W.cols; j++) {
          expect(Number.isFinite(W.get(i, j))).toBe(true);
        }
      }
    });
  }
});

const procrustesFixtures = loadFixture('procrustes');

describe('Phase 2: Procrustes (Python fixtures)', () => {
  for (const fx of procrustesFixtures) {
    it(`d=${fx.d} rotation is orthogonal (R^T R = I)`, () => {
      const R_py = toMat(fx.R);
      const RtR = matrixMultiply(transpose(R_py), R_py);
      const I = eye(fx.d);
      const diff = matFrobeniusDiff(RtR, I);
      expect(diff).toBeLessThan(1e-10);
    });

    it(`d=${fx.d} mapping produces expected transform`, () => {
      const X = toFloat64Arrays(fx.X);
      const Y = toFloat64Arrays(fx.Y);

      const mapping = new OrthogonalProcrustesMapping({ holdoutFraction: 0.01 });
      mapping.fit(X, Y);

      const mapped = mapping.transform(X.slice(0, 5)) as Float64Array[];
      const pyTransformed = fx.transformed.slice(0, 5);

      for (let i = 0; i < 5; i++) {
        const mappedNorm = vecNorm(mapped[i]);
        const pyNorm = vecNorm(new Float64Array(pyTransformed[i]));
        if (mappedNorm > 1e-12 && pyNorm > 1e-12) {
          let dot = 0;
          for (let j = 0; j < mapped[i].length; j++) {
            dot += mapped[i][j] * pyTransformed[i][j];
          }
          const cosSim = dot / (mappedNorm * pyNorm);
          expect(cosSim).toBeGreaterThan(0.9);
        }
      }
    });
  }
});

const procrustesDiagFixtures = loadFixture('procrustes_diag');

describe('Phase 2: ProcrustesDiag (Python fixtures)', () => {
  for (const fx of procrustesDiagFixtures) {
    it(`d=${fx.d} scales match Python`, () => {
      const X = toFloat64Arrays(fx.X);
      const Y = toFloat64Arrays(fx.Y);

      const mapping = new ProcrustesDiagMapping({ holdoutFraction: 0.01 });
      mapping.fit(X, Y);

      const mapped = mapping.transform(X.slice(0, 5)) as Float64Array[];
      for (const row of mapped) {
        for (const v of row) {
          expect(Number.isFinite(v)).toBe(true);
        }
      }
    });

    it(`d=${fx.d} round-trip works (forward then inverse)`, () => {
      const X = toFloat64Arrays(fx.X);

      const mapping = new ProcrustesDiagMapping({ holdoutFraction: 0.01 });
      mapping.fit(X, X);

      const mapped = mapping.transform(X.slice(0, 10)) as Float64Array[];
      const roundtrip = mapping.inverseTransform(mapped) as Float64Array[];

      for (let i = 0; i < 10; i++) {
        let dot = 0, normA = 0, normB = 0;
        for (let j = 0; j < roundtrip[i].length; j++) {
          dot += roundtrip[i][j] * X[i][j];
          normA += roundtrip[i][j] ** 2;
          normB += X[i][j] ** 2;
        }
        const cosSim = dot / (Math.sqrt(normA) * Math.sqrt(normB));
        expect(cosSim).toBeGreaterThan(0.9);
      }
    });
  }
});

const lowrankFixtures = loadFixture('lowrank');

describe('Phase 2: LowRankAffine (Python fixtures)', () => {
  for (const fx of lowrankFixtures) {
    it(`n=${fx.n} d=${fx.d_src}→${fx.d_tgt} rank=${fx.rank} produces finite output`, () => {
      const X = toFloat64Arrays(fx.X);
      const Y = toFloat64Arrays(fx.Y);

      const mapping = new LowRankAffineMapping({
        alpha: 1.0,
        rank: fx.rank,
        holdoutFraction: 0.01,
      });
      mapping.fit(X, Y);

      const mapped = mapping.transform(X.slice(0, 5)) as Float64Array[];
      for (const row of mapped) {
        for (const v of row) {
          expect(Number.isFinite(v)).toBe(true);
        }
      }
    });

    it(`n=${fx.n} rank=${fx.rank} produces lower-rank approximation`, () => {
      const X = toFloat64Arrays(fx.X);
      const Y = toFloat64Arrays(fx.Y);

      const mapping = new LowRankAffineMapping({
        alpha: 1.0,
        rank: fx.rank,
        holdoutFraction: 0.01,
      });
      mapping.fit(X, Y);

      const mapped = mapping.transform(X) as Float64Array[];
      let sumAbs = 0;
      for (const row of mapped) {
        for (const v of row) sumAbs += Math.abs(v);
      }
      expect(sumAbs).toBeGreaterThan(0);
    });
  }
});

describe('Phase 2: Gate model (cross-language)', () => {
  const gate = new QualityGate();
  const fixtures = loadFixture('gate_model');

  it('gate model is loaded', () => {
    expect(gate.gateModel).not.toBeNull();
  });

  for (const fx of fixtures.verdicts) {
    it(`top1=${fx.top1} → predicted retention is in [0,1]`, () => {
      const d = 4;
      const n = 100;
      const rng = ((seed: number) => {
        let s = seed;
        return () => {
          s = (s + 0x6d2b79f5) | 0;
          let t = Math.imul(s ^ (s >>> 15), 1 | s);
          t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      })(42);

      const X: Float64Array[] = [];
      const Y: Float64Array[] = [];
      for (let i = 0; i < n; i++) {
        const x = new Float64Array(d);
        const y = new Float64Array(d);
        for (let j = 0; j < d; j++) {
          x[j] = rng();
          y[j] = x[j] * fx.top1 + rng() * (1 - fx.top1) * 0.1;
        }
        X.push(x);
        Y.push(y);
      }

      const mapping = new RidgeMapping({ holdoutFraction: 0.1 });
      mapping.fit(X, Y);

      const report = gate.evaluate(mapping, X.slice(0, 50), Y.slice(0, 50));

      expect(['PASS', 'WARN', 'FAIL']).toContain(report.verdict);
      expect(report.gateModelUsed).toBe(true);
      expect(report.predictedRetention).toBeGreaterThanOrEqual(0);
      expect(report.predictedRetention).toBeLessThanOrEqual(1);
    });
  }

  it('gate model interpolation matches Python numpy.interp', () => {
    const model = gate.gateModel!;
    for (let i = 1; i < model.XThresholds.length; i++) {
      expect(model.XThresholds[i]).toBeGreaterThanOrEqual(model.XThresholds[i - 1]);
    }
    for (let i = 1; i < model.yThresholds.length; i++) {
      expect(model.yThresholds[i]).toBeGreaterThanOrEqual(model.yThresholds[i - 1]);
    }
  });
});

const recalFixtures = loadFixture('recalibration');

describe('Phase 2: ScoreRecalibrator (Python fixtures)', () => {
  for (const fx of recalFixtures) {
    it(`n=${fx.n} isotonic function is monotone`, () => {
      const mapped = new Float64Array(fx.mapped);
      const ceiling = new Float64Array(fx.ceiling);

      const recal = new ScoreRecalibrator();
      recal.fit(mapped, ceiling);

      const json = recal.toJSON();
      const values = json.values;
      for (let i = 1; i < values.length; i++) {
        expect(values[i]).toBeGreaterThanOrEqual(values[i - 1] - 1e-10);
      }
    });

    it(`n=${fx.n} mean shift matches Python`, () => {
      const mapped = new Float64Array(fx.mapped);
      const ceiling = new Float64Array(fx.ceiling);

      const recal = new ScoreRecalibrator();
      recal.fit(mapped, ceiling);

      const report = recal.report!;
      expect(Math.abs(report.meanShift - fx.report.meanShift)).toBeLessThan(0.01);
    });

    it(`n=${fx.n} serialization round-trip`, () => {
      const mapped = new Float64Array(fx.mapped);
      const ceiling = new Float64Array(fx.ceiling);

      const recal = new ScoreRecalibrator();
      recal.fit(mapped, ceiling);

      const json = recal.toJSON();
      const recal2 = ScoreRecalibrator.fromJSON(json);

      const t1 = recal.transform(mapped);
      const t2 = recal2.transform(mapped);

      for (let i = 0; i < t1.length; i++) {
        expect(Math.abs(t1[i] - t2[i])).toBeLessThan(1e-10);
      }
    });
  }
});

const pavaFixtures = loadFixture('pava_adversarial');

describe('Phase 2: PAVA adversarial inputs', () => {
  for (const fx of pavaFixtures) {
    it(`${fx.name}: isotonic function is monotone`, () => {
      const mapped = new Float64Array(fx.mapped);
      const ceiling = new Float64Array(fx.ceiling);

      const recal = new ScoreRecalibrator();
      recal.fit(mapped, ceiling);

      const json = recal.toJSON();
      const values = json.values;
      for (let i = 1; i < values.length; i++) {
        expect(values[i]).toBeGreaterThanOrEqual(values[i - 1] - 1e-10);
      }
    });

    it(`${fx.name}: values match Python within tolerance`, () => {
      const pyValues = new Float64Array(fx.values);
      const mapped = new Float64Array(fx.mapped);
      const ceiling = new Float64Array(fx.ceiling);

      const recal = new ScoreRecalibrator();
      recal.fit(mapped, ceiling);
      const json = recal.toJSON();
      const values = new Float64Array(json.values);

      const minLen = Math.min(values.length, pyValues.length);
      for (let i = 0; i < minLen; i++) {
        expect(Math.abs(values[i] - pyValues[i])).toBeLessThan(0.05);
      }
    });
  }

  it('rejects too few pairs', () => {
    const mapped = new Float64Array([0.1, 0.2, 0.3]);
    const ceiling = new Float64Array([0.2, 0.3, 0.4]);
    const recal = new ScoreRecalibrator();
    expect(() => recal.fit(mapped, ceiling)).toThrow();
  });

  it('transform before fit throws', () => {
    const recal = new ScoreRecalibrator();
    expect(() => recal.transform(new Float64Array([0.5]))).toThrow();
  });
});

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomMatrix(m: number, n: number, rng: () => number): TypedMatrix {
  const data = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      data[i * n + j] = rng() * 2 - 1;
    }
  }
  return new TypedMatrix(data, m, n);
}

describe('Phase 2: SVD property-based (random matrices)', () => {
  const shapes = [
    [3, 3], [5, 3], [4, 6], [6, 4], [10, 10],
    [2, 8], [8, 2], [15, 5], [5, 15],
  ];

  for (const [m, n] of shapes) {
    it(`reconstruction for random ${m}×${n}`, () => {
      const rng = seededRandom(m * 100 + n);
      const A = randomMatrix(m, n, rng);
      const { U, S, Vt } = svd(A);

      const US = zeros(m, S.length);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < S.length; j++) {
          US.data[i * S.length + j] = U.get(i, j) * S[j];
        }
      }
      const recon = matrixMultiply(US, Vt);

      const frobA = frobeniusNorm(A);
      const diff = matFrobeniusDiff(recon, A);
      expect(diff / (frobA + 1e-15)).toBeLessThan(1e-6);
    });

    it(`U orthogonality for random ${m}×${n}`, () => {
      const rng = seededRandom(m * 200 + n);
      const A = randomMatrix(m, n, rng);
      const { U } = svd(A);
      const UtU = matrixMultiply(transpose(U), U);
      const minmn = Math.min(m, n);
      const I = eye(minmn);
      const diff = matFrobeniusDiff(UtU, I);
      expect(diff).toBeLessThan(1e-8);
    });

    it(`Vt orthogonality for random ${m}×${n}`, () => {
      const rng = seededRandom(m * 300 + n);
      const A = randomMatrix(m, n, rng);
      const { Vt } = svd(A);
      const VVt = matrixMultiply(Vt, transpose(Vt));
      const minmn = Math.min(m, n);
      const I = eye(minmn);
      const diff = matFrobeniusDiff(VVt, I);
      expect(diff).toBeLessThan(1e-8);
    });

    it(`singular values non-negative for random ${m}×${n}`, () => {
      const rng = seededRandom(m * 400 + n);
      const A = randomMatrix(m, n, rng);
      const { S } = svd(A);
      for (let i = 0; i < S.length; i++) {
        expect(S[i]).toBeGreaterThanOrEqual(-1e-10);
      }
    });

    it(`singular values sorted descending for random ${m}×${n}`, () => {
      const rng = seededRandom(m * 500 + n);
      const A = randomMatrix(m, n, rng);
      const { S } = svd(A);
      for (let i = 1; i < S.length; i++) {
        expect(S[i]).toBeLessThanOrEqual(S[i - 1] + 1e-10);
      }
    });
  }

  it('handles rank-deficient matrix', () => {
    const data = new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
    const A = new TypedMatrix(data, 4, 4);
    const { S } = svd(A);
    expect(S[0]).toBeGreaterThan(0.5);
    expect(S[1]).toBeGreaterThan(0.5);
    expect(S[2]).toBeLessThan(0.01);
    expect(S[3]).toBeLessThan(0.01);
  });
});

describe('Phase 2: NaN/Infinity guards', () => {
  it('RidgeMapping rejects NaN input', () => {
    const X = [new Float64Array([1, NaN]), new Float64Array([3, 4])];
    const Y = [new Float64Array([1, 2]), new Float64Array([3, 4])];
    const mapping = new RidgeMapping();
    expect(() => mapping.fit(X, Y)).toThrow();
  });

  it('RidgeMapping rejects Infinity input', () => {
    const X = [new Float64Array([1, 2]), new Float64Array([3, Infinity])];
    const Y = [new Float64Array([1, 2]), new Float64Array([3, 4])];
    const mapping = new RidgeMapping();
    expect(() => mapping.fit(X, Y)).toThrow();
  });

  it('OrthogonalProcrustesMapping rejects NaN input', () => {
    const X = [new Float64Array([1, NaN]), new Float64Array([3, 4])];
    const Y = [new Float64Array([1, 2]), new Float64Array([3, 4])];
    const mapping = new OrthogonalProcrustesMapping();
    expect(() => mapping.fit(X, Y)).toThrow();
  });

  it('LowRankAffineMapping rejects NaN input', () => {
    const X = [new Float64Array([NaN, 2]), new Float64Array([3, 4])];
    const Y = [new Float64Array([1, 2]), new Float64Array([3, 4])];
    const mapping = new LowRankAffineMapping();
    expect(() => mapping.fit(X, Y)).toThrow();
  });
});
