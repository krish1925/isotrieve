/**
 * Stress tests matching Python PyPI test suite behavior.
 * These mirror the exact assertions from test_mapping.py, test_gate.py,
 * test_recalibration.py, test_reranking.py, test_serve.py, etc.
 */

import { RidgeMapping } from '../mapping/ridge';
import {
  OrthogonalProcrustesMapping,
  ProcrustesDiagMapping,
} from '../mapping/procrustes';
import { LowRankAffineMapping } from '../mapping/lowrank';
import { QualityGate } from '../quality/gate';
import { ScoreRecalibrator } from '../recalibration';
import { ConfidenceScorer, confidenceSummary } from '../reranking';
import { QueryAdapter, cslsScores, mergeResults } from '../serve';
import { loadMapping } from '../mapping/registry';
import { writeIsotrieveFile, readIsotrieveHeader } from '../mapping/format';
import { l2Normalize } from '../math/normalize';
import {
  pairwiseCosineStats,
  topkRetention,
  spearmanRho,
  holdoutRankCorrelation,
} from '../math/metrics';
import { cosineSimilarity } from '../math/linalg';
import { mkdirSync, existsSync, unlinkSync, readdirSync } from 'fs';
import { join } from 'path';

// ── Helpers ──────────────────────────────────────────────────────

function seededRandom(seed: number = 42): () => number {
  let s = seed;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Normal distribution via Box-Muller */
function normalRng(seed: number): () => number {
  const base = seededRandom(seed);
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const val = spare;
      spare = null;
      return val;
    }
    let u: number, v: number;
    do { u = base(); } while (u === 0);
    v = base();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.cos(2 * Math.PI * v);
    return mag * Math.sin(2 * Math.PI * v);
  };
}

function matMul(A: Float64Array[], B: Float64Array[]): Float64Array[] {
  const m = A.length, n = B[0].length, k = A[0].length;
  const C: Float64Array[] = [];
  for (let i = 0; i < m; i++) {
    const row = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let l = 0; l < k; l++) sum += A[i][l] * B[l][j];
      row[j] = sum;
    }
    C.push(row);
  }
  return C;
}

function _pairedGaussian(k: number, dSrc: number, dTgt: number, seed: number): {
  X: Float64Array[];
  Y: Float64Array[];
  WTrue: Float64Array[];
} {
  const rng = normalRng(seed);

  // Generate X
  const X: Float64Array[] = [];
  for (let i = 0; i < k; i++) {
    const row = new Float64Array(dSrc);
    for (let j = 0; j < dSrc; j++) row[j] = rng();
    X.push(row);
  }

  // Generate W_true
  const WTrue: Float64Array[] = [];
  for (let i = 0; i < dSrc; i++) {
    const row = new Float64Array(dTgt);
    for (let j = 0; j < dTgt; j++) row[j] = rng();
    WTrue.push(row);
  }

  // Y = X @ W_true + 0.01 * noise
  const raw = matMul(X, WTrue);
  const Y: Float64Array[] = [];
  for (let i = 0; i < k; i++) {
    const row = new Float64Array(dTgt);
    for (let j = 0; j < dTgt; j++) {
      row[j] = raw[i][j] + 0.01 * rng();
    }
    Y.push(row);
  }

  return { X, Y, WTrue };
}

const TMP_DIR = join(__dirname, '../../.tmp_stress');
function setupTmpDir() {
  if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });
}
function cleanupTmpDir() {
  if (existsSync(TMP_DIR)) {
    for (const f of readdirSync(TMP_DIR)) {
      unlinkSync(join(TMP_DIR, f));
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
// MAPPING TESTS (mirror test_mapping.py)
// ═══════════════════════════════════════════════════════════════════

describe('Stress: RidgeMapping', () => {
  beforeAll(setupTmpDir);
  afterAll(cleanupTmpDir);

  it('fit_transform_recovers_targets (cosine > 0.95)', () => {
    const dSrc = 16, dTgt = 24;
    const k = 10 * Math.min(dSrc, dTgt);
    const { X, Y } = _pairedGaussian(k, dSrc, dTgt, 1);
    const m = new RidgeMapping({ alpha: 1.0, seed: 0 });
    m.fit(X, Y);
    const pred = m.transform(X) as Float64Array[];
    const yN = l2Normalize(Y) as Float64Array[];
    let sumSim = 0;
    for (let i = 0; i < k; i++) {
      let dot = 0;
      for (let j = 0; j < dTgt; j++) dot += pred[i][j] * yN[i][j];
      sumSim += dot;
    }
    expect(sumSim / k).toBeGreaterThan(0.95);
  });

  it('rectangular_dims', () => {
    const { X, Y } = _pairedGaussian(200, 16, 32, 2);
    const m = new RidgeMapping({ alpha: 'auto', seed: 0 });
    m.fit(X, Y);
    expect(m.dSrc).toBe(16);
    expect(m.dTarget).toBe(32);
    const Z = m.transform(X.slice(0, 10)) as Float64Array[];
    expect(Z.length).toBe(10);
    expect(Z[0].length).toBe(32);
    for (const row of Z) {
      const norm = Math.sqrt(row.reduce((s, v) => s + v * v, 0));
      expect(norm).toBeCloseTo(1.0, 4);
    }
  });

  it('rejects NaN', () => {
    const { X, Y } = _pairedGaussian(200, 8, 8, 3);
    X[0][0] = NaN;
    expect(() => new RidgeMapping().fit(X, Y)).toThrow(/NaN|finite/i);
  });

  it('seeded_determinism', () => {
    const { X, Y } = _pairedGaussian(200, 12, 12, 4);
    const m1 = new RidgeMapping({ alpha: 1.0, seed: 42 });
    m1.fit(X, Y);
    const m2 = new RidgeMapping({ alpha: 1.0, seed: 42 });
    m2.fit(X, Y);
    const r1 = m1.transform(X) as Float64Array[];
    const r2 = m2.transform(X) as Float64Array[];
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < X[0].length; j++) {
        expect(r1[i][j]).toBeCloseTo(r2[i][j], 10);
      }
    }
  });

  it('save_load_roundtrip', () => {
    const { X, Y } = _pairedGaussian(200, 10, 14, 5);
    const m = new RidgeMapping({ alpha: 0.5, seed: 7 });
    m.fit(X, Y);
    m.setMeta('source_model_id', 'a');
    m.setMeta('target_model_id', 'b');
    const path = join(TMP_DIR, 'roundtrip.isotrieve');
    m.save(path);
    const header = readIsotrieveHeader(path);
    expect(header.mappingType).toBe('ridge');
    expect((header.meta as Record<string, unknown>).source_model_id).toBe('a');
    const loaded = loadMapping(path);
    const orig = m.transform(X) as Float64Array[];
    const rest = loaded.transform(X) as Float64Array[];
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < orig[0].length; j++) {
        expect(rest[i][j]).toBeCloseTo(orig[i][j], 6);
      }
    }
  });

  it('validation_report_present', () => {
    const { X, Y } = _pairedGaussian(200, 8, 8, 9);
    const m = new RidgeMapping({ seed: 1 });
    m.fit(X, Y);
    const r = m.validationReport();
    expect(r).not.toBeNull();
    expect(r!.nHoldout).toBeGreaterThan(0);
    expect(r!.top1Retention).toBeGreaterThanOrEqual(0);
    expect(r!.top1Retention).toBeLessThanOrEqual(1);
  });
});

// ═══════════════════════════════════════════════════════════════════
// PROCRUSTES TESTS (mirror test_mapping.py)
// ═══════════════════════════════════════════════════════════════════

describe('Stress: Procrustes mappings', () => {
  beforeAll(setupTmpDir);
  afterAll(cleanupTmpDir);

  it('OrthogonalProcrustes: square_only', () => {
    const { X, Y } = _pairedGaussian(200, 10, 12, 7);
    expect(() => new OrthogonalProcrustesMapping().fit(X, Y)).toThrow(/d_src.*d_target|d_src == d/i);
  });

  it('OrthogonalProcrustes: fit and transform', () => {
    const { X, Y } = _pairedGaussian(200, 16, 16, 8);
    const m = new OrthogonalProcrustesMapping({ seed: 0 });
    m.fit(X, Y);
    const Z = m.transform(X) as Float64Array[];
    expect(Z.length).toBe(200);
    expect(Z[0].length).toBe(16);
    const t1 = topkRetention(Z, Y, 1);
    expect(t1).toBeGreaterThan(0.5);
  });

  it('OrthogonalProcrustes: inverse', () => {
    const { X, Y } = _pairedGaussian(200, 16, 16, 10);
    const m = new OrthogonalProcrustesMapping({ seed: 0 });
    m.fit(X, Y);
    const Z = m.transform(X) as Float64Array[];
    const XBack = m.inverseTransform(Z) as Float64Array[];
    expect(XBack.length).toBe(200);
    expect(XBack[0].length).toBe(16);
    const xN = l2Normalize(X) as Float64Array[];
    let sumSim = 0;
    for (let i = 0; i < X.length; i++) {
      let dot = 0;
      for (let j = 0; j < X[0].length; j++) dot += XBack[i][j] * xN[i][j];
      sumSim += dot;
    }
    expect(sumSim / X.length).toBeGreaterThan(0.5);
  });

  it('ProcrustesDiag: fit and transform', () => {
    const { X, Y } = _pairedGaussian(200, 16, 16, 12);
    const m = new ProcrustesDiagMapping({ seed: 0 });
    m.fit(X, Y);
    const Z = m.transform(X) as Float64Array[];
    expect(Z.length).toBe(200);
    expect(Z[0].length).toBe(16);
  });

  it('ProcrustesDiag: square_only', () => {
    const { X, Y } = _pairedGaussian(200, 10, 12, 14);
    expect(() => new ProcrustesDiagMapping().fit(X, Y)).toThrow(/d_src.*d_target|d_src == d/i);
  });

  it('ProcrustesDiag: inverse', () => {
    const { X, Y } = _pairedGaussian(200, 16, 16, 13);
    const m = new ProcrustesDiagMapping({ seed: 0 });
    m.fit(X, Y);
    const Z = m.transform(X) as Float64Array[];
    const XBack = m.inverseTransform(Z) as Float64Array[];
    expect(XBack.length).toBe(200);
    expect(XBack[0].length).toBe(16);
  });
});

// ═══════════════════════════════════════════════════════════════════
// LOWRANK TESTS (mirror test_mapping.py)
// ═══════════════════════════════════════════════════════════════════

describe('Stress: LowRankAffineMapping', () => {
  beforeAll(setupTmpDir);
  afterAll(cleanupTmpDir);

  it('fit (cosine > 0.8)', () => {
    const { X, Y } = _pairedGaussian(200, 16, 24, 16);
    const m = new LowRankAffineMapping({ alpha: 1.0, rank: 8, seed: 0 });
    m.fit(X, Y);
    const Z = m.transform(X) as Float64Array[];
    expect(Z.length).toBe(200);
    expect(Z[0].length).toBe(24);
    const yN = l2Normalize(Y) as Float64Array[];
    let sumSim = 0;
    for (let i = 0; i < 200; i++) {
      let dot = 0;
      for (let j = 0; j < 24; j++) dot += Z[i][j] * yN[i][j];
      sumSim += dot;
    }
    expect(sumSim / 200).toBeGreaterThan(0.8);
  });

  it('full_rank_equals_ridge', () => {
    const { X, Y } = _pairedGaussian(200, 16, 16, 17);
    const mLR = new LowRankAffineMapping({ alpha: 1.0, rank: 16, seed: 0 });
    mLR.fit(X, Y);
    const mRidge = new RidgeMapping({ alpha: 1.0, seed: 0 });
    mRidge.fit(X, Y);
    const zLR = mLR.transform(X) as Float64Array[];
    const zRidge = mRidge.transform(X) as Float64Array[];
    for (let i = 0; i < 200; i++) {
      for (let j = 0; j < 16; j++) {
        expect(zLR[i][j]).toBeCloseTo(zRidge[i][j], 3);
      }
    }
  });

  it('rectangular', () => {
    const { X, Y } = _pairedGaussian(400, 32, 64, 18);
    const m = new LowRankAffineMapping({ alpha: 'auto', rank: 16, seed: 0 });
    m.fit(X, Y);
    const Z = m.transform(X.slice(0, 10)) as Float64Array[];
    expect(Z.length).toBe(10);
    expect(Z[0].length).toBe(64);
  });
});

// ═══════════════════════════════════════════════════════════════════
// CROSS-ADAPTER COMPARISON (mirror test_mapping.py)
// ═══════════════════════════════════════════════════════════════════

describe('Stress: All adapters beat random', () => {
  it('all adapters beat random on synthetic', () => {
    const { X, Y } = _pairedGaussian(200, 16, 16, 20);
    const adapters = [
      new RidgeMapping({ alpha: 1.0, seed: 0 }),
      new OrthogonalProcrustesMapping({ seed: 0 }),
      new ProcrustesDiagMapping({ seed: 0 }),
      new LowRankAffineMapping({ alpha: 1.0, rank: 8, seed: 0 }),
    ];
    for (const adapter of adapters) {
      adapter.fit(X, Y);
      const Z = adapter.transform(X) as Float64Array[];
      const t1 = topkRetention(Z, Y, 1);
      expect(t1).toBeGreaterThan(0.3);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// QUALITY GATE TESTS (mirror test_gate.py)
// ═══════════════════════════════════════════════════════════════════

describe('Stress: QualityGate', () => {
  it('pass on good mapping', () => {
    const d = 16, k = 200;
    const rng = normalRng(42);
    const X: Float64Array[] = [];
    for (let i = 0; i < k; i++) {
      const row = new Float64Array(d);
      for (let j = 0; j < d; j++) row[j] = rng();
      X.push(row);
    }
    const W: Float64Array[] = [];
    for (let i = 0; i < d; i++) {
      const row = new Float64Array(d);
      for (let j = 0; j < d; j++) row[j] = rng();
      W.push(row);
    }
    const raw = matMul(X, W);
    const Y: Float64Array[] = raw.map((row, i) => {
      const r = new Float64Array(d);
      for (let j = 0; j < d; j++) r[j] = row[j] + 0.01 * rng();
      return r;
    });

    const m = new RidgeMapping({ alpha: 1.0, seed: 0 });
    m.fit(X, Y);

    const gate = new QualityGate();
    const report = gate.evaluate(m, X.slice(0, 50), Y.slice(0, 50), { holdoutTop1: 0.95 });
    expect(['PASS', 'WARN']).toContain(report.verdict);
    expect(report.top1Retention).toBeGreaterThanOrEqual(0.8);
    expect(report.gateModelUsed).toBe(true);
    expect(report.predictedRetention).toBeGreaterThan(0.6);
  });

  it('gate report exposes margin_compression', () => {
    const d = 16, k = 200;
    const rng = normalRng(42);
    const X: Float64Array[] = [];
    for (let i = 0; i < k; i++) {
      const row = new Float64Array(d);
      for (let j = 0; j < d; j++) row[j] = rng();
      X.push(row);
    }
    const W: Float64Array[] = [];
    for (let i = 0; i < d; i++) {
      const row = new Float64Array(d);
      for (let j = 0; j < d; j++) row[j] = rng();
      W.push(row);
    }
    const raw = matMul(X, W);
    const Y: Float64Array[] = raw.map((row) => new Float64Array(row));

    const m = new RidgeMapping({ alpha: 1.0, seed: 0 });
    m.fit(X, Y);
    const gate = new QualityGate();
    const report = gate.evaluate(m, X.slice(0, 50), Y.slice(0, 50));
    expect(report.marginCompression).not.toBeNull();
    expect(typeof report.marginCompression).toBe('number');
  });

  it('margin_compression returns float not null', () => {
    const rng = normalRng(42);
    const d = 16, k = 100;
    const mapped: Float64Array[] = [];
    const target: Float64Array[] = [];
    for (let i = 0; i < k; i++) {
      const m = new Float64Array(d);
      const t = new Float64Array(d);
      for (let j = 0; j < d; j++) {
        m[j] = rng();
        t[j] = rng();
      }
      mapped.push(m);
      target.push(t);
    }
    const result = QualityGate.computeMarginCompression(mapped, target);
    expect(result).not.toBeNull();
    expect(typeof result).toBe('number');
  });

  it('margin_compression_compressed_case < 1', () => {
    const rng = normalRng(42);
    const d = 16, k = 200;
    const target: Float64Array[] = [];
    for (let i = 0; i < k; i++) {
      const row = new Float64Array(d);
      for (let j = 0; j < d; j++) row[j] = rng();
      target.push(row);
    }
    // Compressed: all mapped vectors point in similar direction
    const base = new Float64Array(d);
    for (let j = 0; j < d; j++) base[j] = rng();
    const mapped: Float64Array[] = [];
    for (let i = 0; i < k; i++) {
      const row = new Float64Array(d);
      for (let j = 0; j < d; j++) row[j] = base[j] + 0.01 * rng();
      mapped.push(row);
    }
    const mc = QualityGate.computeMarginCompression(mapped, target);
    expect(mc).not.toBeNull();
    expect(mc!).toBeLessThan(1.0);
  });
});

// ═══════════════════════════════════════════════════════════════════
// RECALIBRATION TESTS (mirror test_recalibration.py)
// ═══════════════════════════════════════════════════════════════════

describe('Stress: ScoreRecalibrator', () => {
  it('fit and transform reduces error', () => {
    const rng = normalRng(42);
    const n = 1000;
    const ceiling = new Float64Array(n);
    const mapped = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      ceiling[i] = 0.3 + rng() * 0.7;
      mapped[i] = 0.8 * ceiling[i] + rng() * 0.02;
      mapped[i] = Math.max(0, Math.min(1, mapped[i]));
    }

    const recal = new ScoreRecalibrator();
    recal.fit(mapped, ceiling);
    expect(recal.isFitted).toBe(true);

    const recalibrated = recal.transform(mapped);
    let errBefore = 0, errAfter = 0;
    for (let i = 0; i < n; i++) {
      errBefore += Math.abs(mapped[i] - ceiling[i]);
      errAfter += Math.abs(recalibrated[i] - ceiling[i]);
    }
    errBefore /= n;
    errAfter /= n;
    expect(errAfter).toBeLessThan(errBefore);
  });

  it('roundtrip via dict', () => {
    const rng = normalRng(42);
    const n = 500;
    const ceiling = new Float64Array(n);
    const mapped = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      ceiling[i] = 0.3 + rng() * 0.7;
      mapped[i] = 0.85 * ceiling[i] + rng() * 0.01;
    }

    const recal = new ScoreRecalibrator();
    recal.fit(mapped, ceiling);
    const json = recal.toJSON();
    const recal2 = ScoreRecalibrator.fromJSON(json);

    const scores = new Float64Array([0.5, 0.7, 0.9]);
    const t1 = recal.transform(scores);
    const t2 = recal2.transform(scores);
    for (let i = 0; i < 3; i++) {
      expect(t1[i]).toBeCloseTo(t2[i], 10);
    }
  });

  it('report_fields', () => {
    const rng = normalRng(0);
    const n = 2000;
    const ceiling = new Float64Array(n);
    const mapped = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      ceiling[i] = 0.2 + rng() * 0.8;
      mapped[i] = 0.8 * ceiling[i] + rng() * 0.03;
    }

    const recal = new ScoreRecalibrator();
    recal.fit(mapped, ceiling);
    const report = recal.report;
    expect(report).not.toBeNull();
    expect(report!.nPairs).toBe(2000);
    expect(report!.meanShift).toBeGreaterThan(0);
    expect(report!.marginRatio).toBeGreaterThan(0);
    expect(report!.marginRatio).toBeLessThan(2.0);
    expect(Object.keys(report!.thresholdAgreement).length).toBe(5);
    for (const tau of [0.5, 0.6, 0.7, 0.8, 0.9]) {
      expect(report!.thresholdAgreement[tau]).toBeGreaterThanOrEqual(0);
      expect(report!.thresholdAgreement[tau]).toBeLessThanOrEqual(1);
    }
  });

  it('monotonicity', () => {
    const rng = normalRng(1);
    const n = 5000;
    const ceiling = new Float64Array(n);
    const mapped = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      ceiling[i] = rng();
      mapped[i] = 0.7 * ceiling[i] + 0.1 + rng() * 0.05;
      mapped[i] = Math.max(-0.1, Math.min(1.1, mapped[i]));
    }

    const recal = new ScoreRecalibrator();
    recal.fit(mapped, ceiling);

    const testScores = new Float64Array(100);
    for (let i = 0; i < 100; i++) testScores[i] = i / 99;
    const recalibrated = recal.transform(testScores);

    for (let i = 1; i < 100; i++) {
      expect(recalibrated[i]).toBeGreaterThanOrEqual(recalibrated[i - 1] - 1e-10);
    }
  });

  it('clip_out_of_bounds', () => {
    const rng = normalRng(0);
    const n = 500;
    const ceiling = new Float64Array(n);
    const mapped = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      ceiling[i] = 0.3 + rng() * 0.5;
      mapped[i] = 0.3 + rng() * 0.5;
    }

    const recal = new ScoreRecalibrator();
    recal.fit(mapped, ceiling);

    const high = recal.transform(new Float64Array([2.0]));
    expect(high[0]).toBeLessThanOrEqual(ceiling.reduce((a, b) => Math.max(a, b), 0) + 0.01);

    const low = recal.transform(new Float64Array([-1.0]));
    expect(low[0]).toBeGreaterThanOrEqual(ceiling.reduce((a, b) => Math.min(a, b), Infinity) - 0.01);
  });

  it('insufficient_pairs raises', () => {
    const recal = new ScoreRecalibrator();
    expect(() => recal.fit(
      new Float64Array([0.5, 0.5, 0.5, 0.5, 0.5]),
      new Float64Array([0.6, 0.6, 0.6, 0.6, 0.6]),
    )).toThrow(/>=10/);
  });

  it('unfitted raises', () => {
    const recal = new ScoreRecalibrator();
    expect(() => recal.transform(new Float64Array([0.5]))).toThrow(/not fitted|Cannot/);
    expect(() => recal.toJSON()).toThrow(/not fitted|Cannot/);
  });
});

// ═══════════════════════════════════════════════════════════════════
// RERANKING TESTS (mirror test_reranking.py)
// ═══════════════════════════════════════════════════════════════════

describe('Stress: ConfidenceScorer', () => {
  it('basic_scoring (high margin → high)', () => {
    const rng = normalRng(42);
    const nDocs = 100;
    const sims: Float64Array[] = [];
    for (let i = 0; i < 5; i++) {
      const row = new Float64Array(nDocs);
      for (let j = 0; j < nDocs; j++) row[j] = 0.5 + rng() * 0.4;
      sims.push(row);
    }
    // Make query 0 have a clear winner
    sims[0][nDocs - 1] = 0.95;
    sims[0][nDocs - 2] = 0.70;

    const scorer = new ConfidenceScorer();
    const reports = scorer.scoreQueries(
      Array.from({ length: 5 }, (_, i) => `q${i}`),
      sims,
    );
    expect(reports.length).toBe(5);
    expect(reports[0].confidence).toBe('high');
  });

  it('low_margin → low', () => {
    const sims = [new Float64Array(100).fill(0.7)];
    sims[0][99] = 0.701;
    const scorer = new ConfidenceScorer({ marginLow: 0.01 });
    const reports = scorer.scoreQueries(['q0'], sims);
    expect(reports[0].confidence).toBe('low');
  });

  it('empty', () => {
    const scorer = new ConfidenceScorer();
    const reports = scorer.scoreQueries([], []);
    expect(reports.length).toBe(0);
  });

  it('summary', () => {
    const reports = [
      { queryId: 'q1', top1Margin: 0.05, top1Score: 0.9, confidence: 'high' as const, nCandidates: 10 },
      { queryId: 'q2', top1Margin: 0.001, top1Score: 0.4, confidence: 'low' as const, nCandidates: 10 },
      { queryId: 'q3', top1Margin: 0.01, top1Score: 0.7, confidence: 'medium' as const, nCandidates: 10 },
    ];
    const s = confidenceSummary(reports);
    expect(s.n).toBe(3);
    expect(s.nHigh).toBe(1);
    expect(s.nLow).toBe(1);
    expect(s.nMedium).toBe(1);
  });

  it('summary_empty', () => {
    const s = confidenceSummary([]);
    expect(s.n).toBe(0);
  });
});

// ═══════════════════════════════════════════════════════════════════
// SERVE TESTS (mirror test_serve.py)
// ═══════════════════════════════════════════════════════════════════

describe('Stress: Serve', () => {
  function makeBidirectionalMapping(d: number = 16, k: number = 200) {
    const rng = normalRng(42);
    const X: Float64Array[] = [];
    for (let i = 0; i < k; i++) {
      const row = new Float64Array(d);
      for (let j = 0; j < d; j++) row[j] = rng();
      X.push(row);
    }
    const W: Float64Array[] = [];
    for (let i = 0; i < d; i++) {
      const row = new Float64Array(d);
      for (let j = 0; j < d; j++) row[j] = rng();
      W.push(row);
    }
    const raw = matMul(X, W);
    const Y: Float64Array[] = raw.map((row, i) => {
      const r = new Float64Array(d);
      for (let j = 0; j < d; j++) r[j] = row[j] + 0.01 * rng();
      return r;
    });
    const m = new RidgeMapping({ alpha: 1.0, seed: 0 });
    m.fit(X, Y);
    return { m, X, Y };
  }

  it('query_adapter_map_single', () => {
    const { m, X } = makeBidirectionalMapping();
    const qa = new QueryAdapter(m);
    const mapped = qa.mapQuery(X[0]);
    expect(mapped.length).toBe(16);
    const norm = Math.sqrt(mapped.reduce((s, v) => s + v * v, 0));
    expect(norm).toBeCloseTo(1.0, 5);
  });

  it('query_adapter_map_batch', () => {
    const { m, X } = makeBidirectionalMapping();
    const qa = new QueryAdapter(m);
    const mapped = qa.mapQueries(X.slice(0, 5));
    expect(mapped.length).toBe(5);
    for (const row of mapped) {
      const norm = Math.sqrt(row.reduce((s, v) => s + v * v, 0));
      expect(norm).toBeCloseTo(1.0, 5);
    }
  });

  it('query_adapter_properties', () => {
    const { m } = makeBidirectionalMapping();
    const qa = new QueryAdapter(m);
    expect(qa.dNew).toBe(16);
    expect(qa.dLegacy).toBe(16);
  });

  it('query_adapter_dimension_mismatch', () => {
    const { m } = makeBidirectionalMapping();
    const qa = new QueryAdapter(m);
    expect(() => qa.mapQuery(new Float64Array(32))).toThrow(/Dimension mismatch/);
  });

  it('csls_basic', () => {
    const rng = normalRng(42);
    const q: Float64Array[] = [];
    const c: Float64Array[] = [];
    for (let i = 0; i < 10; i++) {
      const row = new Float64Array(16);
      for (let j = 0; j < 16; j++) row[j] = rng();
      q.push(row);
    }
    for (let i = 0; i < 100; i++) {
      const row = new Float64Array(16);
      for (let j = 0; j < 16; j++) row[j] = rng();
      c.push(row);
    }
    const scores = cslsScores(q, c, 5);
    expect(scores.length).toBe(10);
    expect(scores[0].length).toBe(100);
    for (const row of scores) {
      for (const v of row) {
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });

  it('merge_results_dedup', () => {
    const legacy = [
      { id: 'a', score: 0.8 },
      { id: 'b', score: 0.6 },
    ];
    const native = [
      { id: 'a', score: 0.9 },
      { id: 'c', score: 0.7 },
    ];
    const merged = mergeResults(legacy, native, new Set());
    const ids = merged.map((r) => r.id);
    expect(ids).toContain('a');
    expect(ids).toContain('b');
    expect(ids).toContain('c');
    expect(ids.length).toBe(3);
  });

  it('merge_results_weights', () => {
    const legacy = [{ id: 'a', score: 0.5 }];
    const native = [{ id: 'a', score: 0.3 }];
    const merged = mergeResults(legacy, native, new Set(), 2.0);
    const aScore = merged.find((r) => r.id === 'a')!.score;
    expect(aScore).toBe(1.0);
  });
});
