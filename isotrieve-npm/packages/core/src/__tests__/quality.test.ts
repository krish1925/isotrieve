/**
 * Tests for recalibration, calibration, reranking, and higher-level modules
 */

import { ScoreRecalibrator } from '../recalibration';
import { planCalibration, recommendK } from '../calibration/plan';
import { ConfidenceScorer, confidenceSummary } from '../reranking';
import { QualityGate } from '../quality/gate';
import { QueryAdapter, cslsScores, mergeResults } from '../serve';
import { RidgeMapping } from '../mapping/ridge';
import { loadMapping } from '../mapping/registry';
import { writeFileSync, mkdirSync, existsSync, unlinkSync } from 'fs';
import { join } from 'path';

const TMP_DIR = join(__dirname, '../../.tmp');
function setupTmpDir() {
  if (!existsSync(TMP_DIR)) {
    mkdirSync(TMP_DIR, { recursive: true });
  }
}

function seededRandom(seed: number = 42): () => number {
  let s = seed;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('ScoreRecalibrator', () => {
  it('fits and transforms scores', () => {
    const rng = seededRandom();
    const n = 100;
    const mapped = new Float64Array(n);
    const ceiling = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      mapped[i] = rng() * 0.5 + 0.3;
      ceiling[i] = mapped[i] + rng() * 0.1;
    }

    const recal = new ScoreRecalibrator();
    recal.fit(mapped, ceiling);
    expect(recal.isFitted).toBe(true);
    expect(recal.report).not.toBeNull();
    expect(recal.report!.nPairs).toBe(n);

    const transformed = recal.transform(mapped);
    expect(transformed.length).toBe(n);
    expect(transformed.every((v) => Number.isFinite(v))).toBe(true);
  });

  it('serializes and deserializes', () => {
    const rng = seededRandom();
    const n = 50;
    const mapped = new Float64Array(n);
    const ceiling = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      mapped[i] = rng();
      ceiling[i] = rng();
    }

    const recal = new ScoreRecalibrator();
    recal.fit(mapped, ceiling);
    const json = recal.toJSON();
    const recal2 = ScoreRecalibrator.fromJSON(json);

    const t1 = recal.transform(mapped);
    const t2 = recal2.transform(mapped);
    for (let i = 0; i < n; i++) {
      expect(t1[i]).toBeCloseTo(t2[i], 10);
    }
  });
});

describe('planCalibration', () => {
  it('returns valid plan', () => {
    const plan = planCalibration({
      corpusSize: 100_000,
      sourceModel: 'text-embedding-ada-002',
      targetModel: 'text-embedding-3-large',
    });
    expect(plan.corpusSize).toBe(100_000);
    expect(plan.recommendedK).toBeGreaterThan(0);
    expect(plan.estCalibrationUsd).toBeGreaterThan(0);
    expect(plan.estReembedUsd).toBeGreaterThan(0);
    expect(plan.notes.length).toBe(3);
  });
});

describe('recommendK', () => {
  it('scales with dimensions', () => {
    const k1 = recommendK(100_000, 384, 384);
    const k2 = recommendK(100_000, 384, 1024);
    // Larger dims -> more recommended K
    expect(k2).toBeGreaterThanOrEqual(k1);
  });

  it('caps at 20000', () => {
    const k = recommendK(1_000_000, 1000, 1000);
    expect(k).toBeLessThanOrEqual(20000);
  });
});

describe('ConfidenceScorer', () => {
  it('scores queries correctly', () => {
    const scorer = new ConfidenceScorer({ adaptive: false });
    const queryIds = ['q1', 'q2', 'q3'];
    const similarities = [
      new Float64Array([0.9, 0.85, 0.8]), // high margin
      new Float64Array([0.5, 0.49, 0.48]), // low margin
      new Float64Array([0.7, 0.65, 0.6]),  // medium margin
    ];

    const reports = scorer.scoreQueries(queryIds, similarities, 3);
    expect(reports.length).toBe(3);
    expect(reports[0].queryId).toBe('q1');
    expect(reports[0].confidence).toBe('high');
    // q2 margin = 0.01 (0.5-0.49), above marginLow=0.005 → medium
    expect(reports[1].confidence).toBe('medium');
  });

  it('computes summary', () => {
    const reports = [
      { queryId: 'q1', top1Margin: 0.05, top1Score: 0.9, confidence: 'high' as const, nCandidates: 10 },
      { queryId: 'q2', top1Margin: 0.001, top1Score: 0.6, confidence: 'low' as const, nCandidates: 10 },
    ];
    const summary = confidenceSummary(reports);
    expect(summary.n).toBe(2);
    expect(summary.nHigh).toBe(1);
    expect(summary.nLow).toBe(1);
    expect(summary.pctHigh).toBe(0.5);
  });
});

describe('QualityGate', () => {
  it('evaluates mapping and returns gate report', () => {
    const rng = seededRandom();
    const d = 4;
    const n = 50;

    // Generate data
    const X: Float64Array[] = [];
    const Y: Float64Array[] = [];
    for (let i = 0; i < n; i++) {
      const x = new Float64Array(d);
      const y = new Float64Array(d);
      for (let j = 0; j < d; j++) {
        x[j] = rng();
        y[j] = x[j] + rng() * 0.05;
      }
      X.push(x);
      Y.push(y);
    }

    const mapping = new RidgeMapping();
    mapping.fit(X, Y);

    const gate = new QualityGate();
    const report = gate.evaluate(mapping, X.slice(0, 25), Y.slice(0, 25));
    expect(['PASS', 'WARN', 'FAIL']).toContain(report.verdict);
    expect(report.nSample).toBe(25);
    expect(report.predictedRetention).toBeGreaterThanOrEqual(0);
    expect(report.predictedRetention).toBeLessThanOrEqual(1);
    expect(report.cosineMean).toBeGreaterThanOrEqual(-1);
    expect(report.cosineMean).toBeLessThanOrEqual(1);
  });
});

describe('mergeResults', () => {
  it('merges and deduplicates', () => {
    const legacy = [{ id: 'a', score: 0.9 }, { id: 'b', score: 0.8 }];
    const native = [{ id: 'b', score: 0.85 }, { id: 'c', score: 0.7 }];
    const merged = mergeResults(legacy, native, new Set(['b']));
    expect(merged.length).toBe(3);
    // a=0.9 from legacy is highest, b=max(0.8, 0.85)=0.85, c=0.7
    expect(merged[0].id).toBe('a');
    expect(merged[0].score).toBeCloseTo(0.9);
    expect(merged[1].id).toBe('b');
    expect(merged[1].score).toBeCloseTo(0.85);
  });
});

describe('cslsScores', () => {
  it('returns correct shape', () => {
    const q = [new Float64Array([1, 0]), new Float64Array([0, 1])];
    const c = [new Float64Array([1, 0]), new Float64Array([0, 1])];
    const scores = cslsScores(q, c, 1);
    expect(scores.length).toBe(2);
    expect(scores[0].length).toBe(2);
  });

  it('self-similarity is highest', () => {
    const q = [new Float64Array([1, 0])];
    const c = [new Float64Array([1, 0]), new Float64Array([0, 1])];
    const scores = cslsScores(q, c, 1);
    expect(scores[0][0]).toBeGreaterThan(scores[0][1]);
  });
});
