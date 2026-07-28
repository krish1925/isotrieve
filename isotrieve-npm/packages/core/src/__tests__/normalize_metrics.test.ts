/**
 * Tests for math/normalize and math/metrics
 */

import { l2Normalize, checkFinite, augmentBias } from '../math/normalize';
import {
  pairwiseCosineStats,
  topkRetention,
  spearmanRho,
  holdoutRankCorrelation,
  mrrDelta,
} from '../math/metrics';

describe('l2Normalize', () => {
  it('normalizes single vector', () => {
    const v = new Float64Array([3, 4]);
    const n = l2Normalize(v) as Float64Array;
    expect(Math.abs(n[0] - 0.6)).toBeLessThan(1e-10);
    expect(Math.abs(n[1] - 0.8)).toBeLessThan(1e-10);
  });

  it('normalizes array of vectors', () => {
    const vs = [new Float64Array([3, 4]), new Float64Array([0, 5])];
    const ns = l2Normalize(vs) as Float64Array[];
    const norm0 = Math.sqrt(ns[0][0] ** 2 + ns[0][1] ** 2);
    const norm1 = Math.sqrt(ns[1][0] ** 2 + ns[1][1] ** 2);
    expect(norm0).toBeCloseTo(1, 10);
    expect(norm1).toBeCloseTo(1, 10);
  });

  it('handles zero vector', () => {
    const v = new Float64Array([0, 0]);
    const n = l2Normalize(v) as Float64Array;
    expect(n[0]).toBe(0);
    expect(n[1]).toBe(0);
  });
});

describe('checkFinite', () => {
  it('passes for finite values', () => {
    expect(() => checkFinite('test', new Float64Array([1, 2, 3]))).not.toThrow();
  });

  it('throws for NaN', () => {
    expect(() => checkFinite('test', new Float64Array([1, NaN]))).toThrow();
  });

  it('throws for Infinity', () => {
    expect(() => checkFinite('test', new Float64Array([1, Infinity]))).toThrow();
  });
});

describe('augmentBias', () => {
  it('adds bias column', () => {
    const vs = [new Float64Array([1, 2]), new Float64Array([3, 4])];
    const aug = augmentBias(vs);
    expect(aug[0].length).toBe(3);
    expect(aug[0][2]).toBe(1);
    expect(aug[1][2]).toBe(1);
  });
});

describe('pairwiseCosineStats', () => {
  it('computes stats for identical vectors', () => {
    const vs = [
      l2Normalize(new Float64Array([1, 0])) as Float64Array,
      l2Normalize(new Float64Array([0, 1])) as Float64Array,
    ];
    const stats = pairwiseCosineStats(vs, vs);
    expect(stats.mean).toBeCloseTo(1, 5);
    expect(stats.median).toBeCloseTo(1, 5);
  });

  it('computes stats for orthogonal vectors', () => {
    const A = [new Float64Array([1, 0])];
    const B = [new Float64Array([0, 1])];
    const stats = pairwiseCosineStats(A, B);
    expect(stats.mean).toBeCloseTo(0, 5);
  });
});

describe('topkRetention', () => {
  it('returns 1.0 when mapping is perfect', () => {
    const vs = [
      l2Normalize(new Float64Array([1, 0])) as Float64Array,
      l2Normalize(new Float64Array([0, 1])) as Float64Array,
      l2Normalize(new Float64Array([1, 1])) as Float64Array,
    ];
    expect(topkRetention(vs, vs, 1)).toBeCloseTo(1.0, 5);
    expect(topkRetention(vs, vs, 3)).toBeCloseTo(1.0, 5);
  });
});

describe('spearmanRho', () => {
  it('returns 1 for identical ranks', () => {
    const a = new Float64Array([1, 2, 3, 4, 5]);
    const b = new Float64Array([10, 20, 30, 40, 50]);
    expect(spearmanRho(a, b)).toBeCloseTo(1.0, 5);
  });

  it('returns -1 for reversed ranks', () => {
    const a = new Float64Array([1, 2, 3, 4, 5]);
    const b = new Float64Array([50, 40, 30, 20, 10]);
    expect(spearmanRho(a, b)).toBeCloseTo(-1.0, 5);
  });
});

describe('holdoutRankCorrelation', () => {
  it('returns high correlation for identical vectors', () => {
    const vs = [
      l2Normalize(new Float64Array([1, 0])) as Float64Array,
      l2Normalize(new Float64Array([0.5, 0.866])) as Float64Array,
      l2Normalize(new Float64Array([0.707, 0.707])) as Float64Array,
    ];
    const rho = holdoutRankCorrelation(vs, vs);
    expect(rho).toBeCloseTo(1.0, 5);
  });
});

describe('mrrDelta', () => {
  it('returns 0 when rankings are identical', () => {
    const qMapped = [new Float64Array([1, 0])];
    const corpus = [new Float64Array([1, 0]), new Float64Array([0, 1])];
    const qTrue = [new Float64Array([1, 0])];
    const result = mrrDelta(qMapped, corpus, qTrue);
    expect(result.mrrDelta).toBeCloseTo(0, 5);
  });
});
