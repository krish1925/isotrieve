/**
 * Tests for mapping modules
 */

import { writeFileSync, readFileSync, unlinkSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';
import { RidgeMapping } from '../mapping/ridge';
import { OrthogonalProcrustesMapping, ProcrustesDiagMapping } from '../mapping/procrustes';
import { LowRankAffineMapping } from '../mapping/lowrank';
import { writeIsotrieveFile, readIsotrieveHeader, readIsotrievePayload } from '../mapping/format';
import { loadMapping } from '../mapping/registry';

const TMP_DIR = join(__dirname, '../../.tmp');

function setupTmpDir() {
  if (!existsSync(TMP_DIR)) {
    mkdirSync(TMP_DIR, { recursive: true });
  }
}

function cleanupTmpDir() {
  if (existsSync(TMP_DIR)) {
    const files = require('fs').readdirSync(TMP_DIR);
    for (const f of files) {
      unlinkSync(join(TMP_DIR, f));
    }
  }
}

// Generate aligned calibration data: Y = X @ W + noise
function generateCalibrationData(
  n: number,
  dSrc: number,
  dTarget: number,
  seed: number = 42,
): { X: Float64Array[]; Y: Float64Array[]; WTrue: number[][] } {
  const rng = (() => {
    let s = seed;
    return () => {
      s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  })();

  const WTrue: number[][] = [];
  for (let i = 0; i < dSrc; i++) {
    const row: number[] = [];
    for (let j = 0; j < dTarget; j++) {
      row.push((rng() - 0.5) * 0.1);
    }
    WTrue.push(row);
  }

  const X: Float64Array[] = [];
  const Y: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    const x = new Float64Array(dSrc);
    const y = new Float64Array(dTarget);
    for (let j = 0; j < dSrc; j++) x[j] = rng();
    for (let j = 0; j < dTarget; j++) {
      let dot = 0;
      for (let k = 0; k < dSrc; k++) dot += x[k] * WTrue[k][j];
      y[j] = dot + rng() * 0.01;
    }
    X.push(x);
    Y.push(y);
  }
  return { X, Y, WTrue };
}

describe('RidgeMapping', () => {
  beforeAll(setupTmpDir);
  afterAll(cleanupTmpDir);

  it('fits and transforms', () => {
    const { X, Y } = generateCalibrationData(50, 4, 4);
    const mapping = new RidgeMapping();
    mapping.fit(X, Y);

    const result = mapping.transform(X[0]) as Float64Array;
    expect(result.length).toBe(4);
    expect(result.every((v) => Number.isFinite(v))).toBe(true);
  });

  it('produces higher cosine after transform', () => {
    const { X, Y } = generateCalibrationData(50, 4, 4);
    const mapping = new RidgeMapping();
    mapping.fit(X, Y);

    const mapped = mapping.transform(X) as Float64Array[];
    let sumCosBefore = 0;
    let sumCosAfter = 0;
    for (let i = 0; i < X.length; i++) {
      let cosBefore = 0;
      let cosAfter = 0;
      for (let j = 0; j < X[i].length; j++) cosBefore += X[i][j] * Y[i][j];
      for (let j = 0; j < mapped[i].length; j++) cosAfter += mapped[i][j] * Y[i][j];
      sumCosBefore += cosBefore;
      sumCosAfter += cosAfter;
    }
    // After should be better
    expect(sumCosAfter).toBeGreaterThan(sumCosBefore);
  });

  it('can inverse transform', () => {
    const { X, Y } = generateCalibrationData(50, 4, 4);
    const mapping = new RidgeMapping();
    mapping.fit(X, Y);

    const mapped = mapping.transform(X[0]) as Float64Array;
    const inv = mapping.inverseTransform(mapped) as Float64Array;
    expect(inv.length).toBe(4);
    expect(inv.every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe('OrthogonalProcrustesMapping', () => {
  beforeAll(setupTmpDir);
  afterAll(cleanupTmpDir);

  it('fits and transforms', () => {
    const { X, Y } = generateCalibrationData(50, 4, 4);
    const mapping = new OrthogonalProcrustesMapping();
    mapping.fit(X, Y);

    const result = mapping.transform(X[0]) as Float64Array;
    expect(result.length).toBe(4);
    expect(result.every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe('ProcrustesDiagMapping', () => {
  beforeAll(setupTmpDir);
  afterAll(cleanupTmpDir);

  it('fits and transforms', () => {
    const { X, Y } = generateCalibrationData(50, 4, 4);
    const mapping = new ProcrustesDiagMapping();
    mapping.fit(X, Y);

    const result = mapping.transform(X[0]) as Float64Array;
    expect(result.length).toBe(4);
    expect(result.every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe('LowRankAffineMapping', () => {
  beforeAll(setupTmpDir);
  afterAll(cleanupTmpDir);

  it('fits and transforms with rank=2', () => {
    const { X, Y } = generateCalibrationData(50, 4, 4);
    const mapping = new LowRankAffineMapping({ rank: 2 });
    mapping.fit(X, Y);

    const result = mapping.transform(X[0]) as Float64Array;
    expect(result.length).toBe(4);
    expect(result.every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe('format (.isotrieve)', () => {
  beforeAll(setupTmpDir);
  afterAll(cleanupTmpDir);

  it('writes and reads a mapping file', () => {
    const { X, Y } = generateCalibrationData(50, 4, 4);
    const mapping = new RidgeMapping();
    mapping.fit(X, Y);

    const path = join(TMP_DIR, 'test_ridge.isotrieve');
    mapping.save(path);

    // Read back
    const header = readIsotrieveHeader(path);
    expect(header.mappingType).toBe('ridge');
    expect(header.dSrc).toBe(4);
    expect(header.dTarget).toBe(4);
    expect(header.formatVersion).toBe(2);

    // Load mapping from file
    const loaded = loadMapping(path);
    expect(loaded.isFitted).toBe(true);

    // Transform should produce similar results
    const original = mapping.transform(X[0]) as Float64Array;
    const restored = loaded.transform(X[0]) as Float64Array;
    for (let i = 0; i < original.length; i++) {
      expect(restored[i]).toBeCloseTo(original[i], 8);
    }
  });
});
