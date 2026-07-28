import { readFileSync, writeFileSync, unlinkSync } from 'fs';
import { join } from 'path';
import { loadMapping } from '../mapping/registry';
import { readIsotrieveHeader, readIsotrievePayload, writeIsotrieveFile } from '../mapping/format';
import { TypedMatrix } from '../math/matrix';

const FIXTURE_DIR = join(__dirname, 'fixtures', 'xruntime');

function loadFixtureMeta(name: string): any {
  const path = join(FIXTURE_DIR, `${name}.json`);
  return JSON.parse(readFileSync(path, 'utf-8'));
}

function toFloat64Arrays(rows: number[][]): Float64Array[] {
  return rows.map((r) => {
    const buf = new Float64Array(r.length);
    for (let i = 0; i < r.length; i++) buf[i] = r[i];
    return buf;
  });
}

function matMaxAbsDiff(A: Float64Array[], B: Float64Array[]): number {
  if (A.length !== B.length) return Infinity;
  let max = 0;
  for (let i = 0; i < A.length; i++) {
    if (A[i].length !== B[i].length) return Infinity;
    for (let j = 0; j < A[i].length; j++) {
      max = Math.max(max, Math.abs(A[i][j] - B[i][j]));
    }
  }
  return max;
}

describe('Phase 3: Cross-runtime binary format parity', () => {
  const MAPPING_NAMES = [
    'ridge_bias',
    'ridge_nobias',
    'ridge_crossdim',
    'procrustes',
    'procrustes_diag',
    'lowrank',
  ];

  describe.each(MAPPING_NAMES)('%s', (name) => {
    let meta: any;
    let mapping: any;

    beforeAll(() => {
      meta = loadFixtureMeta(name);
      const path = join(FIXTURE_DIR, `${name}.isotrieve`);
      mapping = loadMapping(path);
    });

    test('header keys are normalized (snake_case → camelCase)', () => {
      const path = join(FIXTURE_DIR, `${name}.isotrieve`);
      const header = readIsotrieveHeader(path);
      expect(header.formatVersion).toBeDefined();
      expect(header.mappingType).toBeDefined();
      expect(header.dSrc).toBeDefined();
      expect(header.dTarget).toBeDefined();
      expect(header.matrixShape).toBeDefined();
      expect(header.hasInverse).toBeDefined();
      expect((header as any).format_version).toBeUndefined();
      expect((header as any).mapping_type).toBeUndefined();
      expect((header as any).d_src).toBeUndefined();
      expect((header as any).d_tgt).toBeUndefined();
    });

    test('mapping dimensions match', () => {
      expect(mapping.dSrc).toBe(meta.d_src);
      expect(mapping.dTarget).toBe(meta.d_tgt);
    });

    test('forward transform matches Python output (10 vectors)', () => {
      const X = toFloat64Arrays(meta.X_test);
      const expected = toFloat64Arrays(meta.mapped);
      const got = mapping.transform(X) as Float64Array[];
      const diff = matMaxAbsDiff(got, expected);
      expect(diff).toBeLessThan(1e-7);
    });

    test('inverse transform matches Python output', () => {
      const X = toFloat64Arrays(meta.X_test);
      const mapped = mapping.transform(X) as Float64Array[];
      const expected = meta.inverse;
      if (expected === null) return;
      const expectedArr = toFloat64Arrays(expected);
      const got = mapping.inverseTransform(mapped) as Float64Array[];
      const diff = matMaxAbsDiff(got, expectedArr);
      expect(diff).toBeLessThan(1e-7);
    });

    test('file is loadable via loadMapping', () => {
      const path = join(FIXTURE_DIR, `${name}.isotrieve`);
      const loaded = loadMapping(path);
      expect(loaded.isFitted).toBe(true);
      expect(loaded.dSrc).toBe(meta.d_src);
      expect(loaded.dTarget).toBe(meta.d_tgt);
    });
  });

  describe('round-trip: Python → TS load → TS save → TS load', () => {
    const ROUNDTRIP_MAPS = ['ridge_bias', 'procrustes'];

    test.each(ROUNDTRIP_MAPS)('%s round-trip preserves matrices', (name) => {
      const meta = loadFixtureMeta(name);
      const origPath = join(FIXTURE_DIR, `${name}.isotrieve`);

      const m1 = loadMapping(origPath);

      const tsPath = join(FIXTURE_DIR, `${name}_ts_roundtrip.isotrieve`);
      m1.save(tsPath);

      const m2 = loadMapping(tsPath);

      const X = toFloat64Arrays(meta.X_test);
      const out1 = m1.transform(X) as Float64Array[];
      const out2 = m2.transform(X) as Float64Array[];
      const diff = matMaxAbsDiff(out1, out2);
      expect(diff).toBeLessThan(1e-12);

      const h1 = readIsotrieveHeader(origPath);
      const h2 = readIsotrieveHeader(tsPath);
      expect(h2.mappingType).toBe(h1.mappingType);
      expect(h2.dSrc).toBe(h1.dSrc);
      expect(h2.dTarget).toBe(h1.dTarget);
      expect(h2.matrixShape).toEqual(h1.matrixShape);
      expect(h2.hasInverse).toBe(h1.hasInverse);

      try { unlinkSync(tsPath); } catch {}
    });
  });

  describe('edge cases', () => {
    test('loadMapping throws on non-existent file', () => {
      expect(() => loadMapping('/tmp/nonexistent.isotrieve')).toThrow();
    });

    test('readIsotrieveHeader throws on corrupt file', () => {
      const badPath = join(FIXTURE_DIR, '_bad_header.isotrieve');
      writeFileSync(badPath, Buffer.from('XXXX0000{}'));
      expect(() => readIsotrieveHeader(badPath)).toThrow();
      try { unlinkSync(badPath); } catch {}
    });

    test('readIsotrievePayload throws on truncated file', () => {
      const badPath = join(FIXTURE_DIR, '_truncated.isotrieve');
      const headerJson = JSON.stringify({ matrixShape: [10, 5] });
      const buf = Buffer.alloc(8 + headerJson.length);
      buf.write('ISTR', 0, 'ascii');
      buf.writeUInt32LE(headerJson.length, 4);
      buf.write(headerJson, 8, 'utf-8');
      writeFileSync(badPath, buf);
      expect(() => readIsotrievePayload(badPath)).toThrow();
      try { unlinkSync(badPath); } catch {}
    });
  });

  describe('matrix byte-exact roundtrip', () => {
    test('write → read preserves Float64 values exactly', () => {
      const data = new Float64Array([1.5, -2.718281828, 3.14159265358979, 0, 1e-300, 1e300]);
      const mat = new TypedMatrix(data, 2, 3);
      const path = join(FIXTURE_DIR, '_roundtrip_exact.isotrieve');
      writeIsotrieveFile(path, {
        mappingType: 'ridge',
        dSrc: 3,
        dTarget: 3,
        matrixShape: [2, 3],
        hasInverse: false,
      }, [{ name: 'forward', mat }]);
      const { matrices } = readIsotrievePayload(path);
      const got = matrices.get('forward')!;
      const diff = matMaxAbsDiff([got.data as any as Float64Array], [mat.data as any as Float64Array]);
      // Compare flat data
      let maxD = 0;
      for (let i = 0; i < got.data.length; i++) {
        maxD = Math.max(maxD, Math.abs(got.data[i] - mat.data[i]));
      }
      expect(maxD).toBe(0);
      try { unlinkSync(path); } catch {}
    });
  });
});
