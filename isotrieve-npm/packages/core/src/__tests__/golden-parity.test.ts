/**
 * Golden parity tests (barrage P3): Python-generated fixtures consumed by TS,
 * TS-written binary consumed back by Python (via /tmp artifact), tamper matrix.
 * Pre-registered: transform parity atol 1e-6; tamper matrix all cells correct.
 */
import { readFileSync, writeFileSync, copyFileSync } from 'fs';
import { join } from 'path';
import { createHash } from 'crypto';
import { loadMapping } from '../mapping/registry';
import { readIsotrievePayload, writeIsotrieveFile } from '../mapping/format';

const FIXTURE_DIR = join(__dirname, 'fixtures', 'golden');
const GOLDEN = join(FIXTURE_DIR, 'golden_ridge.isotrieve');
const TS_OUT = '/tmp/iso_ts_roundtrip.isotrieve';

function toF64(rows: number[][]): Float64Array[] {
  return rows.map((r) => {
    const buf = new Float64Array(r.length);
    for (let i = 0; i < r.length; i++) buf[i] = r[i];
    return buf;
  });
}

const meta = JSON.parse(readFileSync(join(FIXTURE_DIR, 'golden_expected.json'), 'utf-8'));

describe('P3-01 python golden consumed by TS', () => {
  it('transforms golden inputs to golden outputs (atol 1e-6)', () => {
    const m = loadMapping(GOLDEN);
    const inputs = toF64(meta.inputs as number[][]);
    const out = m.transform(inputs) as Float64Array[];
    const expected = meta.expected_outputs as number[][];
    expect(out.length).toBe(expected.length);
    let maxErr = 0;
    for (let i = 0; i < out.length; i++) {
      expect(out[i].length).toBe(expected[i].length);
      for (let j = 0; j < out[i].length; j++) {
        maxErr = Math.max(maxErr, Math.abs(out[i][j] - expected[i][j]));
      }
    }
    expect(maxErr).toBeLessThanOrEqual(1e-6);
  });
});

describe('P3-02 TS writes, python verifies', () => {
  it('round-trips golden header+payload through the TS format layer', () => {
    const { header, matrices } = readIsotrievePayload(GOLDEN);
    const mats = Array.from(matrices.entries()).map(([name, mat]) => ({ name, mat }));
    expect(mats.length).toBeGreaterThan(0);
    writeIsotrieveFile(TS_OUT, header, mats);
    const sha = createHash('sha256').update(readFileSync(TS_OUT)).digest('hex');
    writeFileSync('/tmp/iso_ts_roundtrip.sha', sha);
  });
});

describe('P3-03 tamper matrix (TS reader)', () => {
  function loadTampered(pos: number): () => void {
    const p = join('/tmp', `iso_tampered_ts_${pos}.isotrieve`);
    copyFileSync(GOLDEN, p);
    const raw = readFileSync(p);
    raw[pos] ^= 0x01;
    writeFileSync(p, raw);
    return () => loadMapping(p);
  }

  it('accepts the untampered golden (negative control)', () => {
    expect(() => loadMapping(GOLDEN)).not.toThrow();
  });

  it('rejects payload byte flip', () => {
    const raw = readFileSync(GOLDEN);
    expect(loadTampered(raw.length - 5)).toThrow();
  });

  it('rejects header byte flip', () => {
    expect(loadTampered(3)).toThrow();
  });
});

describe('P3-04 shared gate asset identity (TS side)', () => {
  it('sha256 matches the python-recorded hash', () => {
    const gatePath = join(
      __dirname, '..', '..', '..', '..', '..', 'isotrieve-python', 'src', 'isotrieve', 'quality', 'gate_model_v1.json',
    );
    const sha = createHash('sha256').update(readFileSync(gatePath)).digest('hex');
    expect(sha).toBe(meta.gate_model_sha256);
  });
});
