/**
 * Binary .isotrieve file format read/write.
 *
 * Layout (v2):
 *   [4 bytes]   Magic "ISTR"
 *   [4 bytes]   Header length (LE uint32) — unpadded JSON byte length
 *   [4 bytes]   CRC32 over (magic + headerLen + padded JSON + all payload)
 *   [N bytes]   JSON header (UTF-8), padded with trailing spaces to 8-byte boundary
 *   [...]       Float64 matrix payloads (row-major C-order)
 *
 * Matrices are stored sequentially: forward, inverse, then any extras.
 * Each matrix is rows × cols × 8 bytes of Float64 data.
 *
 * Endianness: all multi-byte integers are little-endian (LE).
 * All Float64 values are IEEE 754 double-precision, LE.
 *
 * Header padding ensures payload starts at an 8-byte aligned offset,
 * enabling zero-copy Float64Array views on the read path.
 *
 * v1→v2 changes: CRC32 checksum, header padding, strict key validation.
 * v1 files (no CRC, no padding) are read for backward compatibility.
 */

import { readFileSync, writeFileSync } from 'fs';

// ── Constants ─────────────────────────────────────────────────────

export const ISOTRIEVE_MAGIC = Buffer.from('ISTR');
export const FORMAT_VERSION = 2;
export const HEADER_LEN_STRUCT = '<I';
export const MAX_HEADER_LEN = 1 << 20; // 1 MB ceiling
export const CRC_SIZE = 4;
export const FIXED_HEADER_SIZE = 4 + 4 + CRC_SIZE; // magic + headerLen + crc = 12

// ── Endianness assertion (module init) ────────────────────────────

const _endianTest = new Uint32Array([0x01020304]);
const _isLE = new Uint8Array(_endianTest.buffer)[0] === 0x04;
if (!_isLE) {
  throw new Error(
    '@isotrieve/core requires a little-endian platform. ' +
    'The .isotrieve binary format is defined as LE uint32 / LE Float64.',
  );
}

// ── CRC32 (IEEE 802.3 polynomial) ────────────────────────────────

const CRC_TABLE = new Uint32Array(256);
for (let i = 0; i < 256; i++) {
  let c = i;
  for (let j = 0; j < 8; j++) {
    c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  }
  CRC_TABLE[i] = c;
}

export function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    crc = CRC_TABLE[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

// ── Uint32 helpers ────────────────────────────────────────────────

function encodeUint32(value: number): Buffer {
  const buf = Buffer.alloc(4);
  buf.writeUInt32LE(value, 0);
  return buf;
}

function decodeUint32(buf: Buffer, offset = 0): number {
  return buf.readUInt32LE(offset);
}

// ── Header key normalization (snake_case → camelCase) ─────────────

const SNAKE_TO_CAMEL: Record<string, string> = {
  format_version: 'formatVersion',
  isotrieve_version: 'isotrieveVersion',
  mapping_type: 'mappingType',
  d_src: 'dSrc',
  d_tgt: 'dTarget',
  matrix_shape: 'matrixShape',
  has_inverse: 'hasInverse',
  inverse_matrix_shape: 'inverseMatrixShape',
  extra_matrices: 'extraMatrices',
  score_recal_v1: 'scoreRecalV1',
  fit_date: 'fitDate',
  expires_hint: 'expiresHint',
};

/**
 * Known header keys after normalization (camelCase).
 * Used by the strict reader to reject unknown keys.
 */
const KNOWN_KEYS = new Set([
  'formatVersion', 'isotrieveVersion', 'mappingType',
  'dSrc', 'dTarget', 'bias', 'seed',
  'fitDate', 'expiresHint', 'validation', 'meta',
  'matrixShape', 'hasInverse', 'inverseMatrixShape',
  'extraMatrices', 'scoreRecalV1',
  'srcModel', 'tgtModel', 'crc32',
]);

/**
 * Required header keys that must be present in every .isotrieve file.
 */
const REQUIRED_KEYS = new Set([
  'formatVersion', 'mappingType', 'dSrc', 'dTarget',
  'matrixShape', 'hasInverse',
]);

/**
 * Normalize a .isotrieve JSON header from Python-style snake_case
 * to TypeScript-style camelCase. Keys already in camelCase pass through.
 *
 * Uses Object.create(null) to prevent prototype pollution from
 * adversarial JSON input (__proto__ keys).
 */
function normalizeHeaderKeys(header: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = Object.create(null);
  for (const [key, value] of Object.entries(header)) {
    const mapped = SNAKE_TO_CAMEL[key] ?? key;
    out[mapped] = value;
  }
  return out;
}

/**
 * Validate that all required keys are present and no unknown keys exist.
 * Throws on missing required keys or unexpected keys.
 */
function validateHeaderKeys(header: Record<string, unknown>): void {
  const normalized = normalizeHeaderKeys(header);

  // Check required keys
  for (const key of REQUIRED_KEYS) {
    if (!(key in normalized)) {
      throw new Error(
        `Invalid .isotrieve header: missing required key "${key}"`,
      );
    }
  }

  // Check for unknown keys
  for (const key of Object.keys(normalized)) {
    if (!KNOWN_KEYS.has(key)) {
      throw new Error(
        `Invalid .isotrieve header: unknown key "${key}". ` +
        `If Python added a new field, update KNOWN_KEYS in format.ts.`,
      );
    }
  }

  return;
}

// ── Header padding ────────────────────────────────────────────────

/**
 * Pad a JSON header buffer to the next 8-byte boundary with trailing spaces.
 * This ensures the payload starts at an 8-byte aligned offset.
 */
function padTo8Bytes(buf: Buffer): Buffer {
  const remainder = buf.length % 8;
  if (remainder === 0) return buf;
  const pad = 8 - remainder;
  const padded = Buffer.alloc(buf.length + pad, 0x20); // 0x20 = space
  buf.copy(padded);
  return padded;
}

// ── Matrix serialization ──────────────────────────────────────────

function matrixToBuffer(mat: Float64Array[]): Buffer {
  if (mat.length === 0) return Buffer.alloc(0);
  const rows = mat.length;
  const cols = mat[0].length;
  const buf = Buffer.alloc(rows * cols * 8);
  for (let i = 0; i < rows; i++) {
    const row = mat[i];
    for (let j = 0; j < cols; j++) {
      buf.writeDoubleLE(row[j], (i * cols + j) * 8);
    }
  }
  return buf;
}

function bufferToMatrix(buf: Buffer, rows: number, cols: number): Float64Array[] {
  const mat: Float64Array[] = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Float64Array(cols);
    for (let j = 0; j < cols; j++) {
      row[j] = buf.readDoubleLE((i * cols + j) * 8);
    }
    mat[i] = row;
  }
  return mat;
}

// ── Write ─────────────────────────────────────────────────────────

/**
 * Write a .isotrieve binary file (v2 format).
 *
 * @param path - Filesystem path to write to.
 * @param header - Header object (serialized as JSON).
 * @param matrices - Named matrices to embed after the header.
 *                   The first is forward, second (if present) is inverse,
 *                   remaining are extras.
 */
export function writeIsotrieveFile(
  path: string,
  header: Record<string, unknown>,
  matrices: Array<{ name: string; mat: Float64Array[] }>,
): void {
  // Stamp format version
  header.formatVersion = FORMAT_VERSION;

  const headerJson = JSON.stringify(header);
  const headerBuf = Buffer.from(headerJson, 'utf-8');
  const paddedHeader = padTo8Bytes(headerBuf);

  // Build matrix payloads
  const matrixParts: Buffer[] = [];
  for (const { mat } of matrices) {
    matrixParts.push(matrixToBuffer(mat));
  }
  const payload = Buffer.concat(matrixParts);

  // Assemble file: magic + headerLen + CRCplaceholder + paddedJSON + payload
  const headerLenBuf = encodeUint32(headerBuf.length); // unpadded length

  // CRC32 covers everything except itself: magic + headerLen + paddedJSON + payload
  const part1 = Buffer.concat([ISOTRIEVE_MAGIC, headerLenBuf]); // bytes 0-7
  const part2 = Buffer.concat([paddedHeader, payload]);          // bytes 12-end
  const fullForCrc = Buffer.concat([part1, part2]);
  const checksum = crc32(fullForCrc);

  // Reassemble with real CRC
  const final = Buffer.concat([
    ISOTRIEVE_MAGIC,
    headerLenBuf,
    encodeUint32(checksum),
    paddedHeader,
    payload,
  ]);

  writeFileSync(path, final);
}

// ── Read: header only ─────────────────────────────────────────────

/**
 * Read just the JSON header from a .isotrieve file, without loading matrix data.
 *
 * @param path - Filesystem path to read from.
 * @returns The parsed header object (keys normalized to camelCase).
 * @throws If the file is corrupt, has an invalid magic number, or contains unknown keys.
 */
export function readIsotrieveHeader(path: string): Record<string, unknown> {
  const raw = readFileSync(path);

  if (raw.length < FIXED_HEADER_SIZE) {
    throw new Error('Invalid .isotrieve file: too short');
  }

  // Magic
  const magic = raw.subarray(0, 4);
  if (!magic.equals(ISOTRIEVE_MAGIC)) {
    throw new Error(
      `Invalid .isotrieve file: expected magic "ISTR", got "${magic.toString('ascii')}"`,
    );
  }

  // Header length
  const headerLen = decodeUint32(raw, 4);
  if (headerLen > MAX_HEADER_LEN) {
    throw new Error(
      `Invalid .isotrieve file: header length ${headerLen} exceeds ${MAX_HEADER_LEN} byte limit`,
    );
  }
  if (FIXED_HEADER_SIZE + headerLen > raw.length) {
    throw new Error(
      `Invalid .isotrieve file: header length ${headerLen} exceeds file size`,
    );
  }

  // CRC validation (v2+ only)
  const formatVersion = peekFormatVersion(raw, headerLen);
  if (formatVersion >= 2) {
    const storedCrc = decodeUint32(raw, 8);
    // Compute CRC over everything except the CRC field itself
    const part1 = raw.subarray(0, 8);             // magic + headerLen
    const part2 = raw.subarray(FIXED_HEADER_SIZE); // padded header + payload
    const combined = Buffer.concat([part1, part2]);
    const computedCrc = crc32(combined);
    if (storedCrc !== computedCrc) {
      throw new Error(
        `Invalid .isotrieve file: CRC32 mismatch (stored=0x${storedCrc.toString(16)}, ` +
        `computed=0x${computedCrc.toString(16)})`,
      );
    }
  }

  const headerJson = raw.subarray(FIXED_HEADER_SIZE, FIXED_HEADER_SIZE + headerLen).toString('utf-8');
  const parsed = JSON.parse(headerJson) as Record<string, unknown>;
  validateHeaderKeys(parsed);
  return normalizeHeaderKeys(parsed);
}

/**
 * Peek at formatVersion from the header without full parsing.
 * Returns 0 for v1 files (no formatVersion field).
 */
function peekFormatVersion(raw: Buffer, headerLen: number): number {
  try {
    const json = raw.subarray(FIXED_HEADER_SIZE, FIXED_HEADER_SIZE + headerLen).toString('utf-8');
    const obj = JSON.parse(json);
    return typeof obj.format_version === 'number' ? obj.format_version : 0;
  } catch {
    return 0;
  }
}

// ── Read: full payload ────────────────────────────────────────────

/**
 * Read the full .isotrieve file: header plus all matrix payloads.
 *
 * Matrices are deserialized according to the shapes recorded in the header.
 * Forward matrix, inverse matrix (if hasInverse), and any extra matrices
 * (e.g. mean_X, mean_Y for Procrustes) are returned.
 *
 * @param path - Filesystem path to read from.
 * @returns Object with `header` and `matrices` (a name→matrix map).
 * @throws If the file is corrupt, truncated, or has a CRC mismatch.
 */
export function readIsotrievePayload(path: string): {
  header: Record<string, unknown>;
  matrices: Map<string, Float64Array[]>;
} {
  const raw = readFileSync(path);

  if (raw.length < FIXED_HEADER_SIZE) {
    throw new Error('Invalid .isotrieve file: too short');
  }

  // Magic
  const magic = raw.subarray(0, 4);
  if (!magic.equals(ISOTRIEVE_MAGIC)) {
    throw new Error(
      `Invalid .isotrieve file: expected magic "ISTR", got "${magic.toString('ascii')}"`,
    );
  }

  // Header length
  const headerLen = decodeUint32(raw, 4);
  if (headerLen > MAX_HEADER_LEN) {
    throw new Error(
      `Invalid .isotrieve file: header length ${headerLen} exceeds ${MAX_HEADER_LEN} byte limit`,
    );
  }
  if (FIXED_HEADER_SIZE + headerLen > raw.length) {
    throw new Error(
      `Invalid .isotrieve file: header length ${headerLen} exceeds file size`,
    );
  }

  // CRC validation (v2+)
  const formatVersion = peekFormatVersion(raw, headerLen);
  if (formatVersion >= 2) {
    const storedCrc = decodeUint32(raw, 8);
    const part1 = raw.subarray(0, 8);
    const part2 = raw.subarray(FIXED_HEADER_SIZE);
    const combined = Buffer.concat([part1, part2]);
    const computedCrc = crc32(combined);
    if (storedCrc !== computedCrc) {
      throw new Error(
        `Invalid .isotrieve file: CRC32 mismatch (stored=0x${storedCrc.toString(16)}, ` +
        `computed=0x${computedCrc.toString(16)})`,
      );
    }
  }

  const headerJson = raw.subarray(FIXED_HEADER_SIZE, FIXED_HEADER_SIZE + headerLen).toString('utf-8');
  const parsed = JSON.parse(headerJson) as Record<string, unknown>;
  validateHeaderKeys(parsed);
  const header = normalizeHeaderKeys(parsed);

  // Payload starts after the 8-byte-padded header.
  // headerLen is the unpadded JSON length; padded length is the next multiple of 8.
  const paddedHeaderLen = Math.ceil(headerLen / 8) * 8;
  const payloadStart = FIXED_HEADER_SIZE + paddedHeaderLen;

  // Compute expected payload size from header shapes
  const matrixShape = header.matrixShape as [number, number] | undefined;
  const hasInverse = header.hasInverse as boolean;
  const inverseMatrixShape = header.inverseMatrixShape as [number, number] | undefined;
  const extraMatrices = header.extraMatrices as Record<string, [number, number]> | undefined;

  let expectedPayloadBytes = 0;
  if (matrixShape) {
    expectedPayloadBytes += matrixShape[0] * matrixShape[1] * 8;
  }
  if (hasInverse && inverseMatrixShape) {
    expectedPayloadBytes += inverseMatrixShape[0] * inverseMatrixShape[1] * 8;
  }
  if (extraMatrices) {
    for (const [, shape] of Object.entries(extraMatrices)) {
      expectedPayloadBytes += shape[0] * shape[1] * 8;
    }
  }

  const availablePayloadBytes = raw.length - payloadStart;
  if (expectedPayloadBytes !== availablePayloadBytes) {
    throw new Error(
      `Invalid .isotrieve file: payload size mismatch — ` +
      `header declares ${expectedPayloadBytes} bytes of matrix data, ` +
      `but ${availablePayloadBytes} bytes available after header ` +
      `(expected rows×cols×8 to equal remaining bytes exactly)`,
    );
  }

  let offset = payloadStart;
  const matrices = new Map<string, Float64Array[]>();

  // Forward matrix
  if (matrixShape) {
    const [rows, cols] = matrixShape;
    const bytesNeeded = rows * cols * 8;
    matrices.set('forward', bufferToMatrix(raw.subarray(offset, offset + bytesNeeded), rows, cols));
    offset += bytesNeeded;
  }

  // Inverse matrix
  if (hasInverse && inverseMatrixShape) {
    const [rows, cols] = inverseMatrixShape;
    const bytesNeeded = rows * cols * 8;
    matrices.set('inverse', bufferToMatrix(raw.subarray(offset, offset + bytesNeeded), rows, cols));
    offset += bytesNeeded;
  }

  // Extra matrices
  if (extraMatrices) {
    for (const [name, shape] of Object.entries(extraMatrices)) {
      const [rows, cols] = shape;
      const bytesNeeded = rows * cols * 8;
      matrices.set(name, bufferToMatrix(raw.subarray(offset, offset + bytesNeeded), rows, cols));
      offset += bytesNeeded;
    }
  }

  return { header, matrices };
}
