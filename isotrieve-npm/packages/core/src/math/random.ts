/**
 * Seeded PRNG and data splitting utilities.
 * Uses mulberry32 for deterministic, reproducible random number generation.
 */

/**
 * Mulberry32 PRNG — produces a function that returns [0, 1) floats.
 */
export function createRng(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Fisher-Yates shuffle using seeded RNG.
 */
export function shuffle<T>(arr: readonly T[], rng: () => number): T[] {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

/**
 * Split paired arrays into train/test sets using seeded RNG.
 */
export function trainTestSplit<T>(
  X: readonly T[],
  Y: readonly T[],
  testFraction: number,
  seed: number,
): { XTrain: T[]; XTest: T[]; YTrain: T[]; YTest: T[] } {
  if (X.length !== Y.length) {
    throw new Error(`Sample counts must match: X has ${X.length}, Y has ${Y.length}`);
  }
  if (X.length === 0) {
    throw new Error('Cannot split zero samples');
  }
  if (testFraction <= 0 || testFraction >= 1) {
    throw new Error(`testFraction must be in (0, 1), got ${testFraction}`);
  }

  const n = X.length;
  const indices = shuffle(
    Array.from({ length: n }, (_, i) => i),
    createRng(seed),
  );

  const nTest = Math.max(1, Math.round(n * testFraction));
  const testIdx = new Set(indices.slice(0, nTest));
  const trainIdx = indices.slice(nTest);

  const XTrain = trainIdx.map((i) => X[i]);
  const XTest = testIdx.size > 0 ? [...testIdx].map((i) => X[i]) : [X[X.length - 1]];
  const YTrain = trainIdx.map((i) => Y[i]);
  const YTest = testIdx.size > 0 ? [...testIdx].map((i) => Y[i]) : [Y[Y.length - 1]];

  return { XTrain, XTest, YTrain, YTest };
}
