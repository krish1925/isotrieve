import { cosineSimilarity, transpose, matrixMultiply, vecNorm, TypedMatrix, toTypedMatrix, SingleOrBatch } from './linalg';
import { l2Normalize } from './normalize';

export function pairwiseCosineStats(
  predicted: Float64Array[],
  target: Float64Array[],
): { mean: number; median: number; p5: number; min: number; max: number } {
  if (predicted.length !== target.length) {
    throw new Error(
      `Shape mismatch: ${predicted.length} vectors vs ${target.length} vectors`,
    );
  }
  const n = predicted.length;
  if (n === 0) return { mean: 0, median: 0, p5: 0, min: 0, max: 0 };

  const sims = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    sims[i] = cosineSimilarity(predicted[i], target[i]);
  }

  const sorted = Float64Array.from(sims).sort();
  let sum = 0;
  for (let i = 0; i < n; i++) sum += sims[i];

  return {
    mean: sum / n,
    median: sorted[Math.floor(n / 2)],
    p5: sorted[Math.floor(n * 0.05)],
    min: sorted[0],
    max: sorted[n - 1],
  };
}

export function topkRetention(
  queries: Float64Array[],
  corpus: Float64Array[],
  k = 1,
): number {
  const n = queries.length;
  if (n === 0) return 0;
  k = Math.min(k, n);

  const Q = TypedMatrix.fromRows(queries);
  const C = TypedMatrix.fromRows(corpus);
  const sims = matrixMultiply(Q, transpose(C));

  let correct = 0;
  const simRows = sims.rows;
  const simCols = sims.cols;
  const sd = sims.data;
  for (let i = 0; i < simRows; i++) {
    const row = new Float64Array(sd.subarray(i * simCols, (i + 1) * simCols));
    const topK = quickSelectTopK(row, k);
    if (topK.has(i)) correct++;
  }
  return correct / n;
}

function quickSelectTopK(arr: Float64Array, k: number): Set<number> {
  const n = arr.length;
  if (k >= n) {
    return new Set(Array.from({ length: n }, (_, i) => i));
  }

  const heap: Array<{ val: number; idx: number }> = [];
  for (let i = 0; i < n; i++) {
    if (heap.length < k) {
      heap.push({ val: arr[i], idx: i });
      if (heap.length === k) heapifyUp(heap, 0);
    } else if (arr[i] > heap[0].val) {
      heap[0] = { val: arr[i], idx: i };
      heapifyDown(heap, 0, k);
    }
  }
  return new Set(heap.map((h) => h.idx));
}

function heapifyUp(heap: Array<{ val: number; idx: number }>, i: number): void {
  while (i > 0) {
    const parent = (i - 1) >> 1;
    if (heap[i].val < heap[parent].val) {
      [heap[i], heap[parent]] = [heap[parent], heap[i]];
      i = parent;
    } else break;
  }
}

function heapifyDown(
  heap: Array<{ val: number; idx: number }>,
  i: number,
  size: number,
): void {
  while (true) {
    let smallest = i;
    const left = 2 * i + 1;
    const right = 2 * i + 2;
    if (left < size && heap[left].val < heap[smallest].val) smallest = left;
    if (right < size && heap[right].val < heap[smallest].val) smallest = right;
    if (smallest === i) break;
    [heap[i], heap[smallest]] = [heap[smallest], heap[i]];
    i = smallest;
  }
}

export function spearmanRho(a: Float64Array, b: Float64Array): number {
  if (a.length < 2) return 0;
  const n = a.length;

  const rankA = rank(a);
  const rankB = rank(b);

  let sumA = 0;
  let sumB = 0;
  for (let i = 0; i < n; i++) {
    sumA += rankA[i];
    sumB += rankB[i];
  }
  const meanA = sumA / n;
  const meanB = sumB / n;

  let ssA = 0;
  let ssB = 0;
  let ab = 0;
  for (let i = 0; i < n; i++) {
    const da = rankA[i] - meanA;
    const db = rankB[i] - meanB;
    ssA += da * da;
    ssB += db * db;
    ab += da * db;
  }

  const denom = Math.sqrt(ssA * ssB);
  if (denom < 1e-12) return 0;
  return ab / denom;
}

function rank(arr: Float64Array): Float64Array {
  const n = arr.length;
  const indexed = Array.from({ length: n }, (_, i) => ({ val: arr[i], idx: i }));
  indexed.sort((a, b) => a.val - b.val);

  const result = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j < n - 1 && indexed[j + 1].val === indexed[j].val) j++;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) {
      result[indexed[k].idx] = avgRank;
    }
    i = j + 1;
  }
  return result;
}

export function holdoutRankCorrelation(
  mapped: Float64Array[],
  target: Float64Array[],
  sampleSize: number | null = 200,
  seed = 0,
): number {
  const n = mapped.length;
  if (n < 3) return 0;

  let idx: number[];
  if (sampleSize !== null && sampleSize < n) {
    idx = sampleIndices(n, sampleSize, seed);
  } else {
    idx = Array.from({ length: n }, (_, i) => i);
  }

  const m = l2Normalize(idx.map((i) => mapped[i])) as Float64Array[];
  const t = l2Normalize(idx.map((i) => target[i])) as Float64Array[];

  const simMapped = matrixMultiply(TypedMatrix.fromRows(m), transpose(TypedMatrix.fromRows(t)));
  const simTarget = matrixMultiply(TypedMatrix.fromRows(t), transpose(TypedMatrix.fromRows(t)));

  const len = idx.length;
  const aVals: number[] = [];
  const bVals: number[] = [];

  for (let i = 0; i < len; i++) {
    for (let j = i + 1; j < len; j++) {
      aVals.push(simMapped.data[i * len + j]);
      bVals.push(simTarget.data[i * len + j]);
    }
  }

  return spearmanRho(new Float64Array(aVals), new Float64Array(bVals));
}

function sampleIndices(n: number, k: number, seed: number): number[] {
  let s = seed | 0;
  const sampled = new Set<number>();
  while (sampled.size < k && sampled.size < n) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    sampled.add(s % n);
  }
  return Array.from(sampled);
}

export function mrrDelta(
  mappedQueries: Float64Array[],
  corpus: Float64Array[],
  trueQueries: Float64Array[],
): { mrrMapped: number; mrrTrue: number; mrrDelta: number } {
  const mrrMapped = computeMrr(mappedQueries, corpus);
  const mrrTrue = computeMrr(trueQueries, corpus);
  return {
    mrrMapped,
    mrrTrue,
    mrrDelta: mrrMapped - mrrTrue,
  };
}

function computeMrr(queries: Float64Array[], corpus: Float64Array[]): number {
  const n = queries.length;
  if (n === 0) return 0;

  const Q = TypedMatrix.fromRows(queries);
  const C = TypedMatrix.fromRows(corpus);
  const sims = matrixMultiply(Q, transpose(C));

  let rr = 0;
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(sims.data.buffer, sims.data.byteOffset + i * sims.cols * 8, sims.cols);
    let bestRank = 0;
    let bestSim = -Infinity;
    for (let j = 0; j < row.length; j++) {
      if (row[j] > bestSim) {
        bestSim = row[j];
        bestRank = j;
      }
    }
    let rank = 1;
    for (let j = 0; j < row.length; j++) {
      if (row[j] > row[i]) rank++;
    }
    rr += 1.0 / rank;
  }
  return rr / n;
}

export function retrievalRetentionReport(
  mapped: Float64Array[],
  target: Float64Array[],
): {
  cosine: { mean: number; median: number; p5: number; min: number; max: number };
  top1Retention: number;
  top10Retention: number;
  mrr: { mrrMapped: number; mrrTrue: number; mrrDelta: number };
} {
  const normalizedMapped = l2Normalize(mapped) as Float64Array[];
  const normalizedTarget = l2Normalize(target) as Float64Array[];
  return {
    cosine: pairwiseCosineStats(normalizedMapped, normalizedTarget),
    top1Retention: topkRetention(normalizedMapped, normalizedTarget, 1),
    top10Retention: topkRetention(normalizedMapped, normalizedTarget, Math.min(10, target.length)),
    mrr: mrrDelta(normalizedMapped, normalizedTarget, normalizedTarget),
  };
}
