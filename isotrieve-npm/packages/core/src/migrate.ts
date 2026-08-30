/**
 * Migration tool for vector stores.
 *
 * End-to-end migration with resumability, non-destructive safety,
 * and progress tracking.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { dirname } from 'path';
import type { MigrationManifest, MigrationBatch, VectorRecord, VectorStore } from './types';
import type { Mapping } from './mapping/base';
import { l2Normalize } from './math/normalize';

// ── Helpers ─────────────────────────────────────────────────────

function computeBatchHash(vectors: Float64Array[]): string {
  // Simple deterministic hash: XOR all bytes of first few vectors
  let hash = 0;
  const limit = Math.min(vectors.length, 5);
  for (let i = 0; i < limit; i++) {
    for (let j = 0; j < Math.min(vectors[i].length, 100); j++) {
      hash = ((hash << 5) - hash + (vectors[i][j] * 1000) | 0) | 0;
    }
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}

function timestamp(): string {
  return new Date().toISOString();
}

// ── migrateStore ────────────────────────────────────────────────

export interface MigrateStoreOptions {
  batchSize?: number;
  manifestPath?: string;
  resume?: boolean;
}

/**
 * Migrate vectors from source to target with transformation.
 */
export function migrateStore(
  source: VectorStore,
  target: VectorStore,
  mapping: Mapping,
  options: MigrateStoreOptions = {},
): MigrationManifest {
  const batchSize = options.batchSize ?? 1024;
  const manifestPath = options.manifestPath;
  const resume = options.resume ?? false;

  // Load existing manifest if resuming
  let manifest: MigrationManifest;
  let startIdx = 0;

  if (resume && manifestPath && existsSync(manifestPath)) {
    try {
      const loaded = JSON.parse(readFileSync(manifestPath, 'utf-8'));
      manifest = loaded;
      startIdx = manifest.batchEnd;
    } catch {
      // Start fresh
      manifest = null!;
    }
  } else {
    manifest = null!;
  }

  if (manifest === null) {
    const total = source.count();
    manifest = {
      sourceCollection: 'source',
      targetCollection: 'target',
      sourceModel: '',
      targetModel: '',
      totalVectors: total,
      migratedVectors: 0,
      batchStart: 0,
      batchEnd: 0,
      lastBatchHash: '',
      startedAt: timestamp(),
      completedAt: '',
      batches: [],
    };
  }

  let written = 0;
  let batchNum = 0;
  const tStart = performance.now();

  const batches = source.iterVectors(batchSize);
  const batchArray = Array.isArray(batches) ? batches : Array.from(batches);

  for (const batchRecords of batchArray) {
    // Skip already-migrated batches
    if (batchNum < startIdx) {
      batchNum++;
      continue;
    }

    // Extract vectors
    const vectors = batchRecords.map((r) => Float64Array.from(r.vector));

    // Transform
    const transformed = mapping.transform(vectors) as Float64Array[];

    // Create new records
    const newRecords: VectorRecord[] = batchRecords.map((r, i) => ({
      id: r.id,
      vector: Array.from(transformed[i]),
      text: r.text,
      payload: r.payload,
    }));

    // Write to target
    target.writeVectors(newRecords);

    // Update manifest
    const batchHash = computeBatchHash(transformed);
    manifest.batches.push({
      batchNum,
      startIdx: batchNum * batchSize,
      count: batchRecords.length,
      hash: batchHash,
    });
    manifest.migratedVectors += batchRecords.length;
    manifest.batchEnd = batchNum + 1;
    manifest.lastBatchHash = batchHash;

    // Save manifest
    if (manifestPath) {
      saveManifest(manifestPath, manifest);
    }

    written += batchRecords.length;
    batchNum++;

    // Progress
    const elapsed = (performance.now() - tStart) / 1000;
    const rate = written / elapsed;
    const pct = manifest.totalVectors > 0
      ? (written / manifest.totalVectors) * 100
      : 0;

    process.stdout.write(
      `\r  Migrated ${written}/${manifest.totalVectors} ` +
      `(${pct.toFixed(1)}%) at ${rate.toFixed(0)} vec/s`,
    );
  }

  process.stdout.write('\n');
  manifest.completedAt = timestamp();
  if (manifestPath) {
    saveManifest(manifestPath, manifest);
  }

  return manifest;
}

function saveManifest(path: string, manifest: MigrationManifest): void {
  const dir = dirname(path);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  writeFileSync(path, JSON.stringify(manifest, null, 2));
}
