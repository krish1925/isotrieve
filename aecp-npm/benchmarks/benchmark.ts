/**
 * AECP Performance Benchmarks
 * Compares AECP vs text serialization
 */

import { AECP } from '../packages/core/src/protocol';
import { EmbeddingProvider } from '../packages/core/src/types';
import { performance } from 'perf_hooks';

// Mock embedder for consistent benchmarking
class BenchmarkEmbedder implements EmbeddingProvider {
  constructor(private dimensions: number, private modelId: string) {}

  async embed(text: string): Promise<number[]> {
    const embedding = new Array(this.dimensions).fill(0);
    for (let i = 0; i < Math.min(text.length, this.dimensions); i++) {
      embedding[i] = Math.sin(text.charCodeAt(i) * i);
    }
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map((val) => val / (norm || 1));
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map((t) => this.embed(t)));
  }

  getDimensions(): number {
    return this.dimensions;
  }

  getModelId(): string {
    return this.modelId;
  }
}

interface BenchmarkResult {
  name: string;
  time: number;
  throughput?: number;
  quality?: number;
}

async function benchmarkCalibration(vocabSize: number): Promise<BenchmarkResult> {
  const agent1 = new AECP({ embedder: new BenchmarkEmbedder(384, 'model-1') });
  const agent2 = new AECP({ embedder: new BenchmarkEmbedder(768, 'model-2') });

  const start = performance.now();
  const result = await agent1.calibrateWith(agent2, {
    vocabularySize: vocabSize,
    validationSize: Math.floor(vocabSize * 0.1),
  });
  const time = performance.now() - start;

  return {
    name: `Calibration (${vocabSize} items)`,
    time,
    throughput: vocabSize / (time / 1000), // items per second
    quality: result.qualityMetrics.meanSimilarity,
  };
}

async function benchmarkTransfer(numTransfers: number): Promise<BenchmarkResult> {
  const agent1 = new AECP({ embedder: new BenchmarkEmbedder(384, 'model-1') });
  const agent2 = new AECP({ embedder: new BenchmarkEmbedder(768, 'model-2') });

  await agent1.calibrateWith(agent2, {
    vocabularySize: 100,
    validationSize: 20,
  });

  const embeddings = await agent1.embedBatch(
    Array(numTransfers).fill('benchmark text')
  );

  const start = performance.now();
  for (const embedding of embeddings) {
    await agent1.transferTo(agent2, embedding);
  }
  const time = performance.now() - start;

  return {
    name: `Transfer (${numTransfers} items)`,
    time,
    throughput: numTransfers / (time / 1000),
  };
}

async function benchmarkSimilaritySearch(kbSize: number, queries: number): Promise<BenchmarkResult> {
  const agent = new AECP({ embedder: new BenchmarkEmbedder(384, 'model') });

  const knowledgeBase = await agent.embedBatch(
    Array(kbSize).fill(0).map((_, i) => `document ${i}`)
  );

  const queryEmbeddings = await agent.embedBatch(
    Array(queries).fill(0).map((_, i) => `query ${i}`)
  );

  const start = performance.now();
  for (const query of queryEmbeddings) {
    await agent.findSimilar(query, knowledgeBase, 10);
  }
  const time = performance.now() - start;

  return {
    name: `Similarity Search (KB=${kbSize}, Q=${queries})`,
    time,
    throughput: queries / (time / 1000),
  };
}

async function benchmarkTextBaseline(numItems: number): Promise<BenchmarkResult> {
  const texts = Array(numItems).fill(0).map((_, i) => `text ${i}`);

  const start = performance.now();
  // Simulate text serialization overhead
  for (const text of texts) {
    const serialized = JSON.stringify({ text });
    JSON.parse(serialized);
  }
  const time = performance.now() - start;

  return {
    name: `Text Serialization (${numItems} items)`,
    time,
    throughput: numItems / (time / 1000),
  };
}

async function runBenchmarks() {
  console.log(' AECP Performance Benchmarks\n');
  console.log('='.repeat(80));

  const results: BenchmarkResult[] = [];

  // Calibration benchmarks
  console.log('\n Calibration Performance:');
  for (const size of [100, 500, 1000, 5000]) {
    const result = await benchmarkCalibration(size);
    results.push(result);
    console.log(`  ${result.name}`);
    console.log(`    Time: ${result.time.toFixed(2)}ms`);
    console.log(`    Throughput: ${result.throughput?.toFixed(0)} items/sec`);
    console.log(`    Quality: ${result.quality?.toFixed(4)}`);
  }

  // Transfer benchmarks
  console.log('\n Transfer Performance:');
  for (const count of [10, 100, 1000]) {
    const result = await benchmarkTransfer(count);
    results.push(result);
    console.log(`  ${result.name}`);
    console.log(`    Time: ${result.time.toFixed(2)}ms`);
    console.log(`    Avg per transfer: ${(result.time / count).toFixed(3)}ms`);
    console.log(`    Throughput: ${result.throughput?.toFixed(0)} transfers/sec`);
  }

  // Similarity search benchmarks
  console.log('\n Similarity Search Performance:');
  for (const [kb, queries] of [[100, 10], [1000, 100], [10000, 100]] as [number, number][]) {
    const result = await benchmarkSimilaritySearch(kb, queries);
    results.push(result);
    console.log(`  ${result.name}`);
    console.log(`    Time: ${result.time.toFixed(2)}ms`);
    console.log(`    Avg per query: ${(result.time / queries).toFixed(2)}ms`);
  }

  // Text baseline comparison
  console.log('\n Text Baseline Comparison:');
  const textResult = await benchmarkTextBaseline(1000);
  results.push(textResult);
  console.log(`  ${textResult.name}`);
  console.log(`    Time: ${textResult.time.toFixed(2)}ms`);

  console.log('\n' + '='.repeat(80));
  console.log(' Benchmarks complete!\n');

  // Summary table
  console.log(' Summary Table:\n');
  console.log('| Benchmark | Time (ms) | Throughput |');
  console.log('|-----------|-----------|------------|');
  for (const result of results) {
    const throughput = result.throughput 
      ? `${result.throughput.toFixed(0)}/sec`
      : '-';
    console.log(`| ${result.name.padEnd(45)} | ${result.time.toFixed(2).padStart(8)} | ${throughput.padEnd(10)} |`);
  }

  return results;
}

// Run if called directly
if (require.main === module) {
  runBenchmarks().catch(console.error);
}

export { runBenchmarks, BenchmarkEmbedder };
