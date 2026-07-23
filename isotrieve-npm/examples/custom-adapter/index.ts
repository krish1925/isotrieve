/**
 * Custom Adapter Example
 * Demonstrates creating a custom embedding provider
 */

import { Isotrieve, EmbeddingProvider } from '@isotrieve/core';

/**
 * Simple mock embedder for demonstration
 * In production, this would call your actual embedding service
 */
class CustomEmbedder implements EmbeddingProvider {
  private dimensions: number;
  private modelId: string;

  constructor(dimensions: number = 384, modelId: string = 'custom-model') {
    this.dimensions = dimensions;
    this.modelId = modelId;
  }

  async embed(text: string): Promise<number[]> {
    // Mock: Generate deterministic embedding based on text
    const embedding = new Array(this.dimensions).fill(0);
    
    for (let i = 0; i < text.length && i < this.dimensions; i++) {
      embedding[i] = text.charCodeAt(i) / 255;
    }
    
    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map(val => val / norm);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map(text => this.embed(text)));
  }

  getDimensions(): number {
    return this.dimensions;
  }

  getModelId(): string {
    return this.modelId;
  }
}

async function main() {
  console.log('=== Isotrieve Custom Adapter Example ===\n');

  // Create agents with custom embedders
  const agent1 = new Isotrieve({
    embedder: new CustomEmbedder(384, 'custom-v1'),
  });

  const agent2 = new Isotrieve({
    embedder: new CustomEmbedder(512, 'custom-v2'),
  });

  console.log('Agent 1:', agent1.getCapabilities());
  console.log('Agent 2:', agent2.getCapabilities());
  console.log();

  // Calibrate
  console.log('Calibrating with custom embedders...');
  const calibration = await agent1.calibrateWith(agent2, {
    vocabularySize: 200,
    validationSize: 20,
  });

  console.log(`Quality: ${calibration.qualityMetrics.meanSimilarity.toFixed(4)}`);
  console.log(`Time: ${calibration.calibrationTime}ms\n`);

  // Transfer
  const message = 'Custom embedding transfer test';
  const embedding = await agent1.embed(message);
  const transfer = await agent1.transferTo(agent2, embedding);

  console.log(`Original dimensions: ${embedding.length}`);
  console.log(`Transferred dimensions: ${transfer.embedding.length}`);
  console.log(`Expected quality: ${transfer.expectedSimilarity.toFixed(4)}`);
}

main().catch(console.error);
