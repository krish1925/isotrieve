/**
 * HuggingFace Adapter for Isotrieve
 * Uses @xenova/transformers for local inference
 */

import { pipeline, FeatureExtractionPipeline } from '@xenova/transformers';
import { EmbeddingProvider } from '@isotrieve/core';

export interface HuggingFaceAdapterConfig {
  model?: string;
  quantized?: boolean;
}

export class HuggingFaceAdapter implements EmbeddingProvider {
  private model: string;
  private pipeline: FeatureExtractionPipeline | null = null;
  private dimensions: number;
  private initPromise: Promise<void>;

  constructor(config: HuggingFaceAdapterConfig = {}) {
    this.model = config.model || 'Xenova/all-MiniLM-L6-v2';
    this.dimensions = this.getDimensionsForModel(this.model);
    
    // Initialize pipeline asynchronously
    this.initPromise = this.initialize(config.quantized ?? true);
  }

  private async initialize(quantized: boolean): Promise<void> {
    this.pipeline = await pipeline('feature-extraction', this.model, {
      quantized,
    });
  }

  async embed(text: string): Promise<number[]> {
    await this.initPromise;
    if (!this.pipeline) {
      throw new Error('Pipeline not initialized');
    }

    const output = await this.pipeline(text, {
      pooling: 'mean',
      normalize: true,
    });

    return Array.from(output.data);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    // Process sequentially for now (transformers.js batch support varies)
    return Promise.all(texts.map((text) => this.embed(text)));
  }

  getDimensions(): number {
    return this.dimensions;
  }

  getModelId(): string {
    return this.model;
  }

  private getDimensionsForModel(model: string): number {
    const dimensionMap: Record<string, number> = {
      'Xenova/all-MiniLM-L6-v2': 384,
      'Xenova/all-mpnet-base-v2': 768,
      'Xenova/bge-small-en-v1.5': 384,
      'Xenova/bge-base-en-v1.5': 768,
    };

    // Extract model name if full path provided
    const modelName = model.includes('/') ? model : `Xenova/${model}`;
    return dimensionMap[modelName] || 384;
  }
}
