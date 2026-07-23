/**
 * Cohere Adapter for Isotrieve
 */

import { CohereClient } from 'cohere-ai';
import { EmbeddingProvider } from '@isotrieve/core';

export interface CohereAdapterConfig {
  apiKey: string;
  model?: string;
}

export class CohereAdapter implements EmbeddingProvider {
  private client: CohereClient;
  private model: string;
  private dimensions: number;

  constructor(config: CohereAdapterConfig) {
    this.client = new CohereClient({
      token: config.apiKey,
    });
    this.model = config.model || 'embed-english-v3.0';
    this.dimensions = this.getDimensionsForModel(this.model);
  }

  async embed(text: string): Promise<number[]> {
    const response = await this.client.embed({
      texts: [text],
      model: this.model,
      inputType: 'search_document',
    });

    return (response.embeddings as number[][])[0];
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await this.client.embed({
      texts,
      model: this.model,
      inputType: 'search_document',
    });

    return response.embeddings as number[][];
  }

  getDimensions(): number {
    return this.dimensions;
  }

  getModelId(): string {
    return this.model;
  }

  private getDimensionsForModel(model: string): number {
    const dimensionMap: Record<string, number> = {
      'embed-english-v3.0': 1024,
      'embed-multilingual-v3.0': 1024,
      'embed-english-light-v3.0': 384,
      'embed-multilingual-light-v3.0': 384,
    };
    return dimensionMap[model] || 1024;
  }
}
