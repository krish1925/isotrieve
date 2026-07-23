/**
 * OpenAI Adapter for Isotrieve
 */

import OpenAI from 'openai';
import { EmbeddingProvider } from '@isotrieve/core';

export interface OpenAIAdapterConfig {
  apiKey: string;
  model?: string;
  organization?: string;
  baseURL?: string;
}

export class OpenAIAdapter implements EmbeddingProvider {
  private client: OpenAI;
  private model: string;
  private dimensions: number;

  constructor(config: OpenAIAdapterConfig) {
    this.client = new OpenAI({
      apiKey: config.apiKey,
      organization: config.organization,
      baseURL: config.baseURL,
    });

    this.model = config.model || 'text-embedding-3-small';
    
    // Set dimensions based on model
    this.dimensions = this.getDimensionsForModel(this.model);
  }

  async embed(text: string): Promise<number[]> {
    const response = await this.client.embeddings.create({
      model: this.model,
      input: text,
    });

    return response.data[0].embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    // OpenAI supports batch embedding
    const response = await this.client.embeddings.create({
      model: this.model,
      input: texts,
    });

    return response.data.map((item) => item.embedding);
  }

  getDimensions(): number {
    return this.dimensions;
  }

  getModelId(): string {
    return this.model;
  }

  private getDimensionsForModel(model: string): number {
    const dimensionMap: Record<string, number> = {
      'text-embedding-3-small': 1536,
      'text-embedding-3-large': 3072,
      'text-embedding-ada-002': 1536,
    };

    return dimensionMap[model] || 1536;
  }
}
