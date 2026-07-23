/**
 * Voyage AI Adapter for Isotrieve
 */

import { EmbeddingProvider } from '@isotrieve/core';

export interface VoyageAdapterConfig {
  apiKey: string;
  model?: string;
  baseURL?: string;
}

export class VoyageAdapter implements EmbeddingProvider {
  private apiKey: string;
  private model: string;
  private baseURL: string;
  private dimensions: number;

  constructor(config: VoyageAdapterConfig) {
    this.apiKey = config.apiKey;
    this.model = config.model || 'voyage-2';
    this.baseURL = config.baseURL || 'https://api.voyageai.com/v1';
    this.dimensions = this.getDimensionsForModel(this.model);
  }

  async embed(text: string): Promise<number[]> {
    const response = await fetch(`${this.baseURL}/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        input: text,
        model: this.model,
      }),
    });

    if (!response.ok) {
      throw new Error(`Voyage API error: ${response.statusText}`);
    }

    const data: any = await response.json();
    return data.data[0].embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await fetch(`${this.baseURL}/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        input: texts,
        model: this.model,
      }),
    });

    if (!response.ok) {
      throw new Error(`Voyage API error: ${response.statusText}`);
    }

    const data: any = await response.json();
    return data.data.map((item: any) => item.embedding);
  }

  getDimensions(): number {
    return this.dimensions;
  }

  getModelId(): string {
    return this.model;
  }

  private getDimensionsForModel(model: string): number {
    const dimensionMap: Record<string, number> = {
      'voyage-2': 1024,
      'voyage-large-2': 1536,
      'voyage-code-2': 1536,
    };
    return dimensionMap[model] || 1024;
  }
}
