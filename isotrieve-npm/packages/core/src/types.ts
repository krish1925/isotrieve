/**
 * Core type definitions for Isotrieve protocol
 */

export interface EmbeddingProvider {
  /**
   * Generate embedding for text
   */
  embed(text: string): Promise<number[]>;

  /**
   * Generate embeddings for multiple texts (batch)
   */
  embedBatch(texts: string[]): Promise<number[][]>;

  /**
   * Get embedding dimensions
   */
  getDimensions(): number;

  /**
   * Get model identifier
   */
  getModelId(): string;
}

export interface AgentCapabilities {
  agentId: string;
  embeddingModel: {
    name: string;
    dimensions: number;
  };
  protocolVersion: string;
  maxBatchSize: number;
  minQualityThreshold: number;
  supportsIsotrieve?: boolean;
  fallbackLanguage?: string;
}

export interface CalibrationConfig {
  vocabularySize?: number;
  validationSize?: number;
  qualityThreshold?: number;
  customVocabulary?: string[];
}

export interface TransferMatrix {
  matrixAB: number[][];
  matrixBA: number[][];
  trainingSimilarity: number;
  validationSimilarity: number;
  worstCaseSimilarity: number;
  validUntil: string;
  createdAt?: string;
}

export interface QualityMetrics {
  meanSimilarity: number;
  medianSimilarity: number;
  stdSimilarity: number;
  minSimilarity: number;
  maxSimilarity: number;
}

export interface SemanticTransfer {
  transferId: string;
  embedding: number[];
  sourceAgent: string;
  targetAgent: string;
  expectedSimilarity: number;
  timestamp: string;
  originalNorm?: number;
}

export interface IsotrieveConfig {
  embedder: EmbeddingProvider;
  agentId?: string;
  minQualityThreshold?: number;
  maxBatchSize?: number;
  /** Failures before circuit breaker opens */
  circuitBreakerThreshold?: number;
  /** Milliseconds before circuit breaker half-opens */
  circuitBreakerTimeout?: number;
  /** Maximum retry attempts */
  retryMax?: number;
  /** Base delay between retries in ms */
  retryBaseDelay?: number;
}

export interface CalibrationResult {
  success: boolean;
  transferMatrix: TransferMatrix;
  qualityMetrics: QualityMetrics;
  calibrationTime: number;
  vocabularySize?: number;
  errorMessage?: string;
}

/**
 * Result of sending a message (Isotrieve or text fallback)
 */
export interface MessageResult {
  method: 'isotrieve' | 'text';
  transferId?: string;
  embedding?: number[];
  sourceAgent?: string;
  targetAgent?: string;
  expectedSimilarity?: number;
  timestamp?: string;
  message?: string;
  language?: string;
  fallback?: boolean;
  fallbackReason?: string;
  note?: string;
}

/**
 * Connection health status
 */
export interface ConnectionHealth {
  calibrated: boolean;
  expired: boolean;
  quality: number;
  circuitBreaker: string;
  validUntil: string;
}
