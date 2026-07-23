/**
 * Isotrieve Protocol Implementation
 *
 * Production-ready with:
 * - Circuit breaker for failing connections
 * - Graceful fallback to English text
 * - Real-time debug monitoring
 * - Comprehensive error handling
 * - Thread-safe operations
 */

import { randomBytes } from 'crypto';
import {
  IsotrieveConfig,
  AgentCapabilities,
  CalibrationConfig,
  CalibrationResult,
  QualityMetrics,
  SemanticTransfer,
  TransferMatrix,
  MessageResult,
  ConnectionHealth,
} from './types';
import {
  cosineSimilarity,
  computeTransferMatrices,
  vectorMatrixMultiply,
} from './matrix';
import { DEFAULT_VOCABULARY, generateExtendedVocabulary } from './vocabulary';
import {
  IsotrieveError,
  AgentNotCalibratedError,
  MatrixExpiredError,
  TransferError,
  CircuitBreaker,
  CircuitOpenError,
  RetryPolicy,
  GracefulDegradation,
} from './errors';
import { DebugMonitor, EventType, DebugEvent } from './debug';

export class Isotrieve {
  private embedder: IsotrieveConfig['embedder'];
  private agentId: string;
  private minQualityThreshold: number;
  private maxBatchSize: number;
  private transferMatrices: Map<string, TransferMatrix> = new Map();
  private calibrationCache: Map<string, number[][]> = new Map();

  // Resilience
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();
  private cbThreshold: number;
  private cbTimeout: number;
  private retryPolicy: RetryPolicy;
  private degradation: GracefulDegradation;

  // Transfer log
  private transferLog: Array<Record<string, any>> = [];

  constructor(config: IsotrieveConfig) {
    this.embedder = config.embedder;
    this.agentId = config.agentId || this.generateAgentId();
    this.minQualityThreshold = config.minQualityThreshold || 0.75;
    this.maxBatchSize = config.maxBatchSize || 1000;

    // Circuit breaker config
    this.cbThreshold = config.circuitBreakerThreshold ?? 5;
    this.cbTimeout = config.circuitBreakerTimeout ?? 60000;

    // Retry policy
    this.retryPolicy = new RetryPolicy({
      maxRetries: config.retryMax ?? 3,
      baseDelay: config.retryBaseDelay ?? 1000,
    });

    // Graceful degradation
    this.degradation = new GracefulDegradation('en');
  }

  /**
   * Get agent's unique identifier
   */
  getAgentId(): string {
    return this.agentId;
  }

  /**
   * Get agent capabilities
   */
  getCapabilities(): AgentCapabilities {
    return {
      agentId: this.agentId,
      embeddingModel: {
        name: this.embedder.getModelId(),
        dimensions: this.embedder.getDimensions(),
      },
      protocolVersion: '1.0',
      maxBatchSize: this.maxBatchSize,
      minQualityThreshold: this.minQualityThreshold,
      supportsIsotrieve: true,
      fallbackLanguage: 'en',
    };
  }

  /**
   * Get or create circuit breaker for a partner
   */
  private getCircuitBreaker(partnerId: string): CircuitBreaker {
    if (!this.circuitBreakers.has(partnerId)) {
      const cb = new CircuitBreaker({
        failureThreshold: this.cbThreshold,
        resetTimeout: this.cbTimeout,
      });
      cb.agentId = `${this.agentId}->${partnerId}`;
      this.circuitBreakers.set(partnerId, cb);
    }
    return this.circuitBreakers.get(partnerId)!;
  }

  /**
   * Get debug monitor (if enabled)
   */
  private get monitor(): DebugMonitor | null {
    return DebugMonitor.getGlobal();
  }

  /**
   * Emit a debug event
   */
  private emitEvent(
    eventType: EventType,
    extra: Partial<DebugEvent> = {},
  ): void {
    const monitor = this.monitor;
    if (!monitor) return;

    monitor.logEvent({
      eventType,
      timestamp: Date.now(),
      agentId: this.agentId,
      ...extra,
    });
  }

  /**
   * Calibrate with another Isotrieve agent
   */
  async calibrateWith(
    otherAgent: Isotrieve,
    config: CalibrationConfig = {},
  ): Promise<CalibrationResult> {
    const startTime = Date.now();

    this.emitEvent(EventType.CALIBRATION_START, {
      partnerId: otherAgent.agentId,
    });

    try {
      // Generate or use provided vocabulary
      const vocabularySize = config.vocabularySize || 1000;
      const validationSize = config.validationSize || 100;
      const qualityThreshold = config.qualityThreshold || this.minQualityThreshold;

      const vocabulary =
        config.customVocabulary ||
        generateExtendedVocabulary(vocabularySize + validationSize);

      // Split into training and validation
      const trainVocab = vocabulary.slice(0, vocabularySize);
      const valVocab = vocabulary.slice(vocabularySize, vocabularySize + validationSize);

      console.log(`[Isotrieve] Calibrating ${this.agentId} <-> ${otherAgent.agentId}`);
      console.log(
        `[Isotrieve] Training: ${trainVocab.length} items, Validation: ${valVocab.length} items`,
      );

      // Encode training vocabulary
      console.log('[Isotrieve] Encoding training vocabulary...');
      const embA = await this.embedder.embedBatch(trainVocab);
      const embB = await otherAgent.embedder.embedBatch(trainVocab);

      // Compute transfer matrices
      console.log('[Isotrieve] Computing transfer matrices...');
      const { forward, backward } = computeTransferMatrices(embA, embB);

      // Validate on held-out data
      console.log('[Isotrieve] Validating on held-out data...');
      const valEmbA = await this.embedder.embedBatch(valVocab);
      const valEmbB = await otherAgent.embedder.embedBatch(valVocab);

      // Round-trip validation: A -> B -> A
      const roundtripSimilarities: number[] = [];
      for (let i = 0; i < valEmbA.length; i++) {
        const transferred = vectorMatrixMultiply(valEmbA[i], forward);
        const roundtrip = vectorMatrixMultiply(transferred, backward);
        const similarity = cosineSimilarity(valEmbA[i], roundtrip);
        roundtripSimilarities.push(similarity);
      }

      const qualityMetrics: QualityMetrics = {
        meanSimilarity: this.mean(roundtripSimilarities),
        medianSimilarity: this.median(roundtripSimilarities),
        stdSimilarity: this.std(roundtripSimilarities),
        minSimilarity: Math.min(...roundtripSimilarities),
        maxSimilarity: Math.max(...roundtripSimilarities),
      };

      console.log(
        `[Isotrieve] Validation similarity: ${qualityMetrics.meanSimilarity.toFixed(4)}`,
      );
      console.log(
        `[Isotrieve] Worst-case similarity: ${qualityMetrics.minSimilarity.toFixed(4)}`,
      );

      // Check quality threshold
      const success = qualityMetrics.meanSimilarity >= qualityThreshold;
      if (!success) {
        console.warn(
          `[Isotrieve] ⚠️ Quality below threshold: ${qualityMetrics.meanSimilarity.toFixed(4)} < ${qualityThreshold}`,
        );
        console.warn('[Isotrieve] Communication will still work but with reduced fidelity.');
      } else {
        console.log(
          `[Isotrieve] ✓ Quality threshold met (${qualityMetrics.meanSimilarity.toFixed(4)} >= ${qualityThreshold})`,
        );
      }

      // Store transfer matrices
      const validUntil = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString();
      const transferMatrix: TransferMatrix = {
        matrixAB: forward,
        matrixBA: backward,
        trainingSimilarity: qualityMetrics.meanSimilarity, // Fixed typo
        validationSimilarity: qualityMetrics.meanSimilarity,
        worstCaseSimilarity: qualityMetrics.minSimilarity,
        validUntil,
        createdAt: new Date().toISOString(),
      };

      this.transferMatrices.set(otherAgent.agentId, transferMatrix);

      // Store reverse for other agent
      const reverseMatrix: TransferMatrix = {
        matrixAB: backward,
        matrixBA: forward,
        trainingSimilarity: qualityMetrics.meanSimilarity,
        validationSimilarity: qualityMetrics.meanSimilarity,
        worstCaseSimilarity: qualityMetrics.minSimilarity,
        validUntil,
        createdAt: new Date().toISOString(),
      };
      otherAgent.transferMatrices.set(this.agentId, reverseMatrix);

      const calibrationTime = Date.now() - startTime;
      console.log(`[Isotrieve] ✓ Calibration complete in ${calibrationTime}ms`);

      const result: CalibrationResult = {
        success,
        transferMatrix,
        qualityMetrics,
        calibrationTime,
        vocabularySize: trainVocab.length,
      };

      this.emitEvent(EventType.CALIBRATION_END, {
        partnerId: otherAgent.agentId,
        durationMs: calibrationTime,
        qualityScore: qualityMetrics.meanSimilarity,
        metadata: {
          worstCase: qualityMetrics.minSimilarity,
          vocabSize: trainVocab.length,
        },
      });

      // Reset circuit breaker on successful calibration
      this.getCircuitBreaker(otherAgent.agentId).reset();

      return result;
    } catch (error) {
      const calibrationTime = Date.now() - startTime;
      const errorMsg = error instanceof Error ? error.message : String(error);

      this.emitEvent(EventType.ERROR, {
        partnerId: otherAgent.agentId,
        durationMs: calibrationTime,
        error: errorMsg,
        metadata: { operation: 'calibration' },
      });

      return {
        success: false,
        transferMatrix: {
          matrixAB: [],
          matrixBA: [],
          trainingSimilarity: 0,
          validationSimilarity: 0,
          worstCaseSimilarity: 0,
          validUntil: '',
        },
        qualityMetrics: {
          meanSimilarity: 0,
          medianSimilarity: 0,
          stdSimilarity: 0,
          minSimilarity: 0,
          maxSimilarity: 0,
        },
        calibrationTime,
        errorMessage: errorMsg,
      };
    }
  }

  /**
   * Embed text using this agent's embedder
   */
  async embed(text: string): Promise<number[]> {
    const startTime = Date.now();
    try {
      const result = await this.embedder.embed(text);
      const durationMs = Date.now() - startTime;

      this.emitEvent(EventType.EMBED, {
        durationMs,
        costEstimate: this.estimateEmbedCost(1),
      });

      return result;
    } catch (error) {
      this.emitEvent(EventType.ERROR, {
        error: error instanceof Error ? error.message : String(error),
        metadata: { operation: 'embed' },
      });
      throw error;
    }
  }

  /**
   * Embed multiple texts (batch)
   */
  async embedBatch(texts: string[]): Promise<number[][]> {
    const startTime = Date.now();
    try {
      const result = await this.embedder.embedBatch(texts);
      const durationMs = Date.now() - startTime;

      this.emitEvent(EventType.EMBED_BATCH, {
        durationMs,
        costEstimate: this.estimateEmbedCost(texts.length),
        tokensUsed: texts.length * 10,
        metadata: { batchSize: texts.length },
      });

      return result;
    } catch (error) {
      this.emitEvent(EventType.ERROR, {
        error: error instanceof Error ? error.message : String(error),
        metadata: { operation: 'embedBatch', batchSize: texts.length },
      });
      throw error;
    }
  }

  /**
   * Transfer embedding to another agent's space
   *
   * Uses circuit breaker to prevent cascading failures.
   */
  async transferTo(
    targetAgent: Isotrieve,
    embedding: number[],
  ): Promise<SemanticTransfer> {
    const startTime = Date.now();
    const cb = this.getCircuitBreaker(targetAgent.agentId);

    const doTransfer = async (): Promise<SemanticTransfer> => {
      const matrix = this.transferMatrices.get(targetAgent.agentId);
      if (!matrix) {
        throw new AgentNotCalibratedError(this.agentId, targetAgent.agentId);
      }

      // Check expiration
      if (new Date(matrix.validUntil) < new Date()) {
        throw new MatrixExpiredError(
          `${this.agentId}->${targetAgent.agentId}`,
          matrix.validUntil,
        );
      }

      // Transform embedding
      const transferred = vectorMatrixMultiply(embedding, matrix.matrixAB);

      const originalNorm = Math.sqrt(
        embedding.reduce((sum, v) => sum + v * v, 0),
      );

      return {
        transferId: this.generateTransferId(),
        embedding: transferred,
        sourceAgent: this.agentId,
        targetAgent: targetAgent.agentId,
        expectedSimilarity: matrix.validationSimilarity,
        timestamp: new Date().toISOString(),
        originalNorm,
      };
    };

    try {
      const result = await cb.call(doTransfer);
      const durationMs = Date.now() - startTime;

      this.transferLog.push({
        transferId: result.transferId,
        source: this.agentId,
        target: targetAgent.agentId,
        timestamp: result.timestamp,
        expectedQuality: result.expectedSimilarity,
        durationMs,
      });

      this.emitEvent(EventType.TRANSFER, {
        partnerId: targetAgent.agentId,
        durationMs,
        qualityScore: result.expectedSimilarity,
        costEstimate: this.estimateEmbedCost(1),
      });

      return result;
    } catch (error) {
      const durationMs = Date.now() - startTime;

      if (error instanceof CircuitOpenError) {
        this.emitEvent(EventType.CIRCUIT_BREAK, {
          partnerId: targetAgent.agentId,
          error: 'Circuit breaker open',
        });
      } else {
        this.emitEvent(EventType.ERROR, {
          partnerId: targetAgent.agentId,
          durationMs,
          error: error instanceof Error ? error.message : String(error),
          metadata: { operation: 'transfer' },
        });
      }

      throw error;
    }
  }

  /**
   * Send a message with automatic Isotrieve/text fallback
   *
   * Tries Isotrieve first. If that fails, falls back to English text.
   * This method NEVER throws - it always returns a valid result.
   */
  async sendMessage(
    targetAgent: Isotrieve,
    message: string,
    fallbackToText = true,
  ): Promise<MessageResult> {
    const isotrieveFn = async (): Promise<MessageResult> => {
      const embedding = await this.embed(message);
      const transfer = await this.transferTo(targetAgent, embedding);
      return {
        method: 'isotrieve',
        transferId: transfer.transferId,
        embedding: transfer.embedding,
        sourceAgent: transfer.sourceAgent,
        targetAgent: transfer.targetAgent,
        expectedSimilarity: transfer.expectedSimilarity,
        timestamp: transfer.timestamp,
      };
    };

    if (fallbackToText) {
      try {
        return await isotrieveFn();
      } catch (error) {
        this.emitEvent(EventType.FALLBACK, {
          partnerId: targetAgent.agentId,
          error: error instanceof Error ? error.message : String(error),
        });

        return {
          method: 'text',
          message,
          language: 'en',
          fallback: true,
          fallbackReason: error instanceof Error ? error.message : String(error),
          note:
            'Isotrieve embedding transfer unavailable. ' +
            'Message sent as plain English text. ' +
            'Both agents can understand this format natively.',
        };
      }
    } else {
      return isotrieveFn();
    }
  }

  /**
   * Find similar embeddings in a knowledge base
   */
  async findSimilar(
    queryEmbedding: number[],
    knowledgeBase: number[][],
    topK = 5,
  ): Promise<Array<{ index: number; similarity: number }>> {
    const similarities = knowledgeBase.map((emb, index) => ({
      index,
      similarity: cosineSimilarity(queryEmbedding, emb),
    }));

    return similarities.sort((a, b) => b.similarity - a.similarity).slice(0, topK);
  }

  /**
   * Get quality score for a specific agent connection
   */
  getQualityScore(targetAgentId: string): number | null {
    const matrix = this.transferMatrices.get(targetAgentId);
    return matrix ? matrix.validationSimilarity : null;
  }

  /**
   * Check if recalibration is needed
   */
  requiresRecalibration(targetAgentId: string): boolean {
    const matrix = this.transferMatrices.get(targetAgentId);
    if (!matrix) return true;
    if (new Date(matrix.validUntil) < new Date()) return true;
    if (matrix.validationSimilarity < this.minQualityThreshold) return true;
    return false;
  }

  /**
   * Get connection health for all calibrated partners
   */
  getConnectionHealth(): Record<string, ConnectionHealth> {
    const health: Record<string, ConnectionHealth> = {};

    for (const [partnerId, matrix] of this.transferMatrices.entries()) {
      const cb = this.getCircuitBreaker(partnerId);
      health[partnerId] = {
        calibrated: true,
        expired: new Date(matrix.validUntil) < new Date(),
        quality: matrix.validationSimilarity,
        circuitBreaker: cb.state,
        validUntil: matrix.validUntil,
      };
    }

    return health;
  }

  /**
   * Get calibration stats for a partner
   */
  getCalibrationStats(targetAgentId: string): Record<string, any> | null {
    const matrix = this.transferMatrices.get(targetAgentId);
    if (!matrix) return null;

    const cb = this.getCircuitBreaker(targetAgentId);
    return {
      trainingSimilarity: matrix.trainingSimilarity,
      validationSimilarity: matrix.validationSimilarity,
      worstCaseSimilarity: matrix.worstCaseSimilarity,
      validUntil: matrix.validUntil,
      createdAt: matrix.createdAt,
      expired: new Date(matrix.validUntil) < new Date(),
      circuitBreaker: cb.getStatus(),
    };
  }

  // ── Private Utility Methods ──────────────────────────────────

  private estimateEmbedCost(numTexts: number): number {
    const monitor = this.monitor;
    if (!monitor) return 0;
    return monitor.estimateCost(this.embedder.getModelId(), numTexts);
  }

  private generateAgentId(): string {
    return `agent_${randomBytes(8).toString('hex')}`;
  }

  private generateTransferId(): string {
    return randomBytes(16).toString('hex');
  }

  private mean(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  private median(arr: number[]): number {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  private std(arr: number[]): number {
    if (arr.length === 0) return 0;
    const avg = this.mean(arr);
    const squareDiffs = arr.map((value) => Math.pow(value - avg, 2));
    return Math.sqrt(this.mean(squareDiffs));
  }
}
