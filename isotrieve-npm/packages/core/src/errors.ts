/**
 * Isotrieve Error Handling & Resilience
 *
 * Comprehensive error hierarchy and resilience patterns:
 * - Custom error types for all failure modes
 * - Circuit breaker for failing connections
 * - Retry policies with exponential backoff
 * - Graceful degradation to English text
 */

// ── Error Hierarchy ──────────────────────────────────────────────

export class IsotrieveError extends Error {
  readonly recoverable: boolean;
  readonly context: Record<string, any>;

  constructor(message: string, recoverable = true, context: Record<string, any> = {}) {
    super(message);
    this.name = 'IsotrieveError';
    this.recoverable = recoverable;
    this.context = context;
  }
}

export class CalibrationError extends IsotrieveError {
  constructor(message: string, context: Record<string, any> = {}) {
    super(message, true, context);
    this.name = 'CalibrationError';
  }
}

export class TransferError extends IsotrieveError {
  constructor(message: string, recoverable = true, context: Record<string, any> = {}) {
    super(message, recoverable, context);
    this.name = 'TransferError';
  }
}

export class NegotiationError extends IsotrieveError {
  constructor(message: string, context: Record<string, any> = {}) {
    super(message, true, context);
    this.name = 'NegotiationError';
  }
}

export class AdapterError extends IsotrieveError {
  constructor(message: string, context: Record<string, any> = {}) {
    super(message, true, context);
    this.name = 'AdapterError';
  }
}

export class MatrixExpiredError extends TransferError {
  constructor(agentPair: string, expiredAt: string) {
    super(
      `Transfer matrix for ${agentPair} expired at ${expiredAt}. Recalibration required.`,
      true,
      { agentPair, expiredAt },
    );
    this.name = 'MatrixExpiredError';
  }
}

export class AgentNotCalibratedError extends TransferError {
  constructor(source: string, target: string) {
    super(
      `No calibration found between '${source}' and '${target}'. Call calibrateWith() first.`,
      true,
      { source, target },
    );
    this.name = 'AgentNotCalibratedError';
  }
}

export class CircuitOpenError extends IsotrieveError {
  constructor(agentId: string, failures: number, resetAt: number) {
    const resetStr = new Date(resetAt).toISOString().substr(11, 8);
    super(
      `Circuit breaker OPEN for agent '${agentId}' after ${failures} failures. ` +
      `Will retry at ${resetStr}. Falling back to English text.`,
      true,
      { agentId, failures, resetAt },
    );
    this.name = 'CircuitOpenError';
  }
}

// ── Circuit Breaker ──────────────────────────────────────────────

export enum CircuitState {
  CLOSED = 'closed',
  OPEN = 'open',
  HALF_OPEN = 'half_open',
}

export interface CircuitBreakerConfig {
  failureThreshold?: number;
  resetTimeout?: number;
  halfOpenMaxCalls?: number;
}

export class CircuitBreaker {
  readonly failureThreshold: number;
  readonly resetTimeout: number;
  readonly halfOpenMaxCalls: number;

  private _state: CircuitState = CircuitState.CLOSED;
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime = 0;
  private halfOpenCalls = 0;
  agentId = 'unknown';

  constructor(config: CircuitBreakerConfig = {}) {
    this.failureThreshold = config.failureThreshold ?? 5;
    this.resetTimeout = config.resetTimeout ?? 60000; // ms
    this.halfOpenMaxCalls = config.halfOpenMaxCalls ?? 1;
  }

  get state(): CircuitState {
    if (this._state === CircuitState.OPEN) {
      if (Date.now() - this.lastFailureTime >= this.resetTimeout) {
        this._state = CircuitState.HALF_OPEN;
        this.halfOpenCalls = 0;
      }
    }
    return this._state;
  }

  get isClosed(): boolean {
    return this.state === CircuitState.CLOSED;
  }

  get isOpen(): boolean {
    return this.state === CircuitState.OPEN;
  }

  async call<T>(fn: () => T | Promise<T>): Promise<T> {
    const currentState = this.state;

    if (currentState === CircuitState.OPEN) {
      throw new CircuitOpenError(
        this.agentId,
        this.failureCount,
        this.lastFailureTime + this.resetTimeout,
      );
    }

    if (currentState === CircuitState.HALF_OPEN) {
      if (this.halfOpenCalls >= this.halfOpenMaxCalls) {
        throw new CircuitOpenError(
          this.agentId,
          this.failureCount,
          this.lastFailureTime + this.resetTimeout,
        );
      }
      this.halfOpenCalls++;
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failureCount = 0;
    this.successCount++;
    if (this._state === CircuitState.HALF_OPEN) {
      this._state = CircuitState.CLOSED;
    }
  }

  private onFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();

    if (this._state === CircuitState.HALF_OPEN) {
      this._state = CircuitState.OPEN;
    } else if (this.failureCount >= this.failureThreshold) {
      this._state = CircuitState.OPEN;
    }
  }

  reset(): void {
    this._state = CircuitState.CLOSED;
    this.failureCount = 0;
    this.successCount = 0;
    this.halfOpenCalls = 0;
  }

  getStatus(): Record<string, any> {
    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      agentId: this.agentId,
      failureThreshold: this.failureThreshold,
      resetTimeout: this.resetTimeout,
    };
  }
}

// ── Retry Policy ──────────────────────────────────────────────────

export interface RetryPolicyConfig {
  maxRetries?: number;
  baseDelay?: number;
  maxDelay?: number;
  exponentialBase?: number;
  jitter?: boolean;
}

export class RetryPolicy {
  readonly maxRetries: number;
  readonly baseDelay: number;
  readonly maxDelay: number;
  readonly exponentialBase: number;
  readonly jitter: boolean;

  constructor(config: RetryPolicyConfig = {}) {
    this.maxRetries = config.maxRetries ?? 3;
    this.baseDelay = config.baseDelay ?? 1000;
    this.maxDelay = config.maxDelay ?? 30000;
    this.exponentialBase = config.exponentialBase ?? 2;
    this.jitter = config.jitter ?? true;
  }

  async execute<T>(fn: () => T | Promise<T>): Promise<T> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        if (attempt < this.maxRetries) {
          let delay = Math.min(
            this.baseDelay * Math.pow(this.exponentialBase, attempt),
            this.maxDelay,
          );

          if (this.jitter) {
            delay *= 0.5 + Math.random();
          }

          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError!;
  }
}

// ── Graceful Degradation ──────────────────────────────────────────

export interface TextFallbackResult {
  method: 'text';
  message: string;
  language: string;
  encoding: string;
  fallback: true;
  note: string;
  metadata?: Record<string, any>;
}

export interface IsotrieveResult {
  method: 'isotrieve';
  data: any;
  language: null;
}

export type GracefulResult = TextFallbackResult | IsotrieveResult;

export class GracefulDegradation {
  readonly defaultLanguage: string;
  private isotrieveCount = 0;
  private fallbackCount = 0;

  constructor(defaultLanguage = 'en') {
    this.defaultLanguage = defaultLanguage;
  }

  async send(
    message: string,
    isotrieveFn?: () => any | Promise<any>,
    fallbackMetadata?: Record<string, any>,
  ): Promise<GracefulResult> {
    if (isotrieveFn) {
      try {
        const result = await isotrieveFn();
        this.isotrieveCount++;
        return { method: 'isotrieve', data: result, language: null };
      } catch {
        // Fall through to text fallback
      }
    }

    this.fallbackCount++;
    return this.createTextFallback(message, fallbackMetadata);
  }

  private createTextFallback(
    message: string,
    metadata?: Record<string, any>,
  ): TextFallbackResult {
    return {
      method: 'text',
      message,
      language: this.defaultLanguage,
      encoding: 'utf-8',
      fallback: true,
      note:
        'Isotrieve embedding transfer unavailable. ' +
        'Message sent as plain English text. ' +
        'Both agents can understand this format natively.',
      ...(metadata ? { metadata } : {}),
    };
  }

  getStats(): Record<string, any> {
    const total = this.isotrieveCount + this.fallbackCount;
    return {
      isotrieveCount: this.isotrieveCount,
      fallbackCount: this.fallbackCount,
      total,
      isotrievePercentage: total > 0 ? (this.isotrieveCount / total) * 100 : 0,
      fallbackPercentage: total > 0 ? (this.fallbackCount / total) * 100 : 0,
    };
  }
}
