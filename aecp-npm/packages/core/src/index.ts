/**
 * AECP Core Package
 * Agent Embedding Communication Protocol
 *
 * Production-ready library for semantic transfer between AI agents
 * using different embedding models.
 *
 * Key features:
 * - Automatic protocol negotiation
 * - Graceful English text fallback
 * - Circuit breaker for fault tolerance
 * - Real-time debug monitoring & cost tracking
 * - Comprehensive error handling
 */

// Protocol
export { AECP } from './protocol';

// Types
export * from './types';

// Matrix operations
export * from './matrix';

// Vocabulary
export { DEFAULT_VOCABULARY, generateExtendedVocabulary } from './vocabulary';

// Plugin system
export * from './plugin';
export * from './communication';

// Auto-negotiation
export { AECPNegotiator, enableAECPForAgent } from './negotiation';
export type { CommunicationMethod } from './negotiation';

// Debug monitoring
export {
  DebugMonitor,
  DebugLevel,
  EventType,
  enableDebug,
  disableDebug,
  getMonitor,
} from './debug';
export type { DebugEvent, CostConfig } from './debug';

// Error handling & resilience
export {
  AECPError,
  CalibrationError,
  TransferError,
  NegotiationError,
  AdapterError,
  MatrixExpiredError,
  AgentNotCalibratedError,
  CircuitOpenError,
  CircuitBreaker,
  CircuitState,
  RetryPolicy,
  GracefulDegradation,
} from './errors';
export type {
  CircuitBreakerConfig,
  RetryPolicyConfig,
  TextFallbackResult,
  AECPResult,
  GracefulResult,
} from './errors';
