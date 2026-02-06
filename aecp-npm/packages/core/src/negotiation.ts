/**
 * AECP Auto-Negotiation Module
 * 
 * Enables automatic protocol negotiation between agents.
 * If both agents support AECP, they will use it automatically.
 * Otherwise, they fall back to plain text communication.
 */

import { AECP } from './protocol';
import { CalibrationResult, CalibrationConfig } from './types';

/**
 * Represents the negotiated communication method between two agents
 */
export interface CommunicationMethod {
  usesAECP: boolean;
  agent1Supports: boolean;
  agent2Supports: boolean;
  calibrationResult?: CalibrationResult;
  fallbackReason?: string;
}

/**
 * Result of sending a message
 */
export interface MessageResult {
  method: 'aecp' | 'text';
  transferId?: string;
  embedding?: number[];
  sourceAgent?: string;
  targetAgent?: string;
  expectedSimilarity?: number;
  timestamp?: string;
  message?: string;
  fallbackReason?: string;
}

/**
 * Handles automatic AECP negotiation between agents
 * 
 * This class wraps agent communication and automatically:
 * 1. Detects if both agents support AECP
 * 2. Calibrates if both support it
 * 3. Falls back to text if one doesn't support it
 * 4. Provides clear feedback about the negotiation
 * 
 * @example
 * ```typescript
 * import { AECP } from '@aecp/core';
 * import { AECPNegotiator } from '@aecp/core';
 * 
 * // Agent 1 with AECP support
 * const agent1 = new AECP({ embedder: mockEmbedder1 });
 * 
 * // Agent 2 without AECP (just an object or null)
 * const agent2 = null;
 * 
 * // Auto-negotiate
 * const method = await AECPNegotiator.negotiate(agent1, agent2);
 * // Output: "⚠️  AECP not available: Agent 2 does not support AECP. Falling back to text communication."
 * 
 * // Both with AECP
 * const agent2AECP = new AECP({ embedder: mockEmbedder2 });
 * const method2 = await AECPNegotiator.negotiate(agent1, agent2AECP);
 * // Output: "✓ AECP enabled with 97.3% semantic fidelity"
 * ```
 */
export class AECPNegotiator {
  /**
   * Check if an agent supports AECP
   */
  static isAECPAgent(agent: any): agent is AECP {
    return agent instanceof AECP;
  }

  /**
   * Automatically negotiate communication method between two agents
   * 
   * This method:
   * 1. Checks if both agents support AECP
   * 2. If yes, calibrates them and uses AECP
   * 3. If no, falls back to text and shows a warning
   * 
   * @param agent1 - First agent
   * @param agent2 - Second agent
   * @param options - Negotiation options
   * @returns Communication method describing the negotiated approach
   */
  static async negotiate(
    agent1: AECP | any,
    agent2: AECP | any,
    options: {
      autoCalibrate?: boolean;
      verbose?: boolean;
      calibrationConfig?: CalibrationConfig;
    } = {}
  ): Promise<CommunicationMethod> {
    const { autoCalibrate = true, verbose = true, calibrationConfig = {} } = options;

    const agent1Supports = this.isAECPAgent(agent1);
    const agent2Supports = this.isAECPAgent(agent2);

    // Case 1: Both support AECP
    if (agent1Supports && agent2Supports) {
      if (autoCalibrate) {
        if (verbose) {
          console.log('\n🤝 Both agents support AECP. Calibrating...');
        }

        try {
          const calibrationResult = await agent1.calibrateWith(agent2, calibrationConfig);

          if (calibrationResult.success) {
            const quality = calibrationResult.qualityMetrics.meanSimilarity;
            if (verbose) {
              console.log(`✓ AECP enabled with ${(quality * 100).toFixed(1)}% semantic fidelity`);
            }

            return {
              usesAECP: true,
              agent1Supports: true,
              agent2Supports: true,
              calibrationResult,
            };
          } else {
            if (verbose) {
              console.warn(
                '⚠️  AECP calibration failed. Falling back to text communication.'
              );
            }
            return {
              usesAECP: false,
              agent1Supports: true,
              agent2Supports: true,
              fallbackReason: 'Calibration failed',
            };
          }
        } catch (error) {
          if (verbose) {
            console.warn(
              `⚠️  AECP calibration error: ${error instanceof Error ? error.message : error}. ` +
              'Falling back to text communication.'
            );
          }
          return {
            usesAECP: false,
            agent1Supports: true,
            agent2Supports: true,
            fallbackReason: `Calibration error: ${error instanceof Error ? error.message : error}`,
          };
        }
      } else {
        if (verbose) {
          console.log('🤝 Both agents support AECP (auto-calibrate disabled)');
        }

        return {
          usesAECP: false,
          agent1Supports: true,
          agent2Supports: true,
          fallbackReason: 'Auto-calibration disabled',
        };
      }
    }

    // Case 2: Only one or neither supports AECP
    else {
      const missing: string[] = [];
      if (!agent1Supports) missing.push('Agent 1');
      if (!agent2Supports) missing.push('Agent 2');

      const reason = `${missing.join(' and ')} ${missing.length === 1 ? 'does' : 'do'} not support AECP`;

      if (verbose) {
        console.warn(
          `⚠️  AECP not available: ${reason}. Falling back to text communication.`
        );
      }

      return {
        usesAECP: false,
        agent1Supports,
        agent2Supports,
        fallbackReason: reason,
      };
    }
  }

  /**
   * Send a message from one agent to another using the negotiated method
   * 
   * @param sender - Sending agent
   * @param receiver - Receiving agent
   * @param message - Message to send
   * @param method - Pre-negotiated communication method (will auto-negotiate if not provided)
   * @returns Message result object
   */
  static async sendMessage(
    sender: AECP | any,
    receiver: AECP | any,
    message: string,
    method?: CommunicationMethod
  ): Promise<MessageResult> {
    // Auto-negotiate if method not provided
    if (!method) {
      method = await this.negotiate(sender, receiver);
    }

    // Use AECP if available
    if (method.usesAECP && this.isAECPAgent(sender) && this.isAECPAgent(receiver)) {
      const embedding = await sender.embed(message);
      const transfer = await sender.transferTo(receiver, embedding);

      return {
        method: 'aecp',
        transferId: transfer.transferId,
        embedding: transfer.embedding,
        sourceAgent: transfer.sourceAgent,
        targetAgent: transfer.targetAgent,
        expectedSimilarity: transfer.expectedSimilarity,
        timestamp: transfer.timestamp,
      };
    } else {
      // Fall back to plain text
      return {
        method: 'text',
        message,
        fallbackReason: method.fallbackReason,
      };
    }
  }
}

/**
 * Enable AECP support for an agent
 * 
 * If the agent already supports AECP, returns it unchanged.
 * Otherwise, wraps it in an AECP instance.
 * 
 * @param agent - The agent to enable AECP for
 * @param embedder - Embedding provider to use (required if agent doesn't support AECP)
 * @param agentId - Optional agent ID
 * @returns AECP-enabled agent
 * @throws Error if agent doesn't support AECP and no embedder provided
 */
export function enableAECPForAgent(
  agent: any,
  embedder?: any,
  agentId?: string
): AECP {
  if (agent instanceof AECP) {
    return agent;
  }

  if (!embedder) {
    throw new Error(
      'Cannot enable AECP: agent does not support AECP and no embedder provided'
    );
  }

  return new AECP({ embedder, agentId });
}
