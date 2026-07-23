/**
 * Isotrieve Auto-Negotiation Module
 *
 * Enables automatic protocol negotiation between agents.
 * If both agents support Isotrieve, they will use it automatically.
 * Otherwise, they fall back to plain English text communication.
 *
 * Key features:
 * - Auto-detects Isotrieve support on both sides
 * - Graceful English text fallback (default language)
 * - Circuit breaker integration
 * - Debug monitor integration
 * - Never throws exceptions - always returns a valid result
 */

import { Isotrieve } from './protocol';
import { CalibrationResult, CalibrationConfig, MessageResult } from './types';
import { DebugMonitor, EventType } from './debug';

/**
 * Represents the negotiated communication method between two agents
 */
export interface CommunicationMethod {
  usesIsotrieve: boolean;
  agent1Supports: boolean;
  agent2Supports: boolean;
  calibrationResult?: CalibrationResult;
  fallbackReason?: string;
  fallbackLanguage: string;
  negotiationTimeMs: number;
}

/**
 * Handles automatic Isotrieve negotiation between agents
 *
 * This class wraps agent communication and automatically:
 * 1. Detects if both agents support Isotrieve
 * 2. Calibrates if both support it
 * 3. Falls back to English text if one doesn't support it
 * 4. Provides clear feedback about the negotiation
 * 5. NEVER throws exceptions
 *
 * @example
 * ```typescript
 * import { Isotrieve, IsotrieveNegotiator } from '@isotrieve/core';
 *
 * // Agent 1 with Isotrieve support
 * const agent1 = new Isotrieve({ embedder: mockEmbedder1 });
 *
 * // Agent 2 without Isotrieve (just an object or null)
 * const agent2 = null;
 *
 * // Auto-negotiate - will fall back to English
 * const method = await IsotrieveNegotiator.negotiate(agent1, agent2);
 * // Output: " Agent 2 does not support Isotrieve. Using English text communication (default)."
 *
 * // Both with Isotrieve
 * const agent2Isotrieve = new Isotrieve({ embedder: mockEmbedder2 });
 * const method2 = await IsotrieveNegotiator.negotiate(agent1, agent2Isotrieve);
 * // Output: "✓ Isotrieve enabled with 97.3% semantic fidelity"
 * ```
 */
export class IsotrieveNegotiator {
  /**
   * Check if an agent supports Isotrieve
   *
   * Checks for:
   * 1. Instance of Isotrieve class
   * 2. Has required Isotrieve methods (duck typing)
   * 3. Has supportsIsotrieve flag
   */
  static isIsotrieveAgent(agent: any): agent is Isotrieve {
    if (agent == null) return false;

    // Direct instance check
    if (agent instanceof Isotrieve) return true;

    // Duck typing check
    const requiredMethods = ['calibrateWith', 'transferTo', 'embed', 'getCapabilities'];
    const hasMethods = requiredMethods.every(
      (m) => typeof agent[m] === 'function',
    );
    if (hasMethods) return true;

    // Explicit flag
    if (agent.supportsIsotrieve === true) return true;

    return false;
  }

  /**
   * Automatically negotiate communication method between two agents
   *
   * This method NEVER throws. It always returns a valid CommunicationMethod.
   * If Isotrieve is unavailable for any reason, it gracefully falls back to
   * English text communication.
   *
   * @param agent1 - First agent
   * @param agent2 - Second agent
   * @param options - Negotiation options
   * @returns Communication method describing the negotiated approach
   */
  static async negotiate(
    agent1: Isotrieve | any,
    agent2: Isotrieve | any,
    options: {
      autoCalibrate?: boolean;
      verbose?: boolean;
      fallbackLanguage?: string;
      calibrationConfig?: CalibrationConfig;
    } = {},
  ): Promise<CommunicationMethod> {
    const {
      autoCalibrate = true,
      verbose = true,
      fallbackLanguage = 'en',
      calibrationConfig = {},
    } = options;

    const startTime = Date.now();

    const agent1Supports = this.isIsotrieveAgent(agent1);
    const agent2Supports = this.isIsotrieveAgent(agent2);

    const makeResult = (
      partial: Omit<CommunicationMethod, 'fallbackLanguage' | 'negotiationTimeMs'>,
    ): CommunicationMethod => {
      const result: CommunicationMethod = {
        ...partial,
        fallbackLanguage,
        negotiationTimeMs: Date.now() - startTime,
      };

      // Emit debug event
      const monitor = DebugMonitor.getGlobal();
      if (monitor) {
        const agentId = agent1Supports
          ? (agent1 as Isotrieve).getAgentId()
          : 'agent1';
        const partnerId = agent2Supports
          ? (agent2 as Isotrieve).getAgentId()
          : 'agent2';

        monitor.logEvent({
          eventType: result.usesIsotrieve ? EventType.NEGOTIATION : EventType.FALLBACK,
          timestamp: Date.now(),
          agentId,
          partnerId,
          durationMs: result.negotiationTimeMs,
          qualityScore: result.calibrationResult?.qualityMetrics?.meanSimilarity,
          metadata: {
            usesIsotrieve: result.usesIsotrieve,
            fallbackReason: result.fallbackReason,
          },
        });
      }

      return result;
    };

    // Case 1: Both support Isotrieve
    if (agent1Supports && agent2Supports) {
      if (autoCalibrate) {
        if (verbose) {
          console.log('\n Both agents support Isotrieve. Calibrating...');
        }

        try {
          const calibrationResult = await (agent1 as Isotrieve).calibrateWith(
            agent2 as Isotrieve,
            calibrationConfig,
          );

          if (calibrationResult.success) {
            const quality = calibrationResult.qualityMetrics.meanSimilarity;
            if (verbose) {
              console.log(
                `✓ Isotrieve enabled with ${(quality * 100).toFixed(1)}% semantic fidelity`,
              );
            }

            return makeResult({
              usesIsotrieve: true,
              agent1Supports: true,
              agent2Supports: true,
              calibrationResult,
            });
          } else {
            const reason = `Calibration failed: ${calibrationResult.errorMessage || 'Unknown error'}`;
            if (verbose) {
              console.log(
                `⚠️  ${reason}. Falling back to English text communication.`,
              );
            }
            return makeResult({
              usesIsotrieve: false,
              agent1Supports: true,
              agent2Supports: true,
              fallbackReason: reason,
            });
          }
        } catch (error) {
          const reason = `Calibration error: ${error instanceof Error ? error.message : error}`;
          if (verbose) {
            console.log(
              `⚠️  ${reason}. Falling back to English text communication.`,
            );
          }
          return makeResult({
            usesIsotrieve: false,
            agent1Supports: true,
            agent2Supports: true,
            fallbackReason: reason,
          });
        }
      } else {
        if (verbose) {
          console.log(' Both agents support Isotrieve (auto-calibrate disabled)');
        }

        return makeResult({
          usesIsotrieve: false,
          agent1Supports: true,
          agent2Supports: true,
          fallbackReason: 'Auto-calibration disabled',
        });
      }
    }

    // Case 2: Only one or neither supports Isotrieve
    else {
      const missing: string[] = [];
      if (!agent1Supports) missing.push('Agent 1');
      if (!agent2Supports) missing.push('Agent 2');

      const reason = `${missing.join(' and ')} ${missing.length === 1 ? 'does' : 'do'} not support Isotrieve`;

      if (verbose) {
        console.log(
          `\n ${reason}. Using English text communication (default).`,
        );
        if (!agent1Supports && !agent2Supports) {
          console.log(
            '   Both agents will communicate in plain English. ' +
            'No embedding transfer needed.',
          );
        } else {
          const supporting = agent1Supports ? 'Agent 1' : 'Agent 2';
          console.log(
            `   ${supporting} supports Isotrieve but the other doesn't. ` +
            'Using English text for compatibility.',
          );
        }
      }

      return makeResult({
        usesIsotrieve: false,
        agent1Supports,
        agent2Supports,
        fallbackReason: reason,
      });
    }
  }

  /**
   * Send a message from one agent to another using the best available method
   *
   * If Isotrieve is available, sends as embedding transfer.
   * If not, sends as plain English text.
   *
   * This method NEVER throws - it always returns a valid result.
   */
  static async sendMessage(
    sender: Isotrieve | any,
    receiver: Isotrieve | any,
    message: string,
    method?: CommunicationMethod,
    options: { verbose?: boolean; fallbackLanguage?: string } = {},
  ): Promise<MessageResult> {
    const { verbose = false, fallbackLanguage = 'en' } = options;

    // Auto-negotiate if method not provided
    if (!method) {
      method = await this.negotiate(sender, receiver, { verbose });
    }

    // Use Isotrieve if available
    if (method.usesIsotrieve && this.isIsotrieveAgent(sender) && this.isIsotrieveAgent(receiver)) {
      try {
        // Try using sendMessage which has built-in fallback
        if (typeof sender.sendMessage === 'function') {
          return await sender.sendMessage(receiver, message);
        }

        const embedding = await sender.embed(message);
        const transfer = await sender.transferTo(receiver, embedding);

        return {
          method: 'isotrieve',
          transferId: transfer.transferId,
          embedding: transfer.embedding,
          sourceAgent: transfer.sourceAgent,
          targetAgent: transfer.targetAgent,
          expectedSimilarity: transfer.expectedSimilarity,
          timestamp: transfer.timestamp,
        };
      } catch (error) {
        // Isotrieve failed, fall back to text
        return {
          method: 'text',
          message,
          language: fallbackLanguage,
          fallback: true,
          fallbackReason: `Isotrieve transfer failed: ${error instanceof Error ? error.message : error}`,
          note:
            'Isotrieve embedding transfer failed. ' +
            'Message sent as plain English text instead.',
        };
      }
    } else {
      // Fall back to plain English text
      return {
        method: 'text',
        message,
        language: method.fallbackLanguage || fallbackLanguage,
        fallbackReason: method.fallbackReason,
        note:
          'Message sent as plain English text. ' +
          'Both agents can understand this format.',
      };
    }
  }

  /**
   * Send multiple messages efficiently
   */
  static async batchSend(
    sender: Isotrieve | any,
    receiver: Isotrieve | any,
    messages: string[],
    method?: CommunicationMethod,
    options: { verbose?: boolean } = {},
  ): Promise<MessageResult[]> {
    if (!method) {
      method = await this.negotiate(sender, receiver, {
        verbose: options.verbose ?? false,
      });
    }

    const results: MessageResult[] = [];
    for (const msg of messages) {
      const result = await this.sendMessage(sender, receiver, msg, method, {
        verbose: false,
      });
      results.push(result);
    }

    return results;
  }
}

/**
 * Enable Isotrieve support for an agent
 *
 * If the agent already supports Isotrieve, returns it unchanged.
 * Otherwise, wraps it in an Isotrieve instance.
 */
export function enableIsotrieveForAgent(
  agent: any,
  embedder?: any,
  agentId?: string,
): Isotrieve {
  if (agent instanceof Isotrieve) {
    return agent;
  }

  if (!embedder) {
    throw new Error(
      'Cannot enable Isotrieve: agent does not support Isotrieve and no embedder provided. ' +
      'Please provide an embedder to enable Isotrieve.',
    );
  }

  return new Isotrieve({ embedder, agentId });
}
