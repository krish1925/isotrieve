/**
 * Isotrieve Debug Monitor & Dashboard
 *
 * Real-time monitoring of Isotrieve operations including:
 * - Transfer quality metrics
 * - Cost tracking and savings estimation
 * - Time savings vs text-based communication
 * - Error rates and circuit breaker status
 * - Live operation logging
 *
 * @example
 * ```typescript
 * import { enableDebug, DebugMonitor } from '@isotrieve/core';
 *
 * // Enable global debug mode
 * const monitor = enableDebug('detailed');
 *
 * // ... run Isotrieve operations ...
 *
 * // View dashboard
 * monitor.dashboard();
 * monitor.costReport();
 * monitor.qualityReport();
 * ```
 */

export enum DebugLevel {
  OFF = 0,
  MINIMAL = 1,
  STANDARD = 2,
  DETAILED = 3,
  VERBOSE = 4,
}

export enum EventType {
  HANDSHAKE = 'handshake',
  CALIBRATION_START = 'calibration_start',
  CALIBRATION_END = 'calibration_end',
  TRANSFER = 'transfer',
  RECEIVE = 'receive',
  EMBED = 'embed',
  EMBED_BATCH = 'embed_batch',
  NEGOTIATION = 'negotiation',
  FALLBACK = 'fallback',
  ERROR = 'error',
  WARNING = 'warning',
  CIRCUIT_BREAK = 'circuit_break',
  RETRY = 'retry',
  RECOVERY = 'recovery',
}

export interface DebugEvent {
  eventType: EventType;
  timestamp: number;
  agentId: string;
  partnerId?: string;
  durationMs?: number;
  qualityScore?: number;
  costEstimate?: number;
  tokensUsed?: number;
  error?: string;
  metadata?: Record<string, any>;
}

export interface CostConfig {
  /** Cost per 1M tokens for each provider */
  providerCosts: Record<string, number>;
  /** Average tokens per text */
  avgTokensPerText: number;
  /** Text-based communication cost per message (for comparison) */
  textCommCostPerMsg: number;
}

interface AgentStats {
  embeds: number;
  transfersSent: number;
  transfersReceived: number;
  errors: number;
  totalTimeMs: number;
  totalCost: number;
  avgQuality: number;
  qualitySamples: number;
}

const DEFAULT_COST_CONFIG: CostConfig = {
  providerCosts: {
    'openai:text-embedding-3-small': 0.02,
    'openai:text-embedding-3-large': 0.13,
    'openai:text-embedding-ada-002': 0.10,
    'voyage:voyage-2': 0.10,
    'voyage:voyage-large-2': 0.12,
    'voyage:voyage-code-2': 0.12,
    'cohere:embed-english-v3.0': 0.10,
    'cohere:embed-multilingual-v3.0': 0.10,
    'huggingface:*': 0.0,
    'mock:*': 0.0,
  },
  avgTokensPerText: 10,
  textCommCostPerMsg: 0.003,
};

export class DebugMonitor {
  private static globalInstance: DebugMonitor | null = null;

  readonly level: DebugLevel;
  private costConfig: CostConfig;
  private maxEvents: number;
  private logToConsole: boolean;

  private events: DebugEvent[] = [];
  private stats = {
    totalEvents: 0,
    totalEmbeds: 0,
    totalTransfers: 0,
    totalCalibrations: 0,
    totalNegotiations: 0,
    totalFallbacks: 0,
    totalErrors: 0,
    totalRetries: 0,
    totalTokens: 0,
    totalCost: 0,
    totalTimeMs: 0,
    estimatedTextCost: 0,
  };

  private agentStats: Map<string, AgentStats> = new Map();
  private pairQuality: Map<string, number[]> = new Map();
  private sessionStart: number;

  constructor(options: {
    level?: DebugLevel;
    costConfig?: Partial<CostConfig>;
    maxEvents?: number;
    logToConsole?: boolean;
  } = {}) {
    this.level = options.level ?? DebugLevel.STANDARD;
    this.costConfig = { ...DEFAULT_COST_CONFIG, ...options.costConfig };
    this.maxEvents = options.maxEvents ?? 10000;
    this.logToConsole = options.logToConsole ?? true;
    this.sessionStart = Date.now();
  }

  static getGlobal(): DebugMonitor | null {
    return this.globalInstance;
  }

  static setGlobal(monitor: DebugMonitor | null): void {
    this.globalInstance = monitor;
  }

  logEvent(event: DebugEvent): void {
    // Store event
    if (this.events.length >= this.maxEvents) {
      this.events.shift();
    }
    this.events.push(event);

    // Update stats
    this.stats.totalEvents++;

    switch (event.eventType) {
      case EventType.EMBED:
        this.stats.totalEmbeds++;
        break;
      case EventType.EMBED_BATCH:
        this.stats.totalEmbeds += event.metadata?.batchSize ?? 1;
        break;
      case EventType.TRANSFER:
        this.stats.totalTransfers++;
        break;
      case EventType.CALIBRATION_END:
        this.stats.totalCalibrations++;
        break;
      case EventType.NEGOTIATION:
        this.stats.totalNegotiations++;
        break;
      case EventType.FALLBACK:
        this.stats.totalFallbacks++;
        break;
      case EventType.ERROR:
        this.stats.totalErrors++;
        break;
      case EventType.RETRY:
        this.stats.totalRetries++;
        break;
    }

    if (event.durationMs) this.stats.totalTimeMs += event.durationMs;
    if (event.tokensUsed) this.stats.totalTokens += event.tokensUsed;
    if (event.costEstimate) this.stats.totalCost += event.costEstimate;

    // Per-agent stats
    this.updateAgentStats(event);

    // Console output
    if (this.logToConsole && this.level >= DebugLevel.STANDARD) {
      this.printEvent(event);
    }
  }

  private updateAgentStats(event: DebugEvent): void {
    if (!this.agentStats.has(event.agentId)) {
      this.agentStats.set(event.agentId, {
        embeds: 0, transfersSent: 0, transfersReceived: 0,
        errors: 0, totalTimeMs: 0, totalCost: 0,
        avgQuality: 0, qualitySamples: 0,
      });
    }

    const stats = this.agentStats.get(event.agentId)!;

    if (event.eventType === EventType.EMBED || event.eventType === EventType.EMBED_BATCH) {
      stats.embeds += event.metadata?.batchSize ?? 1;
    } else if (event.eventType === EventType.TRANSFER) {
      stats.transfersSent++;
      if (event.partnerId) {
        if (!this.agentStats.has(event.partnerId)) {
          this.agentStats.set(event.partnerId, {
            embeds: 0, transfersSent: 0, transfersReceived: 0,
            errors: 0, totalTimeMs: 0, totalCost: 0,
            avgQuality: 0, qualitySamples: 0,
          });
        }
        this.agentStats.get(event.partnerId)!.transfersReceived++;
      }
    } else if (event.eventType === EventType.ERROR) {
      stats.errors++;
    }

    if (event.durationMs) stats.totalTimeMs += event.durationMs;
    if (event.costEstimate) stats.totalCost += event.costEstimate;

    if (event.qualityScore != null) {
      const n = stats.qualitySamples;
      stats.avgQuality = (stats.avgQuality * n + event.qualityScore) / (n + 1);
      stats.qualitySamples++;

      if (event.partnerId) {
        const pairKey = `${event.agentId}<->${event.partnerId}`;
        if (!this.pairQuality.has(pairKey)) {
          this.pairQuality.set(pairKey, []);
        }
        this.pairQuality.get(pairKey)!.push(event.qualityScore);
      }
    }
  }

  estimateCost(modelId: string, numTexts: number = 1): number {
    const tokens = numTexts * this.costConfig.avgTokensPerText;
    const costPer1M = this.getCostPer1MTokens(modelId);
    return (tokens * costPer1M) / 1_000_000;
  }

  private getCostPer1MTokens(modelId: string): number {
    if (this.costConfig.providerCosts[modelId] != null) {
      return this.costConfig.providerCosts[modelId];
    }
    const provider = modelId.split(':')[0];
    const wildcard = `${provider}:*`;
    if (this.costConfig.providerCosts[wildcard] != null) {
      return this.costConfig.providerCosts[wildcard];
    }
    return 0.10; // default
  }

  estimateTextCommCost(numMessages: number = 1): number {
    return numMessages * this.costConfig.textCommCostPerMsg;
  }

  // ── Dashboard & Reports ──────────────────────────────────────

  dashboard(compact: boolean = false): string {
    const elapsed = Date.now() - this.sessionStart;
    const elapsedStr = this.formatDuration(elapsed);

    if (compact) {
      const s = this.stats;
      const line =
        `[Isotrieve Monitor] ⏱ ${elapsedStr} | ` +
        ` ${s.totalTransfers} transfers | ` +
        ` $${s.totalCost.toFixed(6)} spent | ` +
        ` ${s.totalErrors} errors | ` +
        ` ${s.totalFallbacks} fallbacks`;
      console.log(line);
      return line;
    }

    const s = this.stats;
    const textCost = this.estimateTextCommCost(s.totalTransfers + s.totalEmbeds);
    const savings = Math.max(0, textCost - s.totalCost);
    const savingsPct = textCost > 0 ? (savings / textCost) * 100 : 0;
    const totalOps = s.totalEmbeds + s.totalTransfers;
    const avgTime = totalOps > 0 ? s.totalTimeMs / totalOps : 0;
    const errorRate = s.totalEvents > 0 ? (s.totalErrors / s.totalEvents) * 100 : 0;

    const lines: string[] = [];
    lines.push('');
    lines.push('╔══════════════════════════════════════════════════════════════╗');
    lines.push('║              Isotrieve Debug Monitor Dashboard                   ║');
    lines.push('╠══════════════════════════════════════════════════════════════╣');
    lines.push(`║  Session Duration: ${elapsedStr.padStart(38)}  ║`);
    lines.push(`║  Debug Level:      ${DebugLevel[this.level].padStart(38)}  ║`);
    lines.push('╠══════════════════════════════════════════════════════════════╣');
    lines.push('║  OPERATIONS                                                  ║');
    lines.push(`║    Total Embeddings:     ${String(s.totalEmbeds).padStart(32)}  ║`);
    lines.push(`║    Total Transfers:      ${String(s.totalTransfers).padStart(32)}  ║`);
    lines.push(`║    Total Calibrations:   ${String(s.totalCalibrations).padStart(32)}  ║`);
    lines.push(`║    Fallbacks to Text:    ${String(s.totalFallbacks).padStart(32)}  ║`);
    lines.push('╠══════════════════════════════════════════════════════════════╣');
    lines.push('║  COST ANALYSIS                                               ║');
    lines.push(`║    Isotrieve Cost:            $${s.totalCost.toFixed(6).padStart(30)}  ║`);
    lines.push(`║    Text Comm Would Cost: $${textCost.toFixed(6).padStart(30)}  ║`);
    lines.push(`║    Estimated Savings:    $${savings.toFixed(6).padStart(30)}  ║`);
    lines.push(`║    Savings Percentage:   ${(savingsPct.toFixed(1) + '%').padStart(31)}  ║`);
    lines.push('╠══════════════════════════════════════════════════════════════╣');
    lines.push('║  PERFORMANCE                                                 ║');
    lines.push(`║    Total Time:           ${(s.totalTimeMs.toFixed(1) + 'ms').padStart(31)}  ║`);
    lines.push(`║    Avg Time/Operation:   ${(avgTime.toFixed(1) + 'ms').padStart(31)}  ║`);
    lines.push('╠══════════════════════════════════════════════════════════════╣');
    lines.push('║  RELIABILITY                                                 ║');
    lines.push(`║    Errors:               ${String(s.totalErrors).padStart(32)}  ║`);
    lines.push(`║    Retries:              ${String(s.totalRetries).padStart(32)}  ║`);
    lines.push(`║    Error Rate:           ${(errorRate.toFixed(2) + '%').padStart(31)}  ║`);
    lines.push('╚══════════════════════════════════════════════════════════════╝');

    const output = lines.join('\n');
    console.log(output);
    return output;
  }

  costReport(): string {
    const s = this.stats;
    const totalOps = s.totalEmbeds + s.totalTransfers;
    const textCost = this.estimateTextCommCost(totalOps);
    const savings = Math.max(0, textCost - s.totalCost);
    const timeTextMs = totalOps * 500;
    const timeSavedMs = Math.max(0, timeTextMs - s.totalTimeMs);

    const lines: string[] = [];
    lines.push('\n Isotrieve Cost Report');
    lines.push('='.repeat(50));
    lines.push(`  Total operations:        ${totalOps.toLocaleString()}`);
    lines.push(`  Total tokens used:       ${s.totalTokens.toLocaleString()}`);
    lines.push(`  Total Isotrieve cost:         $${s.totalCost.toFixed(6)}`);
    lines.push(`  Text-based would cost:   $${textCost.toFixed(6)}`);
    lines.push(`  Estimated savings:       $${savings.toFixed(6)}`);
    if (textCost > 0) {
      lines.push(`  Savings percentage:      ${((savings / textCost) * 100).toFixed(1)}%`);
    }
    lines.push(`\n  ⏱ Time Analysis:`);
    lines.push(`  Isotrieve time:               ${s.totalTimeMs.toFixed(0)}ms`);
    lines.push(`  Text-based estimate:     ${timeTextMs.toFixed(0)}ms`);
    lines.push(`  Time saved:              ${timeSavedMs.toFixed(0)}ms (${(timeSavedMs / 1000).toFixed(1)}s)`);

    const output = lines.join('\n');
    console.log(output);
    return output;
  }

  qualityReport(): string {
    const lines: string[] = [];
    lines.push('\n Isotrieve Quality Report');
    lines.push('='.repeat(50));

    if (this.pairQuality.size === 0) {
      lines.push('  No quality data available yet.');
    } else {
      for (const [pair, scores] of this.pairQuality.entries()) {
        if (scores.length === 0) continue;
        const sorted = [...scores].sort((a, b) => a - b);
        const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
        const min = sorted[0];
        const max = sorted[sorted.length - 1];
        const median = sorted[Math.floor(sorted.length / 2)];

        let assessment: string;
        if (mean >= 0.95) assessment = ' Excellent';
        else if (mean >= 0.90) assessment = ' Very Good';
        else if (mean >= 0.80) assessment = ' Good';
        else if (mean >= 0.70) assessment = ' Acceptable';
        else assessment = ' Poor - Consider recalibration';

        lines.push(`\n  ${pair}:`);
        lines.push(`    Samples:    ${scores.length}`);
        lines.push(`    Mean:       ${mean.toFixed(4)}`);
        lines.push(`    Median:     ${median.toFixed(4)}`);
        lines.push(`    Min:        ${min.toFixed(4)}`);
        lines.push(`    Max:        ${max.toFixed(4)}`);
        lines.push(`    Assessment: ${assessment}`);
      }
    }

    const output = lines.join('\n');
    console.log(output);
    return output;
  }

  getStats(): Record<string, any> {
    return {
      ...this.stats,
      sessionDurationMs: Date.now() - this.sessionStart,
      agentStats: Object.fromEntries(this.agentStats),
      pairQuality: Object.fromEntries(this.pairQuality),
    };
  }

  recentEvents(n: number = 20): DebugEvent[] {
    return this.events.slice(-n);
  }

  exportJson(): string {
    return JSON.stringify({
      sessionStart: new Date(this.sessionStart).toISOString(),
      sessionDurationMs: Date.now() - this.sessionStart,
      stats: this.stats,
      agentStats: Object.fromEntries(this.agentStats),
      pairQuality: Object.fromEntries(this.pairQuality),
      recentEvents: this.recentEvents(100),
    }, null, 2);
  }

  reset(): void {
    this.events = [];
    this.stats = {
      totalEvents: 0, totalEmbeds: 0, totalTransfers: 0,
      totalCalibrations: 0, totalNegotiations: 0, totalFallbacks: 0,
      totalErrors: 0, totalRetries: 0, totalTokens: 0,
      totalCost: 0, totalTimeMs: 0, estimatedTextCost: 0,
    };
    this.agentStats.clear();
    this.pairQuality.clear();
    this.sessionStart = Date.now();
  }

  private printEvent(event: DebugEvent): void {
    const icons: Record<string, string> = {
      handshake: '', calibration_start: '', calibration_end: '',
      transfer: '', receive: '', embed: '', embed_batch: '',
      negotiation: '', fallback: '', error: '', warning: '⚠️',
      circuit_break: '', retry: '', recovery: '',
    };

    const icon = icons[event.eventType] || '';
    const time = new Date(event.timestamp).toISOString().substr(11, 12);
    const parts = [`[${time}]`, icon, event.eventType.toUpperCase()];

    if (event.agentId) parts.push(`agent=${event.agentId}`);
    if (event.partnerId) parts.push(`partner=${event.partnerId}`);
    if (event.durationMs != null) parts.push(`time=${event.durationMs.toFixed(1)}ms`);
    if (event.qualityScore != null) parts.push(`quality=${event.qualityScore.toFixed(4)}`);
    if (event.costEstimate != null) parts.push(`cost=$${event.costEstimate.toFixed(6)}`);
    if (event.error) parts.push(`error=${event.error}`);

    // Skip detailed events unless in detailed mode
    if (
      (event.eventType === EventType.EMBED || event.eventType === EventType.EMBED_BATCH) &&
      this.level < DebugLevel.DETAILED
    ) {
      return;
    }

    console.log(parts.join(' '));
  }

  private formatDuration(ms: number): string {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    const h = Math.floor(m / 60);
    return `${h}h ${m % 60}m ${s % 60}s`;
  }
}

// ── Convenience Functions ──────────────────────────────────────

export function enableDebug(
  level: DebugLevel | 'off' | 'minimal' | 'standard' | 'detailed' | 'verbose' = 'standard',
  options: { logToConsole?: boolean; costConfig?: Partial<CostConfig> } = {},
): DebugMonitor {
  const levelMap: Record<string, DebugLevel> = {
    off: DebugLevel.OFF,
    minimal: DebugLevel.MINIMAL,
    standard: DebugLevel.STANDARD,
    detailed: DebugLevel.DETAILED,
    verbose: DebugLevel.VERBOSE,
  };

  const resolvedLevel = typeof level === 'string' ? levelMap[level] ?? DebugLevel.STANDARD : level;

  const monitor = new DebugMonitor({
    level: resolvedLevel,
    logToConsole: options.logToConsole,
    costConfig: options.costConfig,
  });

  DebugMonitor.setGlobal(monitor);
  return monitor;
}

export function disableDebug(): void {
  DebugMonitor.setGlobal(null);
}

export function getMonitor(): DebugMonitor | null {
  return DebugMonitor.getGlobal();
}
