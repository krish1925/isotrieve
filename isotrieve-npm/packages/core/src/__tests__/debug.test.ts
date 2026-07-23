/**
 * Tests for DebugMonitor and debug functionality
 */

import { DebugMonitor, DebugLevel, EventType, enableDebug, disableDebug, getMonitor } from '../debug';
import { Isotrieve } from '../protocol';

class MockEmbedder {
  private dimensions: number;
  constructor(dimensions: number = 384) {
    this.dimensions = dimensions;
  }
  async embed(text: string): Promise<number[]> {
    return new Array(this.dimensions).fill(0).map(() => Math.random());
  }
  async embedBatch(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map(t => this.embed(t)));
  }
  getDimensions(): number {
    return this.dimensions;
  }
  getModelId(): string {
    return 'mock-model';
  }
}

describe('DebugMonitor', () => {
  beforeEach(() => {
    disableDebug();
  });

  test('creates monitor with default level', () => {
    const monitor = new DebugMonitor({ level: DebugLevel.STANDARD });
    expect(monitor.level).toBe(DebugLevel.STANDARD);
  });

  test('logs events correctly', () => {
    const monitor = new DebugMonitor({ level: DebugLevel.STANDARD, logToConsole: false });
    
    monitor.logEvent({
      eventType: EventType.EMBED,
      timestamp: Date.now(),
      agentId: 'test-agent',
      durationMs: 100,
    });

    const stats = monitor.getStats();
    expect(stats.totalEvents).toBe(1);
    expect(stats.totalEmbeds).toBe(1);
  });

  test('tracks cost estimates', () => {
    const monitor = new DebugMonitor({ level: DebugLevel.STANDARD, logToConsole: false });
    
    monitor.logEvent({
      eventType: EventType.EMBED,
      timestamp: Date.now(),
      agentId: 'test-agent',
      costEstimate: 0.001,
    });

    const stats = monitor.getStats();
    expect(stats.totalCost).toBe(0.001);
  });

  test('dashboard generates output', () => {
    const monitor = new DebugMonitor({ level: DebugLevel.STANDARD, logToConsole: false });
    
    monitor.logEvent({
      eventType: EventType.TRANSFER,
      timestamp: Date.now(),
      agentId: 'agent1',
      partnerId: 'agent2',
      qualityScore: 0.95,
    });

    const output = monitor.dashboard(true);
    expect(output).toContain('Isotrieve Monitor');
  });

  test('cost report generates correctly', () => {
    const monitor = new DebugMonitor({ level: DebugLevel.STANDARD, logToConsole: false });
    
    monitor.logEvent({
      eventType: EventType.EMBED,
      timestamp: Date.now(),
      agentId: 'test',
      costEstimate: 0.001,
    });

    const output = monitor.costReport();
    expect(output).toContain('Cost Report');
  });

  test('global instance works', () => {
    const monitor = enableDebug('standard');
    expect(getMonitor()).toBe(monitor);
    
    disableDebug();
    expect(getMonitor()).toBeNull();
  });

  test('estimates cost correctly', () => {
    const monitor = new DebugMonitor({ level: DebugLevel.STANDARD, logToConsole: false });
    const cost = monitor.estimateCost('openai:text-embedding-3-small', 10);
    expect(cost).toBeGreaterThan(0);
  });
});
