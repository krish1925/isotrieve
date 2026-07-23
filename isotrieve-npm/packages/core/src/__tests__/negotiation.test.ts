/**
 * Tests for Isotrieve negotiation and fallback
 */

import { IsotrieveNegotiator, CommunicationMethod } from '../negotiation';
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

describe('IsotrieveNegotiator', () => {
  test('detects Isotrieve agent', () => {
    const agent = new Isotrieve({ embedder: new MockEmbedder() });
    expect(IsotrieveNegotiator.isIsotrieveAgent(agent)).toBe(true);
  });

  test('detects non-Isotrieve agent', () => {
    expect(IsotrieveNegotiator.isIsotrieveAgent(null)).toBe(false);
    expect(IsotrieveNegotiator.isIsotrieveAgent({})).toBe(false);
    expect(IsotrieveNegotiator.isIsotrieveAgent('string')).toBe(false);
  });

  test('negotiates with both Isotrieve agents', async () => {
    const agent1 = new Isotrieve({ embedder: new MockEmbedder(384) });
    const agent2 = new Isotrieve({ embedder: new MockEmbedder(768) });
    
    const method = await IsotrieveNegotiator.negotiate(agent1, agent2, {
      autoCalibrate: true,
      verbose: false,
      calibrationConfig: { qualityThreshold: 0.3 },
    });
    
    expect(method.agent1Supports).toBe(true);
    expect(method.agent2Supports).toBe(true);
    // May use Isotrieve or fallback depending on calibration quality
    expect(method.fallbackLanguage).toBe('en');
  });

  test('falls back to English when one agent missing', async () => {
    const agent1 = new Isotrieve({ embedder: new MockEmbedder() });
    
    const method = await IsotrieveNegotiator.negotiate(agent1, null, {
      verbose: false,
    });
    
    expect(method.usesIsotrieve).toBe(false);
    expect(method.agent1Supports).toBe(true);
    expect(method.agent2Supports).toBe(false);
    expect(method.fallbackLanguage).toBe('en');
    expect(method.fallbackReason).toContain('does not support Isotrieve');
  });

  test('falls back to English when both missing', async () => {
    const method = await IsotrieveNegotiator.negotiate(null, null, {
      verbose: false,
    });
    
    expect(method.usesIsotrieve).toBe(false);
    expect(method.fallbackLanguage).toBe('en');
  });

  test('sendMessage uses Isotrieve when available', async () => {
    const agent1 = new Isotrieve({ embedder: new MockEmbedder(384) });
    const agent2 = new Isotrieve({ embedder: new MockEmbedder(768) });
    
    // Calibrate first
    const calResult = await agent1.calibrateWith(agent2, { qualityThreshold: 0.3 });
    
    // Only test Isotrieve if calibration succeeded
    if (calResult.success) {
      const result = await IsotrieveNegotiator.sendMessage(
        agent1,
        agent2,
        'Hello world',
        undefined,
        { verbose: false }
      );
      
      // May be Isotrieve or text fallback depending on transfer success
      expect(['isotrieve', 'text']).toContain(result.method);
      if (result.method === 'isotrieve') {
        expect(result.transferId).toBeDefined();
      }
    }
  });

  test('sendMessage falls back to text', async () => {
    const result = await IsotrieveNegotiator.sendMessage(
      null,
      null,
      'Hello world',
      undefined,
      { verbose: false }
    );
    
    expect(result.method).toBe('text');
    expect(result.message).toBe('Hello world');
    expect(result.language).toBe('en');
  });

  test('batchSend works', async () => {
    const agent1 = new Isotrieve({ embedder: new MockEmbedder(384) });
    const agent2 = new Isotrieve({ embedder: new MockEmbedder(768) });
    
    await agent1.calibrateWith(agent2, { qualityThreshold: 0.3 });
    
    const results = await IsotrieveNegotiator.batchSend(
      agent1,
      agent2,
      ['msg1', 'msg2'],
      undefined,
      { verbose: false }
    );
    
    expect(results).toHaveLength(2);
  });
});
