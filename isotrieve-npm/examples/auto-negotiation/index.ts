/**
 * Isotrieve Auto-Negotiation Example
 * 
 * This example demonstrates how Isotrieve automatically negotiates between agents:
 * 1. When both agents support Isotrieve -> Uses Isotrieve
 * 2. When only one agent supports Isotrieve -> Falls back to text
 * 3. When neither agent supports Isotrieve -> Uses text
 */

import { Isotrieve, IsotrieveNegotiator } from '@isotrieve/core';

// Mock embedder for demonstration
class MockEmbedder {
  constructor(private dimensions: number) {}

  getModelId(): string {
    return `mock-${this.dimensions}d`;
  }

  getDimensions(): number {
    return this.dimensions;
  }

  async embed(text: string): Promise<number[]> {
    // Simple mock: hash text to generate deterministic embedding
    const hash = this.simpleHash(text);
    const embedding = new Array(this.dimensions).fill(0);
    for (let i = 0; i < this.dimensions; i++) {
      embedding[i] = Math.sin(hash * (i + 1)) * 0.5 + 0.5;
    }
    return embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map(t => this.embed(t)));
  }

  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash = hash & hash;
    }
    return Math.abs(hash) / 1000000;
  }
}

async function main() {
  console.log('='.repeat(70));
  console.log('Isotrieve Auto-Negotiation Demo');
  console.log('='.repeat(70));

  // Scenario 1: Both agents support Isotrieve
  console.log('\n\n Scenario 1: Both agents support Isotrieve');
  console.log('-'.repeat(70));

  const agentA = new Isotrieve({
    embedder: new MockEmbedder(384),
    agentId: 'agent_a',
  });

  const agentB = new Isotrieve({
    embedder: new MockEmbedder(768),
    agentId: 'agent_b',
  });

  const method1 = await IsotrieveNegotiator.negotiate(agentA, agentB, {
    verbose: true,
    calibrationConfig: { vocabularySize: 100, validationSize: 10 }
  });

  if (method1.usesIsotrieve) {
    const quality = method1.calibrationResult!.qualityMetrics.meanSimilarity;
    console.log(`✓ Using Isotrieve with ${(quality * 100).toFixed(1)}% fidelity`);

    // Send a message using Isotrieve
    const message = 'Hello, how are you?';
    const result = await IsotrieveNegotiator.sendMessage(agentA, agentB, message, method1);
    console.log(`\n Sent via Isotrieve: '${message}'`);
    console.log(`   Transfer ID: ${result.transferId}`);
    console.log(`   Expected similarity: ${(result.expectedSimilarity! * 100).toFixed(1)}%`);
  }

  // Scenario 2: Only one agent supports Isotrieve
  console.log('\n\n Scenario 2: Only one agent supports Isotrieve');
  console.log('-'.repeat(70));

  const agentIsotrieve = new Isotrieve({
    embedder: new MockEmbedder(384),
    agentId: 'agent_isotrieve',
  });

  const agentPlain = { name: 'PlainAgent', type: 'non-isotrieve' }; // Just a regular object

  const method2 = await IsotrieveNegotiator.negotiate(agentIsotrieve, agentPlain, {
    verbose: true,
  });

  if (!method2.usesIsotrieve) {
    console.log('✓ Using text fallback');
    console.log(`   Reason: ${method2.fallbackReason}`);

    // Send a message using text
    const message = "Hello, I don't support Isotrieve";
    const result = await IsotrieveNegotiator.sendMessage(agentIsotrieve, agentPlain, message, method2);
    console.log(`\n Sent via text: '${result.message}'`);
  }

  // Scenario 3: Neither agent supports Isotrieve
  console.log('\n\n Scenario 3: Neither agent supports Isotrieve');
  console.log('-'.repeat(70));

  const agent1Plain = { name: 'Agent1', type: 'non-isotrieve' };
  const agent2Plain = { name: 'Agent2', type: 'non-isotrieve' };

  const method3 = await IsotrieveNegotiator.negotiate(agent1Plain, agent2Plain, {
    verbose: true,
  });

  if (!method3.usesIsotrieve) {
    console.log('✓ Using text fallback');
    console.log(`   Reason: ${method3.fallbackReason}`);
  }

  // Scenario 4: Demonstrate automatic re-negotiation
  console.log('\n\n Scenario 4: Automatic re-negotiation on each message');
  console.log('-'.repeat(70));

  const agentX = new Isotrieve({
    embedder: new MockEmbedder(512),
    agentId: 'agent_x',
  });

  const agentY = new Isotrieve({
    embedder: new MockEmbedder(256),
    agentId: 'agent_y',
  });

  console.log('\nSending message without pre-negotiation (will auto-negotiate)...');
  const result = await IsotrieveNegotiator.sendMessage(
    agentX,
    agentY,
    'This will trigger automatic negotiation'
  );

  if (result.method === 'isotrieve') {
    console.log('✓ Auto-negotiated Isotrieve successfully');
  }

  // Summary
  console.log('\n\n' + '='.repeat(70));
  console.log('Summary');
  console.log('='.repeat(70));
  console.log(`
Key Features of Auto-Negotiation:

1. ✓ Automatic Detection
   - Detects if both agents support Isotrieve
   - No manual configuration needed

2. ✓ Seamless Fallback
   - Falls back to text if one agent doesn't support Isotrieve
   - Provides clear warning messages

3. ✓ Calibration on Demand
   - Automatically calibrates when both agents support Isotrieve
   - Caches matrices for future use

4. ✓ Transparent Communication
   - Returns clear status about which method is being used
   - Provides fallback reasons when Isotrieve is not available

Usage in Your Code:
------------------
import { Isotrieve, IsotrieveNegotiator } from '@isotrieve/core';
import { OpenAIAdapter } from '@isotrieve/adapters-openai';

// Just create your agents normally
const agent1 = new Isotrieve({ embedder: new OpenAIAdapter({ apiKey: '...' }) });
const agent2 = someOtherAgent;  // Could be Isotrieve or not

// Auto-negotiate and send
const result = await IsotrieveNegotiator.sendMessage(agent1, agent2, 'Hello!');

// The library handles everything:
// - Checks if both support Isotrieve
// - Calibrates if needed
// - Uses Isotrieve or falls back to text
// - Returns result with method info
`);
}

main().catch(console.error);
