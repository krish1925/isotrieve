/**
 * AECP Auto-Negotiation Example
 * 
 * This example demonstrates how AECP automatically negotiates between agents:
 * 1. When both agents support AECP -> Uses AECP
 * 2. When only one agent supports AECP -> Falls back to text
 * 3. When neither agent supports AECP -> Uses text
 */

import { AECP, AECPNegotiator } from '@aecp/core';

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
  console.log('AECP Auto-Negotiation Demo');
  console.log('='.repeat(70));

  // Scenario 1: Both agents support AECP
  console.log('\n\n📋 Scenario 1: Both agents support AECP');
  console.log('-'.repeat(70));

  const agentA = new AECP({
    embedder: new MockEmbedder(384),
    agentId: 'agent_a',
  });

  const agentB = new AECP({
    embedder: new MockEmbedder(768),
    agentId: 'agent_b',
  });

  const method1 = await AECPNegotiator.negotiate(agentA, agentB, {
    verbose: true,
    calibrationConfig: { vocabularySize: 100, validationSize: 10 }
  });

  if (method1.usesAECP) {
    const quality = method1.calibrationResult!.qualityMetrics.meanSimilarity;
    console.log(`✓ Using AECP with ${(quality * 100).toFixed(1)}% fidelity`);

    // Send a message using AECP
    const message = 'Hello, how are you?';
    const result = await AECPNegotiator.sendMessage(agentA, agentB, message, method1);
    console.log(`\n📤 Sent via AECP: '${message}'`);
    console.log(`   Transfer ID: ${result.transferId}`);
    console.log(`   Expected similarity: ${(result.expectedSimilarity! * 100).toFixed(1)}%`);
  }

  // Scenario 2: Only one agent supports AECP
  console.log('\n\n📋 Scenario 2: Only one agent supports AECP');
  console.log('-'.repeat(70));

  const agentAECP = new AECP({
    embedder: new MockEmbedder(384),
    agentId: 'agent_aecp',
  });

  const agentPlain = { name: 'PlainAgent', type: 'non-aecp' }; // Just a regular object

  const method2 = await AECPNegotiator.negotiate(agentAECP, agentPlain, {
    verbose: true,
  });

  if (!method2.usesAECP) {
    console.log('✓ Using text fallback');
    console.log(`   Reason: ${method2.fallbackReason}`);

    // Send a message using text
    const message = "Hello, I don't support AECP";
    const result = await AECPNegotiator.sendMessage(agentAECP, agentPlain, message, method2);
    console.log(`\n📤 Sent via text: '${result.message}'`);
  }

  // Scenario 3: Neither agent supports AECP
  console.log('\n\n📋 Scenario 3: Neither agent supports AECP');
  console.log('-'.repeat(70));

  const agent1Plain = { name: 'Agent1', type: 'non-aecp' };
  const agent2Plain = { name: 'Agent2', type: 'non-aecp' };

  const method3 = await AECPNegotiator.negotiate(agent1Plain, agent2Plain, {
    verbose: true,
  });

  if (!method3.usesAECP) {
    console.log('✓ Using text fallback');
    console.log(`   Reason: ${method3.fallbackReason}`);
  }

  // Scenario 4: Demonstrate automatic re-negotiation
  console.log('\n\n📋 Scenario 4: Automatic re-negotiation on each message');
  console.log('-'.repeat(70));

  const agentX = new AECP({
    embedder: new MockEmbedder(512),
    agentId: 'agent_x',
  });

  const agentY = new AECP({
    embedder: new MockEmbedder(256),
    agentId: 'agent_y',
  });

  console.log('\nSending message without pre-negotiation (will auto-negotiate)...');
  const result = await AECPNegotiator.sendMessage(
    agentX,
    agentY,
    'This will trigger automatic negotiation'
  );

  if (result.method === 'aecp') {
    console.log('✓ Auto-negotiated AECP successfully');
  }

  // Summary
  console.log('\n\n' + '='.repeat(70));
  console.log('Summary');
  console.log('='.repeat(70));
  console.log(`
Key Features of Auto-Negotiation:

1. ✓ Automatic Detection
   - Detects if both agents support AECP
   - No manual configuration needed

2. ✓ Seamless Fallback
   - Falls back to text if one agent doesn't support AECP
   - Provides clear warning messages

3. ✓ Calibration on Demand
   - Automatically calibrates when both agents support AECP
   - Caches matrices for future use

4. ✓ Transparent Communication
   - Returns clear status about which method is being used
   - Provides fallback reasons when AECP is not available

Usage in Your Code:
------------------
import { AECP, AECPNegotiator } from '@aecp/core';
import { OpenAIAdapter } from '@aecp/adapters-openai';

// Just create your agents normally
const agent1 = new AECP({ embedder: new OpenAIAdapter({ apiKey: '...' }) });
const agent2 = someOtherAgent;  // Could be AECP or not

// Auto-negotiate and send
const result = await AECPNegotiator.sendMessage(agent1, agent2, 'Hello!');

// The library handles everything:
// - Checks if both support AECP
// - Calibrates if needed
// - Uses AECP or falls back to text
// - Returns result with method info
`);
}

main().catch(console.error);
