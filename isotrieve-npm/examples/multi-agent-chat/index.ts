/**
 * Multi-Agent Chat Example
 * Demonstrates semantic communication between multiple agents
 */

import { Isotrieve } from '@isotrieve/core';
import { OpenAIAdapter } from '@isotrieve/adapters-openai';
import { HuggingFaceAdapter } from '@isotrieve/adapters-huggingface';

class Agent {
  constructor(
    public name: string,
    public isotrieve: Isotrieve,
    public knowledgeBase: Map<string, number[]> = new Map()
  ) {}

  async addKnowledge(key: string, text: string) {
    const embedding = await this.isotrieve.embed(text);
    this.knowledgeBase.set(key, embedding);
    console.log(`[${this.name}] Added knowledge: "${key}"`);
  }

  async sendMessage(recipient: Agent, message: string) {
    console.log(`\n[${this.name}] → [${recipient.name}]: "${message}"`);

    // Embed with my embedder
    const embedding = await this.isotrieve.embed(message);

    // Transfer to recipient's space
    const transfer = await this.isotrieve.transferTo(recipient.isotrieve, embedding);

    // Recipient searches their knowledge base
    const kbEmbeddings = Array.from(recipient.knowledgeBase.values());
    const kbKeys = Array.from(recipient.knowledgeBase.keys());

    if (kbEmbeddings.length > 0) {
      const results = await recipient.isotrieve.findSimilar(
        transfer.embedding,
        kbEmbeddings,
        1
      );

      const topMatch = results[0];
      console.log(
        `[${recipient.name}] Found relevant knowledge: "${kbKeys[topMatch.index]}" (similarity: ${topMatch.similarity.toFixed(4)})`
      );
    }
  }
}

async function main() {
  console.log('=== Isotrieve Multi-Agent Chat Example ===\n');

  // Create agents with different embedding models
  const alice = new Agent(
    'Alice',
    new Isotrieve({
      embedder: new OpenAIAdapter({
        apiKey: process.env.OPENAI_API_KEY!,
        model: 'text-embedding-3-small',
      }),
    })
  );

  const bob = new Agent(
    'Bob',
    new Isotrieve({
      embedder: new HuggingFaceAdapter({
        model: 'Xenova/all-MiniLM-L6-v2',
      }),
    })
  );

  console.log('Calibrating agents...');
  await alice.isotrieve.calibrateWith(bob.isotrieve, {
    vocabularySize: 300,
    validationSize: 30,
  });
  console.log('Calibration complete!\n');

  // Build knowledge bases
  console.log('Building knowledge bases...');
  await alice.addKnowledge('ml-basics', 'Machine learning uses data to train models.');
  await alice.addKnowledge('dl-intro', 'Deep learning uses neural networks with many layers.');

  await bob.addKnowledge('quantum-intro', 'Quantum computing uses qubits for computation.');
  await bob.addKnowledge('ai-ethics', 'AI ethics considers fairness and transparency.');

  // Agents communicate
  console.log('\n--- Communication Phase ---');
  await alice.sendMessage(bob, 'Tell me about quantum computing applications.');
  await bob.sendMessage(alice, 'How do neural networks learn from data?');
}

main().catch(console.error);
