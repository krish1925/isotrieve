/**
 * Basic Transfer Example
 * Demonstrates simple embedding transfer between two agents
 */

import { Isotrieve } from '@isotrieve/core';
import { OpenAIAdapter } from '@isotrieve/adapters-openai';

async function main() {
  console.log('=== Isotrieve Basic Transfer Example ===\n');

  // Create two agents with different embedding models
  const agent1 = new Isotrieve({
    embedder: new OpenAIAdapter({
      apiKey: process.env.OPENAI_API_KEY!,
      model: 'text-embedding-3-small',
    }),
  });

  const agent2 = new Isotrieve({
    embedder: new OpenAIAdapter({
      apiKey: process.env.OPENAI_API_KEY!,
      model: 'text-embedding-3-large',
    }),
  });

  console.log('Agent 1:', agent1.getCapabilities());
  console.log('Agent 2:', agent2.getCapabilities());
  console.log();

  // Calibrate agents (one-time setup)
  console.log('Calibrating agents...');
  const calibration = await agent1.calibrateWith(agent2, {
    vocabularySize: 500,
    validationSize: 50,
  });

  console.log(`Calibration quality: ${calibration.qualityMetrics.meanSimilarity.toFixed(4)}`);
  console.log(`Calibration time: ${calibration.calibrationTime}ms\n`);

  // Transfer embeddings
  const message = 'The quantum computer achieved breakthrough performance.';
  console.log(`Transferring message: "${message}"`);

  const embedding1 = await agent1.embed(message);
  console.log(`Agent 1 embedding dimensions: ${embedding1.length}`);

  const transfer = await agent1.transferTo(agent2, embedding1);
  console.log(`Transferred embedding dimensions: ${transfer.embedding.length}`);
  console.log(`Expected similarity: ${transfer.expectedSimilarity.toFixed(4)}`);

  // Agent 2 can now use this embedding natively
  const knowledgeBase = await agent2.embedBatch([
    'Quantum computing shows promising results.',
    'The weather is sunny today.',
    'Machine learning models improve accuracy.',
  ]);

  const similar = await agent2.findSimilar(transfer.embedding, knowledgeBase, 3);
  console.log('\nMost similar items in knowledge base:');
  similar.forEach(({ index, similarity }) => {
    console.log(`  [${index}] Similarity: ${similarity.toFixed(4)}`);
  });
}

main().catch(console.error);
