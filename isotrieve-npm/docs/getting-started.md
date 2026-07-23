# Getting Started with Isotrieve

## What is Isotrieve?

The Agent Embedding Communication Protocol (Isotrieve) enables agents with different embedding models to communicate semantic information directly, achieving **2x better semantic preservation** compared to text serialization.

## Installation

### Core Package

```bash
npm install @isotrieve/core
```

### With Adapters

```bash
# OpenAI
npm install @isotrieve/core @isotrieve/adapters-openai

# Voyage AI
npm install @isotrieve/core @isotrieve/adapters-voyage

# Cohere
npm install @isotrieve/core @isotrieve/adapters-cohere

# HuggingFace (local)
npm install @isotrieve/core @isotrieve/adapters-huggingface
```

## Quick Start

### 1. Create Agents

```typescript
import { Isotrieve } from '@isotrieve/core';
import { OpenAIAdapter } from '@isotrieve/adapters-openai';

const agent1 = new Isotrieve({
  embedder: new OpenAIAdapter({
    apiKey: process.env.OPENAI_API_KEY,
    model: 'text-embedding-3-small'
  })
});

const agent2 = new Isotrieve({
  embedder: new OpenAIAdapter({
    apiKey: process.env.OPENAI_API_KEY,
    model: 'text-embedding-3-large'
  })
});
```

### 2. Calibrate (One-Time Setup)

```typescript
const calibration = await agent1.calibrateWith(agent2, {
  vocabularySize: 1000,    // Training vocabulary size
  validationSize: 100,     // Validation set size
  qualityThreshold: 0.80   // Minimum acceptable quality
});

console.log(`Quality: ${calibration.qualityMetrics.meanSimilarity}`);
// Output: Quality: 0.9234
```

### 3. Transfer Embeddings

```typescript
// Agent 1 embeds content
const embedding = await agent1.embed("Complex technical information");

// Transfer to Agent 2's space
const transfer = await agent1.transferTo(agent2, embedding);

// Agent 2 uses transferred embedding natively
const knowledgeBase = await agent2.embedBatch([
  "Technical documentation",
  "User manual",
  "API reference"
]);

const similar = await agent2.findSimilar(
  transfer.embedding,
  knowledgeBase,
  3
);
```

## Core Concepts

### Calibration

Calibration is a one-time process where agents:
1. Encode a shared vocabulary
2. Compute transfer matrices using least squares
3. Validate quality on held-out data

**Recommended settings:**
- Vocabulary size: 500-1000 for quick setup, 5000+ for production
- Validation size: 10% of vocabulary size
- Quality threshold: 0.75 minimum, 0.85+ ideal

### Transfer Quality

Quality is measured using cosine similarity:
- **> 0.90**: Excellent - Near-perfect semantic preservation
- **0.80-0.90**: Good - Production-ready
- **0.75-0.80**: Acceptable - Consider recalibration
- **< 0.75**: Poor - Recalibration required

### Recalibration

Recalibrate when:
- Transfer matrix expires (default: 7 days)
- Quality drops below threshold
- Embedding models are updated

```typescript
if (agent1.requiresRecalibration(agent2.agentId)) {
  await agent1.calibrateWith(agent2);
}
```

## Provider-Specific Setup

### OpenAI

```typescript
import { OpenAIAdapter } from '@isotrieve/adapters-openai';

const embedder = new OpenAIAdapter({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'text-embedding-3-large', // 3072 dimensions
  organization: 'org-xxx' // Optional
});
```

### Voyage AI

```typescript
import { VoyageAdapter } from '@isotrieve/adapters-voyage';

const embedder = new VoyageAdapter({
  apiKey: process.env.VOYAGE_API_KEY,
  model: 'voyage-2' // 1024 dimensions
});
```

### Cohere

```typescript
import { CohereAdapter } from '@isotrieve/adapters-cohere';

const embedder = new CohereAdapter({
  apiKey: process.env.COHERE_API_KEY,
  model: 'embed-english-v3.0' // 1024 dimensions
});
```

### HuggingFace (Local)

```typescript
import { HuggingFaceAdapter } from '@isotrieve/adapters-huggingface';

const embedder = new HuggingFaceAdapter({
  model: 'Xenova/all-MiniLM-L6-v2', // 384 dimensions
  quantized: true // Faster inference
});
```

## Custom Adapter

Create your own adapter by implementing `EmbeddingProvider`:

```typescript
import { EmbeddingProvider } from '@isotrieve/core';

class MyAdapter implements EmbeddingProvider {
  async embed(text: string): Promise<number[]> {
    // Your embedding logic
    return [0.1, 0.2, ...];
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map(t => this.embed(t)));
  }

  getDimensions(): number {
    return 384;
  }

  getModelId(): string {
    return 'my-model-v1';
  }
}
```

## Best Practices

### 1. Cache Calibrations

```typescript
// Save transfer matrix
const matrix = agent1.transferMatrices.get(agent2.agentId);
fs.writeFileSync('calibration.json', JSON.stringify(matrix));

// Load later
const savedMatrix = JSON.parse(fs.readFileSync('calibration.json'));
agent1.transferMatrices.set(agent2.agentId, savedMatrix);
```

### 2. Monitor Quality

```typescript
const quality = agent1.getQualityScore(agent2.agentId);
if (quality && quality < 0.80) {
  console.warn('Quality degraded, recalibrating...');
  await agent1.calibrateWith(agent2);
}
```

### 3. Use Appropriate Vocabulary

```typescript
// Domain-specific vocabulary for better quality
const medicalVocab = [
  'diagnosis', 'treatment', 'patient', 'symptoms',
  'The patient shows signs of infection.',
  'Treatment protocol requires antibiotics.',
  // ... more domain terms
];

await agent1.calibrateWith(agent2, {
  customVocabulary: medicalVocab
});
```

### 4. Batch Operations

```typescript
// Batch embeddings for efficiency
const texts = ['text1', 'text2', 'text3'];
const embeddings = await agent1.embedBatch(texts);

// Transfer in batch
const transfers = await Promise.all(
  embeddings.map(emb => agent1.transferTo(agent2, emb))
);
```

## Troubleshooting

### Low Quality Scores

**Problem:** Calibration quality < 0.75

**Solutions:**
1. Increase vocabulary size
2. Use more diverse vocabulary
3. Check if models are too different (e.g., 384D vs 3072D)
4. Consider using models from same family

### Slow Calibration

**Problem:** Calibration takes too long

**Solutions:**
1. Reduce vocabulary size (minimum 200-300)
2. Use batch embedding APIs
3. Cache calibration results

### Memory Issues

**Problem:** Out of memory during calibration

**Solutions:**
1. Reduce vocabulary size
2. Process in smaller batches
3. Use quantized models (HuggingFace)

## Next Steps

- [API Reference](./api-reference.md)
- [Protocol Specification](./protocol-spec.md)
- [Examples](../examples/)

## Support

- GitHub Issues: [github.com/your-org/isotrieve](https://github.com/your-org/isotrieve)
- Documentation: [isotrieve.dev](https://isotrieve.dev)
