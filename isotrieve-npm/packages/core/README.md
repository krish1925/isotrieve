# @isotrieve/core

Core implementation of the Agent Embedding Communication Protocol (Isotrieve).

## Installation

```bash
npm install @isotrieve/core
```

## Quick Start

```typescript
import { Isotrieve } from '@isotrieve/core';

// Create custom embedding provider
const myEmbedder = {
  async embed(text: string): Promise<number[]> {
    // Your embedding logic
    return [0.1, 0.2, 0.3, ...];
  },
  async embedBatch(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map(t => this.embed(t)));
  },
  getDimensions(): number {
    return 384;
  },
  getModelId(): string {
    return 'my-model';
  }
};

const agent = new Isotrieve({ embedder: myEmbedder });
```

## API

### `new Isotrieve(config)`

Create a new Isotrieve agent.

**Config:**
- `embedder: EmbeddingProvider` - Embedding provider implementation
- `agentId?: string` - Optional agent ID (auto-generated if not provided)
- `minQualityThreshold?: number` - Minimum quality threshold (default: 0.75)
- `maxBatchSize?: number` - Maximum batch size (default: 1000)

### `calibrateWith(otherAgent, config?)`

Calibrate with another agent to enable semantic transfer.

**Returns:** `Promise<CalibrationResult>`

### `embed(text)`

Generate embedding for text.

**Returns:** `Promise<number[]>`

### `transferTo(targetAgent, embedding)`

Transfer embedding to another agent's space.

**Returns:** `Promise<SemanticTransfer>`

### `findSimilar(queryEmbedding, knowledgeBase, topK?)`

Find similar embeddings in a knowledge base.

**Returns:** `Promise<Array<{index: number, similarity: number}>>`

### `getQualityScore(targetAgentId)`

Get quality score for connection with another agent.

**Returns:** `number | null`

## License

MIT
