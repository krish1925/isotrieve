# API Reference

## Core Package (@isotrieve/core)

### Isotrieve Class

Main protocol handler for agent communication.

#### Constructor

```typescript
new Isotrieve(config: IsotrieveConfig)
```

**Parameters:**
- `config.embedder: EmbeddingProvider` - Embedding provider implementation (required)
- `config.agentId?: string` - Custom agent ID (auto-generated if omitted)
- `config.minQualityThreshold?: number` - Minimum quality threshold (default: 0.75)
- `config.maxBatchSize?: number` - Maximum batch size (default: 1000)

**Example:**
```typescript
const agent = new Isotrieve({
  embedder: new OpenAIAdapter({ apiKey: 'sk-...' }),
  minQualityThreshold: 0.80
});
```

#### Methods

##### `getCapabilities()`

Get agent capabilities and metadata.

```typescript
getCapabilities(): AgentCapabilities
```

**Returns:**
```typescript
{
  agentId: string;
  embeddingModel: {
    name: string;
    dimensions: number;
  };
  protocolVersion: string;
  maxBatchSize: number;
  minQualityThreshold: number;
}
```

##### `calibrateWith()`

Calibrate with another agent to enable semantic transfer.

```typescript
async calibrateWith(
  otherAgent: Isotrieve,
  config?: CalibrationConfig
): Promise<CalibrationResult>
```

**Parameters:**
- `otherAgent: Isotrieve` - Target agent for calibration
- `config.vocabularySize?: number` - Training vocabulary size (default: 1000)
- `config.validationSize?: number` - Validation set size (default: 100)
- `config.qualityThreshold?: number` - Minimum quality (default: 0.75)
- `config.customVocabulary?: string[]` - Custom vocabulary (overrides generation)

**Returns:**
```typescript
{
  success: boolean;
  transferMatrix: TransferMatrix;
  qualityMetrics: {
    meanSimilarity: number;
    medianSimilarity: number;
    stdSimilarity: number;
    minSimilarity: number;
    maxSimilarity: number;
  };
  calibrationTime: number; // milliseconds
}
```

**Example:**
```typescript
const result = await agent1.calibrateWith(agent2, {
  vocabularySize: 2000,
  qualityThreshold: 0.85
});

if (result.success) {
  console.log(`Quality: ${result.qualityMetrics.meanSimilarity}`);
}
```

##### `embed()`

Generate embedding for text.

```typescript
async embed(text: string): Promise<number[]>
```

**Parameters:**
- `text: string` - Input text

**Returns:** Embedding vector

**Example:**
```typescript
const embedding = await agent.embed("Hello world");
console.log(embedding.length); // e.g., 1536
```

##### `embedBatch()`

Generate embeddings for multiple texts.

```typescript
async embedBatch(texts: string[]): Promise<number[][]>
```

**Parameters:**
- `texts: string[]` - Array of input texts

**Returns:** Array of embedding vectors

**Example:**
```typescript
const embeddings = await agent.embedBatch([
  "First text",
  "Second text",
  "Third text"
]);
```

##### `transferTo()`

Transfer embedding to another agent's space.

```typescript
async transferTo(
  targetAgent: Isotrieve,
  embedding: number[]
): Promise<SemanticTransfer>
```

**Parameters:**
- `targetAgent: Isotrieve` - Target agent
- `embedding: number[]` - Source embedding

**Returns:**
```typescript
{
  transferId: string;
  embedding: number[];
  sourceAgent: string;
  targetAgent: string;
  expectedSimilarity: number;
  timestamp: string;
}
```

**Throws:**
- Error if no calibration exists
- Error if transfer matrix expired

**Example:**
```typescript
const embedding = await agent1.embed("Transfer this");
const transfer = await agent1.transferTo(agent2, embedding);
// agent2 can now use transfer.embedding
```

##### `findSimilar()`

Find similar embeddings in a knowledge base.

```typescript
async findSimilar(
  queryEmbedding: number[],
  knowledgeBase: number[][],
  topK?: number
): Promise<Array<{ index: number; similarity: number }>>
```

**Parameters:**
- `queryEmbedding: number[]` - Query embedding
- `knowledgeBase: number[][]` - Array of embeddings to search
- `topK?: number` - Number of results (default: 5)

**Returns:** Array of matches sorted by similarity (descending)

**Example:**
```typescript
const results = await agent.findSimilar(
  queryEmbedding,
  knowledgeBase,
  3
);

results.forEach(({ index, similarity }) => {
  console.log(`Match ${index}: ${similarity.toFixed(4)}`);
});
```

##### `getQualityScore()`

Get quality score for connection with another agent.

```typescript
getQualityScore(targetAgentId: string): number | null
```

**Parameters:**
- `targetAgentId: string` - Target agent ID

**Returns:** Quality score (0-1) or null if not calibrated

**Example:**
```typescript
const quality = agent1.getQualityScore(agent2.agentId);
if (quality && quality < 0.80) {
  console.warn('Low quality, recalibration recommended');
}
```

##### `requiresRecalibration()`

Check if recalibration is needed.

```typescript
requiresRecalibration(targetAgentId: string): boolean
```

**Parameters:**
- `targetAgentId: string` - Target agent ID

**Returns:** True if recalibration needed

**Example:**
```typescript
if (agent1.requiresRecalibration(agent2.agentId)) {
  await agent1.calibrateWith(agent2);
}
```

---

## Types

### EmbeddingProvider

Interface for embedding providers.

```typescript
interface EmbeddingProvider {
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
  getDimensions(): number;
  getModelId(): string;
}
```

### AgentCapabilities

```typescript
interface AgentCapabilities {
  agentId: string;
  embeddingModel: {
    name: string;
    dimensions: number;
  };
  protocolVersion: string;
  maxBatchSize: number;
  minQualityThreshold: number;
}
```

### CalibrationConfig

```typescript
interface CalibrationConfig {
  vocabularySize?: number;
  validationSize?: number;
  qualityThreshold?: number;
  customVocabulary?: string[];
}
```

### TransferMatrix

```typescript
interface TransferMatrix {
  matrixAB: number[][];
  matrixBA: number[][];
  trainingsimilarity: number;
  validationSimilarity: number;
  worstCaseSimilarity: number;
  validUntil: string; // ISO 8601
}
```

### QualityMetrics

```typescript
interface QualityMetrics {
  meanSimilarity: number;
  medianSimilarity: number;
  stdSimilarity: number;
  minSimilarity: number;
  maxSimilarity: number;
}
```

### SemanticTransfer

```typescript
interface SemanticTransfer {
  transferId: string;
  embedding: number[];
  sourceAgent: string;
  targetAgent: string;
  expectedSimilarity: number;
  timestamp: string; // ISO 8601
}
```

---

## Utility Functions

### Matrix Operations

```typescript
import { cosineSimilarity, computeTransferMatrices } from '@isotrieve/core';
```

#### `cosineSimilarity()`

```typescript
function cosineSimilarity(vec1: number[], vec2: number[]): number
```

Compute cosine similarity between two vectors.

#### `computeTransferMatrices()`

```typescript
function computeTransferMatrices(
  embeddingsSource: number[][],
  embeddingsTarget: number[][]
): { forward: number[][]; backward: number[][] }
```

Compute transfer matrices using least squares.

### Vocabulary

```typescript
import { DEFAULT_VOCABULARY, generateExtendedVocabulary } from '@isotrieve/core';
```

#### `DEFAULT_VOCABULARY`

Pre-curated vocabulary (150+ terms) for calibration.

#### `generateExtendedVocabulary()`

```typescript
function generateExtendedVocabulary(size?: number): string[]
```

Generate extended vocabulary by combining base terms.

**Parameters:**
- `size?: number` - Target vocabulary size (default: 1000)

**Returns:** Array of vocabulary items

---

## Adapters

### OpenAI Adapter

```typescript
import { OpenAIAdapter } from '@isotrieve/adapters-openai';

new OpenAIAdapter({
  apiKey: string;
  model?: string;        // default: 'text-embedding-3-small'
  organization?: string;
  baseURL?: string;
})
```

**Supported Models:**
- `text-embedding-3-small` (1536D)
- `text-embedding-3-large` (3072D)
- `text-embedding-ada-002` (1536D)

### Voyage Adapter

```typescript
import { VoyageAdapter } from '@isotrieve/adapters-voyage';

new VoyageAdapter({
  apiKey: string;
  model?: string;  // default: 'voyage-2'
  baseURL?: string;
})
```

**Supported Models:**
- `voyage-2` (1024D)
- `voyage-large-2` (1536D)
- `voyage-code-2` (1536D)

### Cohere Adapter

```typescript
import { CohereAdapter } from '@isotrieve/adapters-cohere';

new CohereAdapter({
  apiKey: string;
  model?: string;  // default: 'embed-english-v3.0'
})
```

**Supported Models:**
- `embed-english-v3.0` (1024D)
- `embed-multilingual-v3.0` (1024D)
- `embed-english-light-v3.0` (384D)
- `embed-multilingual-light-v3.0` (384D)

### HuggingFace Adapter

```typescript
import { HuggingFaceAdapter } from '@isotrieve/adapters-huggingface';

new HuggingFaceAdapter({
  model?: string;      // default: 'Xenova/all-MiniLM-L6-v2'
  quantized?: boolean; // default: true
})
```

**Supported Models:**
- `Xenova/all-MiniLM-L6-v2` (384D)
- `Xenova/all-mpnet-base-v2` (768D)
- `Xenova/bge-small-en-v1.5` (384D)
- `Xenova/bge-base-en-v1.5` (768D)

---

## Error Handling

### Common Errors

```typescript
try {
  const transfer = await agent1.transferTo(agent2, embedding);
} catch (error) {
  if (error.message.includes('No calibration found')) {
    // Need to calibrate first
    await agent1.calibrateWith(agent2);
  } else if (error.message.includes('expired')) {
    // Recalibration needed
    await agent1.calibrateWith(agent2);
  } else {
    throw error;
  }
}
```

### Quality Checks

```typescript
const result = await agent1.calibrateWith(agent2);

if (!result.success) {
  console.error('Calibration failed');
  console.error(`Quality: ${result.qualityMetrics.meanSimilarity}`);
  
  // Try with larger vocabulary
  await agent1.calibrateWith(agent2, {
    vocabularySize: 5000
  });
}
```
