# @isotrieve/adapters-openai

OpenAI embeddings adapter for Isotrieve.

## Installation

```bash
npm install @isotrieve/core @isotrieve/adapters-openai
```

## Usage

```typescript
import { Isotrieve } from '@isotrieve/core';
import { OpenAIAdapter } from '@isotrieve/adapters-openai';

const agent = new Isotrieve({
  embedder: new OpenAIAdapter({
    apiKey: process.env.OPENAI_API_KEY,
    model: 'text-embedding-3-large', // or 'text-embedding-3-small'
  })
});

// Use the agent
const embedding = await agent.embed("Hello world");
```

## Supported Models

- `text-embedding-3-small` (1536 dimensions)
- `text-embedding-3-large` (3072 dimensions)
- `text-embedding-ada-002` (1536 dimensions)

## Configuration

```typescript
new OpenAIAdapter({
  apiKey: string;           // Required: OpenAI API key
  model?: string;           // Optional: Model name (default: 'text-embedding-3-small')
  organization?: string;    // Optional: OpenAI organization ID
  baseURL?: string;         // Optional: Custom API base URL
})
```

## License

MIT
