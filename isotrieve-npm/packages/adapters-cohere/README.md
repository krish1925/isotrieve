# @isotrieve/adapters-cohere

Cohere embeddings adapter for Isotrieve.

## Installation

```bash
npm install @isotrieve/core @isotrieve/adapters-cohere
```

## Usage

```typescript
import { Isotrieve } from '@isotrieve/core';
import { CohereAdapter } from '@isotrieve/adapters-cohere';

const agent = new Isotrieve({
  embedder: new CohereAdapter({
    apiKey: process.env.COHERE_API_KEY,
    model: 'embed-english-v3.0',
  })
});
```

## Supported Models

- `embed-english-v3.0` (1024 dimensions)
- `embed-multilingual-v3.0` (1024 dimensions)
- `embed-english-light-v3.0` (384 dimensions)
- `embed-multilingual-light-v3.0` (384 dimensions)

## License

MIT
