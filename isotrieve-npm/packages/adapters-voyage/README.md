# @isotrieve/adapters-voyage

Voyage AI embeddings adapter for Isotrieve.

## Installation

```bash
npm install @isotrieve/core @isotrieve/adapters-voyage
```

## Usage

```typescript
import { Isotrieve } from '@isotrieve/core';
import { VoyageAdapter } from '@isotrieve/adapters-voyage';

const agent = new Isotrieve({
  embedder: new VoyageAdapter({
    apiKey: process.env.VOYAGE_API_KEY,
    model: 'voyage-2',
  })
});
```

## Supported Models

- `voyage-2` (1024 dimensions)
- `voyage-large-2` (1536 dimensions)
- `voyage-code-2` (1536 dimensions)

## License

MIT
