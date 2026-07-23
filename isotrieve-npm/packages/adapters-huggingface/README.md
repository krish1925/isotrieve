# @isotrieve/adapters-huggingface

HuggingFace embeddings adapter for Isotrieve using transformers.js (runs locally in browser or Node.js).

## Installation

```bash
npm install @isotrieve/core @isotrieve/adapters-huggingface
```

## Usage

```typescript
import { Isotrieve } from '@isotrieve/core';
import { HuggingFaceAdapter } from '@isotrieve/adapters-huggingface';

const agent = new Isotrieve({
  embedder: new HuggingFaceAdapter({
    model: 'Xenova/all-MiniLM-L6-v2',
    quantized: true, // Use quantized model for faster inference
  })
});

// Wait for model to load
const embedding = await agent.embed("Hello world");
```

## Supported Models

- `Xenova/all-MiniLM-L6-v2` (384 dimensions) - Fast and efficient
- `Xenova/all-mpnet-base-v2` (768 dimensions) - Higher quality
- `Xenova/bge-small-en-v1.5` (384 dimensions)
- `Xenova/bge-base-en-v1.5` (768 dimensions)

## Features

- **Local Inference**: Runs entirely in browser or Node.js
- **No API Keys**: No external API calls required
- **Privacy**: Data never leaves your environment
- **Quantization**: Optional model quantization for faster inference

## License

MIT
