# Isotrieve - Agent Embedding Communication Protocol

> **⚠️ Historical/Experimental Package**
> This NPM package is the original protocol prototype. The actively maintained, benchmark-validated project is [`isotrieve-python/`](../isotrieve-python/) — available on [PyPI as `isotrieve`](https://pypi.org/project/isotrieve/). This package is not actively maintained.

Production-ready NPM package for semantic communication between agents using different embedding models.

## What is Isotrieve?

Isotrieve enables agents with different embedding models to communicate semantic information directly through learned transfer matrices, achieving **2x better semantic preservation** compared to text serialization.

## Quick Start

```bash
npm install @isotrieve/core @isotrieve/adapters-openai
```

```typescript
import { Isotrieve } from '@isotrieve/core';
import { OpenAIAdapter } from '@isotrieve/adapters-openai';

// Initialize agents
const agent1 = new Isotrieve({
  embedder: new OpenAIAdapter({
    apiKey: process.env.OPENAI_KEY,
    model: 'text-embedding-3-large'
  })
});

const agent2 = new Isotrieve({
  embedder: new OpenAIAdapter({
    apiKey: process.env.OPENAI_KEY,
    model: 'text-embedding-3-small'
  })
});

// Calibrate (one-time setup)
await agent1.calibrateWith(agent2);

// Transfer embeddings
const embedding = await agent1.embed("Complex technical state");
const transferred = await agent1.transferTo(agent2, embedding);

// Agent 2 uses transferred embedding natively
const similar = await agent2.findSimilar(transferred, knowledgeBase);
```

###  Auto-Negotiation (NEW)

Isotrieve now automatically detects if both agents support the protocol and falls back to text if needed:

```typescript
import { Isotrieve, IsotrieveNegotiator } from '@isotrieve/core';
import { OpenAIAdapter } from '@isotrieve/adapters-openai';

// Create your agents
const agent1 = new Isotrieve({ embedder: new OpenAIAdapter({ apiKey: '...' }) });
const agent2 = someOtherAgent;  // Could be Isotrieve or not

// Automatically negotiate and send message
const result = await IsotrieveNegotiator.sendMessage(agent1, agent2, 'Hello!');

// Isotrieve automatically:
// ✓ Detects if both support Isotrieve → Uses Isotrieve with 97% fidelity
// ✓ Detects if only one supports Isotrieve → Falls back to text
// ✓ Shows clear warning when falling back
// ✓ Returns result with method info

if (result.method === 'isotrieve') {
  console.log(`✓ Using Isotrieve with ${(result.expectedSimilarity! * 100).toFixed(1)}% fidelity`);
} else {
  console.log(`⚠️  Using text: ${result.fallbackReason}`);
}
```

**Example Output:**
```
# Both support Isotrieve:
 Both agents support Isotrieve. Calibrating...
✓ Isotrieve enabled with 97.3% semantic fidelity

# Only one supports Isotrieve:
⚠️  Isotrieve not available: Agent 2 does not support Isotrieve. Falling back to text communication.
```

## Features

- **2x Better Semantic Preservation** vs text round-trip
- **Provider-Agnostic** - Works with OpenAI, Anthropic, Cohere, HuggingFace
- **Quality Monitoring** - Automatic quality tracking with fallback
- **Lightweight** - Millisecond-level transfer latency
- **Production-Ready** - Validated on 300k vocabulary, 97% fidelity

## Packages

- `@isotrieve/core` - Core protocol implementation
- `@isotrieve/adapters-openai` - OpenAI embeddings adapter
- `@isotrieve/adapters-voyage` - Voyage AI adapter
- `@isotrieve/adapters-cohere` - Cohere adapter
- `@isotrieve/adapters-huggingface` - HuggingFace adapter

## Documentation

- [Protocol Specification](./docs/protocol-spec.md)
- [Getting Started Guide](./docs/getting-started.md)
- [API Reference](./docs/api-reference.md)

## License

MIT
