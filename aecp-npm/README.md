# AECP - Agent Embedding Communication Protocol

> **⚠️ Historical/Experimental Package**
> This NPM package is the original protocol prototype. The actively maintained, benchmark-validated project is [`aecp-python/`](../aecp-python/) — available on [PyPI as `aecp`](https://pypi.org/project/aecp/). This package is not actively maintained.

Production-ready NPM package for semantic communication between agents using different embedding models.

## What is AECP?

AECP enables agents with different embedding models to communicate semantic information directly through learned transfer matrices, achieving **2x better semantic preservation** compared to text serialization.

## Quick Start

```bash
npm install @aecp/core @aecp/adapters-openai
```

```typescript
import { AECP } from '@aecp/core';
import { OpenAIAdapter } from '@aecp/adapters-openai';

// Initialize agents
const agent1 = new AECP({
  embedder: new OpenAIAdapter({
    apiKey: process.env.OPENAI_KEY,
    model: 'text-embedding-3-large'
  })
});

const agent2 = new AECP({
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

AECP now automatically detects if both agents support the protocol and falls back to text if needed:

```typescript
import { AECP, AECPNegotiator } from '@aecp/core';
import { OpenAIAdapter } from '@aecp/adapters-openai';

// Create your agents
const agent1 = new AECP({ embedder: new OpenAIAdapter({ apiKey: '...' }) });
const agent2 = someOtherAgent;  // Could be AECP or not

// Automatically negotiate and send message
const result = await AECPNegotiator.sendMessage(agent1, agent2, 'Hello!');

// AECP automatically:
// ✓ Detects if both support AECP → Uses AECP with 97% fidelity
// ✓ Detects if only one supports AECP → Falls back to text
// ✓ Shows clear warning when falling back
// ✓ Returns result with method info

if (result.method === 'aecp') {
  console.log(`✓ Using AECP with ${(result.expectedSimilarity! * 100).toFixed(1)}% fidelity`);
} else {
  console.log(`⚠️  Using text: ${result.fallbackReason}`);
}
```

**Example Output:**
```
# Both support AECP:
 Both agents support AECP. Calibrating...
✓ AECP enabled with 97.3% semantic fidelity

# Only one supports AECP:
⚠️  AECP not available: Agent 2 does not support AECP. Falling back to text communication.
```

## Features

- **2x Better Semantic Preservation** vs text round-trip
- **Provider-Agnostic** - Works with OpenAI, Anthropic, Cohere, HuggingFace
- **Quality Monitoring** - Automatic quality tracking with fallback
- **Lightweight** - Millisecond-level transfer latency
- **Production-Ready** - Validated on 300k vocabulary, 97% fidelity

## Packages

- `@aecp/core` - Core protocol implementation
- `@aecp/adapters-openai` - OpenAI embeddings adapter
- `@aecp/adapters-voyage` - Voyage AI adapter
- `@aecp/adapters-cohere` - Cohere adapter
- `@aecp/adapters-huggingface` - HuggingFace adapter

## Documentation

- [Protocol Specification](./docs/protocol-spec.md)
- [Getting Started Guide](./docs/getting-started.md)
- [API Reference](./docs/api-reference.md)

## License

MIT
