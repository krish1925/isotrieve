# Isotrieve Plugin Guide

## Creating Custom Adapters

Isotrieve provides a plugin system that lets you create custom embedding provider adapters without modifying the core library.

## Quick Start

### Python

```python
from isotrieve import register_adapter, adapter, Isotrieve
from isotrieve.adapters.base import BaseAdapter
from typing import List

# Method 1: Create and register manually
class MyAdapter(BaseAdapter):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self._dimensions = 384
    
    def _embed_impl(self, text: str) -> List[float]:
        # Your implementation
        return [0.1] * self._dimensions
    
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_impl(t) for t in texts]
    
    def get_dimensions(self) -> int:
        return self._dimensions
    
    def get_model_id(self) -> str:
        return "my-model"

# Register it
register_adapter("my-provider", MyAdapter)

# Use it
from isotrieve import create_adapter
adapter = create_adapter("my-provider", api_key="...")
agent = Isotrieve(adapter)
```

```python
# Method 2: Use decorator (auto-registers)
@adapter("my-provider")
class MyAdapter(BaseAdapter):
    # ... same implementation
```

### TypeScript/JavaScript

```typescript
import { registerAdapter, adapter, Isotrieve, EmbeddingProvider } from '@isotrieve/core';

// Method 1: Create and register manually
class MyAdapter implements EmbeddingProvider {
  private apiKey: string;
  private dimensions: number = 384;

  constructor(config: { apiKey: string }) {
    this.apiKey = config.apiKey;
  }

  async embed(text: string): Promise<number[]> {
    // Your implementation
    return new Array(this.dimensions).fill(0.1);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map(t => this.embed(t)));
  }

  getDimensions(): number {
    return this.dimensions;
  }

  getModelId(): string {
    return 'my-model';
  }
}

// Register it
registerAdapter('my-provider', MyAdapter);

// Use it
import { createAdapter } from '@isotrieve/core';
const adapter = createAdapter('my-provider', { apiKey: '...' });
const agent = new Isotrieve({ embedder: adapter });
```

```typescript
// Method 2: Use decorator (auto-registers)
@adapter('my-provider')
class MyAdapter implements EmbeddingProvider {
  // ... same implementation
}
```

## Plugin System API

### Python

```python
from isotrieve import (
    register_adapter,    # Register an adapter
    get_adapter,         # Get adapter class
    create_adapter,      # Create adapter instance
    list_adapters,       # List all registered adapters
    adapter,             # Decorator for registration
)

# Register
register_adapter("name", AdapterClass)
register_adapter("name", AdapterClass, override=True)  # Replace existing

# Get
AdapterClass = get_adapter("name")
adapter_instance = create_adapter("name", **kwargs)

# List
adapters = list_adapters()  # Dict[str, Type[EmbeddingProvider]]
print(adapters.keys())
```

### TypeScript

```typescript
import {
  registerAdapter,       // Register an adapter
  getAdapter,            // Get adapter class
  createAdapter,         // Create adapter instance
  listAdapters,          // List all registered adapters
  adapter,               // Decorator for registration
  hasAdapter,            // Check if registered
  unregisterAdapter,     // Remove adapter
} from '@isotrieve/core';

// Register
registerAdapter('name', AdapterClass);
registerAdapter('name', AdapterClass, { override: true });  // Replace existing

// Get
const AdapterClass = getAdapter('name');
const adapterInstance = createAdapter('name', { /* config */ });

// List
const adapters = listAdapters();  // Record<string, AdapterConstructor>
console.log(Object.keys(adapters));

// Check
if (hasAdapter('name')) {
  // ...
}
```

## Real-World Example: Anthropic Claude

### Python

```python
from isotrieve.adapters.base import BaseAdapter
from typing import List
import anthropic

@adapter("anthropic")
class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic's Claude models."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet", **kwargs):
        super().__init__(**kwargs)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self._dimensions = 768  # Claude's embedding dimensions
    
    def _embed_impl(self, text: str) -> List[float]:
        # Note: This is pseudocode - Anthropic doesn't have embeddings yet
        # Replace with actual API call when available
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.embedding
    
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def get_dimensions(self) -> int:
        return self._dimensions
    
    def get_model_id(self) -> str:
        return f"anthropic:{self.model}"

# Use it
from isotrieve import Isotrieve, create_adapter

adapter = create_adapter("anthropic", api_key="sk-ant-...")
agent = Isotrieve(adapter)
```

### TypeScript

```typescript
import { adapter, EmbeddingProvider } from '@isotrieve/core';
import Anthropic from '@anthropic-ai/sdk';

@adapter('anthropic')
class AnthropicAdapter implements EmbeddingProvider {
  private client: Anthropic;
  private model: string;
  private dimensions: number = 768;

  constructor(config: { apiKey: string; model?: string }) {
    this.client = new Anthropic({ apiKey: config.apiKey });
    this.model = config.model || 'claude-3-sonnet';
  }

  async embed(text: string): Promise<number[]> {
    // Note: This is pseudocode - Anthropic doesn't have embeddings yet
    const response = await this.client.embeddings.create({
      model: this.model,
      input: text,
    });
    return response.embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await this.client.embeddings.create({
      model: this.model,
      input: texts,
    });
    return response.data.map(item => item.embedding);
  }

  getDimensions(): number {
    return this.dimensions;
  }

  getModelId(): string {
    return `anthropic:${this.model}`;
  }
}

// Use it
import { Isotrieve, createAdapter } from '@isotrieve/core';

const adapter = createAdapter('anthropic', { apiKey: 'sk-ant-...' });
const agent = new Isotrieve({ embedder: adapter });
```

## Publishing Your Adapter

### As a Separate Package

#### Python (PyPI)

```bash
# Package structure
my-isotrieve-adapter/
├── my_isotrieve_adapter/
│   ├── __init__.py
│   └── adapter.py
├── setup.py
└── README.md
```

```python
# my_isotrieve_adapter/adapter.py
from isotrieve import adapter
from isotrieve.adapters.base import BaseAdapter

@adapter("my-provider")
class MyAdapter(BaseAdapter):
    # ... implementation

# setup.py
setup(
    name="my-isotrieve-adapter",
    install_requires=["isotrieve>=1.0.0"],
)
```

Users install with: `pip install my-isotrieve-adapter`

#### TypeScript (npm)

```bash
# Package structure
my-isotrieve-adapter/
├── src/
│   └── index.ts
├── package.json
└── README.md
```

```typescript
// src/index.ts
import { adapter, EmbeddingProvider } from '@isotrieve/core';

@adapter('my-provider')
export class MyAdapter implements EmbeddingProvider {
  // ... implementation
}
```

```json
// package.json
{
  "name": "@your-org/isotrieve-adapter-my-provider",
  "peerDependencies": {
    "@isotrieve/core": "^1.0.0"
  }
}
```

Users install with: `npm install @your-org/isotrieve-adapter-my-provider`

## Built-in Adapters

Both packages auto-register these adapters:

- `openai` - OpenAI embeddings
- `voyage` - Voyage AI embeddings
- `cohere` - Cohere embeddings
- `huggingface` - HuggingFace/SentenceTransformers (Python only)
- `mock` - Mock adapter for testing

## Best Practices

1. **Extend BaseAdapter (Python)** - Inherit from `BaseAdapter` to get input validation, retry logic, etc.

2. **Implement all methods** - Ensure your adapter implements all required methods

3. **Handle errors gracefully** - Wrap API calls in try-catch and provide clear error messages

4. **Support batch operations** - Implement efficient batch processing when possible

5. **Document your adapter** - Include clear docstrings and usage examples

6. **Test thoroughly** - Write tests for your adapter using `MockAdapter` as reference

7. **Version dependencies** - Specify compatible Isotrieve versions in your package

## Examples

See the `examples/` directory:
- Python: `examples/custom_adapter.py`
- TypeScript: `examples/custom-adapter-plugin/index.ts`
