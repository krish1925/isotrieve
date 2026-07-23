/**
 * Example: Creating a Custom Adapter with Plugin System
 * 
 * Shows how to create your own embedding provider adapter
 * and use it with Isotrieve without modifying the core code.
 */

import {
  EmbeddingProvider,
  Isotrieve,
  registerAdapter,
  getAdapter,
  createAdapter,
  listAdapters,
  adapter,
} from '@isotrieve/core';

/**
 * Method 1: Regular class with manual registration
 */
class MyCustomAdapter implements EmbeddingProvider {
  private apiKey: string;
  private model: string;
  private dimensions: number;

  constructor(config: { apiKey: string; model?: string }) {
    this.apiKey = config.apiKey;
    this.model = config.model || 'my-model-v1';
    this.dimensions = 128; // Your model's dimensions
  }

  async embed(text: string): Promise<number[]> {
    // Replace this with your actual API call
    console.log(`Embedding text with ${this.model}: "${text}"`);
    
    // Example: Simple hash-based embedding (replace with real API call)
    const hash = this.simpleHash(text);
    
    // Generate deterministic values
    const embedding: number[] = [];
    for (let i = 0; i < this.dimensions; i++) {
      const val = ((hash + i) % 1000) / 1000.0;
      embedding.push(val);
    }
    
    return embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    // Batch processing - replace with real batch API call
    return Promise.all(texts.map(text => this.embed(text)));
  }

  getDimensions(): number {
    return this.dimensions;
  }

  getModelId(): string {
    return `custom:${this.model}`;
  }

  private simpleHash(text: string): number {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      hash = ((hash << 5) - hash) + text.charCodeAt(i);
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }
}

/**
 * Method 2: Using decorator (auto-registers)
 */
@adapter('my-other-provider')
class AnotherCustomAdapter implements EmbeddingProvider {
  private apiEndpoint: string;
  private dimensions: number = 256;

  constructor(config: { apiEndpoint: string }) {
    this.apiEndpoint = config.apiEndpoint;
  }

  async embed(text: string): Promise<number[]> {
    // Your implementation here
    console.log(`Embedding via ${this.apiEndpoint}: "${text}"`);
    return new Array(this.dimensions).fill(0.1);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map(t => this.embed(t)));
  }

  getDimensions(): number {
    return this.dimensions;
  }

  getModelId(): string {
    return `another:${this.apiEndpoint}`;
  }
}

/**
 * Method 3: Factory function
 */
function createMyAdapter(config: { apiKey: string }): EmbeddingProvider {
  return new MyCustomAdapter(config);
}

// Example 1: Manual registration
async function exampleManualRegistration() {
  console.log('='.repeat(60));
  console.log('Example 1: Manual Registration');
  console.log('='.repeat(60));

  // Register the adapter
  registerAdapter('my-custom-provider', MyCustomAdapter);

  // Create agent using the custom adapter
  const AdapterClass = getAdapter('my-custom-provider');
  const adapter = new AdapterClass({ apiKey: 'fake-key-123' });

  const agent = new Isotrieve({
    embedder: adapter,
    agentId: 'custom_agent',
  });

  // Use it
  const embedding = await agent.embed('Hello from my custom adapter!');
  console.log(`Generated embedding with ${embedding.length} dimensions`);
  console.log(`First 5 values: ${embedding.slice(0, 5)}`);
}

// Example 2: Using decorator-registered adapter
async function exampleDecoratorUsage() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 2: Decorator Registration');
  console.log('='.repeat(60));

  // Already registered via decorator
  const adapter = createAdapter('my-other-provider', {
    apiEndpoint: 'https://api.example.com',
  });

  const agent = new Isotrieve({
    embedder: adapter,
    agentId: 'another_agent',
  });

  const embedding = await agent.embed('Hello from decorator adapter!');
  console.log(`Generated embedding with ${embedding.length} dimensions`);
}

// Example 3: Factory registration
async function exampleFactoryRegistration() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 3: Factory Registration');
  console.log('='.repeat(60));

  // Register a factory function
  registerAdapter('my-factory-provider', createMyAdapter);

  const adapter = createAdapter('my-factory-provider', {
    apiKey: 'factory-key',
  });

  const agent = new Isotrieve({
    embedder: adapter,
    agentId: 'factory_agent',
  });

  const embedding = await agent.embed('Hello from factory!');
  console.log(`Generated embedding with ${embedding.length} dimensions`);
}

// Example 4: List all adapters
function exampleListAdapters() {
  console.log('\n' + '='.repeat(60));
  console.log('Example 4: List All Registered Adapters');
  console.log('='.repeat(60));

  const adapters = listAdapters();
  const names = Object.keys(adapters);
  
  console.log(`Found ${names.length} registered adapters:`);
  names.forEach(name => {
    console.log(`  - ${name}`);
  });
}

// Run all examples
async function main() {
  try {
    await exampleManualRegistration();
    await exampleDecoratorUsage();
    await exampleFactoryRegistration();
    exampleListAdapters();

    console.log('\n' + '='.repeat(60));
    console.log('All examples completed!');
    console.log('='.repeat(60));
  } catch (error) {
    console.error('Error running examples:', error);
  }
}

// Run if executed directly
if (require.main === module) {
  main();
}

export {
  MyCustomAdapter,
  AnotherCustomAdapter,
  createMyAdapter,
};
