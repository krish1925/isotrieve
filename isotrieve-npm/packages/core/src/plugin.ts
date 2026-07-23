/**
 * Isotrieve Plugin System
 * 
 * Provides utilities for creating and registering custom embedding adapters
 * without modifying the core Isotrieve codebase.
 */

import { EmbeddingProvider } from './types';

/**
 * Type for adapter factory functions
 */
export type AdapterFactory = (...args: any[]) => EmbeddingProvider;

/**
 * Type for adapter constructors
 */
export type AdapterConstructor = new (...args: any[]) => EmbeddingProvider;

/**
 * Registry for custom embedding provider plugins
 */
class PluginRegistryClass {
  private adapters: Map<string, AdapterConstructor> = new Map();
  private factories: Map<string, AdapterFactory> = new Map();

  /**
   * Register a custom adapter
   * 
   * @param name - Unique identifier for the adapter
   * @param adapter - Adapter class or factory function
   * @param options - Registration options
   * 
   * @example
   * ```typescript
   * class MyAdapter implements EmbeddingProvider {
   *   // ... implementation
   * }
   * 
   * PluginRegistry.register('my-provider', MyAdapter);
   * ```
   */
  register(
    name: string,
    adapter: AdapterConstructor | AdapterFactory,
    options: { override?: boolean } = {}
  ): void {
    const { override = false } = options;

    if (!override && (this.adapters.has(name) || this.factories.has(name))) {
      throw new Error(
        `Adapter '${name}' already registered. Use override: true to replace it.`
      );
    }

    // Check if it's a class or function
    if (this.isConstructor(adapter)) {
      this.adapters.set(name, adapter as AdapterConstructor);
    } else {
      this.factories.set(name, adapter as AdapterFactory);
    }
  }

  /**
   * Unregister an adapter
   */
  unregister(name: string): void {
    this.adapters.delete(name);
    this.factories.delete(name);
  }

  /**
   * Get a registered adapter class
   */
  get(name: string): AdapterConstructor {
    const adapter = this.adapters.get(name);
    if (!adapter) {
      throw new Error(
        `Adapter '${name}' not found. Available: ${this.listNames().join(', ')}`
      );
    }
    return adapter;
  }

  /**
   * Get a registered adapter factory
   */
  getFactory(name: string): AdapterFactory {
    const factory = this.factories.get(name);
    if (!factory) {
      throw new Error(
        `Factory for '${name}' not found. Available: ${this.listNames().join(', ')}`
      );
    }
    return factory;
  }

  /**
   * Create an adapter instance
   */
  create(name: string, ...args: any[]): EmbeddingProvider {
    if (this.factories.has(name)) {
      return this.getFactory(name)(...args);
    } else if (this.adapters.has(name)) {
      const AdapterClass = this.get(name);
      return new AdapterClass(...args);
    } else {
      throw new Error(`Adapter '${name}' not found`);
    }
  }

  /**
   * Check if an adapter is registered
   */
  has(name: string): boolean {
    return this.adapters.has(name) || this.factories.has(name);
  }

  /**
   * List all registered adapter names
   */
  listNames(): string[] {
    return [
      ...Array.from(this.adapters.keys()),
      ...Array.from(this.factories.keys()),
    ];
  }

  /**
   * Get all registered adapters
   */
  listAdapters(): Record<string, AdapterConstructor | AdapterFactory> {
    const result: Record<string, AdapterConstructor | AdapterFactory> = {};
    
    for (const [name, adapter] of this.adapters.entries()) {
      result[name] = adapter;
    }
    
    for (const [name, factory] of this.factories.entries()) {
      result[name] = factory;
    }
    
    return result;
  }

  /**
   * Clear all registered adapters (mainly for testing)
   */
  clear(): void {
    this.adapters.clear();
    this.factories.clear();
  }

  /**
   * Check if a value is a constructor
   */
  private isConstructor(value: any): boolean {
    try {
      // Check if it's a class (has prototype and can be called with new)
      return !!(value.prototype && value.prototype.constructor === value);
    } catch {
      return false;
    }
  }
}

/**
 * Global plugin registry
 */
export const PluginRegistry = new PluginRegistryClass();

/**
 * Register a custom adapter
 * 
 * @param name - Unique identifier for the adapter
 * @param adapter - Adapter class or factory function
 * @param options - Registration options
 * 
 * @example
 * ```typescript
 * import { registerAdapter, EmbeddingProvider } from '@isotrieve/core';
 * 
 * class MyCustomAdapter implements EmbeddingProvider {
 *   async embed(text: string): Promise<number[]> {
 *     // Your implementation
 *     return [0.1, 0.2, 0.3];
 *   }
 *   
 *   async embedBatch(texts: string[]): Promise<number[][]> {
 *     return texts.map(t => this.embed(t));
 *   }
 *   
 *   getDimensions(): number {
 *     return 3;
 *   }
 *   
 *   getModelId(): string {
 *     return 'my-model';
 *   }
 * }
 * 
 * registerAdapter('my-provider', MyCustomAdapter);
 * ```
 */
export function registerAdapter(
  name: string,
  adapter: AdapterConstructor | AdapterFactory,
  options: { override?: boolean } = {}
): void {
  PluginRegistry.register(name, adapter, options);
}

/**
 * Get a registered adapter class
 * 
 * @example
 * ```typescript
 * import { getAdapter } from '@isotrieve/core';
 * 
 * const MyAdapter = getAdapter('my-provider');
 * const adapter = new MyAdapter({ apiKey: '...' });
 * ```
 */
export function getAdapter(name: string): AdapterConstructor {
  return PluginRegistry.get(name);
}

/**
 * Get a registered adapter factory
 */
export function getAdapterFactory(name: string): AdapterFactory {
  return PluginRegistry.getFactory(name);
}

/**
 * Create an adapter instance
 * 
 * @example
 * ```typescript
 * import { createAdapter } from '@isotrieve/core';
 * 
 * const adapter = createAdapter('my-provider', { apiKey: '...' });
 * ```
 */
export function createAdapter(name: string, ...args: any[]): EmbeddingProvider {
  return PluginRegistry.create(name, ...args);
}

/**
 * List all registered adapters
 * 
 * @example
 * ```typescript
 * import { listAdapters } from '@isotrieve/core';
 * 
 * const adapters = listAdapters();
 * console.log('Available adapters:', Object.keys(adapters));
 * ```
 */
export function listAdapters(): Record<string, AdapterConstructor | AdapterFactory> {
  return PluginRegistry.listAdapters();
}

/**
 * Decorator for registering an adapter class
 * 
 * @example
 * ```typescript
 * import { adapter, EmbeddingProvider } from '@isotrieve/core';
 * 
 * @adapter('my-provider')
 * class MyAdapter implements EmbeddingProvider {
 *   // ... implementation
 * }
 * ```
 */
export function adapter(name: string, options: { override?: boolean } = {}) {
  return function <T extends AdapterConstructor>(target: T): T {
    registerAdapter(name, target, options);
    return target;
  };
}

/**
 * Check if an adapter is registered
 */
export function hasAdapter(name: string): boolean {
  return PluginRegistry.has(name);
}

/**
 * Unregister an adapter
 */
export function unregisterAdapter(name: string): void {
  PluginRegistry.unregister(name);
}
