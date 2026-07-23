# Contributing to Isotrieve

Thank you for your interest in contributing to Isotrieve!

## Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/isotrieve.git
cd isotrieve

# Install dependencies
npm install

# Build all packages
npm run build

# Run tests
npm test
```

## Project Structure

```
isotrieve-npm/
├── packages/
│   ├── core/                    # Core protocol
│   ├── adapters-openai/         # OpenAI adapter
│   ├── adapters-voyage/         # Voyage adapter
│   ├── adapters-cohere/         # Cohere adapter
│   └── adapters-huggingface/    # HuggingFace adapter
├── examples/                    # Usage examples
└── docs/                        # Documentation
```

## Adding a New Adapter

1. Create package directory:
```bash
mkdir -p packages/adapters-yourprovider/src
```

2. Implement `EmbeddingProvider` interface:
```typescript
import { EmbeddingProvider } from '@isotrieve/core';

export class YourAdapter implements EmbeddingProvider {
  async embed(text: string): Promise<number[]> {
    // Your implementation
  }
  
  async embedBatch(texts: string[]): Promise<number[][]> {
    // Your implementation
  }
  
  getDimensions(): number {
    // Return dimensions
  }
  
  getModelId(): string {
    // Return model ID
  }
}
```

3. Add package.json and tsconfig.json

4. Add tests and documentation

5. Submit pull request

## Code Style

- Use TypeScript
- Follow existing code patterns
- Add JSDoc comments
- Keep functions small and focused
- No external dependencies in core package

## Testing

```bash
# Run tests
npm test

# Run linter
npm run lint

# Build
npm run build
```

## Documentation

- Update README.md for new features
- Add examples for new adapters
- Update API reference
- Include inline code comments

## Pull Request Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## Questions?

Open an issue or reach out to maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
