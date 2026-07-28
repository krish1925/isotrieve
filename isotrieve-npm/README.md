# Isotrieve — NPM packages

> **Pre-1.0 / Beta** — The TypeScript packages are ports of the Python [`isotrieve`](https://pypi.org/project/isotrieve/) package under active development. The Python package is the mature, benchmark-validated implementation. Do not depend on these packages in production until the production readiness plan (see root README) is closed.

## Packages

| Package | Description | Status |
|---|---|---|
| [`@isotrieve/core`](./packages/core/) | Mapping, quality gate, migration, recalibration, serve | Beta |
| `@isotrieve/adapters-openai` | OpenAI embedding provider | Legacy (old protocol API) |
| `@isotrieve/adapters-voyage` | Voyage AI embedding provider | Legacy (old protocol API) |
| `@isotrieve/adapters-cohere` | Cohere embedding provider | Legacy (old protocol API) |
| `@isotrieve/adapters-huggingface` | HuggingFace embedding provider | Legacy (old protocol API) |

**Only `@isotrieve/core` has been ported to the new embedding-migration architecture.** The adapter packages still reference the old `Isotrieve` class and `IsotrieveNegotiator` protocol API. They will be updated or removed before any 1.0 release.

## Which package is current?

**Use the Python package (`pip install isotrieve`) for production migrations.** The TypeScript packages are a work in progress. See the root [README.md](../README.md) for the full production readiness plan.

## Development

```bash
npm install
npm run build
npm test
```

## License

Apache-2.0. See [LICENSE](../isotrieve-python/LICENSE).
