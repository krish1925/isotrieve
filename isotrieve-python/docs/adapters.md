# Adapter capability matrix

Automatically generated from adapter class attributes by
`python isotrieve-python/scripts/gen_adapter_matrix.py`. **Do not edit by hand.**

| Adapter | Kind | Serve mode | Offline migration | Idempotency guard | Resume | Rollback strategy | Tested in CI |
|---|---|---|---|---|---|---|---|
| `VectorStoreAdapter (base)` | Base | Yes | Yes | — | — | `none` | — |
| `QdrantAdapter` | Query | Yes | Yes | — | Yes | `snapshot` | Yes |
| `PineconeAdapter` | Query | Yes | Yes | — | — | `shadow` | — |
| `IsotrieveChromaFunction` | Serve | Yes | — | — | — | `none` | Yes |
