# pgvector adapter

`PgvectorAdapter` migrates vectors stored in Postgres with the
[pgvector](https://github.com/pgvector/pgvector) extension, in place, without
re-embedding the corpus.

```bash
pip install isotrieve[pgvector]
```

Requires the `vector` extension enabled on the target database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Safety model: shadow column + atomic swap

Instead of copying the table (which doubles storage and is slow at scale),
migration adds a shadow column `embedding_isotrieve_new vector(D)`, transforms
stored vectors into it in idempotent batches, then swaps the columns
**atomically in a single transaction**:

```sql
BEGIN;
ALTER TABLE items RENAME COLUMN embedding TO embedding_isotrieve_old;
ALTER TABLE items RENAME COLUMN embedding_isotrieve_new TO embedding;
ALTER TABLE items DROP COLUMN IF EXISTS embedding_isotrieve_old;
COMMIT;
```

- **Rollback before the swap** = `DROP COLUMN embedding_isotrieve_new`. The live
  column is untouched until the swap, so the old model keeps serving the
  whole time.
- **Resume** = re-run `migrate()`. Batches are written with `UPDATE ... WHERE
  id = ...`, which is idempotent by primary key, so a killed run continues
  where it left off without double-writing.
- **Rollback after the swap** = restore from your pre-swap snapshot (the
  `embedding_isotrieve_old` column only exists mid-transaction; keep a backup or a
  logical snapshot before migrating for full post-swap rollback).

## Usage

```python
from isotrieve.adapters.pgvector import PgvectorAdapter
from isotrieve.mapping.registry import load_mapping

mapping = load_mapping("map.isotrieve")
adapter = PgvectorAdapter(
    mapping,
    dsn="postgresql://user:pass@localhost:5432/db",
    table="items",
    id_column="id",
    vector_column="embedding",
)

# Preflight: dry run only reports counts/dims
report = adapter.migrate(dry_run=True)

# Real migration: shadow column -> batch transform -> atomic swap
report = adapter.migrate(batch_size=1000)

# Rollback: drop the shadow column (safe before the swap)
adapter.rollback()

# Serve mode: queries mapped on-the-fly against the legacy index
hits = adapter.query(query_vectors, k=10)
```

The `query()` method maps new-model query vectors into legacy space with the
mapping's inverse transform before searching (`<=>` cosine distance). After
migration, construct the adapter with `mode="migrated"` to query the
transformed vectors directly.

## Index rebuild guidance

The column swap changes vector contents but **not** the column's index. After
migration you should rebuild the index so it reflects the new vector
geometry:

- **HNSW** — requires an index rebuild to incorporate the new vectors; the
  `ivfflat`/`hnsw` index on the swapped column is dropped and recreated:

  ```sql
  DROP INDEX IF EXISTS items_embedding_idx;
  CREATE INDEX items_embedding_idx ON items
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
  ```

- **IVFFlat** — the original index was trained on the source-model geometry
  and will mis-target after the swap; rebuild it (and re-run
  `SET ivfflat.probes` tuning):

  ```sql
  DROP INDEX IF EXISTS items_embedding_idx;
  CREATE INDEX items_embedding_idx ON items
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
  ```

Do the rebuild inside a maintenance window after `migrate()` completes. If
you keep the old index until the rebuild finishes, search still works — it
just returns results from the new vectors against the old index structure.

## Capabilities

| Flag | Value |
|---|---|
| Serve mode | ✅ `query()` (inverse-transform on the fly) |
| Offline migration | ✅ `migrate()` (in place) |
| Resume | ✅ idempotent `UPDATE ... WHERE id` |
| Rollback strategy | `shadow` (shadow column, atomic swap) |
| Tested in CI | ✅ dockerized Postgres/pgvector job |
