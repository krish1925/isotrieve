"""LlamaIndex storage-context migration path.

Detects which concrete store LlamaIndex wraps (Chroma, pgvector, etc.)
and delegates to the appropriate Isotrieve adapter rather than reimplementing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from isotrieve.adapters.base import MigrationReport


def migrate_llamaindex_store(
    vector_store: Any,
    mapping: Any,
    *,
    batch_size: int = 1000,
    dry_run: bool = False,
    new_collection: str | None = None,
) -> MigrationReport:
    """Migrate a LlamaIndex VectorStore by detecting the underlying store.

    If LlamaIndex wraps a store we already support (Chroma, pgvector),
    we delegate to the concrete adapter.  Otherwise, we use the generic
    ``VectorStore`` interface via ``iter_vectors``/``write_vectors``.

    Parameters
    ----------
    vector_store:
        A LlamaIndex ``VectorStore`` instance.
    mapping:
        A fitted Isotrieve Mapping.
    batch_size:
        Batch size for streaming migration.
    dry_run:
        If True, count vectors without writing.
    new_collection:
        Target collection/namespace name.

    Returns
    -------
    MigrationReport
    """
    # Try to detect the underlying store type
    store_type = type(vector_store).__module__

    # Chroma detection
    if "chroma" in store_type.lower():
        return _migrate_via_chroma(
            vector_store, mapping, batch_size, dry_run, new_collection
        )

    # Qdrant detection
    if "qdrant" in store_type.lower():
        return _migrate_via_qdrant(
            vector_store, mapping, batch_size, dry_run, new_collection
        )

    # Generic fallback: use LlamaIndex's own interface
    return _migrate_generic(vector_store, mapping, batch_size, dry_run, new_collection)


def _migrate_via_chroma(
    vector_store: Any,
    mapping: Any,
    batch_size: int,
    dry_run: bool,
    new_collection: str | None,
) -> MigrationReport:
    """Delegate to ChromaDB adapter."""
    from isotrieve.adapters.chroma import migrate_collection

    # LlamaIndex Chroma stores expose the underlying client
    client = getattr(vector_store, "client", None)
    collection_name = getattr(vector_store, "collection_name", None)

    if client is None or collection_name is None:
        raise ValueError(
            "Cannot detect ChromaDB client from LlamaIndex VectorStore. "
            "Use the ChromaDB adapter directly."
        )

    return migrate_collection(
        client,
        collection_name,
        mapping,
        batch_size=batch_size,
        dry_run=dry_run,
        new_collection=new_collection,
    )


def _migrate_via_qdrant(
    vector_store: Any,
    mapping: Any,
    batch_size: int,
    dry_run: bool,
    new_collection: str | None,
) -> MigrationReport:
    """Delegate to Qdrant adapter."""
    from isotrieve.adapters.qdrant import QdrantAdapter

    client = getattr(vector_store, "client", None)
    collection = getattr(vector_store, "collection_name", None)

    if client is None or collection is None:
        raise ValueError(
            "Cannot detect Qdrant client from LlamaIndex VectorStore. "
            "Use the Qdrant adapter directly."
        )

    adapter = QdrantAdapter(
        mapping,
        url=getattr(client, "_url", "http://localhost:6333"),
        collection=collection,
    )
    return adapter.migrate(
        batch_size=batch_size,
        dry_run=dry_run,
        new_collection=new_collection,
    )


def _migrate_generic(
    vector_store: Any,
    mapping: Any,
    batch_size: int,
    dry_run: bool,
    new_collection: str | None,
) -> MigrationReport:
    """Generic migration using LlamaIndex's VectorStore interface.

    Uses ``vector_store.get()`` and ``vector_store.add()`` with batching.
    """
    report = MigrationReport(
        source_collection=type(vector_store).__name__,
        target_collection=new_collection or "migrated",
    )

    # Get all vector IDs
    try:
        # LlamaIndex VectorStore.get() returns a VectorStoreQueryResult
        # We need to iterate through all vectors
        all_ids = []
        all_vectors = []
        all_metadatas = []

        # Generic approach: use the store's own query mechanism
        # This is a best-effort migration for unknown store types
        result = vector_store.get()
        if hasattr(result, "ids"):
            all_ids = list(result.ids)
        if hasattr(result, "embeddings"):
            all_vectors = [np.array(e) for e in result.embeddings]
        if hasattr(result, "metadatas"):
            if result.metadatas:
                all_metadatas = list(result.metadatas)
            else:
                all_metadatas = [{}] * len(all_ids)

    except Exception as e:
        report.errors.append(f"Cannot read from store: {e}")
        return report

    if not all_ids:
        return report

    if dry_run:
        report.rows_processed = len(all_ids)
        return report

    # Transform vectors
    vecs = np.array(all_vectors, dtype=np.float64)
    mapped = mapping.transform(vecs)
    report.rows_processed = len(all_ids)

    # Write back (best effort — LlamaIndex stores may not support update)
    try:
        from llama_index.core.vector_stores.types import (
            VectorStoreData,  # type: ignore[import-untyped]
        )

        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i : i + batch_size]
            batch_vecs = mapped[i : i + batch_size]
            if all_metadatas:
                batch_meta = all_metadatas[i : i + batch_size]
            else:
                batch_meta = [{}] * len(batch_ids)

            data = VectorStoreData(
                ids=batch_ids,
                embeddings=batch_vecs.tolist(),
                metadatas=batch_meta,
            )
            vector_store.add(data)
    except ImportError:
        report.errors.append(
            "LlamaIndex VectorStoreData not available. "
            "Use the concrete store adapter directly."
        )
    except Exception as e:
        report.errors.append(f"Write error: {e}")

    return report
