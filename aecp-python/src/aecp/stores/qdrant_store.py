"""Qdrant vector store adapter (WS-4).

Requires: pip install aecp[qdrant]
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from aecp.stores.base import VectorRecord, VectorStore


def _require_qdrant():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance,
            PointStruct,
            VectorParams,
        )

        return QdrantClient, Distance, PointStruct, VectorParams
    except ImportError:
        raise ImportError(
            "Qdrant adapter requires qdrant-client. "
            "Install with: pip install aecp[qdrant]"
        )


class QdrantStore(VectorStore):
    """Qdrant vector store adapter with resumable migration support."""

    def __init__(
        self,
        client: Any,
        collection: str,
        text_field: str = "text",
    ) -> None:
        self._client = client
        self._collection = collection
        self._text_field = text_field

    @classmethod
    def connect(
        cls,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection: str = "vectors",
        text_field: str = "text",
    ) -> QdrantStore:
        """Connect to a Qdrant instance."""
        QdrantClient, _, _, _ = _require_qdrant()
        client = QdrantClient(url=url, api_key=api_key)
        return cls(client, collection, text_field)

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count or 0

    def iter_vectors(self, batch_size: int = 1024) -> Iterator[list[VectorRecord]]:
        """Stream vectors with server-side pagination."""
        offset = None
        while True:
            result, offset = self._client.scroll(
                collection_name=self._collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not result:
                break
            records = []
            for point in result:
                text = None
                payload = point.payload or {}
                if self._text_field in payload:
                    text = payload[self._text_field]
                records.append(
                    VectorRecord(
                        id=str(point.id),
                        vector=np.array(point.vector, dtype=np.float32),
                        text=text,
                        payload=payload,
                    )
                )
            yield records
            if offset is None:
                break

    def write_vectors(
        self,
        records: Iterator[list[VectorRecord]] | list[VectorRecord],
        *,
        batch_size: int = 1024,
    ) -> int:
        """Write records to Qdrant.

        Accepts either:
        - ``Iterator[list[VectorRecord]]``: lazily yields batches
        - ``list[VectorRecord]``: flat list treated as a single batch
        """
        QdrantClient, Distance, PointStruct, VectorParams = _require_qdrant()

        # Normalize both input forms into an iterator of batches
        if isinstance(records, list):

            def _flat_batches() -> Iterator[list[VectorRecord]]:
                yield records

            batches: Iterator[list[VectorRecord]] = _flat_batches()
        else:
            batches = records

        # Lazily peek at first batch to get dimensions
        first_batch = next(batches, None)
        if not first_batch:
            return 0
        dims = len(first_batch[0].vector)

        # Create target collection if needed
        try:
            self._client.get_collection(self._collection)
        except Exception:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=dims,
                    distance=Distance.COSINE,
                ),
            )

        total = 0

        def _all_batches() -> Iterator[list[VectorRecord]]:
            yield first_batch
            yield from batches  # lazy — no materialization

        for batch in _all_batches():
            if not batch:
                continue
            points = [
                PointStruct(
                    id=r.id,
                    vector=r.vector.tolist(),
                    payload=r.payload or {},
                )
                for r in batch
            ]
            self._client.upsert(
                collection_name=self._collection,
                points=points,
            )
            total += len(batch)
        return total

    def create_target(
        self,
        target_collection: str,
        dims: int,
        distance: str = "cosine",
    ) -> None:
        """Create a target collection for migration."""
        QdrantClient, Distance, _, VectorParams = _require_qdrant()
        dist = Distance.COSINE if distance == "cosine" else Distance.EUCLID
        try:
            self._client.get_collection(target_collection)
        except Exception:
            self._client.create_collection(
                collection_name=target_collection,
                vectors_config=VectorParams(size=dims, distance=dist),
            )

    def prepare_target(self, dims: int, distance: str = "cosine") -> str:
        """Create target collection, return its name."""
        target = f"{self._collection}_migrated"
        self.create_target(target, dims, distance)
        return target
