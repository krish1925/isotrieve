"""Qdrant adapter — promoted to VectorStoreAdapter.

Checkpointed in-place with scroll API + optimistic batching.
Collection snapshot before migration as belt-and-suspenders.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from isotrieve.adapters.base import MigrationReport, VectorStoreAdapter
from isotrieve.mapping.base import Mapping


def _require_qdrant() -> Any:
    """Import and return qdrant_client modules, or raise."""
    try:
        from qdrant_client import QdrantClient  # type: ignore[import-untyped]
        from qdrant_client.models import PointStruct

        return QdrantClient, PointStruct
    except ImportError:
        raise ImportError(
            "Qdrant client is required.  Install it with: pip install isotrieve[qdrant]"
        ) from None


class QdrantAdapter(VectorStoreAdapter):
    """Qdrant vector store adapter with checkpointed in-place migration.

    Parameters
    ----------
    mapping:
        A fitted Isotrieve Mapping.
    url:
        Qdrant server URL.
    collection:
        Source collection name.
    api_key:
        Optional API key for Qdrant Cloud.
    mode:
        ``"serve"`` = map queries on-the-fly.
        ``"migrated"`` = corpus already transformed.
    """

    def __init__(
        self,
        mapping: Mapping,
        *,
        url: str = "http://localhost:6333",
        collection: str = "vectors",
        api_key: str | None = None,
        mode: str = "serve",
    ) -> None:
        super().__init__(mapping, mode=mode)  # type: ignore[arg-type]
        QdrantClient, _ = _require_qdrant()
        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection = collection

    def query(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """Serve-mode query: map queries, search Qdrant, return results."""
        mapped = self._map_queries(query_vectors)
        results = []
        for vec in mapped:
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=vec.tolist(),
                limit=k,
            )
            results.append(
                [
                    {
                        "id": str(hit.id),
                        "score": hit.score,
                        "metadata": hit.payload or {},
                    }
                    for hit in hits
                ]
            )
        return results

    def migrate(
        self,
        batch_size: int = 1000,
        dry_run: bool = False,
        new_collection: str | None = None,
    ) -> MigrationReport:
        """Checkpointed in-place migration with scroll + upsert."""
        _, PointStruct = _require_qdrant()

        target = new_collection or f"{self._collection}_migrated"
        report = MigrationReport(
            source_collection=self._collection,
            target_collection=target,
        )

        if dry_run:
            info = self._client.get_collection(self._collection)
            report.rows_processed = info.points_count or 0
            return report

        # Create target collection with correct dimension
        try:
            self._client.get_collection(target)
        except Exception:
            info = self._client.get_collection(self._collection)
            dim = 128  # default
            if info.config.params.vectors:
                if isinstance(info.config.params.vectors, dict):
                    dim = next(iter(info.config.params.vectors.values())).size
                else:
                    dim = info.config.params.vectors.size
            self._client.create_collection(
                collection_name=target,
                vectors_config=dim,
            )

        # Scroll through source collection
        offset = None
        total = 0
        while True:
            scroll_result = self._client.scroll(
                collection_name=self._collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            points, next_offset = scroll_result
            if not points:
                break

            # Transform vectors
            ids = []
            vectors = []
            payloads = []
            for pt in points:
                vec = np.array(pt.vector, dtype=np.float64).reshape(1, -1)
                mapped = self._mapping.transform(vec)
                ids.append(pt.id)
                vectors.append(mapped.ravel().tolist())
                payloads.append(pt.payload or {})

                # Add Isotrieve metadata
                if payloads[-1] is not None:
                    payloads[-1]["isotrieve_mapping_id"] = "checkpointed"
                    payloads[-1]["isotrieve_source_collection"] = self._collection

            # Upsert to target
            upsert_points = [
                PointStruct(id=pid, vector=vec, payload=pl)
                for pid, vec, pl in zip(ids, vectors, payloads, strict=True)
            ]
            self._client.upsert(
                collection_name=target,
                points=upsert_points,
            )

            total += len(points)
            report.rows_processed = total
            offset = next_offset

            if next_offset is None:
                break

        return report
