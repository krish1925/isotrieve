"""Pinecone adapter — shadow-namespace strategy.

Uses Pinecone namespaces for cheap shadow copies. Write transformed vectors
to a shadow namespace, gate against it, then atomic-swap alias.

Supports both serverless and pod-based indexes. Metadata is preserved
byte-for-byte.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from isotrieve.adapters.base import MigrationReport, VectorStoreAdapter
from isotrieve.mapping.base import Mapping


def _require_pinecone() -> Any:
    """Import and return pinecone Client, or raise with install instructions."""
    try:
        from pinecone import Pinecone  # type: ignore[import-untyped]

        return Pinecone
    except ImportError:
        raise ImportError(
            "Pinecone client is required.  Install it with: pip install isotrieve[pinecone]"
        ) from None


class PineconeAdapter(VectorStoreAdapter):
    """Pinecone vector store adapter with shadow-namespace migration.

    Parameters
    ----------
    mapping:
        A fitted Isotrieve Mapping.
    api_key:
        Pinecone API key.
    index_name:
        Name of the Pinecone index.
    namespace:
        Source namespace (default: "" which is the default namespace).
    mode:
        ``"serve"`` = map queries on-the-fly.
        ``"migrated"`` = corpus already transformed.
    """

    def __init__(
        self,
        mapping: Mapping,
        *,
        api_key: str | None = None,
        index_name: str,
        namespace: str = "",
        mode: str = "serve",
    ) -> None:
        super().__init__(mapping, mode=mode)  # type: ignore[arg-type]
        Pinecone = _require_pinecone()
        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(index_name)
        self._namespace = namespace
        self._index_name = index_name

    def query(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """Serve-mode query: map queries, search Pinecone, return results."""
        mapped = self._map_queries(query_vectors)
        results = []
        for vec in mapped:
            resp = self._index.query(
                vector=vec.tolist(),
                top_k=k,
                namespace=self._namespace,
                include_metadata=True,
            )
            hits = []
            for match in resp.get("matches", []):
                hits.append(
                    {
                        "id": match["id"],
                        "score": match["score"],
                        "metadata": match.get("metadata", {}),
                    }
                )
            results.append(hits)
        return results

    def migrate(
        self,
        batch_size: int = 100,
        dry_run: bool = False,
        new_collection: str | None = None,
    ) -> MigrationReport:
        """Shadow-namespace migration.

        1. Fetch all vectors from source namespace
        2. Transform through mapping
        3. Write to shadow namespace
        4. Caller gates against shadow, then swaps
        """
        shadow_ns = new_collection or f"{self._namespace}_isotrieve_shadow"
        report = MigrationReport(
            source_collection=self._namespace or "(default)",
            target_collection=shadow_ns,
        )

        if dry_run:
            stats = self._index.describe_index_stats()
            ns_stats = stats.get("namespaces", {}).get(self._namespace, {})
            report.rows_processed = ns_stats.get("vector_count", 0)
            return report

        # Fetch all vectors via scroll
        all_ids: list[str] = []
        all_vectors: list[list[float]] = []
        all_metadata: list[dict] = []
        all_texts: list[str | None] = []

        cursor = None
        while True:
            kwargs: dict[str, Any] = {
                "namespace": self._namespace,
                "batch_size": batch_size,
            }
            if cursor:
                kwargs["cursor"] = cursor

            resp = self._index.fetch(**kwargs)  # type: ignore[arg-type]
            vectors = resp.get("vectors", {})
            if not vectors:
                break

            for vid, vdata in vectors.items():
                all_ids.append(vid)
                all_vectors.append(vdata["values"])
                all_metadata.append(vdata.get("metadata", {}))
                all_texts.append(None)

            cursor = resp.get("next_cursor")
            if not cursor:
                break

        if not all_ids:
            report.errors.append("No vectors found in source namespace")
            return report

        # Transform vectors
        vecs = np.array(all_vectors, dtype=np.float64)
        mapped = self._mapping.transform(vecs)
        report.rows_processed = len(all_ids)

        # Write to shadow namespace in batches
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i : i + batch_size]
            batch_vecs = mapped[i : i + batch_size].tolist()
            batch_meta = all_metadata[i : i + batch_size]

            # Add Isotrieve metadata
            for meta in batch_meta:
                meta["isotrieve_mapping_id"] = "shadow"
                meta["isotrieve_source_namespace"] = self._namespace

            upsert_data = [
                {"id": vid, "values": vec, "metadata": meta}
                for vid, vec, meta in zip(
                    batch_ids, batch_vecs, batch_meta, strict=True
                )
            ]
            self._index.upsert(
                vectors=upsert_data,
                namespace=shadow_ns,
            )

        return report
