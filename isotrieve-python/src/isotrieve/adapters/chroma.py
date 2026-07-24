"""ChromaDB adapter for Isotrieve.

Two modes:
- **Serve mode**: ``IsotrieveChromaFunction`` — a Chroma ``EmbeddingFunction`` that
  transparently maps new-model queries into legacy space before searching.
- **Offline migration**: ``migrate_collection()`` — transforms stored vectors
  into the new model's space and writes to a new collection.

Requires: ``pip install isotrieve[chroma]`` or ``pip install chromadb``.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from isotrieve.adapters.base import MigrationReport
from isotrieve.mapping.base import Mapping, l2_normalize


def _require_chroma():
    try:
        import chromadb
        from chromadb import Documents, EmbeddingFunction, Embeddings

        return chromadb, Documents, EmbeddingFunction, Embeddings
    except ImportError:
        raise ImportError(
            "ChromaDB adapter requires chromadb. Install with: pip install chromadb"
        )


class IsotrieveChromaFunction:
    """Chroma ``EmbeddingFunction`` that applies an Isotrieve mapping.

    In serve mode, ``__call__`` embeds with the new model then maps into
    legacy space, so the legacy index is searched without modification.

    Usage::

        from isotrieve.adapters.chroma import IsotrieveChromaFunction
        from isotrieve.mapping.base import Mapping

        mapping = Mapping.load("ada002_to_te3.isotrieve")
        ef = IsotrieveChromaFunction(mapping, new_model_embedder=my_embed_fn)
        col = client.get_collection("docs", embedding_function=ef)
        results = col.query(query_texts=["..."], n_results=10)
    """

    def __init__(
        self,
        mapping: Mapping,
        new_model_embedder: Any = None,
        *,
        recalibrate: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        mapping:
            Fitted Isotrieve mapping with inverse direction.
        new_model_embedder:
            Callable that takes ``list[str]`` and returns ``np.ndarray``
            of shape ``(n, d_new)``. If None, the EF embeds with the
            new model (caller must provide this).
        recalibrate:
            If True and the mapping has a recalibrator, post-process
            query scores.
        """
        self._mapping = mapping
        self._embedder = new_model_embedder
        self._recalibrate = recalibrate

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed texts into legacy space (for querying legacy index)."""
        if self._embedder is None:
            raise RuntimeError(
                "No new_model_embedder provided. Pass a callable that "
                "embeds text with the new model."
            )
        # Embed with new model
        new_vecs = np.asarray(self._embedder(input), dtype=np.float64)
        # Map to legacy space
        legacy_vecs = l2_normalize(self._mapping.inverse_transform(new_vecs))
        return legacy_vecs.tolist()

    def embed_query(self, input: str) -> list[float]:
        """Embed a single query into legacy space."""
        return self.__call__([input])[0]

    @property
    def has_recalibrator(self) -> bool:
        return self._mapping.has_recalibrator

    def default_space(self) -> str:
        return "cosine"

    @property
    def name(self) -> str:
        return "isotrieve_chroma"

    def get_config(self) -> dict[str, Any]:
        return {
            "mapping_type": self._mapping.mapping_type,
            "d_src": self._mapping._d_src,
            "d_tgt": self._mapping._d_tgt,
        }


def _hash_mapping(mapping: Mapping) -> str:
    """Short checksum of the mapping matrix for metadata."""
    if mapping._W is None:
        return "unfitted"
    h = hashlib.sha256(mapping._W.tobytes()).hexdigest()[:12]
    return h


def migrate_collection(
    client: Any,
    collection_name: str,
    mapping: Mapping,
    *,
    new_collection: str | None = None,
    batch_size: int = 1000,
    dry_run: bool = False,
    new_model_embedder: Any = None,
) -> MigrationReport:
    """Migrate a Chroma collection from legacy to new model space.

    Reads all vectors from ``collection_name``, applies the forward
    mapping (legacy → new), and writes to ``new_collection``.
    Source collection is never modified.

    Parameters
    ----------
    client:
        ``chromadb.Client`` instance.
    collection_name:
        Source collection (legacy embeddings).
    mapping:
        Fitted Isotrieve mapping (forward direction: legacy → new).
    new_collection:
        Target name. Default: ``{collection_name}_migrated``.
    batch_size:
        Rows per read/write batch.
    dry_run:
        If True, read and transform but don't write.
    new_model_embedder:
        If provided, re-embed texts with new model instead of mapping
        (for comparison / ground-truth baseline).

    Returns
    -------
    MigrationReport with rows processed, timing, and sampled recall.
    """
    chromadb, _, _, _ = _require_chroma()

    t0 = time.perf_counter()
    target_name = new_collection or f"{collection_name}_migrated"
    report = MigrationReport(
        source_collection=collection_name,
        target_collection=target_name,
        mapping_checksum=_hash_mapping(mapping),
    )

    # Open source collection
    src = client.get_collection(collection_name)
    total = src.count()

    if dry_run:
        # Just read a batch and report dimensions
        sample = src.get(
            limit=min(batch_size, total),
            include=["embeddings", "metadatas", "documents"],
        )
        if sample["embeddings"]:
            dims = len(sample["embeddings"][0])
            report.rows_processed = len(sample["embeddings"])
            report.elapsed_seconds = time.perf_counter() - t0
            report.errors.append(
                f"DRY RUN: would migrate {total} vectors of dim {dims}"
            )
        return report

    # Check for double-migration: does the source already have isotrieve metadata?
    sample_meta = src.get(limit=1, include=["metadatas"])
    if sample_meta["metadatas"] and "isotrieve_mapping_id" in (
        sample_meta["metadatas"][0] or {}
    ):
        report.idempotent = False
        report.errors.append(
            "Source collection already has isotrieve_mapping_id metadata. "
            "This may be a double migration. Proceeding anyway."
        )

    # Create target collection with correct dimensions
    # Read first batch to determine dims
    first_batch = src.get(
        limit=batch_size, include=["embeddings", "metadatas", "documents"]
    )
    if src.count() == 0:
        report.errors.append("Source collection is empty")
        return report

    # Create target
    try:
        client.delete_collection(target_name)
    except Exception:
        pass
    target = client.create_collection(
        name=target_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Process in batches
    offset = 0
    rows_written = 0
    sampled_correct = 0
    sampled_total = 0

    while offset < total:
        batch = src.get(
            offset=offset,
            limit=batch_size,
            include=["embeddings", "metadatas", "documents"],
        )
        if not batch["ids"]:
            break

        ids = batch["ids"]
        metadatas = batch["metadatas"] or [{} for _ in ids]
        documents = batch["documents"] or [None for _ in ids]
        embeddings = np.array(batch["embeddings"], dtype=np.float64)

        # Transform: legacy → new space
        mapped = mapping.transform(embeddings)
        mapped_list = mapped.tolist()

        # Add Isotrieve metadata to each row
        for i, meta in enumerate(metadatas):
            if meta is None:
                meta = {}
                metadatas[i] = meta
            meta["isotrieve_mapping_id"] = _hash_mapping(mapping)
            meta["isotrieve_format_version"] = 1
            meta["isotrieve_source_collection"] = collection_name

        # Write to target
        target.add(
            ids=ids,
            embeddings=mapped_list,
            metadatas=metadatas,
            documents=documents,
        )
        rows_written += len(ids)

        # Sample recall check (every 10th batch)
        if rows_written % (batch_size * 10) == 0 or rows_written >= total:
            sample_n = min(50, len(ids))
            src_sample = embeddings[:sample_n]
            tgt_sample = mapped[:sample_n]
            # Recall@10: for each mapped vector, how many of its top-10
            # nearest in mapped space match its top-10 in source space
            src_sims = l2_normalize(src_sample) @ l2_normalize(src_sample).T
            tgt_sims = l2_normalize(tgt_sample) @ l2_normalize(tgt_sample).T
            for i in range(sample_n):
                src_top10 = set(np.argsort(-src_sims[i])[:10])
                tgt_top10 = set(np.argsort(-tgt_sims[i])[:10])
                sampled_correct += len(src_top10 & tgt_top10)
                sampled_total += 10

        offset += batch_size

    report.rows_processed = rows_written
    report.elapsed_seconds = time.perf_counter() - t0
    if sampled_total > 0:
        report.sampled_recall_at_10 = sampled_correct / sampled_total

    return report
