"""LlamaIndex query-time wrapper.

Subclasses ``llama_index.core.base.embeddings.BaseEmbedding`` to map
new-model queries into legacy-model space on-the-fly.  Zero writes to
the vector store.

Usage::

    from isotrieve.wrappers.llamaindex import IsotrieveEmbedding
    from isotrieve.mapping.registry import load_mapping

    mapping = load_mapping("mapping.isotrieve")
    wrapper = IsotrieveEmbedding(
        new_model_embedder=your_llamaindex_embedder,
        transform_artifact_path="mapping.isotrieve",
    )
    # Use wrapper anywhere LlamaIndex expects a BaseEmbedding
    # Queries are mapped; document embeddings raise IsotrieveWrapperUsageError.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from isotrieve.mapping.base import l2_normalize
from isotrieve.mapping.registry import load_mapping
from isotrieve.wrappers import IsotrieveWrapperUsageError


def _check_llamaindex_available() -> None:
    """Warn if LlamaIndex is not installed (wrapper still works via duck-typing)."""
    import contextlib

    with contextlib.suppress(ImportError):
        import llama_index.core  # type: ignore[import-untyped]  # noqa: F401


class IsotrieveEmbedding:
    """Query-time LlamaIndex wrapper.

    Maps new-model queries through the fitted Isotrieve transform into
    legacy-model space.  Document-embedding methods raise
    ``IsotrieveWrapperUsageError`` — this wrapper is query-only.

    Parameters
    ----------
    new_model_embedder:
        Any LlamaIndex ``BaseEmbedding`` instance (the new model).
    transform_artifact_path:
        Path to a fitted ``.isotrieve`` mapping file.
    direction:
        ``"new_to_old"`` (default) maps new-model queries to legacy space.
    """

    def __init__(
        self,
        new_model_embedder: Any,
        transform_artifact_path: str,
        *,
        direction: str = "new_to_old",
    ) -> None:
        _check_llamaindex_available()

        self._embedder = new_model_embedder
        self._direction = direction
        self._mapping = load_mapping(transform_artifact_path)
        self._d_src = self._mapping.d_src
        self._d_tgt = self._mapping.d_tgt

        # Validate dimension compatibility
        try:
            test_vec = self._embedder._get_text_embedding("probe")
            actual_dim = len(test_vec)
        except Exception:
            actual_dim = 0

        if actual_dim > 0 and actual_dim != self._d_tgt:
            raise ValueError(
                f"Embedder output dim ({actual_dim}) does not match "
                f"mapping target dim ({self._d_tgt}).  "
                f"Check that the embedder and mapping are for the same model pair."
            )

    def _map_vectors(self, vecs: np.ndarray) -> np.ndarray:
        """Map vectors through the Isotrieve transform (new→old direction)."""
        if self._direction == "new_to_old":
            return l2_normalize(self._mapping.inverse_transform(vecs))
        return l2_normalize(self._mapping.transform(vecs))

    # ----- Query methods (allowed) -----

    def _get_query_embedding(self, query: str) -> list[float]:
        """Embed query with new model, map to legacy space."""
        raw = self._embedder._get_query_embedding(query)
        vec = np.array(raw, dtype=np.float64).reshape(1, -1)
        mapped = self._map_vectors(vec)
        return mapped.ravel().tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async variant of _get_query_embedding."""
        raw = await self._embedder._aget_query_embedding(query)
        vec = np.array(raw, dtype=np.float64).reshape(1, -1)
        mapped = self._map_vectors(vec)
        return mapped.ravel().tolist()

    # ----- Document methods (blocked) -----

    def _get_text_embedding(self, text: str) -> list[float]:
        raise IsotrieveWrapperUsageError(
            "This wrapper is query-time only.  "
            "Document embeddings must come from the legacy model. "
            "To migrate your corpus, use 'isotrieve migrate'.  "
            "See docs/migration.md for details."
        )

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        raise IsotrieveWrapperUsageError(
            "This wrapper is query-time only.  "
            "Document embeddings must come from the legacy model. "
            "To migrate your corpus, use 'isotrieve migrate'.  "
            "See docs/migration.md for details."
        )

    async def _aget_text_embedding(self, text: str) -> list[float]:
        raise IsotrieveWrapperUsageError(
            "This wrapper is query-time only.  "
            "Document embeddings must come from the legacy model. "
            "To migrate your corpus, use 'isotrieve migrate'.  "
            "See docs/migration.md for details."
        )

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        raise IsotrieveWrapperUsageError(
            "This wrapper is query-time only.  "
            "Document embeddings must come from the legacy model. "
            "To migrate your corpus, use 'isotrieve migrate'.  "
            "See docs/migration.md for details."
        )

    # ----- Properties -----

    @property
    def has_recalibrator(self) -> bool:
        return self._mapping.has_recalibrator

    @property
    def d_src(self) -> int:
        return self._d_src

    @property
    def d_tgt(self) -> int:
        return self._d_tgt

    def __repr__(self) -> str:
        return (
            f"IsotrieveEmbedding(mapping_type={self._mapping.mapping_type!r}, "
            f"d_src={self._d_src}, d_tgt={self._d_tgt}, "
            f"direction={self._direction!r})"
        )
