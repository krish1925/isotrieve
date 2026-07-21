"""LangChain embeddings adapter for AECP.

``AECPEmbeddings`` is a drop-in ``langchain_core.embeddings.Embeddings``
shim that transparently applies an AECP mapping. No index migration
needed — query vectors are mapped on-the-fly.

Requires: ``pip install aecp[langchain]`` or ``pip install langchain-core``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aecp.mapping.base import Mapping, l2_normalize


def _require_langchain():
    try:
        from langchain_core.embeddings import Embeddings

        return Embeddings
    except ImportError:
        raise ImportError(
            "LangChain adapter requires langchain-core. "
            "Install with: pip install langchain-core"
        )


class AECPEmbeddings:
    """Drop-in ``Embeddings`` that applies an AECP mapping.

    Usage::

        from langchain_chroma import Chroma
        from aecp.adapters.langchain import AECPEmbeddings
        from aecp.mapping.base import Mapping

        mapping = Mapping.load("ada002_to_te3.aecp")
        base_embed = OpenAIEmbeddings(model="text-embedding-3-small")
        ae = AECPEmbeddings(mapping, base_embed)

        # Works with any LangChain vector store
        db = Chroma.from_documents(docs, embedding=ae)
        results = db.similarity_search("query", k=10)
    """

    def __init__(
        self,
        mapping: Mapping,
        base_embeddings: Any,
    ) -> None:
        """
        Parameters
        ----------
        mapping:
            Fitted AECP mapping. Forward direction should be
            ``legacy → new`` so that ``mapping.inverse_transform()``
            maps new-model vectors back to legacy space.
        base_embeddings:
            A ``langchain_core.embeddings.Embeddings`` instance for the
            new model (e.g. ``OpenAIEmbeddings``).
        """
        _require_langchain()  # side effect: raises ImportError if langchain missing
        # Validate that base_embeddings satisfies the interface
        if not hasattr(base_embeddings, "embed_documents"):
            raise TypeError("base_embeddings must implement embed_documents()")
        self._mapping = mapping
        self._base = base_embeddings

    def _map_vectors(self, vecs: np.ndarray) -> np.ndarray:
        """Map new-model vectors to legacy space."""
        return l2_normalize(self._mapping.inverse_transform(vecs))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents into legacy space (for legacy index)."""
        new_vecs = np.asarray(self._base.embed_documents(texts), dtype=np.float64)
        legacy_vecs = self._map_vectors(new_vecs)
        return legacy_vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query into legacy space."""
        new_vec = np.asarray(self._base.embed_query(text), dtype=np.float64)
        legacy_vec = self._map_vectors(new_vec.reshape(1, -1)).ravel()
        return legacy_vec.tolist()

    # Async variants — delegate to base + map
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self._base, "aembed_documents"):
            new_vecs = np.asarray(
                await self._base.aembed_documents(texts), dtype=np.float64
            )
        else:
            new_vecs = np.asarray(self._base.embed_documents(texts), dtype=np.float64)
        legacy_vecs = self._map_vectors(new_vecs)
        return legacy_vecs.tolist()

    async def aembed_query(self, text: str) -> list[float]:
        if hasattr(self._base, "aembed_query"):
            new_vec = np.asarray(await self._base.aembed_query(text), dtype=np.float64)
        else:
            new_vec = np.asarray(self._base.embed_query(text), dtype=np.float64)
        legacy_vec = self._map_vectors(new_vec.reshape(1, -1)).ravel()
        return legacy_vec.tolist()

    @property
    def has_recalibrator(self) -> bool:
        return self._mapping.has_recalibrator

    def __repr__(self) -> str:
        base_cls = type(self._base).__name__
        return (
            f"AECPEmbeddings(base={base_cls}, "
            f"mapping={self._mapping.mapping_type}, "
            f"d_src={self._mapping._d_src}, d_tgt={self._mapping._d_tgt})"
        )
