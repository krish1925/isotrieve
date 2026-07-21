"""Voyage AI embedding provider.

Hits the network. Install: ``pip install aecp[voyage]``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from aecp.providers.base import Embedder


class VoyageEmbedder(Embedder):
    """Voyage embeddings API. Requires ``VOYAGE_API_KEY``. Hits the network."""

    def __init__(
        self,
        model_id: str = "voyage-3",
        *,
        api_key: str | None = None,
        batch_size: int = 128,
    ) -> None:
        try:
            import voyageai
        except ImportError as e:
            raise ImportError(
                "voyageai is required. Install with: pip install aecp[voyage]"
            ) from e
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise OSError("VOYAGE_API_KEY is not set")
        self._client = voyageai.Client(api_key=key)
        self._model_id = model_id
        self._batch_size = batch_size
        probe = self._embed_batch(["dim probe"])
        self._dims = int(probe.shape[1])

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dims(self) -> int:
        return self._dims

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embed(texts, model=self._model_id)
        return np.asarray(resp.embeddings, dtype=np.float64)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts via Voyage API (network call unless cached upstream)."""
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, self._dims), dtype=np.float64)
        chunks = []
        for i in range(0, len(texts_list), self._batch_size):
            chunks.append(self._embed_batch(texts_list[i : i + self._batch_size]))
        return np.vstack(chunks)
