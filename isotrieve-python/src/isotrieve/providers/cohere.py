"""Cohere embedding provider.

Hits the network. Install: ``pip install isotrieve[cohere]``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from isotrieve.providers.base import Embedder


class CohereEmbedder(Embedder):
    """Cohere embeddings API. Requires ``COHERE_API_KEY``. Hits the network."""

    def __init__(
        self,
        model_id: str = "embed-english-v3.0",
        *,
        api_key: str | None = None,
        input_type: str = "search_document",
        batch_size: int = 96,
    ) -> None:
        try:
            import cohere
        except ImportError as e:
            raise ImportError(
                "cohere is required. Install with: pip install isotrieve[cohere]"
            ) from e
        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise OSError("COHERE_API_KEY is not set")
        self._client = cohere.ClientV2(api_key=key)
        self._model_id = model_id
        self._input_type = input_type
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
        resp = self._client.embed(
            texts=texts,
            model=self._model_id,
            input_type=self._input_type,
            embedding_types=["float"],
        )
        return np.asarray(resp.embeddings.float, dtype=np.float64)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts via Cohere API (network call unless cached upstream)."""
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, self._dims), dtype=np.float64)
        chunks = []
        for i in range(0, len(texts_list), self._batch_size):
            chunks.append(self._embed_batch(texts_list[i : i + self._batch_size]))
        return np.vstack(chunks)
