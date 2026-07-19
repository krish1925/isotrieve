"""OpenAI embedding provider.

Hits the network. Install: ``pip install aecp[openai]``.
Always wrap with :func:`aecp.providers.cached.with_disk_cache` in production.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from aecp.providers.base import Embedder

# Known dims for common models (avoids a probe round-trip).
_DIMS: dict[str, int] = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings API.

    This method hits the network. Requires ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        model_id: str = "text-embedding-3-large",
        *,
        api_key: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 128,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required. Install with: pip install aecp[openai]"
            ) from e
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=key)
        self._model_id = model_id
        self._batch_size = batch_size
        self._dimensions = dimensions
        if dimensions is not None:
            self._dims = dimensions
        elif model_id in _DIMS:
            self._dims = _DIMS[model_id]
        else:
            # Probe once (network)
            probe = self._embed_batch(["dim probe"])
            self._dims = int(probe.shape[1])

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dims(self) -> int:
        return self._dims

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        kwargs: dict = {"model": self._model_id, "input": texts}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        resp = self._client.embeddings.create(**kwargs)
        rows = sorted(resp.data, key=lambda d: d.index)
        return np.asarray([r.embedding for r in rows], dtype=np.float64)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts via OpenAI API (network call unless cached upstream)."""
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, self._dims), dtype=np.float64)
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts_list), self._batch_size):
            chunks.append(self._embed_batch(texts_list[i : i + self._batch_size]))
        return np.vstack(chunks)
