"""Google Gemini embedding provider.

Hits the network. Install: ``pip install isotrieve[gemini]``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from isotrieve.providers.base import Embedder


class GeminiEmbedder(Embedder):
    """Gemini embeddings. Requires ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``. Hits the network."""

    def __init__(
        self,
        model_id: str = "models/text-embedding-004",
        *,
        api_key: str | None = None,
        batch_size: int = 64,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai is required. Install with: pip install isotrieve[gemini]"
            ) from e
        key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not key:
            raise OSError("GOOGLE_API_KEY or GEMINI_API_KEY is not set")
        genai.configure(api_key=key)
        self._genai = genai
        self._model_id = model_id
        self._batch_size = batch_size
        probe = self._embed_one("dim probe")
        self._dims = int(probe.shape[0])

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dims(self) -> int:
        return self._dims

    def _embed_one(self, text: str) -> np.ndarray:
        result = self._genai.embed_content(model=self._model_id, content=text)
        return np.asarray(result["embedding"], dtype=np.float64)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts via Gemini API (network call unless cached upstream)."""
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, self._dims), dtype=np.float64)
        rows = [self._embed_one(t) for t in texts_list]
        return np.stack(rows, axis=0)
