"""Local sentence-transformers embedder (offline-friendly)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aecp.providers.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Embed with a local ``sentence-transformers`` model.

    Downloads weights on first use (network). Subsequent calls are local.
    Install with ``pip install aecp[sentence-transformers]``.
    """

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install aecp[sentence-transformers]"
            ) from e
        self._model_id = model_id
        self._model = SentenceTransformer(model_id)
        # Probe dims
        probe = self._model.encode(["dim probe"], convert_to_numpy=True)
        self._dims = int(np.asarray(probe).shape[-1])

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dims(self) -> int:
        return self._dims

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts locally (may download model weights on first call)."""
        if not texts:
            return np.zeros((0, self._dims), dtype=np.float64)
        arr = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float64)
