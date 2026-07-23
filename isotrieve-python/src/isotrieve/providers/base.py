"""Embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Embedder(ABC):
    """Abstract embedding provider.

    Any method that hits a network must say so in its docstring and remain
    mockable for offline tests.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable model identifier string."""

    @property
    @abstractmethod
    def dims(self) -> int:
        """Output dimensionality."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts → ``(len(texts), dims)`` float64 array.

        May hit the network depending on the concrete provider.
        """
