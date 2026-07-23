"""Vector store adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VectorRecord:
    """One stored vector with optional payload text."""

    id: str
    vector: np.ndarray
    text: str | None = None
    payload: dict[str, Any] | None = None


class VectorStore(ABC):
    """Abstract vector store: stream read/write, never destructive in-place."""

    @abstractmethod
    def count(self) -> int:
        """Number of vectors in the store."""

    @abstractmethod
    def iter_vectors(self, batch_size: int = 1024) -> Iterator[list[VectorRecord]]:
        """Yield batches of records for streaming transform."""

    @abstractmethod
    def write_vectors(
        self,
        records: Iterator[list[VectorRecord]] | list[VectorRecord],
        *,
        batch_size: int = 1024,
    ) -> int:
        """Write records to this store. Returns number written."""
