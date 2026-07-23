"""Core adapter abstractions.

``EmbeddingAdapter`` wraps any embedding model.
``VectorStoreAdapter`` wraps any vector DB with serve + offline modes.
``MigrationReport`` captures what happened during offline migration.

All mapping/gate/recalibration logic is imported from isotrieve core;
DB-specific code stays thin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from isotrieve.mapping.base import Mapping, l2_normalize


@dataclass
class MigrationReport:
    """Result of an offline migration pass."""

    rows_processed: int = 0
    elapsed_seconds: float = 0.0
    sampled_recall_at_10: float | None = None
    mapping_checksum: str = ""
    source_collection: str = ""
    target_collection: str = ""
    errors: list[str] = field(default_factory=list)
    idempotent: bool = True  # False if double-migration detected

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_processed": self.rows_processed,
            "elapsed_seconds": self.elapsed_seconds,
            "sampled_recall_at_10": self.sampled_recall_at_10,
            "mapping_checksum": self.mapping_checksum,
            "source_collection": self.source_collection,
            "target_collection": self.target_collection,
            "errors": self.errors,
            "idempotent": self.idempotent,
        }


class EmbeddingAdapter(ABC):
    """Wraps any embedding model; source of truth for model identity."""

    model_id: str
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (n, dim) array."""


class VectorStoreAdapter(ABC):
    """Thin per-DB shell. All mapping logic lives in isotrieve core.

    Parameters
    ----------
    mapping:
        A fitted Isotrieve Mapping (with inverse + optional recalibrator).
    mode:
        ``"serve"`` = transform queries on-the-fly, corpus untouched.
        ``"migrated"`` = corpus already transformed, no query mapping needed.
    """

    def __init__(
        self, mapping: Mapping, mode: Literal["serve", "migrated"] = "serve"
    ) -> None:
        self._mapping = mapping
        self._mode = mode

    @property
    def has_recalibrator(self) -> bool:
        return self._mapping.has_recalibrator

    def recalibrate_scores(self, scores: np.ndarray) -> np.ndarray:
        """Apply score recalibration (WS-A) to raw similarity scores."""
        return self._mapping.recalibrate_scores(scores)

    def preflight(self, sample_vectors: np.ndarray, target_vectors: np.ndarray) -> Any:
        """Run quality gate on a sample before serve/migrate.

        Returns a GateReport. Caller should check verdict before proceeding.
        """
        from isotrieve.quality.gate import QualityGate

        gate = QualityGate()
        return gate.evaluate(self._mapping, sample_vectors, target_vectors)

    def _map_query(self, vec: np.ndarray) -> np.ndarray:
        """Map a single query vector from new-model space to legacy space."""
        return l2_normalize(self._mapping.inverse_transform(vec.reshape(1, -1)).ravel())

    def _map_queries(self, vecs: np.ndarray) -> np.ndarray:
        """Batch map query vectors."""
        return l2_normalize(self._mapping.inverse_transform(vecs))

    @abstractmethod
    def query(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """Serve-mode query: map queries, search, return results.

        Returns a list (per query) of lists of result dicts with keys
        ``id``, ``score``, ``metadata``.
        """

    @abstractmethod
    def migrate(
        self,
        batch_size: int = 1000,
        dry_run: bool = False,
        new_collection: str | None = None,
    ) -> MigrationReport:
        """Offline migration: transform corpus vectors and write to new collection.

        Never modifies the source collection. If ``new_collection`` is None,
        appends ``_migrated`` to the source name.
        """
