"""Query-adapter serving mode (WS-3).

Map new-model queries into legacy space for zero-corpus-write migration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from isotrieve.mapping.base import Mapping, l2_normalize
from isotrieve.mapping.registry import load_mapping


class QueryAdapter:
    """Map queries from new-model space to legacy-model space.

    Usage::

        qa = QueryAdapter.load("te3large__to__ada002.isotrieve")
        new_vec = embed_with_new_model(query_text)
        legacy_vec = qa.map_query(new_vec)
        results = qdrant.search(collection="docs", vector=legacy_vec)

    Thread-safe: the mapping matrix is read-only after construction.
    """

    def __init__(self, mapping: Mapping) -> None:
        self._mapping = mapping
        self._d_src = mapping._d_src
        self._d_tgt = mapping._d_tgt

    @classmethod
    def load(cls, path: str | Path) -> QueryAdapter:
        """Load a mapping file (must have inverse direction fitted)."""
        mapping = load_mapping(path)
        if mapping._W_inv is None:
            raise ValueError(
                f"Mapping at {path} has no inverse. "
                "Fit the inverse direction before using serve mode."
            )
        return cls(mapping)

    @property
    def d_new(self) -> int:
        """Dimension of new-model vectors (target space)."""
        return self._d_tgt

    @property
    def d_legacy(self) -> int:
        """Dimension of legacy-model vectors (source space)."""
        return self._d_src

    def map_query(self, vec: np.ndarray) -> np.ndarray:
        """Map a single query vector from new-model space to legacy space.

        Args:
            vec: Query vector from the new model (d_new dims = d_tgt).

        Returns:
            Mapped vector in legacy space (d_legacy dims = d_src), L2-normalized.
        """
        vec = np.asarray(vec, dtype=np.float64).ravel()
        if vec.shape[0] != self._d_tgt:
            raise ValueError(
                f"Dimension mismatch: expected {self._d_tgt} (new model dim), got {vec.shape[0]}. "
                f"Query vectors must be from the new embedding model."
            )
        out = self._mapping.inverse_transform(vec.reshape(1, -1))
        return l2_normalize(out).ravel()

    def map_queries(self, vecs: np.ndarray) -> np.ndarray:
        """Batch map query vectors from new-model space to legacy space.

        Args:
            vecs: Query vectors from the new model (n_queries, d_new).

        Returns:
            Mapped vectors in legacy space (n_queries, d_legacy), L2-normalized.
        """
        vecs = np.asarray(vecs, dtype=np.float64)
        if vecs.ndim == 1:
            return self.map_query(vecs).reshape(1, -1)
        if vecs.shape[1] != self._d_tgt:
            raise ValueError(
                f"Dimension mismatch: expected {self._d_tgt} (new model dim), got {vecs.shape[1]}. "
                f"Query vectors must be from the new embedding model."
            )
        out = self._mapping.inverse_transform(vecs)
        return l2_normalize(out)

    @property
    def has_recalibrator(self) -> bool:
        """Whether the underlying mapping has a fitted score recalibrator."""
        return self._mapping.has_recalibrator

    def recalibrate_scores(self, scores: np.ndarray) -> np.ndarray:
        """Map post-migration scores to ceiling-equivalent scores.

        Pass-through to the mapping's recalibrator. Returns scores unchanged
        if no recalibrator is fitted.
        """
        return self._mapping.recalibrate_scores(scores)

    def score_confidence(
        self,
        query_ids: list[str],
        similarities: np.ndarray,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Score per-query confidence from a similarity matrix.

        Uses margin and score magnitude to flag queries where the mapping
        is likely unreliable.

        Returns a dict with per-query reports and an aggregate summary.
        """
        from isotrieve.reranking import ConfidenceScorer, confidence_summary

        scorer = ConfidenceScorer()
        reports = scorer.score_queries(query_ids, similarities, top_k=top_k)
        return {
            "reports": reports,
            "summary": confidence_summary(reports),
        }


def csls_scores(
    query_vecs: np.ndarray,
    candidate_vecs: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Compute CSLS scores for query-candidate pairs.

    CSLS (Cross-Domain Similarity Local Scaling) corrects for hubness
    in cross-lingual retrieval. Reference: Joulin et al., "Loss in
    Translation: Learning Bilingual Word Embeddings with (almost) No
    Bilingual Data" (ACL 2016).

    Args:
        query_vecs: Mapped query vectors (n_queries, d).
        candidate_vecs: Candidate vectors to score against (n_candidates, d).
        k: Number of neighbors for hubness estimation.

    Returns:
        CSLS scores matrix (n_queries, n_candidates).
    """
    query_vecs = l2_normalize(np.asarray(query_vecs, dtype=np.float64))
    candidate_vecs = l2_normalize(np.asarray(candidate_vecs, dtype=np.float64))

    # Cosine similarity matrix
    S = query_vecs @ candidate_vecs.T

    # Hubness penalty: mean cosine of each candidate to its k nearest queries
    n_candidates = candidate_vecs.shape[0]
    n_queries = query_vecs.shape[0]
    k_eff = min(k, n_queries)

    # For each candidate, find k nearest queries
    # S.T shape: (n_candidates, n_queries)
    top_k_idx = np.argpartition(S.T, -k_eff, axis=1)[:, -k_eff:]
    r_T = np.zeros(n_candidates)
    for j in range(n_candidates):
        r_T[j] = np.mean(S.T[j, top_k_idx[j]])

    # CSLS: 2 * cos(q, x) - r_T(x)
    csls = 2 * S - r_T[np.newaxis, :]
    return csls


def merge_results(
    legacy_results: list[dict],
    native_results: list[dict],
    migrated_ids: set[str],
    legacy_weight: float = 1.0,
    native_weight: float = 1.0,
) -> list[dict]:
    """Merge results from legacy (mapped) and native (new-model) collections.

    For progressive migration: some docs are in legacy space, some in native.

    Args:
        legacy_results: Results from legacy collection (with mapped queries).
        native_results: Results from native collection (with native queries).
        migrated_ids: Set of doc IDs already migrated to native space.
        legacy_weight: Weight for legacy scores.
        native_weight: Weight for native scores.

    Returns:
        Merged results, deduplicated, weighted score.
    """
    scores: dict[str, float] = {}

    for r in legacy_results:
        doc_id = r.get("id", "")
        score = r.get("score", 0.0) * legacy_weight
        scores[doc_id] = max(scores.get(doc_id, 0.0), score)

    for r in native_results:
        doc_id = r.get("id", "")
        score = r.get("score", 0.0) * native_weight
        scores[doc_id] = max(scores.get(doc_id, 0.0), score)

    merged = [{"id": k, "score": v} for k, v in scores.items()]
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged
