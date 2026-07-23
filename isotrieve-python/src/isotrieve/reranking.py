"""Cross-encoder reranking and per-query confidence scoring for Isotrieve.

WS-B implementation: rerank Isotrieve top-k results with a cross-encoder, and
flag low-confidence queries where the mapping is likely unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ConfidenceReport:
    """Per-query confidence metadata."""

    query_id: str
    top1_margin: float
    top1_score: float
    confidence: str  # "high", "medium", "low"
    n_candidates: int


class CrossEncoderReranker:
    """Cross-encoder reranker for Isotrieve top-k results.

    Wraps ``sentence_transformers.CrossEncoder``.  The cross-encoder scores
    (query, document) pairs directly, correcting for mapping-induced
    score distortions.

    Parameters
    ----------
    model_name:
        HuggingFace cross-encoder model name.  Default:
        ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``.
    top_k:
        Number of candidates to rerank.  Default: 20.
    device:
        Device for inference (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 20,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._device = device
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "CrossEncoder reranking requires sentence-transformers: "
                "pip install sentence-transformers"
            )
        self._model = CrossEncoder(self._model_name, device=self._device)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rerank candidates using the cross-encoder.

        Parameters
        ----------
        query:
            The query text.
        candidates:
            List of dicts with at least ``"text"`` and ``"score"`` keys.

        Returns
        -------
        List of dicts sorted by cross-encoder score (descending), with an
        added ``"rerank_score"`` key.
        """
        self._ensure_model()
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        reranked = []
        for c, s in zip(candidates, scores):
            reranked.append({**c, "rerank_score": float(s)})
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked


class ConfidenceScorer:
    """Score per-query confidence based on margin and score characteristics.

    Flags queries where the Isotrieve mapping is likely unreliable.

    Two modes:
    - **Fixed thresholds**: pass ``margin_low``, ``margin_high``, ``score_low``.
    - **Adaptive (percentile-based)**: pass ``adaptive=True`` (default).
      Computes P33/P67 of the margin distribution across all queries,
      which transfers across pairs with different score ranges.

    Parameters
    ----------
    margin_low:
        Fixed: margin below this → "low". Default: 0.005.
    margin_high:
        Fixed: margin above this → "high". Default: 0.025.
    score_low:
        Fixed: top-1 score below this → "low". Default: 0.8.
    adaptive:
        If True, ignore fixed thresholds and use percentile-based cutoffs.
    """

    def __init__(
        self,
        margin_low: float = 0.005,
        margin_high: float = 0.025,
        score_low: float = 0.8,
        adaptive: bool = True,
    ) -> None:
        self._margin_low = margin_low
        self._margin_high = margin_high
        self._score_low = score_low
        self._adaptive = adaptive

    def score_queries(
        self,
        query_ids: list[str],
        similarities: np.ndarray,
        top_k: int = 10,
    ) -> list[ConfidenceReport]:
        """Score confidence for each query.

        Parameters
        ----------
        query_ids:
            List of query IDs.
        similarities:
            (n_queries, n_docs) similarity matrix.
        top_k:
            Number of top results to consider for margin computation.

        Returns
        -------
        List of ConfidenceReport objects.
        """
        # Compute margins for all queries first
        all_margins = []
        all_top1 = []
        for qi in range(len(query_ids)):
            scores = np.sort(similarities[qi])[-top_k:]
            top1 = float(scores[-1])
            top2 = float(scores[-2]) if len(scores) >= 2 else top1
            all_margins.append(top1 - top2)
            all_top1.append(top1)

        all_margins = np.array(all_margins)
        all_top1 = np.array(all_top1)

        if self._adaptive and len(all_margins) > 10:
            margin_low = float(np.percentile(all_margins, 33))
            margin_high = float(np.percentile(all_margins, 67))
        else:
            margin_low = self._margin_low
            margin_high = self._margin_high

        reports = []
        for qi, qid in enumerate(query_ids):
            margin = all_margins[qi]
            top1 = all_top1[qi]

            if margin >= margin_high:
                conf = "high"
            elif margin <= margin_low:
                conf = "low"
            else:
                conf = "medium"

            reports.append(
                ConfidenceReport(
                    query_id=qid,
                    top1_margin=margin,
                    top1_score=top1,
                    confidence=conf,
                    n_candidates=top_k,
                )
            )
        return reports


def confidence_summary(reports: list[ConfidenceReport]) -> dict[str, Any]:
    """Aggregate confidence reports into a summary."""
    n = len(reports)
    if n == 0:
        return {"n": 0}
    n_high = sum(1 for r in reports if r.confidence == "high")
    n_medium = sum(1 for r in reports if r.confidence == "medium")
    n_low = sum(1 for r in reports if r.confidence == "low")
    return {
        "n": n,
        "n_high": n_high,
        "n_medium": n_medium,
        "n_low": n_low,
        "pct_high": n_high / n,
        "pct_low": n_low / n,
        "mean_margin": float(np.mean([r.top1_margin for r in reports])),
        "mean_top1_score": float(np.mean([r.top1_score for r in reports])),
    }
