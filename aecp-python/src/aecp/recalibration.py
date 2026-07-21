"""Score recalibration: map post-migration scores to ceiling-equivalent scores.

After embedding-space mapping, similarity scores are systematically compressed
(see Q10 study: margins shrink 39%, top-1 scores drop ~0.036). This breaks
downstream absolute-score thresholds silently.

ScoreRecalibrator fits an isotonic regression on holdout (mapped_score, ceiling_score)
pairs and provides a monotone map that restores score interpretability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RecalibrationReport:
    """Diagnostics for the fitted recalibrator."""

    n_pairs: int
    mean_mapped_score: float
    mean_ceiling_score: float
    mean_shift: float
    pre_margin_mean: float
    post_margin_mean: float
    margin_ratio: float
    threshold_agreement: dict[float, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "mean_mapped_score": self.mean_mapped_score,
            "mean_ceiling_score": self.mean_ceiling_score,
            "mean_shift": self.mean_shift,
            "pre_margin_mean": self.pre_margin_mean,
            "post_margin_mean": self.post_margin_mean,
            "margin_ratio": self.margin_ratio,
            "threshold_agreement": {str(k): v for k, v in self.threshold_agreement.items()},
        }


class ScoreRecalibrator:
    """Monotone map from mapped-space similarity scores to ceiling-equivalent scores.

    Uses isotonic regression trained on holdout query-document similarity pairs.

    Usage::

        # During calibration (automatic):
        recal = ScoreRecalibrator.fit_from_holdout(
            mapped_scores, ceiling_scores, query_ids, doc_ids, qrels
        )

        # At search time:
        calibrated = recal.transform(raw_scores)

        # Serialize:
        recal.save_dict()  # → stored in .aecp header under "score_recal_v1"
        recal2 = ScoreRecalibrator.from_dict(data)
    """

    def __init__(self) -> None:
        self._thresholds: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._report: RecalibrationReport | None = None

    @property
    def is_fitted(self) -> bool:
        return self._thresholds is not None

    @property
    def report(self) -> RecalibrationReport | None:
        return self._report

    def fit(
        self,
        mapped_scores: np.ndarray,
        ceiling_scores: np.ndarray,
    ) -> ScoreRecalibrator:
        """Fit isotonic regression on (mapped_score → ceiling_score) pairs.

        Parameters
        ----------
        mapped_scores:
            Similarity scores from the mapped retrieval (n_pairs,).
        ceiling_scores:
            Corresponding similarity scores from ceiling retrieval (n_pairs,).
        """
        from sklearn.isotonic import IsotonicRegression

        mapped_scores = np.asarray(mapped_scores, dtype=np.float64).ravel()
        ceiling_scores = np.asarray(ceiling_scores, dtype=np.float64).ravel()

        if len(mapped_scores) < 10:
            raise ValueError(f"Need ≥10 score pairs, got {len(mapped_scores)}")

        # Fit isotonic regression (increasing=True: higher mapped → higher ceiling)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(mapped_scores, ceiling_scores)

        self._thresholds = iso.X_thresholds_.copy()
        self._values = iso.y_thresholds_.copy()

        # Compute report
        pre = mapped_scores
        post = iso.predict(mapped_scores)

        # Margin computation (sorted pairs, consecutive differences)
        pre_sorted = np.sort(pre)
        post_sorted = np.sort(post)
        pre_margin = float(np.mean(np.diff(pre_sorted[-20:])))  # top-20 margin
        post_margin = float(np.mean(np.diff(post_sorted[-20:])))

        # Threshold agreement
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        agreement = {}
        for tau in thresholds:
            pre_decisions = pre >= tau
            post_decisions = post >= tau
            ceiling_decisions = ceiling_scores >= tau
            # Agreement = fraction where recalibrated matches ceiling
            agree = np.mean(post_decisions == ceiling_decisions)
            agreement[tau] = float(agree)

        self._report = RecalibrationReport(
            n_pairs=len(mapped_scores),
            mean_mapped_score=float(np.mean(mapped_scores)),
            mean_ceiling_score=float(np.mean(ceiling_scores)),
            mean_shift=float(np.mean(post) - np.mean(mapped_scores)),
            pre_margin_mean=pre_margin,
            post_margin_mean=post_margin,
            margin_ratio=post_margin / pre_margin if pre_margin > 1e-12 else 1.0,
            threshold_agreement=agreement,
        )

        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Recalibrate scores: mapped → ceiling-equivalent."""
        if not self.is_fitted:
            raise RuntimeError("ScoreRecalibrator not fitted; call fit() first")
        scores = np.asarray(scores, dtype=np.float64)
        # Isotonic interpolation using stored breakpoints
        return np.interp(scores, self._thresholds, self._values)

    @classmethod
    def fit_from_holdout(
        cls,
        doc_src: np.ndarray,
        doc_tgt: np.ndarray,
        qry_tgt: np.ndarray,
        doc_ids: list[str],
        query_ids: list[str],
        qrels: dict[str, set[str]],
        mapped_docs: np.ndarray,
        n_random_pairs: int = 50_000,
        seed: int = 0,
    ) -> ScoreRecalibrator:
        """Build recalibrator from holdout data.

        Samples random query-document pairs + all top-20 neighbors to cover
        the full score range.
        """
        from aecp.mapping.base import l2_normalize

        rng = np.random.default_rng(seed)
        n_docs = len(doc_ids)
        n_queries = len(query_ids)

        # Normalize
        qt_n = l2_normalize(qry_tgt)
        dt_n = l2_normalize(doc_tgt)
        dm_n = l2_normalize(mapped_docs)

        # Ceiling similarities: qry × doc_tgt
        ceil_sims = qt_n @ dt_n.T  # (n_queries, n_docs)
        # Mapped similarities: qry × doc_mapped
        map_sims = qt_n @ dm_n.T  # (n_queries, n_docs)

        mapped_pairs = []
        ceiling_pairs = []

        # Sample random pairs (cover the low-score region)
        n_random = min(n_random_pairs, n_queries * n_docs)
        rand_q = rng.integers(0, n_queries, size=n_random)
        rand_d = rng.integers(0, n_docs, size=n_random)
        for qi, di in zip(rand_q, rand_d):
            mapped_pairs.append(float(map_sims[qi, di]))
            ceiling_pairs.append(float(ceil_sims[qi, di]))

        # Add all top-20 neighbor pairs (cover the high-score region)
        top_k = min(20, n_docs)
        for qi in range(n_queries):
            top_d = np.argsort(-map_sims[qi])[:top_k]
            for di in top_d:
                mapped_pairs.append(float(map_sims[qi, di]))
                ceiling_pairs.append(float(ceil_sims[qi, di]))

        recal = cls()
        recal.fit(np.array(mapped_pairs), np.array(ceiling_pairs))
        return recal

    def save_dict(self) -> dict[str, Any]:
        """Serialize to dict for .aecp header storage."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize unfitted recalibrator")
        return {
            "thresholds": self._thresholds.tolist(),
            "values": self._values.tolist(),
            "report": self._report.to_dict() if self._report else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoreRecalibrator:
        """Load from serialized dict."""
        recal = cls()
        recal._thresholds = np.array(data["thresholds"], dtype=np.float64)
        recal._values = np.array(data["values"], dtype=np.float64)
        if "report" in data and data["report"] is not None:
            recal._report = RecalibrationReport(**data["report"])
        return recal

    def save(self, path: str | Path) -> None:
        """Save recalibrator to JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.save_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ScoreRecalibrator:
        """Load from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
