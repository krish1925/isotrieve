"""Quality gate with proxy-based prediction (WS-1)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from aecp.mapping.base import Mapping
from aecp.quality.metrics import (
    holdout_rank_correlation,
    pairwise_cosine_stats,
    topk_retention,
)


class GateVerdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


def _load_thresholds() -> dict[str, Any]:
    path = Path(__file__).with_name("thresholds.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "pass_retention": 0.75,
        "warn_retention": 0.55,
        "max_optimism_gap": 0.20,
        "provisional": True,
    }


def _load_gate_model() -> dict[str, Any] | None:
    path = Path(__file__).with_name("gate_model_v1.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


@dataclass
class GateReport:
    verdict: GateVerdict
    predicted_retention: float
    prediction_interval: tuple[float, float]
    cosine_mean: float
    cosine_median: float
    cosine_p5: float
    top1_retention: float
    top10_retention: float
    holdout_rank_corr: float
    n_sample: int
    rationale: str
    holdout_top1: float | None = None
    optimism_gap: float | None = None
    provisional_thresholds: bool = True
    gate_model_used: bool = False
    gate_model_scope: str | None = None
    lopo_error: float | None = None
    thresholds_used: dict[str, Any] = field(default_factory=dict)
    margin_compression: float | None = None
    score_recal_recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "predicted_retention": self.predicted_retention,
            "prediction_interval": list(self.prediction_interval),
            "cosine_mean": self.cosine_mean,
            "cosine_median": self.cosine_median,
            "cosine_p5": self.cosine_p5,
            "top1_retention": self.top1_retention,
            "top10_retention": self.top10_retention,
            "holdout_rank_corr": self.holdout_rank_corr,
            "n_sample": self.n_sample,
            "rationale": self.rationale,
            "holdout_top1": self.holdout_top1,
            "optimism_gap": self.optimism_gap,
            "provisional_thresholds": self.provisional_thresholds,
            "gate_model_used": self.gate_model_used,
            "gate_model_scope": self.gate_model_scope,
            "lopo_error": self.lopo_error,
            "thresholds_used": self.thresholds_used,
            "margin_compression": self.margin_compression,
            "score_recal_recommendation": self.score_recal_recommendation,
        }


class QualityGate:
    """Evaluate a fitted mapping before full corpus migration.

    Uses a **fresh sample disjoint from calibration** (caller responsibility).
    Predicts retention from holdout proxies using an isotonic regression model
    fitted on benchmark data. Reports prediction intervals for honesty.
    """

    def __init__(self, thresholds: dict[str, Any] | None = None) -> None:
        self.thresholds = _load_thresholds()
        if thresholds:
            self.thresholds.update(thresholds)
        self.gate_model = _load_gate_model()
        if self.gate_model and self.gate_model.get("scope") == "local_model_pairs_only":
            import logging

            logging.getLogger(__name__).info(
                "Gate model scope: local model pairs only. "
                "Not validated on API model pairs (ada-002, te3-large, etc.)."
            )

    def _predict_retention(
        self, top1_retention: float, margin_compression: float | None = None
    ) -> tuple[float, tuple[float, float]]:
        """Predict retention from top1_retention using isotonic regression.

        Returns (predicted_retention, (lower_bound, upper_bound)).

        When margin_compression is provided (< 1.0 = compressed), the
        prediction interval is widened to reflect pair-level uncertainty
        that the univariate isotonic model cannot capture.
        """
        if self.gate_model is None:
            # Fallback: use top1_retention directly as prediction
            return top1_retention, (top1_retention - 0.1, top1_retention + 0.1)

        X_thresholds = np.array(self.gate_model["X_thresholds"])
        y_thresholds = np.array(self.gate_model["y_thresholds"])

        # Isotonic regression prediction via interpolation
        predicted = float(np.interp(top1_retention, X_thresholds, y_thresholds))

        # Prediction interval from LOPO stats
        lopo = self.gate_model.get("lipo", {})
        interval_hw = lopo.get("interval_half_width_80", 0.1)

        # Gate v3: widen interval when margin compression is severe
        # Compression < 0.85 means raw scores are significantly compressed,
        # which the univariate isotonic model doesn't capture. This adds
        # pair-level uncertainty to the interval.
        if margin_compression is not None and margin_compression < 0.85:
            compression_penalty = (0.85 - margin_compression) * 0.5
            interval_hw += compression_penalty

        lo = max(0.0, predicted - interval_hw)
        hi = min(1.0, predicted + interval_hw)
        return predicted, (lo, hi)

    def evaluate(
        self,
        mapping: Mapping,
        X_sample: np.ndarray,
        Y_sample: np.ndarray,
        *,
        holdout_top1: float | None = None,
    ) -> GateReport:
        """Run gate on paired source/target embeddings of the same texts.

        ``X_sample`` / ``Y_sample`` must not overlap the calibration fit set.
        """
        mapped = mapping.transform(X_sample)
        cos = pairwise_cosine_stats(mapped, Y_sample)
        t1 = topk_retention(mapped, Y_sample, k=1)
        t10 = topk_retention(mapped, Y_sample, k=min(10, len(Y_sample)))
        rank_corr = holdout_rank_correlation(mapped, Y_sample)

        optimism_gap: float | None = None
        if holdout_top1 is not None:
            optimism_gap = float(holdout_top1 - t1)

        # Compute margin compression before prediction (used as covariate)
        mc = self._compute_margin_compression(mapped, Y_sample)

        # Predict retention from proxies (uses margin compression for v3 interval)
        predicted, (lower, upper) = self._predict_retention(t1, mc)

        pass_r = float(self.thresholds.get("pass_retention", 0.75))
        warn_r = float(self.thresholds.get("warn_retention", 0.55))
        provisional = bool(self.thresholds.get("provisional", True))

        # Verdict based on prediction interval
        if lower >= pass_r:
            verdict = GateVerdict.PASS
            rationale = (
                f"PASS: predicted retention={predicted:.3f} "
                f"(80% CI: [{lower:.3f}, {upper:.3f}]). "
                f"Holdout proxies indicate good mapping quality."
            )
        elif upper < warn_r:
            verdict = GateVerdict.FAIL
            rationale = (
                f"FAIL: predicted retention={predicted:.3f} "
                f"(80% CI: [{lower:.3f}, {upper:.3f}]). "
                f"Mapping quality too low for migration. Recommend full re-embedding."
            )
        else:
            verdict = GateVerdict.WARN
            rationale = (
                f"WARN: predicted retention={predicted:.3f} "
                f"(80% CI: [{lower:.3f}, {upper:.3f}]). "
                f"Usable for recall-tolerant workloads only. "
                f"Consider more in-domain calibration."
            )

        lopo_error = None
        if self.gate_model:
            lopo_error = self.gate_model.get("lipo", {}).get("mae")

        return GateReport(
            verdict=verdict,
            predicted_retention=predicted,
            prediction_interval=(lower, upper),
            cosine_mean=cos["mean"],
            cosine_median=cos["median"],
            cosine_p5=cos["p5"],
            top1_retention=t1,
            top10_retention=t10,
            holdout_rank_corr=rank_corr,
            n_sample=len(X_sample),
            rationale=rationale,
            holdout_top1=holdout_top1,
            optimism_gap=optimism_gap,
            provisional_thresholds=provisional,
            gate_model_used=self.gate_model is not None,
            gate_model_scope=self.gate_model.get("scope") if self.gate_model else None,
            lopo_error=lopo_error,
            thresholds_used={
                "pass_retention": pass_r,
                "warn_retention": warn_r,
            },
            margin_compression=mc,
            score_recal_recommendation=self._score_recal_recommendation(mc),
        )

    @staticmethod
    def _compute_margin_compression(
        mapped: np.ndarray, target: np.ndarray
    ) -> float | None:
        """Estimate margin compression from mapped vs target cosine distributions.

        Returns ratio of mapped margin to target margin (< 1 = compression).

        Uses cosine similarity between paired mapped→target vectors (not
        self-similarity, which is always 1 after L2 norm). The reference
        distribution is random target→target cosine similarities, giving
        a meaningful baseline for variance comparison.
        """
        from aecp.mapping.base import l2_normalize

        m_n = l2_normalize(mapped)
        t_n = l2_normalize(target)
        k = len(m_n)

        # Paired cosine: each mapped vector vs its corresponding target
        paired_cos = np.sum(m_n * t_n, axis=1)
        paired_var = float(np.var(paired_cos))

        # Reference: variance of cosine similarities between random target pairs
        rng = np.random.default_rng(0)
        idx_a = rng.integers(0, k, size=k)
        idx_b = rng.integers(0, k, size=k)
        ref_cos = np.sum(t_n[idx_a] * t_n[idx_b], axis=1)
        ref_var = float(np.var(ref_cos))

        if ref_var < 1e-12:
            return None

        return paired_var / ref_var

    @staticmethod
    def _score_recal_recommendation(margin_ratio: float | None) -> str | None:
        if margin_ratio is None:
            return None
        if margin_ratio < 0.8:
            return (
                f"Score margins are compressed (ratio={margin_ratio:.2f}). "
                "If your application uses absolute score thresholds, "
                "enable score recalibration or re-tune thresholds."
            )
        return None
