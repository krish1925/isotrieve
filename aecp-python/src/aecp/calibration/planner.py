"""Calibration planner — recommend K and estimate costs."""

from __future__ import annotations

from dataclasses import dataclass

# Rough public list prices ($ / 1M tokens). Update when providers change;
# these are planning estimates, not invoices. Documented as estimates in CLI.
_PRICE_PER_MILLION: dict[str, float] = {
    "text-embedding-ada-002": 0.10,
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "voyage-3": 0.06,
    "default": 0.05,
}


@dataclass
class CalibrationPlan:
    corpus_size: int
    recommended_k: int
    source_model: str
    target_model: str
    est_calibration_calls: int
    est_reembed_calls: int
    est_calibration_usd: float
    est_reembed_usd: float
    notes: list[str]


def estimate_embed_cost(
    model_id: str, n_texts: int, tokens_per_text: int = 100
) -> float:
    """Estimate USD cost for embedding ``n_texts`` (rough token heuristic)."""
    price = _PRICE_PER_MILLION.get(model_id, _PRICE_PER_MILLION["default"])
    # Try suffix match
    for key, val in _PRICE_PER_MILLION.items():
        if key != "default" and key in model_id:
            price = val
            break
    tokens = n_texts * tokens_per_text
    return (tokens / 1_000_000.0) * price


def recommend_k(corpus_size: int, d_src: int, d_tgt: int) -> int:
    """Recommend calibration size K from dims and corpus size."""
    min_dim = min(d_src, d_tgt)
    floor = 10 * min_dim
    # Prefer at least 2k when corpus allows; cap at 20k or corpus size
    target = max(floor, 2000)
    target = min(target, 20000, max(corpus_size, floor))
    return int(target)


def plan_calibration(
    *,
    corpus_size: int,
    source_model: str,
    target_model: str,
    d_src: int = 384,
    d_tgt: int = 1024,
    tokens_per_text: int = 100,
) -> CalibrationPlan:
    """Produce a calibration vs re-embed cost comparison."""
    k = recommend_k(corpus_size, d_src, d_tgt)
    cal_calls = 2 * k
    reembed_calls = corpus_size
    cal_usd = estimate_embed_cost(
        source_model, k, tokens_per_text
    ) + estimate_embed_cost(target_model, k, tokens_per_text)
    reembed_usd = estimate_embed_cost(target_model, corpus_size, tokens_per_text)
    notes = [
        "Costs are rough public-list estimates, not invoices.",
        "Prefer in-domain sampling from your corpus over the generic set.",
        "AECP cannot beat true re-embedding; run the quality gate before migrating.",
    ]
    return CalibrationPlan(
        corpus_size=corpus_size,
        recommended_k=k,
        source_model=source_model,
        target_model=target_model,
        est_calibration_calls=cal_calls,
        est_reembed_calls=reembed_calls,
        est_calibration_usd=cal_usd,
        est_reembed_usd=reembed_usd,
        notes=notes,
    )
