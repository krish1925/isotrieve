"""Quality metrics for embedding-space mappings.

Cosine similarity alone is misleading for migration decisions. Prefer
top-k retrieval retention: the fraction of queries whose true nearest
neighbor is preserved after mapping.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aecp.mapping.base import l2_normalize


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without a scipy dependency."""
    if a.size < 2:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt(np.sum(ra**2) * np.sum(rb**2)))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(ra * rb) / denom)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"Size mismatch: {a.size} vs {b.size}")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def pairwise_cosine_stats(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Row-wise cosine between predicted and target; return mean/median/p5."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: {predicted.shape} vs {target.shape}"
        )
    p = l2_normalize(predicted)
    t = l2_normalize(target)
    sims = np.sum(p * t, axis=1)
    return {
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "p5": float(np.percentile(sims, 5)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
    }


def topk_retention(
    queries: np.ndarray,
    corpus: np.ndarray,
    k: int = 1,
) -> float:
    """Fraction of rows whose true match (same index) is in the top-k neighbors.

    For each query ``queries[i]``, retrieve top-k nearest neighbors in
    ``corpus`` by cosine similarity and check whether index ``i`` appears.
    Self-match exclusion is not applied: the intended neighbor *is* index i.
    """
    queries = l2_normalize(np.asarray(queries, dtype=np.float64))
    corpus = l2_normalize(np.asarray(corpus, dtype=np.float64))
    n = queries.shape[0]
    if n == 0:
        return 0.0
    k = min(k, n)
    sims = queries @ corpus.T
    # argsort ascending; take last k
    top = np.argpartition(sims, -k, axis=1)[:, -k:]
    correct = 0
    for i in range(n):
        if i in top[i]:
            correct += 1
    return float(correct) / float(n)


def rank_correlation(
    queries_a: np.ndarray,
    corpus_a: np.ndarray,
    queries_b: np.ndarray,
    corpus_b: np.ndarray,
    sample_queries: int | None = 50,
    seed: int = 0,
) -> dict[str, float]:
    """Spearman rank correlation of neighbor rankings between two spaces."""
    rng = np.random.default_rng(seed)
    n = queries_a.shape[0]
    idx = np.arange(n)
    if sample_queries is not None and sample_queries < n:
        idx = rng.choice(n, size=sample_queries, replace=False)

    qa = l2_normalize(queries_a[idx])
    ca = l2_normalize(corpus_a)
    qb = l2_normalize(queries_b[idx])
    cb = l2_normalize(corpus_b)

    sims_a = qa @ ca.T
    sims_b = qb @ cb.T
    rhos: list[float] = []
    for i in range(len(idx)):
        rho = _spearman_rho(sims_a[i], sims_b[i])
        if np.isfinite(rho):
            rhos.append(float(rho))
    if not rhos:
        return {"mean_spearman": 0.0, "n": 0.0}
    return {"mean_spearman": float(np.mean(rhos)), "n": float(len(rhos))}


def holdout_rank_correlation(
    mapped: np.ndarray,
    target: np.ndarray,
    sample_size: int | None = 200,
    seed: int = 0,
) -> float:
    """Spearman rank correlation of pairwise similarity matrices.

    Compares the neighbor structure of mapped vectors vs true target vectors.
    Captures neighborhood-structure preservation better than cosine alone.
    """
    rng = np.random.default_rng(seed)
    n = mapped.shape[0]
    if n < 3:
        return 0.0
    if sample_size is not None and sample_size < n:
        idx = rng.choice(n, size=sample_size, replace=False)
    else:
        idx = np.arange(n)

    m = l2_normalize(mapped[idx])
    t = l2_normalize(target[idx])

    # Pairwise cosine similarities (upper triangle only for efficiency)
    sim_mapped = m @ m.T
    sim_target = t @ t.T

    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(len(idx), k=1)
    a = sim_mapped[triu_idx]
    b = sim_target[triu_idx]

    return _spearman_rho(a, b)


def mrr_delta(
    queries_mapped: np.ndarray,
    corpus_target: np.ndarray,
    queries_true: np.ndarray,
) -> dict[str, float]:
    """Mean reciprocal rank for true-index retrieval: mapped vs true queries."""

    def _mrr(queries: np.ndarray, corpus: np.ndarray) -> float:
        q = l2_normalize(queries)
        c = l2_normalize(corpus)
        sims = q @ c.T
        n = sims.shape[0]
        rr = 0.0
        for i in range(n):
            order = np.argsort(-sims[i])
            rank = int(np.where(order == i)[0][0]) + 1
            rr += 1.0 / rank
        return rr / n if n else 0.0

    mrr_mapped = _mrr(queries_mapped, corpus_target)
    mrr_true = _mrr(queries_true, corpus_target)
    return {
        "mrr_mapped": float(mrr_mapped),
        "mrr_true": float(mrr_true),
        "mrr_delta": float(mrr_mapped - mrr_true),
    }


def retrieval_retention_report(
    mapped: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    """Bundle of migration-relevant quality metrics."""
    cos = pairwise_cosine_stats(mapped, target)
    return {
        "cosine": cos,
        "top1_retention": topk_retention(mapped, target, k=1),
        "top10_retention": topk_retention(
            mapped, target, k=min(10, len(target))
        ),
        "mrr": mrr_delta(mapped, target, target),
    }
