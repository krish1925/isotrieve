"""Tests for quality metrics and provisional gate."""

from __future__ import annotations

import numpy as np

from aecp.mapping.linear import RidgeMapping
from aecp.quality.gate import GateVerdict, QualityGate
from aecp.quality.metrics import (
    cosine_similarity,
    pairwise_cosine_stats,
    topk_retention,
)


def test_cosine_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_topk_perfect():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    assert topk_retention(X, X, k=1) == 1.0


def test_topk_random_low():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(40, 8))
    B = rng.normal(size=(40, 8))
    # Unrelated spaces: top-1 retention should be near chance (1/40)
    assert topk_retention(A, B, k=1) < 0.25


def test_gate_pass_on_good_mapping():
    rng = np.random.default_rng(2)
    d = 8
    k = 10 * d
    X = rng.normal(size=(k, d))
    W = rng.normal(size=(d, d))
    Y = X @ W
    m = RidgeMapping(alpha=0.1, seed=0).fit(X, Y)
    # Fresh sample
    X2 = rng.normal(size=(80, d))
    Y2 = X2 @ W
    report = QualityGate().evaluate(m, X2, Y2)
    assert report.verdict in (GateVerdict.PASS, GateVerdict.WARN)
    assert report.gate_model_used is True
    assert report.provisional_thresholds is False


def test_pairwise_stats_keys():
    a = np.eye(5)
    s = pairwise_cosine_stats(a, a)
    assert set(s) >= {"mean", "median", "p5"}
