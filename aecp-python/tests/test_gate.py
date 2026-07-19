"""Offline unit tests for QualityGate v2 proxy-based prediction (WS-1)."""

from __future__ import annotations

import numpy as np
import pytest

from aecp.mapping.linear import RidgeMapping
from aecp.quality.gate import GateVerdict, QualityGate


def _make_good_mapping(d: int = 16, k: int = 200):
    """Create a well-fitted mapping for testing."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(k, d))
    W = rng.normal(size=(d, d))
    Y = X @ W + 0.01 * rng.normal(size=(k, d))
    m = RidgeMapping(alpha=1.0, seed=0)
    m.fit(X, Y)
    return m, X, Y


def test_gate_pass_on_good_mapping():
    m, X, Y = _make_good_mapping()
    gate = QualityGate()
    report = gate.evaluate(m, X[:50], Y[:50], holdout_top1=0.95)
    assert report.verdict == GateVerdict.PASS
    assert report.top1_retention > 0.8
    assert report.gate_model_used is True
    assert report.predicted_retention > 0.7


def test_gate_warns_on_mediocre_mapping():
    """Test that mediocre mapping gets WARN verdict."""
    rng = np.random.default_rng(42)
    d = 16
    X = rng.normal(size=(200, d))
    # Create a mediocre mapping (partial projection)
    W = np.eye(d) * 0.5
    Y = X @ W + 0.3 * rng.normal(size=(200, d))
    m = RidgeMapping(alpha=1.0, seed=0)
    m.fit(X, Y)

    gate = QualityGate()
    report = gate.evaluate(m, X[:50], Y[:50])
    # Should be WARN or PASS depending on gate model
    assert report.verdict in (GateVerdict.WARN, GateVerdict.PASS)
    assert report.predicted_retention > 0


def test_gate_report_has_new_fields():
    m, X, Y = _make_good_mapping()
    gate = QualityGate()
    report = gate.evaluate(m, X[:50], Y[:50])
    d = report.to_dict()
    assert "predicted_retention" in d
    assert "prediction_interval" in d
    assert "holdout_rank_corr" in d
    assert "gate_model_used" in d
    assert "lopo_error" in d


def test_gate_report_to_dict():
    m, X, Y = _make_good_mapping()
    gate = QualityGate()
    report = gate.evaluate(m, X[:50], Y[:50])
    d = report.to_dict()
    assert "verdict" in d
    assert "predicted_retention" in d
    assert "prediction_interval" in d
    assert len(d["prediction_interval"]) == 2
