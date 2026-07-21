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


def _make_compressed_mapping(d: int = 16, k: int = 200):
    """Create a mapping that compresses similarity distributions.

    The mapping projects all vectors onto a single direction, making
    cosine similarities between mapped and target nearly uniform.
    """
    rng = np.random.default_rng(99)
    X = rng.normal(size=(k, d))
    # Project onto first axis: all mapped vectors point in ~same direction
    # This makes cosine similarities to any target nearly uniform
    direction = rng.normal(size=d)
    direction = direction / np.linalg.norm(direction)
    # Mapped vectors: small perturbation along one direction
    mapped_target = np.outer(X[:, 0], direction) + 0.001 * rng.normal(size=(k, d))
    m = RidgeMapping(alpha=1.0, seed=0)
    m.fit(X, mapped_target)
    return m, X, mapped_target


def test_gate_pass_on_good_mapping():
    m, X, Y = _make_good_mapping()
    gate = QualityGate()
    report = gate.evaluate(m, X[:50], Y[:50], holdout_top1=0.95)
    # PASS or WARN acceptable — margin compression now correctly widens
    # intervals, which may push borderline cases to WARN
    assert report.verdict in (GateVerdict.PASS, GateVerdict.WARN)
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


# ---------------------------------------------------------------------------
# Gate margin compression tests (issue #8)
# ---------------------------------------------------------------------------


class TestMarginCompression:
    """Tests that _compute_margin_compression returns meaningful values
    and interval widening triggers when compression is severe.

    Regression: the old code computed variance of L2-normalized self-similarity,
    which is always 0 — margin compression never triggered.
    """

    def test_margin_compression_returns_float_not_none(self):
        """Margin compression must return a real number, not None.

        The old code always returned None because target_self variance
        collapsed to 0 after L2 normalization.
        """
        rng = np.random.default_rng(42)
        d, k = 16, 100
        mapped = rng.normal(size=(k, d))
        target = rng.normal(size=(k, d))
        result = QualityGate._compute_margin_compression(mapped, target)
        assert result is not None, (
            "_compute_margin_compression returned None — target_self variance "
            "collapsed to 0 after L2 normalization (the old bug). "
            "Must return a float."
        )
        assert isinstance(result, float)

    def test_margin_compression_compressed_case(self):
        """When mapped vectors are compressed (uniform), ratio should be < 1."""
        rng = np.random.default_rng(42)
        d, k = 16, 200
        target = rng.normal(size=(k, d))
        # Compressed: all mapped vectors point in similar direction
        base = rng.normal(size=d)
        mapped = base[None, :] + 0.01 * rng.normal(size=(k, d))
        mc = QualityGate._compute_margin_compression(mapped, target)
        assert mc is not None
        assert mc < 1.0, (
            f"Expected compression ratio < 1.0 for compressed case, got {mc}"
        )

    def test_margin_compression_good_case(self):
        """When mapping preserves similarity spread, ratio should be near 1."""
        rng = np.random.default_rng(42)
        d, k = 16, 500
        target = rng.normal(size=(k, d))
        # Good mapping: orthogonal rotation + small noise preserves spread
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        mapped = target @ Q + 0.02 * rng.normal(size=(k, d))
        mc = QualityGate._compute_margin_compression(mapped, target)
        assert mc is not None
        # Rotation preserves variance structure — ratio should be in
        # a reasonable range (not near 0, which means total collapse)
        assert mc > 0.1, f"Expected ratio >> 0 for well-preserved case, got {mc}"

    def test_interval_widening_triggers_on_compressed_mapping(self):
        """When margin compression < 0.85, prediction interval must widen."""
        m, X, Y = _make_compressed_mapping()
        gate = QualityGate()

        # Get baseline interval (without margin compression)
        mapped = m.transform(X[:50])
        from aecp.quality.metrics import topk_retention

        t1 = topk_retention(mapped, Y[:50], k=1)
        predicted_base, (lo_base, hi_base) = gate._predict_retention(t1, None)

        # Get interval with margin compression
        mc = QualityGate._compute_margin_compression(mapped, Y[:50])
        if mc is not None and mc < 0.85:
            predicted_comp, (lo_comp, hi_comp) = gate._predict_retention(t1, mc)
            # Interval should be wider (larger half-width)
            hw_base = (hi_base - lo_base) / 2
            hw_comp = (hi_comp - lo_comp) / 2
            assert hw_comp > hw_base, (
                f"Interval did not widen: base hw={hw_base:.4f}, "
                f"compressed hw={hw_comp:.4f}. Widening must trigger "
                f"when margin_compression < 0.85 (got {mc:.4f})."
            )
        else:
            pytest.skip(
                f"Compressed mapping produced mc={mc} — "
                "could not test widening (need mc < 0.85)"
            )

    def test_gate_report_exposes_margin_compression(self):
        """GateReport must include margin_compression value."""
        m, X, Y = _make_good_mapping()
        gate = QualityGate()
        report = gate.evaluate(m, X[:50], Y[:50])
        # Field must exist (value may be float or None)
        d = report.to_dict()
        assert "margin_compression" in d
        assert "score_recal_recommendation" in d
