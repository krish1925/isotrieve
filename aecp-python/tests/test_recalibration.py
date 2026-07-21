"""Tests for ScoreRecalibrator (WS-A)."""

from __future__ import annotations

import numpy as np
import pytest

from aecp.recalibration import ScoreRecalibrator


class TestScoreRecalibrator:
    def test_fit_and_transform(self):
        rng = np.random.default_rng(42)
        # Simulate compressed scores: mapped = 0.8 * ceiling + noise
        ceiling = rng.uniform(0.3, 1.0, 1000)
        mapped = 0.8 * ceiling + rng.normal(0, 0.02, 1000)
        mapped = np.clip(mapped, 0, 1)

        recal = ScoreRecalibrator()
        recal.fit(mapped, ceiling)
        assert recal.is_fitted

        # Recalibrated scores should be closer to ceiling
        recalibrated = recal.transform(mapped)
        err_before = np.mean(np.abs(mapped - ceiling))
        err_after = np.mean(np.abs(recalibrated - ceiling))
        assert err_after < err_before, (
            f"Recalibration made things worse: {err_after} > {err_before}"
        )

    def test_roundtrip_via_dict(self):
        rng = np.random.default_rng(42)
        ceiling = rng.uniform(0.3, 1.0, 500)
        mapped = 0.85 * ceiling + rng.normal(0, 0.01, 500)

        recal = ScoreRecalibrator()
        recal.fit(mapped, ceiling)

        # Save/load via dict
        data = recal.save_dict()
        recal2 = ScoreRecalibrator.from_dict(data)

        scores = np.array([0.5, 0.7, 0.9])
        np.testing.assert_array_equal(recal.transform(scores), recal2.transform(scores))

    def test_report_fields(self):
        rng = np.random.default_rng(0)
        ceiling = rng.uniform(0.2, 1.0, 2000)
        mapped = 0.8 * ceiling + rng.normal(0, 0.03, 2000)

        recal = ScoreRecalibrator()
        recal.fit(mapped, ceiling)
        report = recal.report
        assert report is not None
        assert report.n_pairs == 2000
        assert report.mean_shift > 0  # recalibrated mean > mapped mean
        assert 0.0 < report.margin_ratio < 2.0
        assert len(report.threshold_agreement) == 5
        for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
            assert tau in report.threshold_agreement
            assert 0.0 <= report.threshold_agreement[tau] <= 1.0

    def test_monotonicity(self):
        """Recalibrated scores must be non-decreasing with input scores."""
        rng = np.random.default_rng(1)
        ceiling = rng.uniform(0.0, 1.0, 5000)
        mapped = 0.7 * ceiling + 0.1 + rng.normal(0, 0.05, 5000)
        mapped = np.clip(mapped, -0.1, 1.1)

        recal = ScoreRecalibrator()
        recal.fit(mapped, ceiling)

        test_scores = np.linspace(0, 1, 100)
        recalibrated = recal.transform(test_scores)
        diffs = np.diff(recalibrated)
        assert np.all(diffs >= -1e-10), f"Non-monotonic: {diffs[diffs < 0]}"

    def test_clip_out_of_bounds(self):
        """Scores outside training range should be clipped."""
        rng = np.random.default_rng(0)
        ceiling = rng.uniform(0.3, 0.8, 500)
        mapped = rng.uniform(0.3, 0.8, 500)

        recal = ScoreRecalibrator()
        recal.fit(mapped, ceiling)

        # Very high score should clip to max training ceiling
        high = recal.transform(np.array([2.0]))
        assert high[0] <= ceiling.max() + 0.01

        # Very low score should clip to min training ceiling
        low = recal.transform(np.array([-1.0]))
        assert low[0] >= ceiling.min() - 0.01

    def test_insufficient_pairs(self):
        recal = ScoreRecalibrator()
        with pytest.raises(ValueError, match="Need ≥10"):
            recal.fit(np.array([0.5] * 5), np.array([0.6] * 5))

    def test_unfitted_raises(self):
        recal = ScoreRecalibrator()
        with pytest.raises(RuntimeError, match="not fitted|Cannot serialize"):
            recal.transform(np.array([0.5]))
        with pytest.raises(RuntimeError, match="not fitted|Cannot serialize"):
            recal.save_dict()
