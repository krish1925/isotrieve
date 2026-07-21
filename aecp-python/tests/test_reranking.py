"""Tests for aecp.reranking (WS-B)."""

import numpy as np
import pytest

from aecp.reranking import ConfidenceReport, ConfidenceScorer, confidence_summary


class TestConfidenceScorer:
    def test_basic_scoring(self):
        rng = np.random.default_rng(42)
        sims = rng.uniform(0.5, 0.9, (5, 100))
        # Make query 0 have a clear winner
        sims[0, -1] = 0.95
        sims[0, -2] = 0.70

        scorer = ConfidenceScorer()
        reports = scorer.score_queries([f"q{i}" for i in range(5)], sims)
        assert len(reports) == 5
        assert isinstance(reports[0], ConfidenceReport)
        assert reports[0].confidence == "high"

    def test_low_margin(self):
        sims = np.ones((1, 100)) * 0.7
        sims[0, -1] = 0.701  # tiny margin

        scorer = ConfidenceScorer(margin_low=0.01)
        reports = scorer.score_queries(["q0"], sims)
        assert reports[0].confidence == "low"

    def test_empty(self):
        scorer = ConfidenceScorer()
        reports = scorer.score_queries([], np.array([]).reshape(0, 0))
        assert reports == []

    def test_summary(self):
        reports = [
            ConfidenceReport("q1", 0.05, 0.9, "high", 10),
            ConfidenceReport("q2", 0.001, 0.4, "low", 10),
            ConfidenceReport("q3", 0.01, 0.7, "medium", 10),
        ]
        s = confidence_summary(reports)
        assert s["n"] == 3
        assert s["n_high"] == 1
        assert s["n_low"] == 1
        assert s["n_medium"] == 1

    def test_summary_empty(self):
        s = confidence_summary([])
        assert s["n"] == 0
