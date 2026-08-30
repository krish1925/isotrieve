"""Tests for gate and doctor CLI commands."""

from __future__ import annotations

import json

import numpy as np
from tests.fakes import make_mapping, save_mapping
from typer.testing import CliRunner

from isotrieve.cli import app

runner = CliRunner()


class TestGateCommand:
    def test_gate_pass(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12, k=200)
        save_mapping(m, tmp_path)

        rng = np.random.default_rng(99)
        X = rng.normal(size=(50, 8))
        Y = rng.normal(size=(50, 12))
        np.save(tmp_path / "X.npy", X)
        np.save(tmp_path / "Y.npy", Y)

        result = runner.invoke(
            app,
            [
                "gate",
                "--mapping",
                str(tmp_path / "map.isotrieve"),
                "--source-vectors",
                str(tmp_path / "X.npy"),
                "--target-vectors",
                str(tmp_path / "Y.npy"),
                "--format",
                "json",
            ],
        )
        # Should not crash (exit 0 = pass, exit 1 = gate raised)
        assert result.exit_code in (0, 1)
        # JSON output should be valid in both cases
        data = json.loads(result.output)
        assert "verdict" in data

    def test_gate_requires_vectors(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12, k=200)
        save_mapping(m, tmp_path)

        result = runner.invoke(
            app,
            [
                "gate",
                "--mapping",
                str(tmp_path / "map.isotrieve"),
            ],
        )
        assert result.exit_code == 2

    def test_gate_queries_only(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12, k=200)
        save_mapping(m, tmp_path)

        rng = np.random.default_rng(99)
        np.save(tmp_path / "queries.npy", rng.normal(size=(50, 8)))
        np.save(tmp_path / "corpus.npy", rng.normal(size=(50, 12)))

        result = runner.invoke(
            app,
            [
                "gate",
                "--mapping",
                str(tmp_path / "map.isotrieve"),
                "--queries",
                str(tmp_path / "queries.npy"),
                "--corpus",
                str(tmp_path / "corpus.npy"),
                "--format",
                "json",
            ],
        )
        assert result.exit_code in (0, 1)

    def test_gate_seed_sensitivity(self, tmp_path):
        """--seed-sensitivity refits on subsamples and emits per-seed JSON."""
        m = make_mapping(d_src=8, d_tgt=12, k=200)
        save_mapping(m, tmp_path)

        rng = np.random.default_rng(99)
        X = rng.normal(size=(60, 8))
        W = rng.normal(size=(8, 12))
        Y = X @ W
        np.save(tmp_path / "X.npy", X)
        np.save(tmp_path / "Y.npy", Y)

        result = runner.invoke(
            app,
            [
                "gate",
                "--mapping",
                str(tmp_path / "map.isotrieve"),
                "--source-vectors",
                str(tmp_path / "X.npy"),
                "--target-vectors",
                str(tmp_path / "Y.npy"),
                "--seed-sensitivity",
                "--seed-sensitivity-runs",
                "3",
                "--seed-sensitivity-threshold",
                "1.0",
                "--format",
                "json",
            ],
        )
        assert result.exit_code in (0, 1)
        data = json.loads(result.output)
        ss = data["seed_sensitivity"]
        assert ss["runs"] == 3
        assert len(ss["per_seed_retention"]) == 3
        assert ss["unstable"] is False
        assert "mean_retention" in ss

    def test_gate_seed_sensitivity_requires_paired_vectors(self, tmp_path):
        """--seed-sensitivity refuses queries/corpus mode (no paired calib)."""
        m = make_mapping(d_src=8, d_tgt=12, k=200)
        save_mapping(m, tmp_path)

        rng = np.random.default_rng(99)
        np.save(tmp_path / "queries.npy", rng.normal(size=(50, 8)))
        np.save(tmp_path / "corpus.npy", rng.normal(size=(50, 12)))

        result = runner.invoke(
            app,
            [
                "gate",
                "--mapping",
                str(tmp_path / "map.isotrieve"),
                "--queries",
                str(tmp_path / "queries.npy"),
                "--corpus",
                str(tmp_path / "corpus.npy"),
                "--seed-sensitivity",
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 2

    def test_gate_seed_sensitivity_without_mapping(self, tmp_path):
        """--seed-sensitivity runs standalone without a mapping."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(60, 8))
        W = rng.normal(size=(8, 12))
        Y = X @ W
        np.save(tmp_path / "X.npy", X)
        np.save(tmp_path / "Y.npy", Y)

        result = runner.invoke(
            app,
            [
                "gate",
                "--source-vectors",
                str(tmp_path / "X.npy"),
                "--target-vectors",
                str(tmp_path / "Y.npy"),
                "--seed-sensitivity",
                "--seed-sensitivity-runs",
                "3",
                "--seed-sensitivity-threshold",
                "1.0",
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "seed_sensitivity" in data


class TestCalibrateQueriesOnly:
    def test_queries_only_success(self, tmp_path):
        rng = np.random.default_rng(42)
        queries = rng.normal(size=(200, 8))
        target = rng.normal(size=(200, 12))
        np.save(tmp_path / "queries.npy", queries)
        np.save(tmp_path / "target.npy", target)

        result = runner.invoke(
            app,
            [
                "calibrate",
                "--queries-only",
                "--queries",
                str(tmp_path / "queries.npy"),
                "--target-vectors",
                str(tmp_path / "target.npy"),
                "-o",
                str(tmp_path / "out.isotrieve"),
                "--k",
                "200",
            ],
        )
        assert result.exit_code == 0
        assert (tmp_path / "out.isotrieve").exists()

    def test_queries_only_requires_queries(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "calibrate",
                "--queries-only",
                "-o",
                str(tmp_path / "out.isotrieve"),
            ],
        )
        assert result.exit_code == 2


class TestDoctorCommand:
    def test_doctor_numpy(self, tmp_path):
        from isotrieve.stores.base import VectorRecord
        from isotrieve.stores.numpy_files import NumpyFileStore

        store = NumpyFileStore(tmp_path / "store", create=True)
        rng = np.random.default_rng(42)
        records = [
            VectorRecord(id=str(i), vector=rng.normal(size=(8,)), text=f"doc {i}")
            for i in range(10)
        ]
        store.write_vectors([records])

        result = runner.invoke(
            app,
            [
                "doctor",
                "--store",
                "numpy",
                "--url",
                str(tmp_path / "store"),
            ],
        )
        assert result.exit_code == 0
        assert "10" in result.output  # vector count

    def test_doctor_json(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "doctor",
                "--store",
                "numpy",
                "--url",
                str(tmp_path / "nonexistent"),
                "--json",
            ],
        )
        # Should still return valid JSON even with error
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["store_type"] == "numpy"
