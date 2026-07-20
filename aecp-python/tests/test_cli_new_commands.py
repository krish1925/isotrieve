"""Tests for gate and doctor CLI commands."""

from __future__ import annotations

import json

import numpy as np
from tests.fakes import make_mapping, save_mapping
from typer.testing import CliRunner

from aecp.cli import app

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

        result = runner.invoke(app, [
            "gate",
            "--mapping", str(tmp_path / "map.aecp"),
            "--source-vectors", str(tmp_path / "X.npy"),
            "--target-vectors", str(tmp_path / "Y.npy"),
            "--format", "json",
        ])
        # Should not crash (exit 0 or 1 depending on gate verdict)
        assert result.exit_code in (0, 1)
        # JSON output should be valid
        if result.exit_code == 0:
            data = json.loads(result.output)
            assert "verdict" in data

    def test_gate_requires_vectors(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12, k=200)
        save_mapping(m, tmp_path)

        result = runner.invoke(app, [
            "gate",
            "--mapping", str(tmp_path / "map.aecp"),
        ])
        assert result.exit_code == 2

    def test_gate_queries_only(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12, k=200)
        save_mapping(m, tmp_path)

        rng = np.random.default_rng(99)
        np.save(tmp_path / "queries.npy", rng.normal(size=(50, 8)))
        np.save(tmp_path / "corpus.npy", rng.normal(size=(50, 12)))

        result = runner.invoke(app, [
            "gate",
            "--mapping", str(tmp_path / "map.aecp"),
            "--queries", str(tmp_path / "queries.npy"),
            "--corpus", str(tmp_path / "corpus.npy"),
            "--format", "json",
        ])
        assert result.exit_code in (0, 1)


class TestDoctorCommand:
    def test_doctor_numpy(self, tmp_path):
        from aecp.stores.base import VectorRecord
        from aecp.stores.numpy_files import NumpyFileStore

        store = NumpyFileStore(tmp_path / "store", create=True)
        rng = np.random.default_rng(42)
        records = [
            VectorRecord(id=str(i), vector=rng.normal(size=(8,)), text=f"doc {i}")
            for i in range(10)
        ]
        store.write_vectors([records])

        result = runner.invoke(app, [
            "doctor",
            "--store", "numpy",
            "--url", str(tmp_path / "store"),
        ])
        assert result.exit_code == 0
        assert "10" in result.output  # vector count

    def test_doctor_json(self, tmp_path):
        result = runner.invoke(app, [
            "doctor",
            "--store", "numpy",
            "--url", str(tmp_path / "nonexistent"),
            "--json",
        ])
        # Should still return valid JSON even with error
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["store_type"] == "numpy"
