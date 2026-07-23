"""Release validation tests — run before PyPI publish.

These tests verify the package is importable, the CLI works,
core functionality is intact, and the version is consistent.
They should pass on a clean install without optional deps.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


class TestPackageImport:
    """Verify the package imports cleanly."""

    def test_import_isotrieve(self):
        import isotrieve

        assert isotrieve.__version__ == "0.2.1"

    def test_import_mapping(self):
        from isotrieve.mapping.linear import RidgeMapping

        assert RidgeMapping.mapping_type == "ridge"

    def test_import_gate(self):
        from isotrieve.quality.gate import GateVerdict

        assert GateVerdict.PASS.value == "PASS"

    def test_import_serve(self):
        from isotrieve.serve import QueryAdapter

        assert QueryAdapter is not None

    def test_import_migrate(self):
        from isotrieve.migrate import migrate_store

        assert callable(migrate_store)

    def test_import_stores(self):
        from isotrieve.stores.numpy_files import NumpyFileStore

        assert NumpyFileStore is not None


class TestCoreFunctionality:
    """Verify core features work end-to-end."""

    def test_ridge_fit_transform(self):
        from isotrieve.mapping.linear import RidgeMapping

        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 8))
        W = rng.normal(size=(8, 12))
        Y = X @ W
        m = RidgeMapping(alpha="auto", seed=0)
        m.fit(X, Y)
        Z = m.transform(X[:10])
        assert Z.shape == (10, 12)
        assert np.isfinite(Z).all()

    def test_save_load_roundtrip(self, tmp_path):
        from isotrieve.mapping.linear import RidgeMapping

        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 8))
        W = rng.normal(size=(8, 12))
        Y = X @ W
        m = RidgeMapping(alpha="auto", seed=0)
        m.fit(X, Y)
        path = tmp_path / "test.isotrieve"
        m.save(path)
        loaded = RidgeMapping.load(path)
        Z1 = m.transform(X[:5])
        Z2 = loaded.transform(X[:5])
        np.testing.assert_allclose(Z1, Z2, atol=1e-6)

    def test_quality_gate(self):
        from isotrieve.mapping.linear import RidgeMapping
        from isotrieve.quality.gate import QualityGate

        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 8))
        W = rng.normal(size=(8, 12))
        Y = X @ W
        m = RidgeMapping(alpha="auto", seed=0)
        m.fit(X, Y)
        gate = QualityGate()
        report = gate.evaluate(m, X[:50], Y[:50])
        assert report.verdict.value in ("PASS", "WARN", "FAIL")
        assert 0.0 <= report.predicted_retention <= 1.0

    def test_numpy_store_roundtrip(self, tmp_path):
        from isotrieve.stores.numpy_files import NumpyFileStore

        vecs = np.random.randn(10, 8)
        texts = [f"doc {i}" for i in range(10)]
        store = NumpyFileStore.from_arrays(tmp_path / "store", vecs, texts=texts)
        assert store.count() == 10
        loaded = list(store.iter_vectors(batch_size=5))
        assert len(loaded) == 2

    def test_migration_manifest(self, tmp_path):
        from isotrieve.mapping.linear import RidgeMapping
        from isotrieve.migrate import migrate_store
        from isotrieve.stores.numpy_files import NumpyFileStore

        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 8))
        W = rng.normal(size=(8, 12))
        Y = X @ W
        m = RidgeMapping(alpha="auto", seed=0).fit(X, Y)

        source = NumpyFileStore.from_arrays(
            tmp_path / "src", X[:50], texts=[f"doc {i}" for i in range(50)]
        )
        target = NumpyFileStore(tmp_path / "tgt", create=True)

        manifest = migrate_store(source, target, m, batch_size=20)
        assert manifest.migrated_vectors == 50


def _isotrieve_bin() -> str | None:
    """Find the isotrieve CLI script next to the current Python interpreter."""
    isotrieve = Path(sys.executable).parent / "isotrieve"
    if isotrieve.exists():
        return str(isotrieve)
    return shutil.which("isotrieve")


def _run_isotrieve(*args: str) -> subprocess.CompletedProcess[str]:
    bin_path = _isotrieve_bin()
    if bin_path is None:
        pytest.skip("isotrieve CLI not installed")
    return subprocess.run([bin_path, *args], capture_output=True, text=True, timeout=10)


class TestCLI:
    """Verify CLI entry points work."""

    def test_cli_version(self):
        result = _run_isotrieve("version")
        assert result.returncode == 0
        assert "0.2.1" in result.stdout

    def test_cli_help(self):
        result = _run_isotrieve("--help")
        assert result.returncode == 0
        assert "isotrieve" in result.stdout.lower()

    def test_cli_plan(self):
        result = _run_isotrieve(
            "plan",
            "--source-model",
            "all-MiniLM-L6-v2",
            "--target-model",
            "bge-large",
            "--corpus-size",
            "10000",
        )
        assert result.returncode == 0
        assert "10,000" in result.stdout


class TestVersionConsistency:
    """Verify version is consistent across the package."""

    def test_version_in_pyproject(self):
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        assert 'version = "0.2.1"' in content

    def test_version_in_init(self):
        init = Path(__file__).parent.parent / "src" / "isotrieve" / "__init__.py"
        content = init.read_text()
        assert "0.2.1" in content
