"""CLI smoke tests (no network)."""

from __future__ import annotations

import numpy as np
from typer.testing import CliRunner

from aecp import __version__
from aecp.cli import app
from aecp.mapping.linear import RidgeMapping

runner = CliRunner()


def test_version():
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0
    assert __version__ in r.stdout


def test_plan_json():
    r = runner.invoke(
        app,
        [
            "plan",
            "--source-model",
            "text-embedding-ada-002",
            "--target-model",
            "text-embedding-3-large",
            "--corpus-size",
            "1000000",
            "--json",
        ],
    )
    assert r.exit_code == 0
    assert "recommended_k" in r.stdout


def test_calibrate_from_npy_and_inspect(tmp_path):
    rng = np.random.default_rng(0)
    d = 8
    k = 10 * d
    X = rng.normal(size=(k, d))
    Y = X @ rng.normal(size=(d, d)) + 0.01 * rng.normal(size=(k, d))
    xp = tmp_path / "X.npy"
    yp = tmp_path / "Y.npy"
    np.save(xp, X)
    np.save(yp, Y)
    out = tmp_path / "map.aecp"
    r = runner.invoke(
        app,
        [
            "calibrate",
            "--source-vectors",
            str(xp),
            "--target-vectors",
            str(yp),
            "-o",
            str(out),
            "--seed",
            "0",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    assert out.exists()
    r2 = runner.invoke(app, ["inspect", str(out), "--json"])
    assert r2.exit_code == 0
    assert "ridge" in r2.stdout


def test_transform_cli(tmp_path):
    from aecp.stores.numpy_files import NumpyFileStore

    rng = np.random.default_rng(1)
    d = 8
    k = 10 * d
    X = rng.normal(size=(k, d))
    Y = X @ rng.normal(size=(d, 10))
    m = RidgeMapping(alpha=0.1, seed=0).fit(X, Y)
    map_path = tmp_path / "m.aecp"
    m.save(map_path)
    NumpyFileStore.from_arrays(tmp_path / "src", rng.normal(size=(200, d)))
    r = runner.invoke(
        app,
        [
            "transform",
            "--mapping",
            str(map_path),
            "--source-dir",
            str(tmp_path / "src"),
            "--target-dir",
            str(tmp_path / "dst"),
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    assert (tmp_path / "dst" / "vectors.npy").exists()
