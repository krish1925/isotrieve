"""CLI smoke tests (no network)."""

from __future__ import annotations

import numpy as np
from typer.testing import CliRunner

from isotrieve import __version__
from isotrieve.cli import app
from isotrieve.mapping.linear import RidgeMapping

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
    out = tmp_path / "map.isotrieve"
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


def test_calibrate_queries_only_no_source_docs(tmp_path):
    """--queries-only fits from query log + stored vectors, no source docs."""
    rng = np.random.default_rng(5)
    d_src, d_tgt = 8, 12
    k = 10 * d_src
    W = rng.normal(size=(d_src, d_tgt))
    queries = rng.normal(size=(k, d_src))  # query log in source space
    stored = queries @ W  # stored vectors are in target space already
    qp = tmp_path / "queries.npy"
    sp = tmp_path / "stored.npy"
    np.save(qp, queries)
    np.save(sp, stored)
    out = tmp_path / "map.isotrieve"

    # No --source-vectors, no --texts, no --source-model — queries only
    r = runner.invoke(
        app,
        [
            "calibrate",
            "--queries-only",
            "--queries",
            str(qp),
            "--target-vectors",
            str(sp),
            "-o",
            str(out),
            "--seed",
            "0",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    assert out.exists()

    # Fitted mapping can be inspected and maps d_src -> d_tgt
    r2 = runner.invoke(app, ["inspect", str(out), "--json"])
    assert r2.exit_code == 0
    assert "ridge" in r2.stdout


def test_calibrate_queries_only_requires_queries(tmp_path):
    """--queries-only without --queries fails fast with exit 2."""
    rng = np.random.default_rng(6)
    np.save(tmp_path / "stored.npy", rng.normal(size=(80, 12)))
    r = runner.invoke(
        app,
        [
            "calibrate",
            "--queries-only",
            "--target-vectors",
            str(tmp_path / "stored.npy"),
            "-o",
            str(tmp_path / "map.isotrieve"),
        ],
    )
    assert r.exit_code == 2


def test_transform_cli(tmp_path):
    from isotrieve.stores.numpy_files import NumpyFileStore

    rng = np.random.default_rng(1)
    d = 8
    k = 10 * d
    X = rng.normal(size=(k, d))
    Y = X @ rng.normal(size=(d, 10))
    m = RidgeMapping(alpha=0.1, seed=0).fit(X, Y)
    map_path = tmp_path / "m.isotrieve"
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
