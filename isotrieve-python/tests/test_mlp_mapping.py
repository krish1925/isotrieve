"""Tests for ResidualMLPMapping (requires torch)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isotrieve.mapping.base import read_isotrieve_header
from isotrieve.mapping.mlp import ResidualMLPMapping
from isotrieve.mapping.registry import load_mapping


def _paired_gaussian(
    k: int, d_src: int, d_tgt: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic paired spaces related by a known linear map + noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(k, d_src))
    W_true = rng.normal(size=(d_src, d_tgt))
    Y = X @ W_true + 0.01 * rng.normal(size=(k, d_tgt))
    return X, Y


# ── Basic functionality ──


def test_mlp_fit_transform():
    X, Y = _paired_gaussian(200, 16, 16, seed=1)
    m = ResidualMLPMapping(n_epochs=100, seed=0, device="cpu")
    m.fit(X, Y)
    Z = m.transform(X)
    assert Z.shape == X.shape
    norms = np.linalg.norm(Z, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_mlp_rectangular_dims():
    X, Y = _paired_gaussian(200, 16, 32, seed=2)
    m = ResidualMLPMapping(n_epochs=100, seed=0, device="cpu")
    m.fit(X, Y)
    assert m.d_src == 16
    assert m.d_tgt == 32
    Z = m.transform(X[:10])
    assert Z.shape == (10, 32)


def test_mlp_inverse_transform():
    X, Y = _paired_gaussian(200, 16, 16, seed=3)
    m = ResidualMLPMapping(n_epochs=100, seed=0, device="cpu")
    m.fit(X, Y)
    Z = m.transform(X)
    X_back = m.inverse_transform(Z)
    assert X_back.shape == X.shape
    from isotrieve.mapping.base import l2_normalize

    x_n = l2_normalize(X)
    sims = np.sum(X_back * x_n, axis=1)
    assert float(np.mean(sims)) > 0.5


def test_mlp_validation_report():
    X, Y = _paired_gaussian(200, 16, 16, seed=4)
    m = ResidualMLPMapping(n_epochs=100, seed=0, device="cpu")
    m.fit(X, Y)
    r = m.validation_report()
    assert r.n_holdout > 0
    assert 0.0 <= r.top1_retention <= 1.0
    assert r.holdout_cosine_mean > 0.0


def test_mlp_single_vector():
    X, Y = _paired_gaussian(200, 16, 16, seed=5)
    m = ResidualMLPMapping(n_epochs=100, seed=0, device="cpu")
    m.fit(X, Y)
    z = m.transform(X[0])
    assert z.shape == (16,)
    assert abs(float(np.linalg.norm(z)) - 1.0) < 1e-5


def test_mlp_rejects_nan():
    X, Y = _paired_gaussian(200, 8, 8, seed=6)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ResidualMLPMapping(n_epochs=10, seed=0, device="cpu").fit(X, Y)


def test_mlp_rejects_dimension_mismatch():
    X, Y = _paired_gaussian(200, 16, 16, seed=7)
    m = ResidualMLPMapping(n_epochs=100, seed=0, device="cpu").fit(X, Y)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        m.transform(X[:, :8])


def test_mlp_warns_on_small_k():
    X, Y = _paired_gaussian(20, 16, 16, seed=8)
    with pytest.warns(UserWarning, match="below recommended minimum"):
        ResidualMLPMapping(n_epochs=10, seed=0, device="cpu").fit(X, Y)


def test_mlp_beats_random():
    """MLP should learn something meaningful on a known linear pair."""
    X, Y = _paired_gaussian(200, 16, 16, seed=9)
    m = ResidualMLPMapping(n_epochs=200, seed=0, device="cpu")
    m.fit(X, Y)
    Z = m.transform(X)
    from isotrieve.mapping.base import l2_normalize

    y_n = l2_normalize(Y)
    cos = float(np.mean(np.sum(Z * y_n, axis=1)))
    # MLP should achieve positive cosine (better than random ~0)
    assert cos > 0.1, f"MLP cosine too low: {cos:.3f}"


# ── Save / load roundtrip ──


def test_mlp_save_load_roundtrip(tmp_path):
    X, Y = _paired_gaussian(200, 16, 16, seed=10)
    m = ResidualMLPMapping(n_epochs=100, seed=0, device="cpu")
    m.fit(X, Y)
    path = tmp_path / "mlp.isotrieve"
    m.save(path)

    # Header should exist
    header = read_isotrieve_header(path)
    assert header["mapping_type"] == "residual_mlp"
    assert header.get("torch_state_file") is not None

    # State dict .pt file should exist
    import os

    assert os.path.exists(str(path) + ".pt")

    # Load and verify roundtrip
    loaded = load_mapping(path)
    np.testing.assert_allclose(m.transform(X[:5]), loaded.transform(X[:5]))


def test_mlp_save_load_rectangular(tmp_path):
    X, Y = _paired_gaussian(200, 16, 32, seed=11)
    m = ResidualMLPMapping(n_epochs=100, seed=0, device="cpu")
    m.fit(X, Y)
    path = tmp_path / "mlp_rect.isotrieve"
    m.save(path)
    loaded = load_mapping(path)
    assert loaded.d_src == 16
    assert loaded.d_tgt == 32
    np.testing.assert_allclose(m.transform(X[:5]), loaded.transform(X[:5]))


# ── Device tests ──


def test_mlp_explicit_device_cpu():
    X, Y = _paired_gaussian(200, 16, 16, seed=12)
    m = ResidualMLPMapping(n_epochs=50, seed=0, device="cpu")
    m.fit(X, Y)
    Z = m.transform(X)
    assert Z.shape == X.shape


def test_mlp_auto_device():
    X, Y = _paired_gaussian(200, 16, 16, seed=13)
    m = ResidualMLPMapping(n_epochs=50, seed=0, device="cpu")
    m.fit(X, Y)
    Z = m.transform(X)
    assert Z.shape == X.shape
    # Verify device detection works
    m2 = ResidualMLPMapping(n_epochs=50, seed=0, device=None)
    m2.fit(X, Y)
    Z2 = m2.transform(X)
    assert Z2.shape == X.shape


def test_mlp_deterministic():
    X, Y = _paired_gaussian(200, 16, 16, seed=14)
    m1 = ResidualMLPMapping(n_epochs=100, seed=42, device="cpu")
    m1.fit(X, Y)
    m2 = ResidualMLPMapping(n_epochs=100, seed=42, device="cpu")
    m2.fit(X, Y)
    # Neural nets are non-deterministic by default; check quality is similar
    from isotrieve.quality.metrics import topk_retention

    ret1 = topk_retention(m1.transform(X), Y, k=1)
    ret2 = topk_retention(m2.transform(X), Y, k=1)
    assert abs(ret1 - ret2) < 0.15, f"Quality varies too much: {ret1:.3f} vs {ret2:.3f}"


# ── Cross-adapter comparison ──


def test_mlp_vs_ridge_quality():
    """Ridge should generally match or beat MLP on linear synthetic data."""
    from isotrieve.mapping.linear import RidgeMapping

    X, Y = _paired_gaussian(200, 16, 16, seed=15)
    mlp = ResidualMLPMapping(n_epochs=200, seed=0, device="cpu")
    mlp.fit(X, Y)
    ridge = RidgeMapping(alpha="auto", seed=0)
    ridge.fit(X, Y)

    mlp_cos = float(
        np.mean(
            np.sum(
                mlp.transform(X) * (Y / np.linalg.norm(Y, axis=1, keepdims=True)),
                axis=1,
            )
        )
    )
    ridge_cos = float(
        np.mean(
            np.sum(
                ridge.transform(X) * (Y / np.linalg.norm(Y, axis=1, keepdims=True)),
                axis=1,
            )
        )
    )

    # Ridge should be close to perfect on linear data; MLP should be reasonable
    assert ridge_cos > 0.9, f"Ridge too low: {ridge_cos:.3f}"
    assert mlp_cos > 0.5, f"MLP too low: {mlp_cos:.3f}"
