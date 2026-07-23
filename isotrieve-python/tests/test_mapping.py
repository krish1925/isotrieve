"""Offline unit tests for all mapping types + .isotrieve I/O."""

from __future__ import annotations

import numpy as np
import pytest

from isotrieve.mapping.base import read_isotrieve_header
from isotrieve.mapping.linear import (
    LowRankAffineMapping,
    OrthogonalProcrustesMapping,
    ProcrustesDiagMapping,
    RidgeMapping,
)
from isotrieve.mapping.registry import load_mapping
from isotrieve.quality.metrics import topk_retention


def _paired_gaussian(
    k: int, d_src: int, d_tgt: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic paired spaces related by a known linear map + noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(k, d_src))
    W_true = rng.normal(size=(d_src, d_tgt))
    Y = X @ W_true + 0.01 * rng.normal(size=(k, d_tgt))
    return X, Y


# ── RidgeMapping ──


def test_ridge_fit_transform_recovers_targets():
    d_src, d_tgt = 16, 24
    k = 10 * min(d_src, d_tgt)
    X, Y = _paired_gaussian(k, d_src, d_tgt, seed=1)
    m = RidgeMapping(alpha=1.0, seed=0)
    m.fit(X, Y)
    pred = m.transform(X)
    from isotrieve.mapping.base import l2_normalize

    y_n = l2_normalize(Y)
    sims = np.sum(pred * y_n, axis=1)
    assert float(np.mean(sims)) > 0.95


def test_ridge_rectangular_dims():
    X, Y = _paired_gaussian(200, 16, 32, seed=2)
    m = RidgeMapping(alpha="auto", seed=0)
    m.fit(X, Y)
    assert m.d_src == 16
    assert m.d_tgt == 32
    Z = m.transform(X[:10])
    assert Z.shape == (10, 32)
    norms = np.linalg.norm(Z, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_ridge_warns_on_small_k():
    X = np.random.randn(20, 16)
    Y = np.random.randn(20, 16)
    with pytest.warns(UserWarning, match="below the recommended minimum"):
        RidgeMapping().fit(X, Y)


def test_ridge_rejects_nan():
    X, Y = _paired_gaussian(200, 8, 8, seed=3)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        RidgeMapping().fit(X, Y)


def test_seeded_determinism():
    X, Y = _paired_gaussian(200, 12, 12, seed=4)
    m1 = RidgeMapping(alpha=1.0, seed=42).fit(X, Y)
    m2 = RidgeMapping(alpha=1.0, seed=42).fit(X, Y)
    np.testing.assert_allclose(m1.transform(X), m2.transform(X))


def test_save_load_roundtrip(tmp_path):
    X, Y = _paired_gaussian(200, 10, 14, seed=5)
    m = RidgeMapping(alpha=0.5, seed=7).fit(X, Y)
    m.set_meta(source_model_id="a", target_model_id="b")
    path = tmp_path / "map.isotrieve"
    m.save(path)
    header = read_isotrieve_header(path)
    assert header["mapping_type"] == "ridge"
    assert header["meta"]["source_model_id"] == "a"
    loaded = load_mapping(path)
    np.testing.assert_allclose(m.transform(X), loaded.transform(X))


def test_streaming_equals_batch():
    X, Y = _paired_gaussian(200, 10, 10, seed=6)
    m = RidgeMapping(alpha=1.0, seed=0).fit(X, Y)
    batch = m.transform(X)
    parts = list(m.transform_batches([X[:50], X[50:120], X[120:]]))
    streamed = np.vstack(parts)
    np.testing.assert_allclose(batch, streamed)


def test_validation_report_present():
    X, Y = _paired_gaussian(200, 8, 8, seed=9)
    m = RidgeMapping(seed=1).fit(X, Y)
    r = m.validation_report()
    assert r.n_holdout > 0
    assert 0.0 <= r.top1_retention <= 1.0


# ── OrthogonalProcrustesMapping ──


def test_procrustes_square_only():
    X, Y = _paired_gaussian(200, 10, 12, seed=7)
    with pytest.raises(ValueError, match="d_src == d_tgt"):
        OrthogonalProcrustesMapping().fit(X, Y)


def test_procrustes_fit():
    X, Y = _paired_gaussian(200, 16, 16, seed=8)
    m = OrthogonalProcrustesMapping(seed=0).fit(X, Y)
    Z = m.transform(X)
    assert Z.shape == X.shape
    assert topk_retention(Z, Y, k=1) > 0.5


def test_procrustes_inverse():
    X, Y = _paired_gaussian(200, 16, 16, seed=10)
    m = OrthogonalProcrustesMapping(seed=0).fit(X, Y)
    Z = m.transform(X)
    X_back = m.inverse_transform(Z)
    assert X_back.shape == X.shape
    from isotrieve.mapping.base import l2_normalize

    x_n = l2_normalize(X)
    sims = np.sum(X_back * x_n, axis=1)
    assert float(np.mean(sims)) > 0.5


def test_procrustes_save_load(tmp_path):
    X, Y = _paired_gaussian(200, 16, 16, seed=11)
    m = OrthogonalProcrustesMapping(seed=0).fit(X, Y)
    path = tmp_path / "proc.isotrieve"
    m.save(path)
    loaded = load_mapping(path)
    np.testing.assert_allclose(m.transform(X), loaded.transform(X))


# ── ProcrustesDiagMapping ──


def test_procrustes_diag_fit():
    X, Y = _paired_gaussian(200, 16, 16, seed=12)
    m = ProcrustesDiagMapping(seed=0).fit(X, Y)
    Z = m.transform(X)
    assert Z.shape == X.shape
    # Should beat pure Procrustes due to diagonal scaling
    t1_pure = topk_retention(
        OrthogonalProcrustesMapping(seed=0).fit(X, Y).transform(X), Y, k=1
    )
    t1_diag = topk_retention(Z, Y, k=1)
    assert t1_diag >= t1_pure - 0.05  # at least comparable


def test_procrustes_diag_inverse():
    X, Y = _paired_gaussian(200, 16, 16, seed=13)
    m = ProcrustesDiagMapping(seed=0).fit(X, Y)
    Z = m.transform(X)
    X_back = m.inverse_transform(Z)
    assert X_back.shape == X.shape


def test_procrustes_diag_square_only():
    X, Y = _paired_gaussian(200, 10, 12, seed=14)
    with pytest.raises(ValueError, match="d_src == d_tgt"):
        ProcrustesDiagMapping().fit(X, Y)


def test_procrustes_diag_save_load(tmp_path):
    X, Y = _paired_gaussian(200, 16, 16, seed=15)
    m = ProcrustesDiagMapping(seed=0).fit(X, Y)
    path = tmp_path / "pd.isotrieve"
    m.save(path)
    loaded = load_mapping(path)
    np.testing.assert_allclose(m.transform(X), loaded.transform(X))


# ── LowRankAffineMapping ──


def test_lowrank_affine_fit():
    X, Y = _paired_gaussian(200, 16, 24, seed=16)
    m = LowRankAffineMapping(alpha=1.0, rank=8, seed=0).fit(X, Y)
    Z = m.transform(X)
    assert Z.shape == (200, 24)
    from isotrieve.mapping.base import l2_normalize

    y_n = l2_normalize(Y)
    sims = np.sum(Z * y_n, axis=1)
    assert float(np.mean(sims)) > 0.8


def test_lowrank_affine_full_rank_equals_ridge():
    X, Y = _paired_gaussian(200, 16, 16, seed=17)
    m_lr = LowRankAffineMapping(alpha=1.0, rank=16, seed=0).fit(X, Y)
    m_ridge = RidgeMapping(alpha=1.0, seed=0).fit(X, Y)
    # Full-rank low-rank should be very close to ridge
    np.testing.assert_allclose(m_lr.transform(X), m_ridge.transform(X), atol=1e-4)


def test_lowrank_affine_rectangular():
    X, Y = _paired_gaussian(400, 32, 64, seed=18)
    m = LowRankAffineMapping(alpha="auto", rank=16, seed=0).fit(X, Y)
    Z = m.transform(X[:10])
    assert Z.shape == (10, 64)


def test_lowrank_affine_save_load(tmp_path):
    X, Y = _paired_gaussian(200, 16, 24, seed=19)
    m = LowRankAffineMapping(alpha=1.0, rank=8, seed=0).fit(X, Y)
    path = tmp_path / "lr.isotrieve"
    m.save(path)
    loaded = load_mapping(path)
    np.testing.assert_allclose(m.transform(X), loaded.transform(X))


# ── Cross-adapter comparison ──


def test_all_adapters_beat_random_on_synthetic():
    """All adapters should significantly beat random on a known linear pair."""
    X, Y = _paired_gaussian(200, 16, 16, seed=20)
    adapters = [
        RidgeMapping(alpha=1.0, seed=0),
        OrthogonalProcrustesMapping(seed=0),
        ProcrustesDiagMapping(seed=0),
        LowRankAffineMapping(alpha=1.0, rank=8, seed=0),
    ]
    for adapter in adapters:
        adapter.fit(X, Y)
        Z = adapter.transform(X)
        t1 = topk_retention(Z, Y, k=1)
        assert t1 > 0.5, f"{adapter.mapping_type} failed: top-1={t1:.3f}"
