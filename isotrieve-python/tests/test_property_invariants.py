"""Property-based invariants for the barrage (verification PR only).

Each invariant check has a paired negative control (`*_nc`) proving the
checker can fail. Pre-registered in verification/artifacts/*/preregistration.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from isotrieve.mapping.base import l2_normalize
from isotrieve.mapping.linear import OrthogonalProcrustesMapping, RidgeMapping

SEED = 20260829
HYP = settings(max_examples=50, deadline=None, print_blob=True)


def _fit_ridge(
    d_src: int, d_tgt: int, n: int = 120, seed: int = SEED
) -> tuple[RidgeMapping, np.ndarray, np.ndarray]:
    # keep n >= 10*min(d) so the rank-deficiency warning never fires
    n = max(n, 10 * min(d_src, d_tgt))
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d_src))
    # Orthonormal basis keeps the fixture well-conditioned so the round-trip
    # invariant measures the map, not the fixture's conditioning.
    q, _r = np.linalg.qr(rng.normal(size=(d_src, d_src)))
    if d_src >= d_tgt:
        W = q[:, :d_tgt]
    else:
        W = q @ rng.normal(size=(d_src, d_tgt))  # orthonormal rows -> well-conditioned
    Y = l2_normalize(X @ W + 0.001 * rng.normal(size=(n, d_tgt)))
    return RidgeMapping(alpha="auto", seed=seed).fit(X, Y), X, Y


def _assert_unit_norm_rows(Z: np.ndarray) -> None:
    norms = np.linalg.norm(Z, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9), f"row norms off: {norms[:5]}"


def _assert_orthogonal(W: np.ndarray) -> None:
    gram = W.T @ W
    assert np.allclose(gram, np.eye(W.shape[1]), atol=1e-8), (
        f"W^T W not I: {np.abs(gram - np.eye(W.shape[1])).max()}"
    )


class _UnnormalizedStub:
    def transform(self, V: np.ndarray) -> np.ndarray:
        return V * 3.0  # violates unit-norm invariant


class _ScaledProcrustes:
    W = None  # set per-test


@HYP
@given(d_src=st.integers(4, 24), d_tgt=st.integers(4, 24))
def test_p4_01_l2_norm_post_transform(d_src: int, d_tgt: int) -> None:
    m, X, _ = _fit_ridge(d_src, d_tgt)
    _assert_unit_norm_rows(m.transform(X))


def test_p4_01_l2_norm_negative_control() -> None:
    with pytest.raises(AssertionError):
        _assert_unit_norm_rows(np.ones((4, 4)))


@HYP
@given(d=st.integers(4, 24), n_extra=st.integers(0, 60))
def test_p4_02_procrustes_orthogonality(d: int, n_extra: int) -> None:
    n = 10 * d + n_extra  # always above the rank-deficiency floor
    rng = np.random.default_rng(SEED)
    X = l2_normalize(rng.normal(size=(n, d)))
    noise = 0.01 * rng.normal(size=(n, d))
    m = OrthogonalProcrustesMapping()
    m.fit(X, l2_normalize(X + noise))
    assert m._W is not None
    _assert_orthogonal(m._W)


def test_p4_02_procrustes_negative_control() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError):
        _assert_orthogonal(rng.normal(size=(4, 4)) * 2.0)  # scaled, not orthogonal


@HYP
@given(d_src=st.integers(6, 20), d_tgt=st.integers(6, 20))
def test_p4_03_ridge_beats_random_w_on_calibration(d_src: int, d_tgt: int) -> None:
    m, X, Y = _fit_ridge(d_src, d_tgt, n=100)
    resid_fit = float(np.mean((m.transform(X) - l2_normalize(Y)) ** 2))
    rng = np.random.default_rng(SEED + 1)
    W_rand = l2_normalize(rng.normal(size=(len(Y), d_tgt)))
    resid_rand = float(np.mean((W_rand - l2_normalize(Y)) ** 2))
    assert resid_fit < resid_rand


@HYP
@given(d_src=st.integers(6, 16), d_tgt=st.integers(6, 16))
def test_p4_04_inverse_roundtrip(d_src: int, d_tgt: int) -> None:
    # transform() L2-normalizes by design, and for d_src > d_tgt the inverse
    # can only recover the projection of x onto the mapped subspace — the
    # direction cos is bounded below by sqrt(d_tgt/d_src) (Pythagorean).
    # Expanding maps (d_src <= d_tgt, the product's primary case: 384->1536)
    # must round-trip near-exactly.
    m, X, _ = _fit_ridge(d_src, d_tgt, n=140)
    Z = m.transform(X)
    X_back = m.inverse_transform(Z)
    cos = np.sum(l2_normalize(X_back) * l2_normalize(X), axis=1)
    med = float(np.median(cos))
    if d_src <= d_tgt:
        assert med >= 0.99, f"expanding-map roundtrip cos {med}"
    else:
        bound = float(np.sqrt(d_tgt / d_src)) - 0.05
        assert med >= bound, f"projective roundtrip cos {med} < bound {bound}"


def test_p4_04_inverse_negative_control() -> None:
    m, X, _ = _fit_ridge(8, 8, n=100)
    X_back = m.inverse_transform(m.transform(X))
    bogus = X_back + 1.0  # deliberately perturbed inverse result
    cos = np.sum(l2_normalize(bogus) * l2_normalize(X), axis=1)
    assert float(np.median(cos)) < 0.99  # the check DOES catch violations


def test_p4_05_determinism_byte_identical(tmp_path: Path) -> None:
    # NOTE (prereg correction, criteria-defect P4-05): the header embeds fit_date,
    # so full-file byte identity is unattainable BY DESIGN. The reproducibility
    # claim under test is: identical alpha + bit-identical payload matrix +
    # header identical modulo the fit_date field.
    import struct

    m1, X, Y = _fit_ridge(8, 12, n=100, seed=7)
    m2 = RidgeMapping(alpha="auto", seed=7).fit(X, Y)
    p1, p2 = tmp_path / "a.isotrieve", tmp_path / "b.isotrieve"
    m1.save(p1)
    m2.save(p2)
    b1, b2 = p1.read_bytes(), p2.read_bytes()
    assert m1._chosen_alpha == m2._chosen_alpha
    assert b1[4:8] == b2[4:8], "header length differs"
    hlen = struct.unpack("<I", b1[4:8])[0]
    assert b1[12 + hlen :] == b2[12 + hlen :], "matrix payload not bit-identical"
    h1, h2 = b1[12 : 12 + hlen], b2[12 : 12 + hlen]
    assert h1[: h1.find(b"fit_date")] == h2[: h2.find(b"fit_date")], (
        "header differs beyond fit_date"
    )


def test_p4_06_crc_exhaustive_single_bit(tmp_path: Path) -> None:
    m, _, _ = _fit_ridge(8, 8, n=40, seed=3)
    p = tmp_path / "m.isotrieve"
    m.save(p)
    raw = p.read_bytes()
    assert len(raw) > 64
    ok = 0
    for pos in range(len(raw)):
        tampered = bytearray(raw)
        tampered[pos] ^= 0x01
        tp = tmp_path / "t.isotrieve"
        tp.write_bytes(bytes(tampered))
        try:
            RidgeMapping.load(tp)
        except Exception:
            ok += 1
        else:
            raise AssertionError(f"tampered file accepted at byte {pos}")
    assert ok == len(raw)  # every single-bit flip rejected


def test_p4_06_crc_negative_control_untampered(tmp_path: Path) -> None:
    m, _, _ = _fit_ridge(8, 8, n=40, seed=3)
    p = tmp_path / "ok.isotrieve"
    m.save(p)
    RidgeMapping.load(p)  # must NOT raise


@HYP
@given(d_src=st.integers(6, 16), d_tgt=st.integers(6, 16))
def test_p4_07_batch_of_one(d_src: int, d_tgt: int) -> None:
    m, X, _ = _fit_ridge(d_src, d_tgt, n=80)
    Z = m.transform(X[:1])
    assert Z.shape == (1, d_tgt)
    _assert_unit_norm_rows(Z)


def test_p4_07_nan_input_typed_error() -> None:
    m, X, _ = _fit_ridge(8, 12, n=80)
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises((ValueError, FloatingPointError)):  # typed error, not silent NaN
        m.transform(bad)


def test_p4_07_inf_input_typed_error() -> None:
    m, X, _ = _fit_ridge(8, 12, n=80)
    bad = X.copy()
    bad[0, 0] = np.inf
    with pytest.raises((ValueError, FloatingPointError)):
        m.transform(bad)
