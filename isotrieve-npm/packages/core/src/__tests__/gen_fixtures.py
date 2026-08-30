"""
Generate fixture files for cross-language numerical correctness testing.

Run this script to produce JSON fixtures that the TypeScript test suite
loads and compares against. Requires: pip install isotrieve numpy

Usage:
    cd isotrieve-npm/packages/core
    python ../../isotrieve-python/tests/gen_fixtures.py
"""

import json
import os
import numpy as np
from isotrieve.mapping.linear import (
    RidgeMapping,
    OrthogonalProcrustesMapping,
    ProcrustesDiagMapping,
    LowRankAffineMapping,
)
from isotrieve.quality.gate import QualityGate
from isotrieve.recalibration import ScoreRecalibrator

OUT_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)


def rng(seed=42):
    return np.random.RandomState(seed)


def make_data(n, d_src, d_tgt, seed=42):
    """Generate paired calibration data: Y = X @ W_true + noise."""
    r = rng(seed)
    X = r.randn(n, d_src).astype(np.float64)
    W_true = r.randn(d_src, d_tgt) * 0.1
    Y = X @ W_true + r.randn(n, d_tgt) * 0.01
    return X, Y


def fixture_svd():
    """SVD correctness: reconstruct, orthogonality, known matrices."""
    cases = []
    shapes = [(3, 3), (5, 3), (4, 6), (6, 4), (2, 2), (1, 1)]
    for i, (m, n) in enumerate(shapes):
        X = rng(i).randn(m, n).astype(np.float64)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        # Reconstruct
        recon = U @ np.diag(S) @ Vt
        cases.append({
            "m": m, "n": n,
            "X": X.tolist(),
            "U": U.tolist(),
            "S": S.tolist(),
            "Vt": Vt.tolist(),
            "recon": recon.tolist(),
            "UtU": (U.T @ U).tolist(),
            "VtV": (Vt @ Vt.T).tolist(),
            "frob_norm_X": float(np.linalg.norm(X, "fro")),
            "sum_sq_S": float(np.sum(S ** 2)),
        })
    return cases


def fixture_ridge():
    """Ridge regression: fixed alpha, compare TS closed-form vs sklearn."""
    cases = []
    for i, (n, d_src, d_tgt) in enumerate([
        (50, 4, 4), (100, 8, 3), (30, 5, 7), (20, 3, 3),
    ]):
        X, Y = make_data(n, d_src, d_tgt, seed=i)
        for alpha in [0.01, 1.0, 10.0]:
            # sklearn Ridge
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(X, Y)
            W_sk = model.coef_.T  # (d_src, d_tgt)

            # Also fit with bias
            X_aug = np.hstack([X, np.ones((n, 1))])
            model_b = Ridge(alpha=alpha, fit_intercept=False)
            model_b.fit(X_aug, Y)
            W_sk_b = model_b.coef_.T  # (d_src+1, d_tgt)

            cases.append({
                "n": n, "d_src": d_src, "d_tgt": d_tgt, "alpha": alpha,
                "X": X.tolist(),
                "Y": Y.tolist(),
                "W_no_bias": W_sk.tolist(),
                "W_with_bias": W_sk_b.tolist(),
                "transformed_no_bias": (X @ W_sk).tolist(),
                "transformed_with_bias": (X_aug @ W_sk_b).tolist(),
            })
    return cases


def fixture_ridge_gcv():
    """Ridge GCV: verify alpha selection produces reasonable results."""
    cases = []
    for i, (n, d_src, d_tgt) in enumerate([
        (100, 4, 4), (200, 8, 3),
    ]):
        X, Y = make_data(n, d_src, d_tgt, seed=i)
        from sklearn.linear_model import RidgeCV
        alpha_grid = np.logspace(-3, 3, 25)
        model = RidgeCV(alphas=alpha_grid, fit_intercept=False)
        model.fit(X, Y)
        W = model.coef_.T
        cases.append({
            "n": n, "d_src": d_src, "d_tgt": d_tgt,
            "best_alpha": float(model.alpha_),
            "X": X.tolist(),
            "Y": Y.tolist(),
            "W": W.tolist(),
        })
    return cases


def fixture_procrustes():
    """Procrustes: orthogonal rotation, round-trip."""
    cases = []
    for i, d in enumerate([4, 8, 3]):
        n = 50 + i * 10
        X, Y = make_data(n, d, d, seed=i)
        M = X.T @ Y
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt
        XR = X @ R
        cases.append({
            "n": n, "d": d,
            "X": X.tolist(),
            "Y": Y.tolist(),
            "R": R.tolist(),
            "RtR": (R.T @ R).tolist(),
            "transformed": XR.tolist(),
            "inverse_R": R.T.tolist(),
        })
    return cases


def fixture_procrustes_diag():
    """ProcrustesDiag: rotation + diagonal scaling."""
    cases = []
    for i, d in enumerate([4, 6]):
        n = 60 + i * 10
        X, Y = make_data(n, d, d, seed=i)
        M = X.T @ Y
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt
        XR = X @ R
        denom = np.sum(XR * XR, axis=0)
        numer = np.sum(XR * Y, axis=0)
        s = numer / np.maximum(denom, 1e-8)
        s_inv = np.where(np.abs(s) > 1e-8, 1.0 / s, 0.0)
        W = R * s[np.newaxis, :]
        W_inv = R.T * s_inv[np.newaxis, :]
        cases.append({
            "n": n, "d": d,
            "X": X.tolist(),
            "Y": Y.tolist(),
            "scales": s.tolist(),
            "scales_inv": s_inv.tolist(),
            "W": W.tolist(),
            "W_inv": W_inv.tolist(),
            "transformed": (X @ W).tolist(),
            "roundtrip": (X @ W @ W_inv).tolist(),
        })
    return cases


def fixture_lowrank():
    """LowRankAffineMapping: ridge + TSVD truncation."""
    cases = []
    for i, (n, d_src, d_tgt, rank) in enumerate([
        (80, 6, 6, 3), (100, 8, 4, 2),
    ]):
        X, Y = make_data(n, d_src, d_tgt, seed=i)
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0, fit_intercept=False)
        model.fit(X, Y)
        W_full = model.coef_.T

        # TSVD truncation
        U, S, Vt = np.linalg.svd(W_full, full_matrices=False)
        r = min(rank, len(S))
        W_lr = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]

        cases.append({
            "n": n, "d_src": d_src, "d_tgt": d_tgt, "rank": rank,
            "X": X.tolist(),
            "Y": Y.tolist(),
            "W_full": W_full.tolist(),
            "W_lr": W_lr.tolist(),
            "transformed_full": (X @ W_full).tolist(),
            "transformed_lr": (X @ W_lr).tolist(),
        })
    return cases


def fixture_gate_model():
    """Gate model: load JSON, verify interpolation and verdicts."""
    gate = QualityGate()

    # Test with various top1 retention values
    test_values = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    predictions = []
    for v in test_values:
        pred, interval = gate._predict_retention(v)
        predictions.append({
            "top1": v,
            "predicted": float(pred),
            "lower": float(interval[0]),
            "upper": float(interval[1]),
        })

    # Test verdicts on synthetic data
    verdicts = []
    for v in test_values:
        pred, interval = gate._predict_retention(v)
        lower, upper = interval
        if lower >= 0.75:
            verdict = "PASS"
        elif upper < 0.55:
            verdict = "FAIL"
        else:
            verdict = "WARN"
        verdicts.append({
            "top1": v,
            "predicted": float(pred),
            "lower": float(lower),
            "upper": float(upper),
            "verdict": verdict,
        })

    return {
        "predictions": predictions,
        "verdicts": verdicts,
        "gate_model_used": gate.gate_model is not None,
    }


def fixture_recalibration():
    """ScoreRecalibrator: PAVA isotonic regression."""
    cases = []
    for n in [50, 100]:
        r = rng(n)
        mapped = r.rand(n).astype(np.float64) * 0.5 + 0.3
        ceiling = mapped + r.rand(n) * 0.1
        recal = ScoreRecalibrator()
        recal.fit(mapped, ceiling)
        transformed = recal.transform(mapped)
        report_dict = recal.report.to_dict() if hasattr(recal.report, 'to_dict') else recal.report
        cases.append({
            "n": n,
            "mapped": mapped.tolist(),
            "ceiling": ceiling.tolist(),
            "thresholds": recal._thresholds.tolist(),
            "values": recal._values.tolist(),
            "transformed": transformed.tolist(),
            "report": {
                "nPairs": report_dict["n_pairs"],
                "meanMappedScore": float(report_dict["mean_mapped_score"]),
                "meanCeilingScore": float(report_dict["mean_ceiling_score"]),
                "meanShift": float(report_dict["mean_shift"]),
            },
        })
    return cases


def fixture_pava_adversarial():
    """PAVA adversarial inputs: already-sorted, reverse-sorted, all-equal, etc."""
    cases = []

    # Already sorted (monotone increasing)
    mapped = np.linspace(0.1, 0.9, 50)
    ceiling = np.linspace(0.2, 1.0, 50)
    recal = ScoreRecalibrator()
    recal.fit(mapped, ceiling)
    cases.append({
        "name": "already_sorted",
        "mapped": mapped.tolist(),
        "ceiling": ceiling.tolist(),
        "values": recal._values.tolist(),
    })

    # Reverse sorted
    mapped = np.linspace(0.9, 0.1, 50)
    ceiling = np.linspace(1.0, 0.2, 50)
    recal2 = ScoreRecalibrator()
    recal2.fit(mapped, ceiling)
    cases.append({
        "name": "reverse_sorted",
        "mapped": mapped.tolist(),
        "ceiling": ceiling.tolist(),
        "values": recal2._values.tolist(),
    })

    # All equal
    mapped = np.full(50, 0.5)
    ceiling = np.full(50, 0.7)
    recal3 = ScoreRecalibrator()
    recal3.fit(mapped, ceiling)
    cases.append({
        "name": "all_equal",
        "mapped": mapped.tolist(),
        "ceiling": ceiling.tolist(),
        "values": recal3._values.tolist(),
    })

    return cases


if __name__ == "__main__":
    print("Generating SVD fixtures...")
    with open(os.path.join(OUT_DIR, "svd.json"), "w") as f:
        json.dump(fixture_svd(), f)

    print("Generating ridge fixtures...")
    with open(os.path.join(OUT_DIR, "ridge.json"), "w") as f:
        json.dump(fixture_ridge(), f)

    print("Generating ridge GCV fixtures...")
    with open(os.path.join(OUT_DIR, "ridge_gcv.json"), "w") as f:
        json.dump(fixture_ridge_gcv(), f)

    print("Generating Procrustes fixtures...")
    with open(os.path.join(OUT_DIR, "procrustes.json"), "w") as f:
        json.dump(fixture_procrustes(), f)

    print("Generating ProcrustesDiag fixtures...")
    with open(os.path.join(OUT_DIR, "procrustes_diag.json"), "w") as f:
        json.dump(fixture_procrustes_diag(), f)

    print("Generating LowRank fixtures...")
    with open(os.path.join(OUT_DIR, "lowrank.json"), "w") as f:
        json.dump(fixture_lowrank(), f)

    print("Generating gate model fixtures...")
    with open(os.path.join(OUT_DIR, "gate_model.json"), "w") as f:
        json.dump(fixture_gate_model(), f)

    print("Generating recalibration fixtures...")
    with open(os.path.join(OUT_DIR, "recalibration.json"), "w") as f:
        json.dump(fixture_recalibration(), f)

    print("Generating PAVA adversarial fixtures...")
    with open(os.path.join(OUT_DIR, "pava_adversarial.json"), "w") as f:
        json.dump(fixture_pava_adversarial(), f)

    print(f"All fixtures written to {OUT_DIR}/")
