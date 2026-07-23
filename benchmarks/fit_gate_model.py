#!/usr/bin/env python3
"""Fit QualityGate v2: predict retention from holdout proxies.

Reads all benchmark results, extracts proxies + true retention,
fits isotonic regression, reports LOPO cross-validation stats.

Usage:
    python fit_gate_model.py [--results-dir DIR] [--output PATH]
"""
import argparse
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_results(results_dir: Path) -> list[dict]:
    """Load all result JSONs."""
    results = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            results.append(d)
        except Exception:
            pass
    return results


def extract_features(d: dict) -> dict | None:
    """Extract proxy features + target from a result dict."""
    val = d.get("validation", {})
    retention = d.get("retention", {})
    config = d.get("config", {})

    # Target: nDCG@10 retention
    target = retention.get("nDCG@10")
    if target is None:
        return None

    # Proxies (computable at calibration time)
    features = {
        "holdout_cosine_mean": val.get("holdout_cosine_mean"),
        "holdout_cosine_p5": val.get("holdout_cosine_p5"),
        "top1_retention": val.get("top1_retention"),
        "top10_retention": val.get("top10_retention"),
        "k_cal": d.get("k_cal"),
        "d_src": d.get("d_src"),
        "d_tgt": d.get("d_tgt"),
        "n_docs": d.get("n_docs"),
        "n_queries": d.get("n_queries"),
        "adapter": d.get("adapter", "unknown"),
    }

    # Check all required features are present
    if any(v is None for v in features.values()):
        return None

    return {"features": features, "target": target}


def build_pair_key(d: dict) -> str:
    """Build a key identifying the model pair + dataset."""
    src = d.get("source_model", "")
    tgt = d.get("target_model", "")
    ds = d.get("dataset", "")
    if not src and not tgt:
        # Fallback: use adapter + dims
        adapter = d.get("validation", {}).get("adapter", "unknown")
        d_src = d.get("d_src", 0)
        d_tgt = d.get("d_tgt", 0)
        return f"{adapter}::{d_src}→{d_tgt}"
    return f"{src}→{tgt}::{ds}"


def fit_isotonic(X: np.ndarray, y: np.ndarray) -> dict:
    """Fit isotonic regression on the best single proxy.

    Returns model params for serialization.
    """
    from sklearn.isotonic import IsotonicRegression

    # Use holdout_top1 as primary proxy (most direct predictor)
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(X, y)

    return {
        "type": "isotonic",
        "feature_idx": 0,  # holdout_top1 is column 0
        "X_thresholds": ir.X_thresholds_.tolist(),
        "y_thresholds": ir.y_thresholds_.tolist(),
        "n_train": len(X),
    }


def predict_isotonic(model: dict, x: np.ndarray) -> float:
    """Predict using fitted isotonic regression."""
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.X_thresholds_ = np.array(model["X_thresholds"])
    ir.y_thresholds_ = np.array(model["y_thresholds"])
    ir.n_iter_ = len(model["X_thresholds"])
    return float(ir.predict(np.array([[x]]))[0])


def lopo_cross_validation(
    data: list[dict],
    feature_name: str = "top1_retention",
) -> dict:
    """Leave-One-Pair-Out cross-validation.

    For each model pair, train on N-1 pairs, predict the held-out pair.
    """
    # Group by pair
    by_pair = defaultdict(list)
    for d in data:
        pair_key = f"{d['features']['adapter']}::{build_pair_key(d)}"
        by_pair[pair_key].append(d)

    pair_keys = list(by_pair.keys())
    n_pairs = len(pair_keys)
    if n_pairs < 2:
        return {"error": "need at least 2 pairs for LOPO"}

    from sklearn.isotonic import IsotonicRegression

    all_preds = []
    all_trues = []
    all_pair_keys = []

    for held_out in pair_keys:
        # Train on all pairs except held_out
        train_data = [d for d in data if f"{d['features']['adapter']}::{build_pair_key(d)}" != held_out]
        test_data = by_pair[held_out]

        X_train = np.array([
            d["features"][feature_name] for d in train_data
        ])
        y_train = np.array([
            d["target"] for d in train_data
        ])

        X_test = np.array([
            d["features"][feature_name] for d in test_data
        ])
        y_test = np.array([
            d["target"] for d in test_data
        ])

        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(X_train, y_train)
        preds = ir.predict(X_test)

        all_preds.extend(preds.tolist())
        all_trues.extend(y_test.tolist())
        all_pair_keys.extend([held_out] * len(y_test))

    preds = np.array(all_preds)
    trues = np.array(all_trues)

    residuals = preds - trues
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # Empirical 80% prediction interval
    sorted_res = np.sort(np.abs(residuals))
    idx_80 = int(0.8 * len(sorted_res))
    interval_half_width = float(sorted_res[min(idx_80, len(sorted_res) - 1)])

    return {
        "n_pairs": n_pairs,
        "n_runs": len(preds),
        "mae": mae,
        "rmse": rmse,
        "interval_half_width_80": interval_half_width,
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "residuals_by_pair": {
            pair: float(np.mean(np.abs(np.array([
                preds[i] for i in range(len(preds)) if all_pair_keys[i] == pair
            ]) - np.array([
                trues[i] for i in range(len(trues)) if all_pair_keys[i] == pair
            ])))) for pair in pair_keys
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "src" / "isotrieve" / "quality" / "gate_model_v1.json",
    )
    args = parser.parse_args()

    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} results")

    # Extract features + targets
    data = []
    for r in results:
        d = extract_features(r)
        if d is not None:
            data.append(d)

    print(f"Extracted {len(data)} valid rows")

    # Group by pair (adapter + model pair)
    by_pair = defaultdict(list)
    for d in data:
        pair_key = f"{d['features']['adapter']}::{build_pair_key(d)}"
        by_pair[pair_key].append(d)

    print(f"Pairs: {len(by_pair)}")
    for pair, items in sorted(by_pair.items()):
        targets = [i["target"] for i in items]
        print(f"  {pair}: {len(items)} runs, retention={np.mean(targets):.4f} ± {np.std(targets):.4f}")

    # LOPO CV using top1_retention as primary proxy
    lopo_stats = lopo_cross_validation(data, feature_name="top1_retention")

    print(f"\nLOPO Cross-Validation:")
    print(f"  N pairs: {lopo_stats['n_pairs']}")
    print(f"  MAE: {lopo_stats['mae']:.4f}")
    print(f"  RMSE: {lopo_stats['rmse']:.4f}")
    print(f"  80% interval half-width: {lopo_stats['interval_half_width_80']:.4f}")
    print(f"  Mean residual: {lopo_stats['mean_residual']:.4f}")

    # Fit final model on all data
    X_all = np.array([d["features"]["top1_retention"] for d in data])
    y_all = np.array([d["target"] for d in data])

    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(X_all, y_all)

    model = {
        "type": "isotonic",
        "feature": "top1_retention",
        "X_thresholds": ir.X_thresholds_.tolist(),
        "y_thresholds": ir.y_thresholds_.tolist(),
        "lipo": lopo_stats,
        "training_pairs": list(by_pair.keys()),
        "n_training_runs": len(data),
    }

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2))
    print(f"\nSaved gate model to {args.output}")

    # Verify monotonicity
    X_test = np.linspace(0, 1, 101)
    preds = ir.predict(X_test)
    diffs = np.diff(preds)
    if np.any(diffs < -1e-6):
        print("WARNING: Model not monotonic!")
    else:
        print("Monotonicity verified: better proxies never lower predicted retention")


if __name__ == "__main__":
    main()
