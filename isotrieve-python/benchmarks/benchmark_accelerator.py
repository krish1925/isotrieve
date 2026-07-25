#!/usr/bin/env python
"""Benchmark PyTorch accelerator: CPU vs MPS (Apple Silicon) vs CUDA.

Compares ResidualMLPMapping training speed and quality across devices.
Requires torch; install with ``pip install isotrieve[mlp]``.

Usage:
    python benchmarks/benchmark_accelerator.py
    python benchmarks/benchmark_accelerator.py --k 4000 --n-epochs 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"


def _paired_gaussian(
    k: int, d_src: int, d_tgt: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(k, d_src))
    W_true = rng.normal(size=(d_src, d_tgt))
    Y = X @ W_true + 0.01 * rng.normal(size=(k, d_tgt))
    return X, Y


def _available_devices() -> list[str]:
    """List available torch devices."""
    import torch

    devices = ["cpu"]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def benchmark_device(
    device: str,
    X: np.ndarray,
    Y: np.ndarray,
    n_epochs: int,
    seed: int,
    n_warmup: int = 2,
    n_trials: int = 3,
) -> dict:
    """Benchmark training on a specific device."""
    from isotrieve.mapping.mlp import ResidualMLPMapping
    from isotrieve.mapping.base import l2_normalize
    from isotrieve.quality.metrics import topk_retention

    fit_times = []
    transform_times = []
    qualities = []

    for trial in range(n_warmup + n_trials):
        m = ResidualMLPMapping(n_epochs=n_epochs, seed=seed, device=device)

        t0 = time.perf_counter()
        m.fit(X, Y)
        fit_t = time.perf_counter() - t0

        t1 = time.perf_counter()
        Z = m.transform(X)
        transform_t = time.perf_counter() - t1

        ret = topk_retention(Z, Y, k=1)
        cos = float(np.mean(np.sum(l2_normalize(Z) * l2_normalize(Y), axis=1)))

        if trial >= n_warmup:
            fit_times.append(fit_t)
            transform_times.append(transform_t)
            qualities.append({"top1_retention": ret, "cosine_mean": cos})

    return {
        "device": device,
        "n_epochs": n_epochs,
        "n_samples": len(X),
        "d_src": X.shape[1],
        "d_tgt": Y.shape[1],
        "fit_time_mean_s": round(float(np.mean(fit_times)), 3),
        "fit_time_std_s": round(float(np.std(fit_times)), 3),
        "transform_time_mean_s": round(float(np.mean(transform_times)), 4),
        "quality_mean": {
            k: round(float(np.mean([q[k] for q in qualities])), 4)
            for k in qualities[0]
        },
        "quality_std": {
            k: round(float(np.std([q[k] for q in qualities])), 4)
            for k in qualities[0]
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch device acceleration")
    parser.add_argument("--k", type=int, default=2000, help="Number of calibration pairs")
    parser.add_argument("--d", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--n-epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--n-trials", type=int, default=3, help="Timing trials per device")
    args = parser.parse_args()

    devices = _available_devices()
    print(f"Available devices: {devices}")

    seeds = [int(s) for s in args.seeds.split(",")]
    X, Y = _paired_gaussian(args.k, args.d, args.d, seed=42)
    print(f"Synthetic data: {args.k} samples, {args.d}d")

    all_results = []
    for device in devices:
        for seed in seeds:
            print(f"\nBenchmarking {device} (seed={seed}, epochs={args.n_epochs})...")
            result = benchmark_device(
                device, X, Y, args.n_epochs, seed, n_trials=args.n_trials
            )
            all_results.append(result)

            fname = f"accelerator__{device}__k{args.k}__d{args.d}__epochs{args.n_epochs}__seed{seed}.json"
            out_path = RESULTS_DIR / fname
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  fit: {result['fit_time_mean_s']:.3f}±{result['fit_time_std_s']:.3f}s")
            print(f"  transform: {result['transform_time_mean_s']:.4f}s")
            print(f"  top1_ret: {result['quality_mean']['top1_retention']:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("DEVICE COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Device':<8} {'Fit (s)':>12} {'Transform (s)':>14} {'Top1 Ret':>10} {'Speedup':>8}")
    print("-" * 60)

    cpu_time = None
    for device in devices:
        results = [r for r in all_results if r["device"] == device]
        fit_mean = np.mean([r["fit_time_mean_s"] for r in results])
        fit_std = np.mean([r["fit_time_std_s"] for r in results])
        transform_mean = np.mean([r["transform_time_mean_s"] for r in results])
        ret_mean = np.mean([r["quality_mean"]["top1_retention"] for r in results])
        ret_std = np.mean([r["quality_std"]["top1_retention"] for r in results])

        if device == "cpu":
            cpu_time = fit_mean

        speedup = f"{cpu_time / fit_mean:.1f}x" if cpu_time and fit_mean > 0 else "—"

        print(
            f"{device:<8} {fit_mean:>5.3f}±{fit_std:.3f}  "
            f"{transform_mean:>10.4f}     "
            f"{ret_mean:.3f}±{ret_std:.3f} {speedup:>7}"
        )

    # Save summary
    summary = {
        "description": "PyTorch device acceleration benchmark",
        "devices": devices,
        "config": {
            "k": args.k,
            "d": args.d,
            "n_epochs": args.n_epochs,
            "seeds": seeds,
        },
        "results": all_results,
    }
    summary_path = RESULTS_DIR / "accelerator_comparison.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
