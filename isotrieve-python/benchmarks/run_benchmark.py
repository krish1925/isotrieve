#!/usr/bin/env python
"""Isotrieve benchmark on BEIR/SciFact.

Loads SciFact via ir_datasets, embeds with sentence-transformers,
fits isotrieve mappings, and evaluates retrieval retention.

Usage:
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --mapping ridge --k 2000 --seeds 0,1,2
    python benchmarks/run_benchmark.py --mapping all --k 4000 --device mps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ir_datasets
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from isotrieve.mapping.base import l2_normalize
from isotrieve.mapping.linear import (
    LowRankAffineMapping,
    OrthogonalProcrustesMapping,
    ProcrustesDiagMapping,
    RidgeMapping,
)

RESULTS_DIR = Path(__file__).parent / "results"


def _load_scifact():
    """Load SciFact corpus, queries, and qrels."""
    ds = ir_datasets.load("beir/scifact")
    docs = list(ds.docs_iter())
    queries = list(ds.queries_iter())

    # Test split has qrels
    test_ds = ir_datasets.load("beir/scifact/test")
    qrels = list(test_ds.qrels_iter())

    return docs, queries, qrels


def _embed_corpus(model_name: str, docs: list, queries: list) -> tuple[np.ndarray, np.ndarray]:
    """Embed all documents and queries."""
    model = SentenceTransformer(model_name)

    doc_texts = [d.text for d in docs]
    doc_vecs = model.encode(doc_texts, show_progress_bar=True, normalize_embeddings=True)

    query_texts = [q.text for q in queries]
    query_vecs = model.encode(query_texts, show_progress_bar=True, normalize_embeddings=True)

    return doc_vecs, query_vecs


def _build_qrel_matrix(
    qrels: list, n_queries: int, n_docs: int, doc_id_to_idx: dict
) -> np.ndarray:
    """Build binary relevance matrix (n_queries, n_docs)."""
    R = np.zeros((n_queries, n_docs), dtype=np.float32)
    for qrel in qrels:
        qi = int(qrel.query_id)
        di = doc_id_to_idx.get(qrel.doc_id)
        if di is not None and qrel.relevance > 0 and 0 <= qi < n_queries:
            R[qi, di] = 1.0
    return R


def ndcg_at_k(scores: np.ndarray, R: np.ndarray, k: int = 10) -> float:
    """Compute nDCG@k averaged over queries with at least one relevant doc."""
    n_queries = R.shape[0]
    total = 0.0
    count = 0
    for i in range(n_queries):
        rels = R[i]
        if rels.sum() == 0:
            continue
        order = np.argsort(-scores[i])[:k]
        dcg = float(np.sum(rels[order] / np.log2(np.arange(2, k + 2))))
        ideal = float(np.sort(rels)[::-1][:k] @ (1.0 / np.log2(np.arange(2, k + 2))))
        if ideal > 0:
            total += dcg / ideal
            count += 1
    return total / count if count > 0 else 0.0


def topk_retention(mapped_queries, target_queries, k=10):
    """Fraction of queries whose true nearest neighbor is in top-k after mapping."""
    m = l2_normalize(np.asarray(mapped_queries, dtype=np.float64))
    t = l2_normalize(np.asarray(target_queries, dtype=np.float64))
    n = m.shape[0]
    k = min(k, n)
    sims = m @ t.T
    top = np.argpartition(sims, -k, axis=1)[:, -k:]
    correct = sum(1 for i in range(n) if i in top[i])
    return float(correct) / float(n)


def run_single(
    mapping_name: str,
    X_docs: np.ndarray,
    Y_docs: np.ndarray,
    X_queries: np.ndarray,
    Y_queries: np.ndarray,
    R: np.ndarray,
    *,
    k: int,
    seed: int,
    device: str | None = None,
) -> dict:
    """Run a single benchmark: fit mapping, evaluate nDCG@10 and top-k retention."""
    # Subsample calibration set
    rng = np.random.default_rng(seed)
    n_cal = min(k, len(X_docs))
    idx = rng.choice(len(X_docs), size=n_cal, replace=False)
    X_cal = X_docs[idx]
    Y_cal = Y_docs[idx]

    # Fit mapping
    t0 = time.perf_counter()
    if mapping_name == "ridge":
        m = RidgeMapping(alpha="auto", seed=seed)
    elif mapping_name == "lowrank":
        m = LowRankAffineMapping(alpha="auto", rank=512, seed=seed)
    elif mapping_name == "orthogonal_procrustes":
        m = OrthogonalProcrustesMapping(seed=seed)
    elif mapping_name == "procrustes_diag":
        m = ProcrustesDiagMapping(seed=seed)
    elif mapping_name == "mlp":
        from isotrieve.mapping.mlp import ResidualMLPMapping

        m = ResidualMLPMapping(n_epochs=200, seed=seed, device=device)
    else:
        raise ValueError(f"Unknown mapping: {mapping_name}")

    m.fit(X_cal, Y_cal)
    fit_time = time.perf_counter() - t0

    # Transform queries
    t1 = time.perf_counter()
    mapped_queries = m.transform(X_queries)
    transform_time = time.perf_counter() - t1

    # Evaluate
    cos_sims = np.sum(l2_normalize(mapped_queries) * l2_normalize(Y_queries), axis=1)
    ndcg = ndcg_at_k(mapped_queries @ Y_docs.T, R, k=10)
    ret1 = topk_retention(mapped_queries, Y_queries, k=1)
    ret10 = topk_retention(mapped_queries, Y_queries, k=min(10, len(Y_queries)))

    # Validation report
    vr = m.validation_report()

    return {
        "mapping": mapping_name,
        "seed": seed,
        "k_calibration": k,
        "n_docs": len(X_docs),
        "n_queries": len(X_queries),
        "fit_time_s": round(fit_time, 3),
        "transform_time_s": round(transform_time, 3),
        "holdout_cosine_mean": round(vr.holdout_cosine_mean, 4),
        "holdout_cosine_median": round(vr.holdout_cosine_median, 4),
        "query_cosine_mean": round(float(np.mean(cos_sims)), 4),
        "query_cosine_median": round(float(np.median(cos_sims)), 4),
        "ndcg_at_10": round(ndcg, 4),
        "top1_retention": round(ret1, 4),
        "top10_retention": round(ret10, 4),
        "alpha": vr.alpha,
        "n_train": vr.n_train,
        "n_holdout": vr.n_holdout,
    }


def main():
    parser = argparse.ArgumentParser(description="Isotrieve benchmark on BEIR/SciFact")
    parser.add_argument(
        "--mapping",
        default="ridge",
        choices=["ridge", "lowrank", "orthogonal_procrustes", "procrustes_diag", "mlp", "all"],
        help="Mapping type to benchmark",
    )
    parser.add_argument("--k", type=int, default=4000, help="Calibration set size")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    parser.add_argument(
        "--source-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Source embedding model",
    )
    parser.add_argument(
        "--target-model",
        default="BAAI/bge-large-en-v1.5",
        help="Target embedding model",
    )
    parser.add_argument("--device", default=None, help="PyTorch device (cpu, mps, cuda)")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    mappings = (
        ["ridge", "lowrank", "orthogonal_procrustes", "procrustes_diag", "mlp"]
        if args.mapping == "all"
        else [args.mapping]
    )

    # Source model short name for filenames
    src_short = args.source_model.split("/")[-1].replace("-", "_")
    tgt_short = args.target_model.split("/")[-1].replace("-", "_")

    print(f"Loading SciFact...")
    docs, queries, qrels = _load_scifact()
    print(f"  {len(docs)} docs, {len(queries)} queries, {len(qrels)} qrels")

    print(f"Embedding with source model: {args.source_model}")
    X_docs, X_queries = _embed_corpus(args.source_model, docs, queries)

    print(f"Embedding with target model: {args.target_model}")
    Y_docs, Y_queries = _embed_corpus(args.target_model, docs, queries)

    # Build qrel matrix
    doc_id_to_idx = {d.doc_id: i for i, d in enumerate(docs)}
    R = _build_qrel_matrix(qrels, len(queries), len(docs), doc_id_to_idx)
    n_relevant = int(R.sum())
    print(f"  {n_relevant} relevant pairs across {(R > 0).sum()} query-doc pairs")

    # Subsample for evaluation (all queries, but cap docs for matrix size).
    # Truncate both X_docs/Y_docs and R to the SAME doc subset so ndcg scores
    # and the relevance matrix stay index-aligned.
    n_eval_docs = min(5000, len(Y_docs))
    X_docs = X_docs[:n_eval_docs]
    Y_docs = Y_docs[:n_eval_docs]
    R_eval = R[:, :n_eval_docs]

    # Run benchmarks
    all_results = []
    for mapping_name in mappings:
        for seed in seeds:
            print(f"\nRunning {mapping_name} (seed={seed}, k={args.k})...")
            result = run_single(
                mapping_name,
                X_docs,
                Y_docs,
                X_queries,
                Y_queries,
                R_eval,
                k=args.k,
                seed=seed,
                device=args.device,
            )
            all_results.append(result)

            # Save individual result
            fname = f"beir_scifact_{src_short}_to_{tgt_short}__{mapping_name}__k{args.k}__seed{seed}__{time.strftime('%Y%m%d_%H%M%S')}.json"
            out_path = output_dir / fname
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {out_path.name}")
            print(f"  nDCG@10={result['ndcg_at_10']:.4f}, top10_ret={result['top10_retention']:.4f}, fit={result['fit_time_s']:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mapping':<25} {'nDCG@10':>8} {'Top10 Ret':>10} {'Fit (s)':>8}")
    print("-" * 70)
    for mapping_name in mappings:
        results = [r for r in all_results if r["mapping"] == mapping_name]
        ndcg = np.mean([r["ndcg_at_10"] for r in results])
        ndcg_std = np.std([r["ndcg_at_10"] for r in results])
        ret10 = np.mean([r["top10_retention"] for r in results])
        ret10_std = np.std([r["top10_retention"] for r in results])
        fit_t = np.mean([r["fit_time_s"] for r in results])
        print(f"{mapping_name:<25} {ndcg:.3f}±{ndcg_std:.3f} {ret10:.3f}±{ret10_std:.3f} {fit_t:>7.1f}")


if __name__ == "__main__":
    main()
