#!/usr/bin/env python3
"""Diagnose retention failures: mapping error vs margin/precision error.

For each query where true top-1 drops out of top-10 after mapping, compute:
  - cosine(mapped_query, true_target_vec_in_target_space)
  - cosine(ceiling_query, true_target_vec_in_target_space)
  - margin: cosine_to_true - cosine_to_nearest_non_relevant

Scatter plot: true-neighbor ceiling rank vs mapped cosine-to-true.
High cosine (>0.9) means mapping is fine — the problem is retrieval margin.
Low cosine means mapping error — better math could help.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "isotrieve-python" / "src"))

from isotrieve.mapping.base import l2_normalize
from isotrieve.mapping.linear import RidgeMapping

CACHE_DIR = ROOT / "benchmarks" / ".embed_cache"
OUT_DIR = ROOT / "benchmarks"


def load_cached(model_pair: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    tag = hashlib.sha256(model_pair.encode()).hexdigest()[:10]
    d = CACHE_DIR / tag
    meta = json.loads((d / "meta.json").read_text())
    return (
        np.load(d / "doc_src.npy"),
        np.load(d / "doc_tgt.npy"),
        np.load(d / "qry_tgt.npy"),
        meta,
    )


def load_scifact_qrels():
    import ir_datasets
    ds = ir_datasets.load("beir/scifact/test")
    qrels = {}
    for qrel in ds.qrels_iter():
        qid = qrel.query_id
        did = qrel.doc_id
        if qid not in qrels:
            qrels[qid] = set()
        qrels[qid].add(did)
    return qrels


def load_scifact_docs_queries():
    import ir_datasets
    corpus_ds = ir_datasets.load("beir/scifact")
    eval_ds = ir_datasets.load("beir/scifact/test")
    doc_ids = []
    for doc in corpus_ds.docs_iter():
        doc_ids.append(doc.doc_id)
    query_ids = []
    qrels_qids = set()
    for qrel in eval_ds.qrels_iter():
        qrels_qids.add(qrel.query_id)
    for q in eval_ds.queries_iter():
        if q.query_id in qrels_qids:
            query_ids.append(q.query_id)
    return doc_ids, query_ids


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pair", choices=["bge-e5", "minilm-bge"], default="bge-e5")
    args = p.parse_args()

    if args.pair == "bge-e5":
        source_model = "BAAI/bge-large-en-v1.5"
        target_model = "intfloat/e5-large-v2"
        k_cal = 2000
    else:
        source_model = "sentence-transformers/all-MiniLM-L6-v2"
        target_model = "BAAI/bge-large-en-v1.5"
        k_cal = 4000

    model_pair = f"{source_model}|{target_model}"

    print("Loading cached embeddings...")
    doc_src, doc_tgt, qry_tgt, meta = load_cached(model_pair)
    print(f"  doc_src: {doc_src.shape}, doc_tgt: {doc_tgt.shape}, qry_tgt: {qry_tgt.shape}")

    doc_ids, query_ids = load_scifact_docs_queries()
    qrels = load_scifact_qrels()
    print(f"  docs={len(doc_ids)}, queries={len(query_ids)}, qrels={len(qrels)}")

    # Align: benchmark used all corpus docs but only qrels queries
    n_docs = min(len(doc_ids), doc_src.shape[0])
    n_queries = min(len(query_ids), qry_tgt.shape[0])
    doc_ids = doc_ids[:n_docs]
    query_ids = query_ids[:n_queries]
    doc_src = doc_src[:n_docs]
    doc_tgt = doc_tgt[:n_docs]
    qry_tgt = qry_tgt[:n_queries]

    # Fit mapping (same as benchmark: K=2000, seed=0)
    k_cal = 2000
    rng = np.random.default_rng(0)
    cal_idx = rng.choice(n_docs, size=k_cal, replace=False)
    X = doc_src[cal_idx]
    Y = doc_tgt[cal_idx]

    print("Fitting RidgeMapping...")
    mapping = RidgeMapping(alpha="auto", seed=0)
    mapping.fit(X, Y)
    val = mapping.validation_report()
    print(f"  alpha={val.alpha}, holdout_cosine_mean={val.holdout_cosine_mean:.4f}")

    # Transform all source docs
    doc_mapped = mapping.transform(doc_src)

    # Normalize everything
    doc_tgt_n = l2_normalize(doc_tgt)
    doc_mapped_n = l2_normalize(doc_mapped)
    qry_tgt_n = l2_normalize(qry_tgt)

    # Compute similarity matrices
    # Ceiling: qry_tgt vs doc_tgt (target queries vs target docs)
    ceiling_sims = qry_tgt_n @ doc_tgt_n.T  # (n_queries, n_docs)
    # Isotrieve: qry_tgt vs doc_mapped (target queries vs mapped docs)
    isotrieve_sims = qry_tgt_n @ doc_mapped_n.T  # (n_queries, n_docs)

    # Per-query analysis
    results = []
    for qi, qid in enumerate(query_ids):
        rel = qrels.get(qid, set())
        if not rel:
            continue

        # Find indices of relevant docs
        rel_indices = [di for di, did in enumerate(doc_ids) if did in rel]
        if not rel_indices:
            continue

        # Ceiling: rank of each relevant doc
        ceiling_row = ceiling_sims[qi]
        ceiling_sorted = np.argsort(-ceiling_row)
        ceiling_ranks = {did: int(np.where(ceiling_sorted == di)[0][0]) for di, did in enumerate(doc_ids) if did in rel}

        # Isotrieve: rank of each relevant doc
        isotrieve_row = isotrieve_sims[qi]
        isotrieve_sorted = np.argsort(-isotrieve_row)
        isotrieve_ranks = {did: int(np.where(isotrieve_sorted == di)[0][0]) for di, did in enumerate(doc_ids) if did in rel}

        # True top-1 in ceiling space
        true_top1_idx = rel_indices[0]
        # Actually, find the relevant doc with highest ceiling similarity
        best_rel_idx = max(rel_indices, key=lambda di: ceiling_row[di])
        true_top1_ceiling_rank = ceiling_ranks[doc_ids[best_rel_idx]]
        true_top1_isotrieve_rank = isotrieve_ranks[doc_ids[best_rel_idx]]

        # Cosine to true target
        cos_mapped_to_true = float(isotrieve_row[best_rel_idx])
        cos_ceiling_to_true = float(ceiling_row[best_rel_idx])

        # Margin: cosine to true top-1 minus cosine to nearest non-relevant
        non_rel_mask = np.ones(n_docs, dtype=bool)
        for ri in rel_indices:
            non_rel_mask[ri] = False
        max_non_rel_cos = float(np.max(isotrieve_row[non_rel_mask])) if np.any(non_rel_mask) else 0.0
        margin = cos_mapped_to_true - max_non_rel_cos

        # Is this a failure? (true top-1 not in top-10 after mapping)
        is_failure = true_top1_isotrieve_rank >= 10

        # Also check: ceiling top-1 rank (should be ~1 for relevant docs)
        ceiling_top1_for_query = int(ceiling_sorted[0])
        ceiling_top1_is_relevant = doc_ids[ceiling_top1_for_query] in rel

        results.append({
            "query_id": qid,
            "ceiling_rank": true_top1_ceiling_rank,
            "isotrieve_rank": true_top1_isotrieve_rank,
            "cos_mapped_to_true": cos_mapped_to_true,
            "cos_ceiling_to_true": cos_ceiling_to_true,
            "margin": margin,
            "is_failure": is_failure,
            "ceiling_top1_is_relevant": ceiling_top1_is_relevant,
            "n_relevant": len(rel_indices),
        })

    # Summary stats
    failures = [r for r in results if r["is_failure"]]
    successes = [r for r in results if not r["is_failure"]]
    print(f"\n=== FAILURE ANALYSIS ===")
    print(f"Total queries with relevance: {len(results)}")
    print(f"Failures (top-1 dropped from top-10): {len(failures)} ({100*len(failures)/len(results):.1f}%)")
    print(f"Successes: {len(successes)} ({100*len(successes)/len(results):.1f}%)")

    if failures:
        cos_vals = [f["cos_mapped_to_true"] for f in failures]
        margin_vals = [f["margin"] for f in failures]
        ceiling_ranks = [f["ceiling_rank"] for f in failures]

        print(f"\n--- Failure subgroup analysis ---")
        high_cos = [f for f in failures if f["cos_mapped_to_true"] > 0.9]
        mid_cos = [f for f in failures if 0.7 < f["cos_mapped_to_true"] <= 0.9]
        low_cos = [f for f in failures if f["cos_mapped_to_true"] <= 0.7]

        print(f"High cosine (>0.9): {len(high_cos)} — mapping is fine, retrieval margin problem")
        print(f"Mid cosine (0.7-0.9): {len(mid_cos)} — partial mapping error + margin")
        print(f"Low cosine (<0.7): {len(low_cos)} — mapping error, better math could help")

        print(f"\nFailure stats:")
        print(f"  cos_mapped_to_true:  mean={np.mean(cos_vals):.4f}  median={np.median(cos_vals):.4f}  p5={np.percentile(cos_vals, 5):.4f}")
        print(f"  margin:              mean={np.mean(margin_vals):.4f}  median={np.median(margin_vals):.4f}  p5={np.percentile(margin_vals, 5):.4f}")
        print(f"  ceiling_rank:        mean={np.mean(ceiling_ranks):.1f}  median={np.median(ceiling_ranks):.1f}")

        # Check: in ceiling space, is the true top-1 actually rank 1?
        ceiling_top1_correct = sum(1 for f in results if f["ceiling_top1_is_relevant"])
        print(f"\nCeiling top-1 is relevant: {ceiling_top1_correct}/{len(results)} ({100*ceiling_top1_correct/len(results):.1f}%)")

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Scatter — ceiling rank vs mapped cosine-to-true
    ax = axes[0]
    s_colors = ["#2196F3" if not f["is_failure"] else "#F44336" for f in results]
    s_sizes = [20 if not f["is_failure"] else 60 for f in results]
    ax.scatter(
        [r["ceiling_rank"] for r in results],
        [r["cos_mapped_to_true"] for r in results],
        c=s_colors, s=s_sizes, alpha=0.5, edgecolors="none",
    )
    ax.set_xlabel("True neighbor ceiling rank")
    ax.set_ylabel("Cosine(mapped_query, true_target)")
    ax.set_title("Per-query: ceiling rank vs mapping cosine")
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="cos=0.9")
    ax.axvline(x=10, color="gray", linestyle="--", alpha=0.5, label="rank=10")
    ax.legend(fontsize=8)

    # Plot 2: Histogram of cosine-to-true for failures vs successes
    ax = axes[1]
    if successes:
        ax.hist(
            [r["cos_mapped_to_true"] for r in successes],
            bins=30, alpha=0.6, color="#2196F3", label=f"Success (n={len(successes)})", density=True,
        )
    if failures:
        ax.hist(
            [f["cos_mapped_to_true"] for f in failures],
            bins=30, alpha=0.6, color="#F44336", label=f"Failure (n={len(failures)})", density=True,
        )
    ax.set_xlabel("Cosine(mapped_query, true_target)")
    ax.set_ylabel("Density")
    ax.set_title("Cosine-to-true distribution")
    ax.legend(fontsize=8)

    # Plot 3: Margin distribution for failures
    ax = axes[2]
    if failures:
        ax.hist(
            [f["margin"] for f in failures],
            bins=30, alpha=0.7, color="#FF9800",
        )
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="margin=0 (tied)")
        ax.set_xlabel("Margin: cos_to_true - cos_to_nearest_non_relevant")
        ax.set_ylabel("Count")
        ax.set_title(f"Failure margins (n={len(failures)})")
        ax.legend(fontsize=8)

    plt.suptitle(f"Retention failure diagnosis: {source_model.split('/')[-1]} → {target_model.split('/')[-1]}", fontsize=13)
    plt.tight_layout()
    out_path = OUT_DIR / "failure_diagnosis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    # Also print the top-10 worst failures
    if failures:
        print(f"\n--- Top-10 worst failures (by isotrieve_rank) ---")
        worst = sorted(failures, key=lambda f: -f["isotrieve_rank"])[:10]
        for f in worst:
            print(f"  qid={f['query_id']:>10s}  isotrieve_rank={f['isotrieve_rank']:>3d}  ceiling_rank={f['ceiling_rank']:>2d}  "
                  f"cos_mapped={f['cos_mapped_to_true']:.4f}  margin={f['margin']:.4f}")


if __name__ == "__main__":
    main()
