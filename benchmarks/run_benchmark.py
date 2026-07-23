#!/usr/bin/env python3
"""Isotrieve benchmark harness — supports local + API models, K-sweep, calibration modes.

Protocol (per product contract §5):
  1. Embed corpus with source; queries with target (post-migration reality).
  2. Floor: target queries vs raw source vectors (expect near-zero / impossible if dims differ).
  3. Ceiling: target queries vs true target-embedded corpus.
  4. Isotrieve: map source corpus → retrieve with target queries.
  5. Report nDCG@10 / Recall@10 retention = Isotrieve / ceiling.
  6. K-sweep: vary K to understand quality-vs-cost tradeoff.
  7. Calibration modes: in-domain (from corpus) vs generic (external vocabulary).
  8. 3 seeds; mean ± std (embeddings cached; only fit/eval re-run per seed).

Writes one JSON per run under benchmarks/results/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isotrieve-python" / "src"))

from isotrieve.mapping.base import l2_normalize  # noqa: E402
from isotrieve.mapping.linear import RidgeMapping  # noqa: E402
from isotrieve.providers.factory import create_embedder  # noqa: E402
from beir_loaders import load_beir_dataset  # noqa: E402

# Model-specific prefixes required for correct embeddings.
# e5 models require "query: " / "passage: " prefixes; without them
# retrieval quality degrades severely (ceiling nDCG@10 drops from ~0.74 to 0.36).
# bge models use [CLS] token pooling and don't need text prefixes.
MODEL_PREFIXES: dict[str, dict[str, str]] = {
    "intfloat/e5-large-v2": {"query": "query: ", "passage": "passage: "},
    "intfloat/e5-base-v2": {"query": "query: ", "passage": "passage: "},
    "intfloat/e5-small-v2": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-large": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-base": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-small": {"query": "query: ", "passage": "passage: "},
}


def _get_prefix(model_id: str, text_type: str) -> str:
    """Return the required prefix for a model and text type (query or passage)."""
    entry = MODEL_PREFIXES.get(model_id, {})
    return entry.get(text_type, "")


CACHE_PREFIX_VERSION = "v2"  # bump to invalidate caches after prefix changes


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def load_scifact(max_docs: int | None = None):
    """Load SciFact corpus + test queries + qrels via ir_datasets."""
    import ir_datasets

    # Parent has full corpus; /test has queries + qrels for evaluation.
    corpus_ds = ir_datasets.load("beir/scifact")
    eval_ds = ir_datasets.load("beir/scifact/test")
    docs = []
    for doc in corpus_ds.docs_iter():
        text = f"{doc.title} {doc.text}".strip()
        docs.append({"id": doc.doc_id, "text": text})
        if max_docs and len(docs) >= max_docs:
            break
    queries = [{"id": q.query_id, "text": q.text} for q in eval_ds.queries_iter()]
    qrels: dict[str, set[str]] = {}
    for qrel in eval_ds.qrels_iter():
        if int(qrel.relevance) > 0:
            qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)
    return docs, queries, qrels, "beir/scifact/test@ir_datasets"


def embed_texts(
    model, texts: list[str], batch_size: int = 64, prefix: str = ""
) -> np.ndarray:
    if prefix:
        texts = [prefix + t for t in texts]
    return np.asarray(
        model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float64,
    )


def ndcg_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = 0.0
    for i, did in enumerate(ranked_doc_ids[:k]):
        if did in relevant:
            dcg += 1.0 / np.log2(i + 2)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return float(dcg / ideal) if ideal > 0 else 0.0


def recall_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for d in ranked_doc_ids[:k] if d in relevant)
    return float(hit / len(relevant))


def retrieve(
    query_vecs: np.ndarray,
    doc_vecs: np.ndarray,
    doc_ids: list[str],
    k: int = 100,
) -> list[list[str]]:
    q = l2_normalize(query_vecs)
    d = l2_normalize(doc_vecs)
    sims = q @ d.T
    results = []
    kk = min(k, sims.shape[1])
    for i in range(sims.shape[0]):
        top = np.argpartition(sims[i], -kk)[-kk:]
        order = top[np.argsort(-sims[i, top])]
        results.append([doc_ids[j] for j in order])
    return results


def evaluate(
    rankings: list[list[str]],
    query_ids: list[str],
    qrels: dict[str, set[str]],
) -> dict[str, float]:
    ndcgs, r10s, r100s = [], [], []
    for qid, ranked in zip(query_ids, rankings):
        rel = qrels.get(qid, set())
        if not rel:
            continue
        ndcgs.append(ndcg_at_k(ranked, rel, 10))
        r10s.append(recall_at_k(ranked, rel, 10))
        r100s.append(recall_at_k(ranked, rel, 100))
    if not ndcgs:
        return {
            "nDCG@10": 0.0,
            "Recall@10": 0.0,
            "Recall@100": 0.0,
            "n_eval": 0,
        }
    return {
        "nDCG@10": float(np.mean(ndcgs)),
        "Recall@10": float(np.mean(r10s)),
        "Recall@100": float(np.mean(r100s)),
        "n_eval": len(ndcgs),
    }


def cache_paths(cache_dir: Path, source_model: str, target_model: str) -> dict[str, Path]:
    tag = hashlib.sha256(f"{source_model}|{target_model}".encode()).hexdigest()[:10]
    d = cache_dir / tag
    d.mkdir(parents=True, exist_ok=True)
    return {
        "dir": d,
        "doc_src": d / "doc_src.npy",
        "doc_tgt": d / "doc_tgt.npy",
        "qry_tgt": d / "qry_tgt.npy",
        "meta": d / "meta.json",
    }


def load_or_embed(
    *,
    source_model: str,
    target_model: str,
    doc_texts: list[str],
    query_texts: list[str],
    cache_dir: Path,
    use_api: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load cached embeddings or compute them (local or API).

    Applies model-specific prefixes (e.g., "query: " / "passage: " for e5).
    Cache is invalidated when prefix_version changes (CACHE_PREFIX_VERSION).
    """
    paths = cache_paths(cache_dir, source_model, target_model)
    meta_ok = False
    if paths["meta"].exists() and all(
        paths[k].exists() for k in ("doc_src", "doc_tgt", "qry_tgt")
    ):
        meta = json.loads(paths["meta"].read_text())
        if (
            meta.get("n_docs") == len(doc_texts)
            and meta.get("n_queries") == len(query_texts)
            and meta.get("source_model") == source_model
            and meta.get("target_model") == target_model
            and meta.get("prefix_version") == CACHE_PREFIX_VERSION
        ):
            meta_ok = True
        elif meta.get("prefix_version") != CACHE_PREFIX_VERSION:
            print(f"Cache invalidated: prefix_version changed ({meta.get('prefix_version')} → {CACHE_PREFIX_VERSION})")
    if meta_ok:
        print(f"Loading cached embeddings from {paths['dir']}")
        return (
            np.load(paths["doc_src"]),
            np.load(paths["doc_tgt"]),
            np.load(paths["qry_tgt"]),
            0.0,
        )

    src_prefix = _get_prefix(source_model, "passage")
    tgt_prefix_doc = _get_prefix(target_model, "passage")
    tgt_prefix_qry = _get_prefix(target_model, "query")
    if src_prefix:
        print(f"Using source prefix: {src_prefix!r}")
    if tgt_prefix_doc:
        print(f"Using target passage prefix: {tgt_prefix_doc!r}")
    if tgt_prefix_qry:
        print(f"Using target query prefix: {tgt_prefix_qry!r}")

    t0 = time.perf_counter()
    if use_api:
        doc_src = _embed_api(source_model, doc_texts)
        doc_tgt = _embed_api(target_model, doc_texts)
        qry_tgt = _embed_api(target_model, query_texts)
    else:
        from sentence_transformers import SentenceTransformer

        print("Embedding corpus (will cache for seed re-runs)…")
        src_m = SentenceTransformer(source_model)
        tgt_m = SentenceTransformer(target_model)
        doc_src = embed_texts(src_m, doc_texts, prefix=src_prefix)
        doc_tgt = embed_texts(tgt_m, doc_texts, prefix=tgt_prefix_doc)
        qry_tgt = embed_texts(tgt_m, query_texts, prefix=tgt_prefix_qry)
    embed_s = time.perf_counter() - t0

    np.save(paths["doc_src"], doc_src)
    np.save(paths["doc_tgt"], doc_tgt)
    np.save(paths["qry_tgt"], qry_tgt)
    paths["meta"].write_text(
        json.dumps(
            {
                "source_model": source_model,
                "target_model": target_model,
                "n_docs": len(doc_texts),
                "n_queries": len(query_texts),
                "use_api": use_api,
                "prefix_version": CACHE_PREFIX_VERSION,
                "source_prefix": src_prefix,
                "target_doc_prefix": tgt_prefix_doc,
                "target_query_prefix": tgt_prefix_qry,
            }
        ),
        encoding="utf-8",
    )
    print(f"Cached embeddings → {paths['dir']}")
    return doc_src, doc_tgt, qry_tgt, embed_s


def _embed_api(model_id: str, texts: list[str], batch_size: int = 128) -> np.ndarray:
    """Embed texts using Isotrieve provider adapters (with disk cache)."""
    embedder = create_embedder(model_id, cached=True)
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = embedder.embed(batch)
        all_vecs.append(vecs)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} with {model_id}")
    return np.vstack(all_vecs)


def create_mapping(
    adapter_name: str,
    d_src: int,
    d_tgt: int,
    seed: int,
) -> "Mapping":
    """Create a mapping adapter by name."""
    from isotrieve.mapping.linear import (
        LowRankAffineMapping,
        OrthogonalProcrustesMapping,
        ProcrustesDiagMapping,
        RidgeMapping,
    )

    is_square = d_src == d_tgt

    if adapter_name == "ridge":
        return RidgeMapping(alpha="auto", seed=seed)
    elif adapter_name == "procrustes":
        if not is_square:
            raise ValueError(f"Procrustes requires d_src==d_tgt, got {d_src}!={d_tgt}")
        return OrthogonalProcrustesMapping(seed=seed)
    elif adapter_name == "procrustes_diag":
        if not is_square:
            raise ValueError(f"ProcrustesDiag requires d_src==d_tgt, got {d_src}!={d_tgt}")
        return ProcrustesDiagMapping(seed=seed)
    elif adapter_name == "lowrank":
        return LowRankAffineMapping(alpha="auto", rank=min(256, d_src, d_tgt), seed=seed)
    elif adapter_name == "mlp":
        try:
            from isotrieve.mapping.mlp import ResidualMLPMapping
            return ResidualMLPMapping(seed=seed)
        except ImportError:
            raise ImportError("ResidualMLPMapping requires torch (pip install isotrieve[mlp])")
    else:
        raise ValueError(f"Unknown adapter: {adapter_name!r}")


def adapters_for_dims(d_src: int, d_tgt: int) -> list[str]:
    """Return adapter names valid for these dimensions."""
    is_square = d_src == d_tgt
    adapters = ["ridge", "lowrank"]
    if is_square:
        adapters.extend(["procrustes", "procrustes_diag"])
    try:
        import torch  # noqa: F401
        adapters.append("mlp")
    except ImportError:
        pass
    return adapters


def run_seed(
    *,
    doc_src: np.ndarray,
    doc_tgt: np.ndarray,
    qry_tgt: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, set[str]],
    dataset_id: str,
    source_model: str,
    target_model: str,
    k_cal: int,
    seed: int,
    embed_s: float,
    out_dir: Path,
    adapter: str = "ridge",
) -> dict:
    rng = np.random.default_rng(seed)
    k_cal = min(k_cal, len(doc_ids))
    min_dim = min(doc_src.shape[1], doc_tgt.shape[1])
    if k_cal < 10 * min_dim:
        print(f"  Warning: K={k_cal} < 10×min_dim={10*min_dim}. Mapping may be rank-deficient.")
    cal_idx = rng.choice(len(doc_ids), size=k_cal, replace=False)
    X = doc_src[cal_idx]
    Y = doc_tgt[cal_idx]

    mapping = create_mapping(adapter, doc_src.shape[1], doc_tgt.shape[1], seed)
    t1 = time.perf_counter()
    mapping.fit(X, Y)
    fit_s = time.perf_counter() - t1
    val = mapping.validation_report().to_dict()

    t2 = time.perf_counter()
    doc_mapped = mapping.transform(doc_src)
    transform_s = time.perf_counter() - t2
    throughput = len(doc_ids) / transform_s if transform_s > 0 else float("inf")

    if doc_src.shape[1] == qry_tgt.shape[1]:
        floor_metrics = evaluate(
            retrieve(qry_tgt, doc_src, doc_ids), query_ids, qrels
        )
    else:
        floor_metrics = {
            "nDCG@10": 0.0,
            "Recall@10": 0.0,
            "Recall@100": 0.0,
            "n_eval": len([q for q in query_ids if q in qrels]),
            "note": "dims differ; floor set to 0 (cross-space raw retrieval impossible)",
        }

    isotrieve_metrics = evaluate(
        retrieve(qry_tgt, doc_mapped, doc_ids), query_ids, qrels
    )
    ceiling_metrics = evaluate(
        retrieve(qry_tgt, doc_tgt, doc_ids), query_ids, qrels
    )

    def retention(a: float, c: float) -> float | None:
        if c <= 1e-12:
            return None
        return float(a / c)

    # Bidirectional: query→legacy direction (WS-3)
    query_legacy_metrics = None
    query_legacy_retention = None
    if mapping._W_inv is not None:
        qry_legacy = mapping.inverse_transform(qry_tgt)
        query_legacy_metrics = evaluate(
            retrieve(qry_legacy, doc_src, doc_ids), query_ids, qrels
        )
        # Retention vs target-model ceiling (what users get with full re-embed)
        # For serve-mode, compare against ceiling (target model's native retrieval)
        query_legacy_retention = {}
        for key in ["nDCG@10", "Recall@10", "Recall@100"]:
            q_val = query_legacy_metrics.get(key, 0.0)
            c_val = ceiling_metrics.get(key, 1.0)
            query_legacy_retention[key] = retention(q_val, c_val) if c_val > 1e-12 else None

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "dataset": dataset_id,
        "source_model": source_model,
        "target_model": target_model,
        "adapter": adapter,
        "calibration_mode": "in_domain",
        "protocol": "fixed_calibration_per_seed",
        "k_cal": int(k_cal),
        "seed": seed,
        "n_docs": len(doc_ids),
        "n_queries": len(query_ids),
        "d_src": int(doc_src.shape[1]),
        "d_tgt": int(doc_tgt.shape[1]),
        "timings_s": {
            "embed": embed_s,
            "fit": fit_s,
            "transform": transform_s,
        },
        "transform_vectors_per_s": throughput,
        "validation": val,
        "floor": floor_metrics,
        "isotrieve": isotrieve_metrics,
        "ceiling": ceiling_metrics,
        "retention": {
            "nDCG@10": retention(isotrieve_metrics["nDCG@10"], ceiling_metrics["nDCG@10"]),
            "Recall@10": retention(
                isotrieve_metrics["Recall@10"], ceiling_metrics["Recall@10"]
            ),
            "Recall@100": retention(
                isotrieve_metrics["Recall@100"], ceiling_metrics["Recall@100"]
            ),
        },
        "calibration_calls": 2 * k_cal,
        "reembed_calls": len(doc_ids),
    }

    # Add bidirectional results if available
    if query_legacy_metrics is not None:
        result["query_legacy"] = query_legacy_metrics
        result["query_legacy_retention"] = query_legacy_retention

    cfg_hash = hashlib.sha256(
        json.dumps(
            {
                "dataset": dataset_id,
                "source": source_model,
                "target": target_model,
                "k": k_cal,
                "seed": seed,
                "adapter": adapter,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    result["config_hash"] = cfg_hash

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_dataset = dataset_id.replace("/", "_").replace("@", "_")
    fname = (
        f"{safe_dataset}__{source_model.replace('/', '_')}__to__"
        f"{target_model.replace('/', '_')}__{adapter}__k{k_cal}__seed{seed}__{cfg_hash}.json"
    )
    path = out_dir / fname
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    print(
        f"seed={seed} retention nDCG@10={result['retention']['nDCG@10']} "
        f"floor={floor_metrics['nDCG@10']:.4f} "
        f"isotrieve={isotrieve_metrics['nDCG@10']:.4f} "
        f"ceiling={ceiling_metrics['nDCG@10']:.4f}"
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    p.add_argument(
        "--target-model",
        default="BAAI/bge-large-en-v1.5",
    )
    p.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[4000],
        help="Calibration sizes (supports K-sweep: --k 500 1000 2000 4000)",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmarks" / "results",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "benchmarks" / ".embed_cache",
    )
    p.add_argument(
        "--use-api",
        action="store_true",
        help="Use API providers (OpenAI/Voyage/Cohere/Gemini) instead of local models",
    )
    p.add_argument(
        "--dataset",
        default="scifact",
        choices=["scifact", "nfcorpus", "fiqa"],
        help="BEIR dataset to evaluate on",
    )
    p.add_argument(
        "--adapter",
        nargs="+",
        default=["ridge"],
        help="Adapter(s) to benchmark: ridge, procrustes, procrustes_diag, lowrank, mlp, or 'all'",
    )
    args = p.parse_args()

    docs, queries, qrels, dataset_id = load_beir_dataset(
        args.dataset, max_docs=args.max_docs
    )
    doc_ids = [d["id"] for d in docs]
    doc_texts = [d["text"] for d in docs]
    qrels_qids = set(qrels.keys())
    queries = [q for q in queries if q["id"] in qrels_qids]
    query_ids = [q["id"] for q in queries]
    query_texts = [q["text"] for q in queries]
    print(f"Dataset={args.dataset} Docs={len(docs)} eval_queries={len(queries)} qrels={len(qrels)}")

    doc_src, doc_tgt, qry_tgt, embed_s = load_or_embed(
        source_model=args.source_model,
        target_model=args.target_model,
        doc_texts=doc_texts,
        query_texts=query_texts,
        cache_dir=args.cache_dir,
        use_api=args.use_api,
    )

    # Resolve adapter list
    adapter_list = args.adapter
    if adapter_list == ["all"]:
        adapter_list = adapters_for_dims(doc_src.shape[1], doc_tgt.shape[1])
    elif "all" in adapter_list:
        adapter_list = adapters_for_dims(doc_src.shape[1], doc_tgt.shape[1])

    results = []
    for adapter_name in adapter_list:
        print(f"\n=== ADAPTER: {adapter_name} ===")
        for k in args.k:
            for seed in args.seeds:
                try:
                    results.append(
                        run_seed(
                            doc_src=doc_src,
                            doc_tgt=doc_tgt,
                            qry_tgt=qry_tgt,
                            doc_ids=doc_ids,
                            query_ids=query_ids,
                            qrels=qrels,
                            dataset_id=dataset_id,
                            source_model=args.source_model,
                            target_model=args.target_model,
                            k_cal=k,
                            seed=seed,
                            embed_s=embed_s,
                            out_dir=args.out_dir,
                            adapter=adapter_name,
                        )
                    )
                except Exception as e:
                    print(f"  FAILED: {adapter_name} seed={seed} K={k}: {e}")

    # Summary
    if len(adapter_list) > 1:
        print("\n=== ADAPTER COMPARISON ===")
        for adapter_name in adapter_list:
            a_results = [r for r in results if r["adapter"] == adapter_name]
            ret = [r["retention"]["nDCG@10"] for r in a_results if r["retention"]["nDCG@10"] is not None]
            if ret:
                print(f"  {adapter_name:>16}: nDCG@10 retention = {np.mean(ret):.4f} ± {np.std(ret):.4f}")
            else:
                print(f"  {adapter_name:>16}: no results")

    if len(args.k) > 1:
        print("\n=== K-SWEEP SUMMARY (by adapter) ===")
        for adapter_name in adapter_list:
            print(f"\n  {adapter_name}:")
            for k in args.k:
                k_results = [r for r in results if r["k_cal"] == k and r["adapter"] == adapter_name]
                ret = [r["retention"]["nDCG@10"] for r in k_results if r["retention"]["nDCG@10"] is not None]
                if ret:
                    print(f"    K={k:>5}: nDCG@10 retention = {np.mean(ret):.4f} ± {np.std(ret):.4f}")
    elif len(results) > 1 and len(adapter_list) == 1:
        ret = [r["retention"]["nDCG@10"] for r in results if r["retention"]["nDCG@10"] is not None]
        if ret:
            print(
                f"\nSummary nDCG@10 retention: "
                f"mean={np.mean(ret):.4f} ± {np.std(ret):.4f} over {len(ret)} seeds"
            )


if __name__ == "__main__":
    main()
