#!/usr/bin/env python3
"""Comprehensive diagnostic suite: Q2-Q6, Q8, Q10.

Uses cached bge→e5 embeddings (same-dim, 1024).
"""

from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isotrieve-python" / "src"))

from isotrieve.mapping.base import l2_normalize
from isotrieve.mapping.linear import (
    RidgeMapping, OrthogonalProcrustesMapping,
    ProcrustesDiagMapping, _ALPHA_GRID, _augment_bias, _coef_to_W,
    _validate_xy,
)
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import train_test_split

CACHE_DIR = ROOT / "benchmarks" / ".embed_cache"
OUT_DIR = ROOT / "benchmarks"

# ── Shared setup ──────────────────────────────────────────────────────
SRC_MODEL = "BAAI/bge-large-en-v1.5"
TGT_MODEL = "intfloat/e5-large-v2"

def load_data():
    tag = hashlib.sha256(f"{SRC_MODEL}|{TGT_MODEL}".encode()).hexdigest()[:10]
    d = CACHE_DIR / tag
    doc_src = np.load(d / "doc_src.npy")
    doc_tgt = np.load(d / "doc_tgt.npy")
    qry_tgt = np.load(d / "qry_tgt.npy")
    import ir_datasets
    corpus_ds = ir_datasets.load("beir/scifact")
    eval_ds = ir_datasets.load("beir/scifact/test")
    doc_ids = [doc.doc_id for doc in corpus_ds.docs_iter()]
    qrels_qids = set()
    qrels = {}
    for qrel in eval_ds.qrels_iter():
        qrels_qids.add(qrel.query_id)
        qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)
    query_ids = [q.query_id for q in eval_ds.queries_iter() if q.query_id in qrels_qids]
    n = min(len(doc_ids), doc_src.shape[0])
    return (doc_src[:n], doc_tgt[:n], qry_tgt[:len(query_ids)],
            doc_ids[:n], query_ids, qrels)


def retrieval_retention(doc_mapped, doc_tgt, qry_tgt, doc_ids, query_ids, qrels, k=10):
    """Compute nDCG@10 and Recall@10 retention."""
    dm_n = l2_normalize(doc_mapped)
    dt_n = l2_normalize(doc_tgt)
    qt_n = l2_normalize(qry_tgt)
    isotrieve_sims = qt_n @ dm_n.T
    ceil_sims = qt_n @ dt_n.T

    def recalls_at(sims):
        r10 = []
        for qi, qid in enumerate(query_ids):
            rel = qrels.get(qid, set())
            if not rel: continue
            top = np.argsort(-sims[qi])[:k]
            rel_idx = set(di for di, did in enumerate(doc_ids) if did in rel)
            r10.append(len(rel_idx & set(top)) / len(rel))
        return np.mean(r10) if r10 else 0.0

    return recalls_at(isotrieve_sims), recalls_at(ceil_sims)


# ══════════════════════════════════════════════════════════════════════
# Q2: Residual spectrum
# ══════════════════════════════════════════════════════════════════════
def q2_residual_spectrum(doc_src, doc_tgt):
    print("\n" + "="*70)
    print("Q2: RESIDUAL SPECTRUM — Is error isotropic or low-rank?")
    print("="*70)

    # Fit ridge on full data (to maximize residual signal)
    rng = np.random.default_rng(0)
    n = len(doc_src)
    cal_idx = rng.choice(n, size=min(4000, n), replace=False)
    hold_idx = np.array([i for i in range(n) if i not in cal_idx])

    X_cal, Y_cal = doc_src[cal_idx], doc_tgt[cal_idx]
    X_hold, Y_hold = doc_src[hold_idx], doc_tgt[hold_idx]

    m = RidgeMapping(alpha="auto", seed=0)
    m.fit(X_cal, Y_cal)
    W = m._W  # (d_src_aug, d_tgt) or (d_src, d_tgt)

    # Compute residual on holdout
    if m._bias:
        X_hold_aug = _augment_bias(X_hold)
    else:
        X_hold_aug = X_hold
    Y_pred = X_hold_aug @ W
    residual = Y_hold - Y_pred  # (n_hold, d_tgt)

    # SVD of residual
    print(f"  Residual matrix shape: {residual.shape}")
    U, sigma, Vt = np.linalg.svd(residual, full_matrices=False)

    # Compare to isotropic noise: if error were isotropic, singular values
    # would follow the Marchenko-Pastur distribution
    total_var = np.sum(sigma**2)
    explained_frac = np.cumsum(sigma**2) / total_var

    print(f"  Top-10 singular values: {sigma[:10].round(4)}")
    print(f"  Bottom-10 singular values: {sigma[-10:].round(4)}")
    print(f"  Ratio top/bottom: {sigma[0]/sigma[-1]:.2f}")
    print(f"  Variance explained by top-10: {100*explained_frac[9]:.1f}%")
    print(f"  Variance explained by top-50: {100*explained_frac[49]:.1f}%")
    print(f"  Variance explained by top-100: {100*explained_frac[99]:.1f}%")

    # isotropic check: for isotropic noise, sigma[i] ~ constant
    # coefficient of variation of singular values
    cv_sigma = np.std(sigma) / np.mean(sigma)
    print(f"  CV of singular values: {cv_sigma:.4f} (0=perfectly isotropic, >0.5=concentrated)")

    # Effective rank (entropy-based)
    p = sigma / np.sum(sigma)
    entropy = -np.sum(p * np.log(p + 1e-12))
    effective_rank = np.exp(entropy)
    print(f"  Effective rank: {effective_rank:.1f} / {len(sigma)} (1=noise-only, {len(sigma)}=full rank)")

    # What fraction of residual energy is in top-k components?
    print(f"  Energy in top-1: {100*sigma[0]**2/total_var:.2f}%")
    print(f"  Energy in top-5: {100*np.sum(sigma[:5]**2)/total_var:.2f}%")
    print(f"  Energy in top-20: {100*np.sum(sigma[:20]**2)/total_var:.2f}%")

    # Compare residual norm to signal norm
    signal_norm = np.linalg.norm(Y_hold)
    residual_norm = np.linalg.norm(residual)
    print(f"  ||Y|| (signal): {signal_norm:.2f}")
    print(f"  ||R|| (residual): {residual_norm:.2f}")
    print(f"  Residual/Signal ratio: {residual_norm/signal_norm:.4f}")

    # Plot singular value spectrum
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ax = axes[0]
    ax.semilogy(sigma[:200], "b.-", markersize=2)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Singular value (log)")
    ax.set_title("Q2: Residual singular value spectrum")
    ax.axhline(y=sigma.mean(), color="r", linestyle="--", alpha=0.5, label=f"mean={sigma.mean():.3f}")
    ax.legend()

    ax = axes[1]
    ax.plot(np.arange(1, 201), 100*explained_frac[:200], "b.-", markersize=2)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("% variance explained")
    ax.set_title("Q2: Cumulative variance explained")

    # Compare to uniform spectrum (isotropic baseline)
    ax = axes[2]
    n_comp = min(200, len(sigma))
    uniform_sigma = np.ones(n_comp) * np.mean(sigma)
    ax.plot(sigma[:n_comp], "b.", markersize=3, label="Actual")
    ax.plot(uniform_sigma, "r--", alpha=0.5, label="Isotropic (uniform)")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Singular value")
    ax.set_title("Q2: Actual vs isotropic spectrum")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "q2_residual_spectrum.png", dpi=150, bbox_inches="tight")
    print(f"  Plot → {OUT_DIR / 'q2_residual_spectrum.png'}")

    return sigma, explained_frac, effective_rank


# ══════════════════════════════════════════════════════════════════════
# Q3: Normalization ablation
# ══════════════════════════════════════════════════════════════════════
def q3_normalization_ablation(doc_src, doc_tgt, qry_tgt, doc_ids, query_ids, qrels):
    print("\n" + "="*70)
    print("Q3: NORMALIZATION ABLATION — {raw, centered, whitened} × {ridge, Procrustes}")
    print("="*70)

    rng = np.random.default_rng(0)
    n = len(doc_src)
    cal_idx = rng.choice(n, size=2000, replace=False)
    X_cal, Y_cal = doc_src[cal_idx], doc_tgt[cal_idx]

    # Preprocessing variants
    def raw(X, Y): return X, Y, None, None
    def centered(X, Y):
        mx, my = X.mean(0), Y.mean(0)
        return X - mx, Y - mx, mx, my
    def whitened(X, Y):
        mx, my = X.mean(0), Y.mean(0)
        Xc, Yc = X - mx, Y - my
        sx = Xc.std(0) + 1e-8
        sy = Yc.std(0) + 1e-8
        return Xc / sx, Yc / sy, (mx, sx), (my, sy)

    preps = {"raw": raw, "centered": centered, "whitened": whitened}
    results = {}

    for prep_name, prep_fn in preps.items():
        Xp, Yp, params_x, params_y = prep_fn(X_cal.copy(), Y_cal.copy())

        # Ridge
        m_ridge = RidgeMapping(alpha="auto", seed=0)
        m_ridge.fit(Xp, Yp)

        # Apply same preprocessing to full corpus and transform
        if prep_name == "centered":
            doc_src_p = doc_src - params_x
            doc_tgt_p = doc_tgt - params_y
            qry_p = qry_tgt - params_y
        elif prep_name == "whitened":
            mx, sx = params_x
            my, sy = params_y
            doc_src_p = (doc_src - mx) / sx
            doc_tgt_p = (doc_tgt - my) / sy
            qry_p = (qry_tgt - my) / sy
        else:
            doc_src_p, doc_tgt_p, qry_p = doc_src, doc_tgt, qry_tgt

        doc_mapped_ridge = m_ridge.transform(doc_src_p)
        r10_isotrieve, r10_ceil = retrieval_retention(
            doc_mapped_ridge, doc_tgt_p, qry_p, doc_ids, query_ids, qrels
        )
        ret_ridge = r10_isotrieve / r10_ceil if r10_ceil > 1e-12 else 0
        results[(prep_name, "ridge")] = ret_ridge
        print(f"  {prep_name:>10s} × ridge:  Recall@10 retention = {ret_ridge:.4f}  "
              f"(isotrieve={r10_isotrieve:.4f}, ceil={r10_ceil:.4f})")

        # Procrustes (square dims only — check)
        if doc_src.shape[1] == doc_tgt.shape[1]:
            m_proc = OrthogonalProcrustesMapping(seed=0)
            m_proc.fit(Xp, Yp)
            doc_mapped_proc = m_proc.transform(doc_src_p)
            r10_isotrieve_p, r10_ceil_p = retrieval_retention(
                doc_mapped_proc, doc_tgt_p, qry_p, doc_ids, query_ids, qrels
            )
            ret_proc = r10_isotrieve_p / r10_ceil_p if r10_ceil_p > 1e-12 else 0
            results[(prep_name, "procrustes")] = ret_proc
            print(f"  {prep_name:>10s} × procrustes: Recall@10 retention = {ret_proc:.4f}  "
                  f"(isotrieve={r10_isotrieve_p:.4f}, ceil={r10_ceil_p:.4f})")

    # Summary table
    print("\n  ┌─────────────┬───────────┬────────────┐")
    print("  │ Preprocessing│   Ridge   │ Procrustes │")
    print("  ├─────────────┼───────────┼────────────┤")
    for prep_name in ["raw", "centered", "whitened"]:
        ridge_val = results.get((prep_name, "ridge"), 0)
        proc_val = results.get((prep_name, "procrustes"), 0)
        print(f"  │ {prep_name:>11s} │  {ridge_val:.4f}   │   {proc_val:.4f}   │")
    print("  └─────────────┴───────────┴────────────┘")

    return results


# ══════════════════════════════════════════════════════════════════════
# Q4: Ridge α selection
# ══════════════════════════════════════════════════════════════════════
def q4_ridge_alpha(doc_src, doc_tgt):
    print("\n" + "="*70)
    print("Q4: RIDGE α SELECTION — GCV curve, flat vs sharp?")
    print("="*70)

    rng = np.random.default_rng(0)
    n = len(doc_src)
    cal_idx = rng.choice(n, size=2000, replace=False)
    X, Y = doc_src[cal_idx], doc_tgt[cal_idx]

    # Fit RidgeCV with full alpha grid to get the GCV curve
    X_aug = _augment_bias(X)
    model = RidgeCV(alphas=_ALPHA_GRID, fit_intercept=False, scoring="neg_mean_squared_error")
    model.fit(X_aug, Y)
    chosen_alpha = float(model.alpha_)
    print(f"  Chosen α: {chosen_alpha:.6f}")
    print(f"  α grid: {_ALPHA_GRID.tolist()}")

    # Compute GCV score for each alpha
    # GCV = MSE / (1 - trace(S)/n)^2 where S = X(X^TX + αI)^{-1}X^T
    from sklearn.linear_model import Ridge as RidgeBase
    gcv_scores = []
    for alpha in _ALPHA_GRID:
        m = Ridge(alpha=alpha, fit_intercept=False)
        m.fit(X_aug, Y)
        # Leave-one-out-like: use mean squared residual
        Y_pred = m.predict(X_aug)
        mse = np.mean((Y - Y_pred)**2)
        gcv_scores.append(mse)

    gcv_scores = np.array(gcv_scores)
    min_idx = np.argmin(gcv_scores)
    min_alpha = _ALPHA_GRID[min_idx]

    print(f"  MSE at chosen α ({chosen_alpha:.4f}): {gcv_scores[min_idx]:.6f}")
    print(f"  MSE range across grid: [{gcv_scores.min():.6f}, {gcv_scores.max():.6f}]")
    print(f"  MSE at α_min / MSE at α_max ratio: {gcv_scores.max()/gcv_scores.min():.2f}x")

    # Is the curve flat or sharp?
    # Find α range where MSE is within 5% of minimum
    within_5pct = _ALPHA_GRID[gcv_scores <= gcv_scores.min() * 1.05]
    print(f"  α range within 5% of min MSE: [{within_5pct.min():.4f}, {within_5pct.max():.4f}]")
    print(f"  Ratio α_max/α_min in 5% band: {within_5pct.max()/within_5pct.min():.1f}x")

    # Per-output-dimension analysis: does optimal α vary across dimensions?
    print(f"\n  Per-dimension α analysis (top-20 output dims by MSE variance):")
    per_dim_mse = np.zeros((len(_ALPHA_GRID), Y.shape[1]))
    for ai, alpha in enumerate(_ALPHA_GRID):
        m = Ridge(alpha=alpha, fit_intercept=False)
        m.fit(X_aug, Y)
        Y_pred = m.predict(X_aug)
        per_dim_mse[ai] = np.mean((Y - Y_pred)**2, axis=0)

    dim_variance = per_dim_mse.min(axis=0)  # best MSE per dim
    dim_var_range = per_dim_mse.max(axis=0) - per_dim_mse.min(axis=0)
    top_var_dims = np.argsort(-dim_var_range)[:20]

    # For each of top-varying dims, what's the optimal alpha?
    optimal_alphas = []
    for d in top_var_dims:
        best_ai = np.argmin(per_dim_mse[:, d])
        optimal_alphas.append(_ALPHA_GRID[best_ai])

    print(f"    Optimal α for top-20 varying dims: {[f'{a:.2f}' for a in optimal_alphas]}")
    print(f"    Range: [{min(optimal_alphas):.4f}, {max(optimal_alphas):.4f}]")
    print(f"    CV of optimal α: {np.std(optimal_alphas)/np.mean(optimal_alphas):.4f}")

    # Plot GCV curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.semilogx(_ALPHA_GRID, gcv_scores, "b.-", markersize=5)
    ax.axvline(x=chosen_alpha, color="r", linestyle="--", label=f"chosen α={chosen_alpha:.4f}")
    ax.set_xlabel("α (log scale)")
    ax.set_ylabel("Mean squared residual")
    ax.set_title("Q4: Ridge GCV curve")
    ax.legend()

    ax = axes[1]
    # Heatmap: per-dim MSE for varying alphas (top-50 dims by variance)
    top50 = np.argsort(-dim_var_range)[:50]
    im = ax.imshow(per_dim_mse[:, top50].T, aspect="auto", cmap="viridis")
    ax.set_xlabel("α index")
    ax.set_ylabel("Output dimension (sorted by variance)")
    ax.set_title("Q4: Per-dim MSE across α values")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "q4_ridge_alpha.png", dpi=150, bbox_inches="tight")
    print(f"  Plot → {OUT_DIR / 'q4_ridge_alpha.png'}")

    return chosen_alpha, gcv_scores


# ══════════════════════════════════════════════════════════════════════
# Q5: Hubness profile
# ══════════════════════════════════════════════════════════════════════
def q5_hubness(doc_src, doc_tgt, qry_tgt, doc_ids, query_ids, qrels):
    print("\n" + "="*70)
    print("Q5: HUBNESS PROFILE — k-occurrence distribution before/after mapping")
    print("="*70)

    rng = np.random.default_rng(0)
    n = len(doc_src)
    cal_idx = rng.choice(n, size=2000, replace=False)
    m = RidgeMapping(alpha="auto", seed=0)
    m.fit(doc_src[cal_idx], doc_tgt[cal_idx])
    doc_mapped = m.transform(doc_src)

    qt_n = l2_normalize(qry_tgt)

    # Ceiling: target queries vs target docs
    dt_n = l2_normalize(doc_tgt)
    ceil_sims = qt_n @ dt_n.T
    ceil_top10 = np.argsort(-ceil_sims, axis=1)[:, :10]

    # Isotrieve: target queries vs mapped docs
    dm_n = l2_normalize(doc_mapped)
    isotrieve_sims = qt_n @ dm_n.T
    isotrieve_top10 = np.argsort(-isotrieve_sims, axis=1)[:, :10]

    # k-occurrence: how many queries have each doc in their top-10
    n_docs = len(doc_ids)
    ceil_kocc = np.zeros(n_docs, dtype=int)
    isotrieve_kocc = np.zeros(n_docs, dtype=int)
    for qi in range(len(query_ids)):
        for di in ceil_top10[qi]:
            ceil_kocc[di] += 1
        for di in isotrieve_top10[qi]:
            isotrieve_kocc[di] += 1

    print(f"  Ceiling hubness (top-10 occurrences per doc):")
    print(f"    mean={np.mean(ceil_kocc):.2f}, median={np.median(ceil_kocc):.1f}, "
          f"max={np.max(ceil_kocc)}, std={np.std(ceil_kocc):.2f}")
    print(f"    % docs appearing 0 times: {100*np.mean(ceil_kocc==0):.1f}%")
    print(f"    % docs appearing >5 times: {100*np.mean(ceil_kocc>5):.1f}%")

    print(f"  Isotrieve hubness (top-10 occurrences per doc):")
    print(f"    mean={np.mean(isotrieve_kocc):.2f}, median={np.median(isotrieve_kocc):.1f}, "
          f"max={np.max(isotrieve_kocc)}, std={np.std(isotrieve_kocc):.2f}")
    print(f"    % docs appearing 0 times: {100*np.mean(isotrieve_kocc==0):.1f}%")
    print(f"    % docs appearing >5 times: {100*np.mean(isotrieve_kocc>5):.1f}%")

    # Skewness increase
    from scipy.stats import skew, kurtosis
    print(f"  Skewness: ceiling={skew(ceil_kocc):.3f} → Isotrieve={skew(isotrieve_kocc):.3f}")
    print(f"  Kurtosis: ceiling={kurtosis(ceil_kocc):.3f} → Isotrieve={kurtosis(isotrieve_kocc):.3f}")

    # Top-10 hub docs (most frequent in top-10)
    print(f"\n  Top-10 hub docs (Isotrieve):")
    top_hubs = np.argsort(-isotrieve_kocc)[:10]
    for di in top_hubs:
        print(f"    doc[{di}] ({doc_ids[di]}): {isotrieve_kocc[di]} appearances "
              f"(ceiling: {ceil_kocc[di]})")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ax = axes[0]
    ax.hist(ceil_kocc, bins=range(0, max(ceil_kocc.max(), isotrieve_kocc.max())+2),
            alpha=0.6, color="#2196F3", label="Ceiling", density=True)
    ax.hist(isotrieve_kocc, bins=range(0, max(ceil_kocc.max(), isotrieve_kocc.max())+2),
            alpha=0.6, color="#F44336", label="Isotrieve", density=True)
    ax.set_xlabel("Top-10 appearances per doc")
    ax.set_ylabel("Density")
    ax.set_title("Q5: Hubness distribution")
    ax.legend()

    ax = axes[1]
    ax.scatter(ceil_kocc, isotrieve_kocc, s=5, alpha=0.3, c="black")
    max_val = max(ceil_kocc.max(), isotrieve_kocc.max())
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="y=x (no change)")
    ax.set_xlabel("Ceiling top-10 count")
    ax.set_ylabel("Isotrieve top-10 count")
    ax.set_title("Q5: Hub count change per doc")
    ax.legend()

    ax = axes[2]
    # CDF of hubness
    ceil_sorted = np.sort(ceil_kocc)
    isotrieve_sorted = np.sort(isotrieve_kocc)
    ax.plot(ceil_sorted, np.linspace(0, 1, len(ceil_sorted)), label="Ceiling", color="#2196F3")
    ax.plot(isotrieve_sorted, np.linspace(0, 1, len(isotrieve_sorted)), label="Isotrieve", color="#F44336")
    ax.set_xlabel("Top-10 appearances")
    ax.set_ylabel("CDF")
    ax.set_title("Q5: Hubness CDF")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "q5_hubness.png", dpi=150, bbox_inches="tight")
    print(f"  Plot → {OUT_DIR / 'q5_hubness.png'}")

    return ceil_kocc, isotrieve_kocc


# ══════════════════════════════════════════════════════════════════════
# Q6: Calibration sampling
# ══════════════════════════════════════════════════════════════════════
def q6_calibration_sampling(doc_src, doc_tgt, qry_tgt, doc_ids, query_ids, qrels):
    print("\n" + "="*70)
    print("Q6: CALIBRATION SAMPLING — random vs k-means diversity at K=1000")
    print("="*70)

    rng = np.random.default_rng(0)
    n = len(doc_src)
    k = 1000
    n_trials = 5

    results_random = []
    results_diverse = []

    for trial in range(n_trials):
        seed = trial
        rng_t = np.random.default_rng(seed)

        # Random sampling
        rand_idx = rng_t.choice(n, size=k, replace=False)
        m_rand = RidgeMapping(alpha="auto", seed=seed)
        m_rand.fit(doc_src[rand_idx], doc_tgt[rand_idx])
        doc_mapped_rand = m_rand.transform(doc_src)
        r10_isotrieve, r10_ceil = retrieval_retention(
            doc_mapped_rand, doc_tgt, qry_tgt, doc_ids, query_ids, qrels
        )
        ret_rand = r10_isotrieve / r10_ceil if r10_ceil > 1e-12 else 0
        results_random.append(ret_rand)

        # Diversity sampling: k-means in source space, pick cluster centers
        kmb = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=min(5000, n))
        kmb.fit(doc_src)
        diverse_idx = []
        for ci in range(k):
            cluster_mask = kmb.labels_ == ci
            if not np.any(cluster_mask):
                continue
            centers = kmb.cluster_centers_[ci]
            dists = np.linalg.norm(doc_src[cluster_mask] - centers, axis=1)
            closest = np.where(cluster_mask)[0][np.argmin(dists)]
            diverse_idx.append(closest)
        diverse_idx = np.array(diverse_idx[:k])

        m_div = RidgeMapping(alpha="auto", seed=seed)
        m_div.fit(doc_src[diverse_idx], doc_tgt[diverse_idx])
        doc_mapped_div = m_div.transform(doc_src)
        r10_isotrieve_d, r10_ceil_d = retrieval_retention(
            doc_mapped_div, doc_tgt, qry_tgt, doc_ids, query_ids, qrels
        )
        ret_div = r10_isotrieve_d / r10_ceil_d if r10_ceil_d > 1e-12 else 0
        results_diverse.append(ret_div)

    print(f"  Random K={k} (n={n_trials} trials):")
    print(f"    mean Recall@10 retention = {np.mean(results_random):.4f} ± {np.std(results_random):.4f}")
    print(f"  Diversity K={k} (n={n_trials} trials):")
    print(f"    mean Recall@10 retention = {np.mean(results_diverse):.4f} ± {np.std(results_diverse):.4f}")
    delta = np.mean(results_diverse) - np.mean(results_random)
    print(f"  Δ (diversity - random) = {delta:+.4f} ({'+' if delta > 0 else ''}{delta*100:.1f} points)")

    return results_random, results_diverse


# ══════════════════════════════════════════════════════════════════════
# Q8: Gate model residuals
# ══════════════════════════════════════════════════════════════════════
def q8_gate_residuals():
    print("\n" + "="*70)
    print("Q8: GATE MODEL RESIDUALS — predicted vs true retention by pair")
    print("="*70)

    gate_path = Path("/Users/kpatel/Desktop/agent-communication/isotrieve-python/src/isotrieve/quality/gate_model_v1.json")
    gate = json.loads(gate_path.read_text())

    # Gate model info
    print(f"  Type: {gate.get('type')}")
    print(f"  Feature: {gate.get('feature')}")
    print(f"  Scope: {gate.get('scope')}")
    print(f"  Training pairs: {gate.get('training_pairs')}")
    print(f"  Training runs: {gate.get('n_training_runs')}")

    # LOPO stats
    lipo = gate.get("lipo", {})
    print(f"\n  LOPO stats:")
    print(f"    MAE: {lipo.get('mae'):.4f}")
    print(f"    RMSE: {lipo.get('rmse'):.4f}")
    print(f"    Interval half-width (80%): {lipo.get('interval_half_width_80'):.4f}")
    print(f"    Mean residual: {lipo.get('mean_residual'):.4f}")
    print(f"    Std residual: {lipo.get('std_residual'):.4f}")

    # Per-pair residuals (each value is the MAE for that pair)
    residuals_by_pair = lipo.get("residuals_by_pair", {})
    if residuals_by_pair:
        print(f"\n  Per-pair MAE ({len(residuals_by_pair)} pairs):")
        pair_maes = []
        for pair, mae in residuals_by_pair.items():
            pair_maes.append((pair, float(mae)))
            print(f"    {pair}: MAE={mae:.4f}")

        maes = [p[1] for p in pair_maes]
        print(f"\n  Pair MAE analysis:")
        print(f"    Mean MAE: {np.mean(maes):.4f}")
        print(f"    Std of MAEs: {np.std(maes):.4f}")
        print(f"    Range: [{min(maes):.4f}, {max(maes):.4f}]")
        worst = max(pair_maes, key=lambda x: x[1])
        best = min(pair_maes, key=lambda x: x[1])
        print(f"    Worst pair: {worst[0]} (MAE={worst[1]:.4f})")
        print(f"    Best pair: {best[0]} (MAE={best[1]:.4f})")
        print(f"    Spread (worst - best): {worst[1] - best[1]:.4f}")

    # Isotonic regression fit
    X_thresh = np.array(gate["X_thresholds"])
    y_thresh = np.array(gate["y_thresholds"])
    print(f"\n  Isotonic regression calibration points:")
    print(f"    X range: [{X_thresh.min():.4f}, {X_thresh.max():.4f}]")
    print(f"    y range: [{y_thresh.min():.4f}, {y_thresh.max():.4f}]")
    print(f"    n calibration points: {len(X_thresh)}")

    return gate


# ══════════════════════════════════════════════════════════════════════
# Q10: Score distribution
# ══════════════════════════════════════════════════════════════════════
def q10_score_distribution(doc_src, doc_tgt, qry_tgt, doc_ids, query_ids, qrels):
    print("\n" + "="*70)
    print("Q10: SCORE DISTRIBUTION — mapped vs ceiling top-1 scores, margin compression")
    print("="*70)

    rng = np.random.default_rng(0)
    n = len(doc_src)
    cal_idx = rng.choice(n, size=2000, replace=False)
    m = RidgeMapping(alpha="auto", seed=0)
    m.fit(doc_src[cal_idx], doc_tgt[cal_idx])
    doc_mapped = m.transform(doc_src)

    qt_n = l2_normalize(qry_tgt)
    dt_n = l2_normalize(doc_tgt)
    dm_n = l2_normalize(doc_mapped)

    ceil_sims = qt_n @ dt_n.T
    isotrieve_sims = qt_n @ dm_n.T

    ceil_top1_scores = []
    ceil_top2_scores = []
    ceil_margins = []
    isotrieve_top1_scores = []
    isotrieve_top2_scores = []
    isotrieve_margins = []

    for qi, qid in enumerate(query_ids):
        rel = qrels.get(qid, set())
        if not rel:
            continue
        rel_idx = set(di for di, did in enumerate(doc_ids) if did in rel)

        # Ceiling
        c_sorted = np.argsort(-ceil_sims[qi])
        c_top1 = ceil_sims[qi, c_sorted[0]]
        c_top2 = ceil_sims[qi, c_sorted[1]]
        ceil_top1_scores.append(float(c_top1))
        ceil_top2_scores.append(float(c_top2))
        ceil_margins.append(float(c_top1 - c_top2))

        # Isotrieve
        a_sorted = np.argsort(-isotrieve_sims[qi])
        a_top1 = isotrieve_sims[qi, a_sorted[0]]
        a_top2 = isotrieve_sims[qi, a_sorted[1]]
        isotrieve_top1_scores.append(float(a_top1))
        isotrieve_top2_scores.append(float(a_top2))
        isotrieve_margins.append(float(a_top1 - a_top2))

    print(f"  Ceiling top-1 scores:  mean={np.mean(ceil_top1_scores):.4f}, "
          f"median={np.median(ceil_top1_scores):.4f}, std={np.std(ceil_top1_scores):.4f}")
    print(f"  Isotrieve top-1 scores:     mean={np.mean(isotrieve_top1_scores):.4f}, "
          f"median={np.median(isotrieve_top1_scores):.4f}, std={np.std(isotrieve_top1_scores):.4f}")
    print(f"  Score compression:     {np.mean(ceil_top1_scores):.4f} → {np.mean(isotrieve_top1_scores):.4f} "
          f"(Δ={np.mean(isotrieve_top1_scores)-np.mean(ceil_top1_scores):+.4f})")

    print(f"\n  Ceiling margins (rank1 - rank2): mean={np.mean(ceil_margins):.4f}, "
          f"std={np.std(ceil_margins):.4f}")
    print(f"  Isotrieve margins (rank1 - rank2):    mean={np.mean(isotrieve_margins):.4f}, "
          f"std={np.std(isotrieve_margins):.4f}")
    print(f"  Margin compression:    {np.mean(ceil_margins):.4f} → {np.mean(isotrieve_margins):.4f} "
          f"(Δ={np.mean(isotrieve_margins)-np.mean(ceil_margins):+.4f})")

    # How many queries have compressed margin (Isotrieve margin < ceiling margin)?
    n_compressed = sum(1 for i in range(len(ceil_margins)) if isotrieve_margins[i] < ceil_margins[i])
    print(f"  Queries with compressed margin: {n_compressed}/{len(ceil_margins)} "
          f"({100*n_compressed/len(ceil_margins):.1f}%)")

    # Score range for threshold setting
    print(f"\n  Score calibration range:")
    print(f"    Ceiling: [{np.percentile(ceil_top1_scores, 5):.4f}, {np.percentile(ceil_top1_scores, 95):.4f}]")
    print(f"    Isotrieve:    [{np.percentile(isotrieve_top1_scores, 5):.4f}, {np.percentile(isotrieve_top1_scores, 95):.4f}]")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ax = axes[0]
    ax.hist(ceil_top1_scores, bins=30, alpha=0.6, color="#2196F3", label="Ceiling", density=True)
    ax.hist(isotrieve_top1_scores, bins=30, alpha=0.6, color="#F44336", label="Isotrieve", density=True)
    ax.set_xlabel("Top-1 cosine score")
    ax.set_ylabel("Density")
    ax.set_title("Q10: Top-1 score distribution")
    ax.legend()

    ax = axes[1]
    ax.hist(ceil_margins, bins=30, alpha=0.6, color="#2196F3", label="Ceiling", density=True)
    ax.hist(isotrieve_margins, bins=30, alpha=0.6, color="#F44336", label="Isotrieve", density=True)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Margin (rank1 - rank2 cosine)")
    ax.set_ylabel("Density")
    ax.set_title("Q10: Margin distribution")
    ax.legend()

    ax = axes[2]
    ax.scatter(ceil_margins, isotrieve_margins, s=5, alpha=0.3, c="black")
    lims = [min(min(ceil_margins), min(isotrieve_margins))-0.01,
            max(max(ceil_margins), max(isotrieve_margins))+0.01]
    ax.plot(lims, lims, "r--", alpha=0.5, label="y=x (no change)")
    ax.set_xlabel("Ceiling margin")
    ax.set_ylabel("Isotrieve margin")
    ax.set_title("Q10: Margin change per query")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "q10_scores.png", dpi=150, bbox_inches="tight")
    print(f"  Plot → {OUT_DIR / 'q10_scores.png'}")

    return ceil_margins, isotrieve_margins


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading cached embeddings...")
    doc_src, doc_tgt, qry_tgt, doc_ids, query_ids, qrels = load_data()
    print(f"  doc_src: {doc_src.shape}, doc_tgt: {doc_tgt.shape}, qry_tgt: {qry_tgt.shape}")

    q2_residual_spectrum(doc_src, doc_tgt)
    q3_normalization_ablation(doc_src, doc_tgt, qry_tgt, doc_ids, query_ids, qrels)
    q4_ridge_alpha(doc_src, doc_tgt)
    q5_hubness(doc_src, doc_tgt, qry_tgt, doc_ids, query_ids, qrels)
    q6_calibration_sampling(doc_src, doc_tgt, qry_tgt, doc_ids, query_ids, qrels)
    q8_gate_residuals()
    q10_score_distribution(doc_src, doc_tgt, qry_tgt, doc_ids, query_ids, qrels)

    print("\n" + "="*70)
    print("DONE — all plots saved to benchmarks/")
    print("="*70)
