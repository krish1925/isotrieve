# CLAIMS.md — Public quantitative claims → benchmark artifacts

Every number that appears in README, docs, or marketing must have a row here
pointing at a committed file under `benchmarks/results/`.

## Active claims

| Claim text (exact or paraphrase) | Where used | Artifact path | Notes |
|----------------------------------|------------|---------------|-------|
| SciFact MiniLM→bge-large nDCG@10 retention ≈ 0.871 ± 0.006 (K=3840, Ridge adapter, 3 seeds) | `aecp-python/README.md` | `benchmarks/results/beir_scifact_*__ridge__k3840__seed{0,1,2}__*.json` | Phase 1 gate. Floor=0.0, ceiling≈0.725, AECP≈0.634. |
| LowRank adapter nDCG@10 retention ≈ 0.862 ± 0.010 (K=3840, 3 seeds) | same | `benchmarks/results/beir_scifact_*__lowrank__k3840__seed{0,1,2}__*.json` | Phase 2 adapter sweep. |
| MLP adapter nDCG@10 retention ≈ 0.729 ± 0.008 (K=3840, 3 seeds) | same | `benchmarks/results/beir_scifact_*__mlp__k3840__seed{0,1,2}__*.json` | Phase 2. No hyperparameter tuning. |
| K-sweep: K=500 retention 0.667±0.039, K=1000 0.732±0.056, K=2000 0.788±0.054, K=4000 0.817±0.064 | same | `benchmarks/results/beir_scifact_*__k{500,1000,2000,4000}__*.json` | Phase 2. All adapters averaged across 3 seeds. Monotonic improvement with K. |
| Floor nDCG@10 = 0.0 when dims differ (384≠1024) | same | same | Raw cross-space vectors cannot be queried. |
| Ceiling (full re-embed) nDCG@10 ≈ 0.725 | same | same | Quality upper bound for this model pair. |
| Same-dim pair (bge-large→e5-large, 1024→1024): floor=0.0, AECP=0.656, ceiling=0.722, retention=0.908 | `aecp-python/README.md` | `benchmarks/results/beir_scifact_*__BAAI_bge-large-en-v1.5__to__intfloat_e5-large-v2__ridge__k2000__seed0__*.json` | With e5 prefixes. Without prefixes, ceiling=0.355 and retention=0.95 (broken). |
| Gate model trained on local pairs only (no API model pairs) | `aecp-python/src/aecp/quality/gate_model_v1.json` | `benchmarks/results/gate_lopo.json` | Gate model valid for local model pairs. API pair performance may differ. |
| WS-A: bge→e5 raw scores agree at 100% for τ≤0.75; recalibration helps at τ=0.80 (+4.7% agreement, +2.36pt recall) | `README.md` | `benchmarks/results/ws_a_bge_to_e5_recall_tables.json` | MAE=0.095, margin compression=0.83x. |
| WS-A: MiniLM→bge rectangular pair — raw scores severely compressed (mean 0.157 vs ceiling 0.521, MAE=0.364). Recalibration essential. | `README.md` | `benchmarks/results/ws_a_minilm_to_bge_recall_tables.json` | τ=0.60 goes 78%→100% (+22%), τ=0.70 goes 27%→67% (+40%). |
| WS-B: Confidence flags (adaptive P33/P67 margins) are predictive across both pairs. bge→e5 high=0.955/low=0.637; MiniLM→bge high=0.875/low=0.651 | `README.md` | `benchmarks/results/ws_b_confidence_flags.json` | 50-56ms/query latency. |
| WS-B: Cross-encoder reranking NULL RESULT (-10.7pts bge→e5, -9.8pts MiniLM→bge — MS MARCO domain-mismatched for sci-text) | `DECISIONS.md` | `benchmarks/results/ws_b_cross_encoder.json` | Not shipped. |
| WS-C: Independent inverse-α: +2.17pts (bge→e5), +2.23pts (MiniLM→bge). Consistent across pairs. | `README.md` | `benchmarks/results/ws_c_independent_inv_alpha.json` | Optimal inv alpha differs from forward (0.178 vs 0.316 on bge→e5). |
| WS-C: TSVD shrinkage NULL RESULT (rank=512 only -0.33pt) | `DECISIONS.md` | `benchmarks/results/ws_c_tsvd_shrinkage.json` | Not worth complexity. |
| WS-E: Rectangular pair re-validation — MiniLM→bge: 86% retention, margin compression 0.85x, independent inverse-α +2.23pts | `README.md` | `benchmarks/results/ws_e_minilm_to_bge_revalidation.json` | |
| WS-D: Gate v3 — margin compression <0.85 widens prediction interval | `DECISIONS.md` | `src/aecp/quality/gate_model_v1.json` | `_predict_retention()` accepts `margin_compression` parameter. |

## Retired / deleted claims (must not reappear)

These appeared in prior docs without reproducible artifacts and were removed:

- "97% semantic fidelity" / "97.2%" / "97.35%"
- "<10ms" / "<1ms" transfer latency as a product claim
- "85% Top-1" abort threshold as a validated number
- "86% corpus fidelity" / "43% text baseline"
- "150x faster" / "200x"
- "Validated on 300k vocab, zero overfitting"
- Ridge ">90%" / future MLP ">99%" as stated facts

Replacement rule: only cite numbers produced by `benchmarks/run_benchmark.py`
and stored under `benchmarks/results/`.
