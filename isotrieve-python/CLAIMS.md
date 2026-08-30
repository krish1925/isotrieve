# CLAIMS.md — Public quantitative claims → benchmark artifacts

Every number that appears in README, docs, or marketing must have a row here
pointing at a committed file under `benchmarks/results/`.

## Active claims

| Claim text (exact or paraphrase) | Where used | Artifact path | Verified | Notes |
|----------------------------------|------------|---------------|----------|-------|
| SciFact MiniLM→bge-large nDCG@10 retention ≈ 0.871 ± 0.006 (K=4000, Ridge adapter, 3 seeds) | `isotrieve-python/README.md` | `benchmarks/results/beir_scifact_*__ridge__k4000__seed{0,1,2}__*.json` | 2026-07-21 | Floor=0.0, ceiling=0.735, Isotrieve=0.634–0.646. Actual: 0.872, 0.863, 0.879. |
| LowRank adapter nDCG@10 retention ≈ 0.857 ± 0.009 (K=4000, 3 seeds) | same | `benchmarks/results/beir_scifact_*__lowrank__k4000__seed{0,1,2}__*.json` | 2026-07-21 | Phase 2 adapter sweep. Actual: 0.867, 0.846, 0.858. |
| MLP adapter nDCG@10 retention ≈ 0.727 ± 0.007 (K=4000, 3 seeds) | same | `benchmarks/results/beir_scifact_*__mlp__k4000__seed{0,1,2}__*.json` | 2026-07-21 | No hyperparameter tuning. Actual: 0.736, 0.724, 0.721. |
| K-sweep (ridge only): K=500 0.704±0.004, K=1000 0.781±0.008, K=2000 0.818±0.007, K=4000 0.871±0.006 | same | `benchmarks/results/beir_scifact_*__ridge__k{500,1000,2000,4000}__*.json` | 2026-07-21 | Ridge-only K-sweep. Monotonic improvement with K. |
| K-sweep (all adapters avg): K=500 0.671±0.041, K=1000 0.735±0.058, K=2000 0.785±0.052, K=4000 0.832±0.061 | same | `benchmarks/results/beir_scifact_*__k{500,1000,2000,4000}__*.json` | 2026-07-21 | All adapters averaged across 3 seeds. |
| Floor nDCG@10 = 0.0 when dims differ (384≠1024) | same | same | 2026-07-21 | Raw cross-space vectors cannot be queried. |
| Ceiling (full re-embed) nDCG@10 ≈ 0.735 | same | same | 2026-07-21 | Quality upper bound for this model pair. |
| Same-dim pair (bge-large→e5-large, 1024→1024): floor=0.0, Isotrieve≈0.667, ceiling=0.722, retention=0.923±0.010 | `isotrieve-python/README.md` | `benchmarks/results/beir_scifact_*__BAAI_bge-large-en-v1.5__to__intfloat_e5-large-v2__ridge__k2000__seed{0,1,2}__*.json` | 2026-07-21 | Actual: 0.908, 0.930, 0.931. With e5 prefixes. Without prefixes, ceiling=0.355 and retention=0.95 (broken). |
| Gate model trained on local pairs only (no API model pairs) | `isotrieve-python/src/isotrieve/quality/gate_model_v1.json` | `benchmarks/results/gate_lopo.json` | — | Gate model valid for local model pairs. API pair performance may differ. |
| WS-A: bge→e5 raw scores agree at 100% for τ≤0.75; recalibration helps at τ=0.80 (+4.7% agreement, +2.36pt recall) | `README.md` | `benchmarks/results/ws_a_bge_to_e5_recall_tables.json` | — | MAE=0.095, margin compression=0.83x. |
| WS-A: MiniLM→bge rectangular pair — raw scores severely compressed (mean 0.157 vs ceiling 0.521, MAE=0.364). Recalibration essential. | `README.md` | `benchmarks/results/ws_a_minilm_to_bge_recall_tables.json` | — | τ=0.60 goes 78%→100% (+22%), τ=0.70 goes 27%→67% (+40%). |
| WS-B: Confidence flags (adaptive P33/P67 margins) are predictive across both pairs. bge→e5 high=0.955/low=0.637; MiniLM→bge high=0.875/low=0.651 | `README.md` | `benchmarks/results/ws_b_confidence_flags.json` | — | 50-56ms/query latency. |
| WS-B: Cross-encoder reranking NULL RESULT (-10.7pts bge→e5, -9.8pts MiniLM→bge — MS MARCO domain-mismatched for sci-text) | `DECISIONS.md` | `benchmarks/results/ws_b_cross_encoder.json` | — | Not shipped. |
| WS-C: Independent inverse-α: +2.17pts (bge→e5), +2.23pts (MiniLM→bge). Consistent across pairs. | `README.md` | `benchmarks/results/ws_c_independent_inv_alpha.json` | — | Optimal inv alpha differs from forward (0.178 vs 0.316 on bge→e5). |
| WS-C: TSVD shrinkage NULL RESULT (rank=512 only -0.33pt) | `DECISIONS.md` | `benchmarks/results/ws_c_tsvd_shrinkage.json` | — | Not worth complexity. |
| WS-E: Rectangular pair re-validation — MiniLM→bge: 86% retention, margin compression 0.85x, independent inverse-α +2.23pts | `README.md` | `benchmarks/results/ws_e_minilm_to_bge_revalidation.json` | — | |
| WS-D: Gate v3 — margin compression <0.85 widens prediction interval | `DECISIONS.md` | `src/isotrieve/quality/gate_model_v1.json` | — | `_predict_retention()` accepts `margin_compression` parameter. |
| Domain matrix (medical): SciFact MiniLM→bge-large K=500 max-docs=2000 (2 seeds) — Ridge nDCG@10 retention 0.813±0.020 | `benchmarks/README.md` | `benchmarks/results/beir_scifact_*__ridge__k500__seed0__bb10a373fec9.json, benchmarks/results/beir_scifact_*__ridge__k500__seed1__d0192d6990fd.json` | 2026-08-05 | Ceiling=0.362, floor=0.0. Actual: 0.827, 0.799. |
| Domain matrix (medical): SciFact MiniLM→bge-large K=500 max-docs=2000 (2 seeds) — LowRank nDCG@10 retention 0.810±0.022 | `benchmarks/README.md` | `benchmarks/results/beir_scifact_*__lowrank__k500__seed0__c57d7af77649.json, benchmarks/results/beir_scifact_*__lowrank__k500__seed1__75e1721e9e76.json` | 2026-08-05 | Ceiling=0.362, floor=0.0. Actual: 0.826, 0.795. |
| Domain matrix (general): FiQA MiniLM→bge-large K=500 max-docs=2000 (2 seeds) — Ridge nDCG@10 retention 0.773±0.005 | `benchmarks/README.md` | `benchmarks/results/beir_fiqa_*__ridge__k500__seed0__64dba3ab41be.json, benchmarks/results/beir_fiqa_*__ridge__k500__seed1__af47a546e2ec.json` | 2026-08-05 | Ceiling=0.032 (truncation-limited; most qrels docs beyond first 2k). Actual: 0.769, 0.776. |
| Domain matrix (general): FiQA MiniLM→bge-large K=500 max-docs=2000 (2 seeds) — LowRank nDCG@10 retention 0.768±0.012 | `benchmarks/README.md` | `benchmarks/results/beir_fiqa_*__lowrank__k500__seed0__88e8338191d1.json, benchmarks/results/beir_fiqa_*__lowrank__k500__seed1__53f0d0b2496f.json` | 2026-08-05 | Ceiling=0.032. Actual: 0.759, 0.777. |
| Domain matrix (code): offline synthetic probe corpus (520 docs) MiniLM→bge-large K=500 (2 seeds) — Ridge/LowRank nDCG@10 retention 1.022±0.114 | `benchmarks/README.md` | `benchmarks/results/probe_code_*__k500__seed{0,1}__*.json` | 2026-08-05 | Synthetic corpus; mapped retrieval can exceed target ceiling. Ridge and LowRank identical here. |
| Domain matrix (legal): offline synthetic probe corpus (520 docs) MiniLM→bge-large K=500 (2 seeds) — Ridge/LowRank nDCG@10 retention 1.037±0.126 | `benchmarks/README.md` | `benchmarks/results/probe_legal_*__k500__seed{0,1}__*.json` | 2026-08-05 | Synthetic corpus; mapped retrieval can exceed target ceiling. Ridge and LowRank identical here. |
| Identifier-level probe retention (same-index true-match top-1/top-10) reported per run for all four domains; in-domain > cross-domain. E.g. medical-fit mapping keeps medical probes top-1 0.65 / code probes 0.82; code-fit keeps code probes top-1 1.0 but general/legal/medical drop to 0.11–0.25 | `benchmarks/README.md` | `benchmarks/results/beir_scifact_*__ridge__k500__seed0__bb10a373fec9.json, benchmarks/results/beir_scifact_*__ridge__k500__seed1__d0192d6990fd.json, benchmarks/results/beir_scifact_*__lowrank__k500__seed0__c57d7af77649.json, benchmarks/results/beir_scifact_*__lowrank__k500__seed1__75e1721e9e76.json, benchmarks/results/beir_fiqa_*__ridge__k500__seed0__64dba3ab41be.json, benchmarks/results/beir_fiqa_*__ridge__k500__seed1__af47a546e2ec.json, benchmarks/results/beir_fiqa_*__lowrank__k500__seed0__88e8338191d1.json, benchmarks/results/beir_fiqa_*__lowrank__k500__seed1__53f0d0b2496f.json, benchmarks/results/probe_code_*__k500__seed{0,1}__*.json, benchmarks/results/probe_legal_*__k500__seed{0,1}__*.json` | 2026-08-05 | Probe defs in `benchmarks/domain_probes.py`; deterministic, offline, no model downloads beyond harness models. |

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
