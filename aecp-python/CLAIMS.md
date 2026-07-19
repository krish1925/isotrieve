# CLAIMS.md — Public quantitative claims → benchmark artifacts

Every number that appears in README, docs, or marketing must have a row here
pointing at a committed file under `benchmarks/results/`.

## Active claims

| Claim text (exact or paraphrase) | Where used | Artifact path | Notes |
|----------------------------------|------------|---------------|-------|
| SciFact MiniLM→bge-large nDCG@10 retention ≈ 0.871 ± 0.006 (K=4000, Ridge adapter, 3 seeds) | `aecp-python/README.md` | `benchmarks/results/beir_scifact_*__ridge__k4000__seed{0,1,2}__*.json` | Phase 1 gate. Floor=0.0, ceiling≈0.725, AECP≈0.634. |
| LowRank adapter nDCG@10 retention ≈ 0.857 ± 0.009 (K=4000, 3 seeds) | same | `benchmarks/results/beir_scifact_*__lowrank__k4000__seed{0,1,2}__*.json` | Phase 2 adapter sweep. |
| MLP adapter nDCG@10 retention ≈ 0.719 ± 0.008 (K=4000, 3 seeds) | same | `benchmarks/results/beir_scifact_*__mlp__k4000__seed{0,1,2}__*.json` | Phase 2. No hyperparameter tuning. |
| K-sweep: K=500 retention 0.667±0.039, K=1000 0.732±0.056, K=2000 0.788±0.054, K=4000 0.814±0.068 | same | `benchmarks/results/beir_scifact_*__k{500,1000,2000,4000}__*.json` | Phase 2. All adapters averaged across 3 seeds. Monotonic improvement with K. |
| Floor nDCG@10 = 0.0 when dims differ (384≠1024) | same | same | Raw cross-space vectors cannot be queried. |
| Ceiling (full re-embed) nDCG@10 ≈ 0.725 | same | same | Quality upper bound for this model pair. |
| Same-dim pair (bge-large→e5-large, 1024→1024): floor=0.0, AECP=0.656, ceiling=0.722, retention=0.908 | `aecp-python/README.md` | `benchmarks/results/beir_scifact_*__BAAI_bge-large-en-v1.5__to__intfloat_e5-large-v2__ridge__k2000__seed0__*.json` | With e5 prefixes. Without prefixes, ceiling=0.355 and retention=0.95 (broken). |
| Gate model trained on local pairs only (no API model pairs) | `aecp-python/src/aecp/quality/gate_model_v1.json` | `benchmarks/results/gate_lopo.json` | Gate model valid for local model pairs. API pair performance may differ. |

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
