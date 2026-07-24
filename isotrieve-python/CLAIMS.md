# CLAIMS.md — Public quantitative claims → benchmark artifacts

Every number that appears in README, docs, or marketing must have a row here
pointing at a committed file under `benchmarks/results/`.

## Active claims

### Embedding Stability & Number Preservation

| Claim | Value | Methodology | Artifact | Date |
|-------|-------|-------------|----------|------|
| Source self-retrieval R@1 | 100.0% | Embed 200 FiQA docs with all-MiniLM-L6-v2, compute pairwise cosine, check if each doc retrieves itself | Inline (reproducible, seed=42) | 2026-07-24 |
| Target self-retrieval R@1 | 100.0% | Same as above with all-mpnet-base-v2 | Inline | 2026-07-24 |
| Re-embed exact match (source) | True (max diff: 0.00e+00) | Re-embed 100 docs, compare to original vectors element-wise | Inline | 2026-07-24 |
| Re-embed exact match (target) | True (max diff: 1.27e-07) | Same as above for target model | Inline | 2026-07-24 |
| Mapped self-retrieval R@1 | 95.0% | Fit RidgeMapping on 200 paired embeddings, transform source vectors, check self-retrieval in target space | Inline | 2026-07-24 |

**Methodology:** Embeddings are deterministic — same input text produces identical vectors (zero drift). The 1.27e-07 max diff on target model is floating-point rounding, not model drift. After isotrieve mapping, 95% of documents still retrieve themselves in the target space.

### Mapping Fidelity (Cosine Similarity)

| Claim | Value | Methodology | Date |
|-------|-------|-------------|------|
| In-domain mapping fidelity (FiQA) | mean=0.8574, p5=0.5753, min=0.3051 | Fit RidgeMapping on 200 FiQA doc pairs, compute cosine between mapped source and target vectors | 2026-07-24 |
| Cross-domain mapping fidelity (FiQA→NQ) | mean=0.4813, p5=0.3193, min=0.1418 | Fit on FiQA, transform NQ vectors, compute cosine | 2026-07-24 |

**Methodology:** Mapping fidelity measures how close the mapped vectors are to the true target embeddings. In-domain fidelity of 0.86 means the mapped vectors are 86% similar to the real target vectors on average. Cross-domain fidelity drops to 0.48 because the mapping is domain-specific.

### Retrieval Retention (FiQA, text-overlap relevance)

| Claim | K | Gate | R@1 | R@10 | MRR | Retention R@10 | Date |
|-------|---|------|-----|------|-----|----------------|------|
| FiQA in-domain | 500 | WARN (pred=0.695) | 0.095 | 0.390 | 0.171 | 91.8% | 2026-07-24 |
| FiQA in-domain | 1000 | WARN (pred=0.795) | 0.120 | 0.430 | 0.196 | 101.2% | 2026-07-24 |
| FiQA in-domain | 2000 | WARN (pred=0.838) | 0.095 | 0.420 | 0.181 | 98.8% | 2026-07-24 |

**Methodology:**
- Dataset: FiQA (financial QA), 2000 docs sampled from 57,638, 200 queries from 6,648
- Relevance: text-overlap (Jaccard similarity of significant words, top-10 per query)
- Ceiling: target-model queries vs target-model corpus
- Model pair: all-MiniLM-L6-v2 (384d) → all-mpnet-base-v2 (768d)
- Gate: isotrieve QualityGate with default thresholds (pass=0.75, warn=0.55)
- Retention >100% at K=1000 is possible because the mapping can improve ranking by smoothing noise

### SciFact Benchmarks (BEIR, nDCG@10 retention)

| Claim | Value | Methodology | Artifact | Date |
|-------|-------|-------------|----------|------|
| SciFact MiniLM→bge-large nDCG@10 retention | 0.871 ± 0.006 (K=4000, Ridge, 3 seeds) | BEIR benchmark, SciFact dataset | `benchmarks/results/beir_scifact_*__ridge__k4000__seed{0,1,2}__*.json` | 2026-07-21 |
| LowRank adapter nDCG@10 retention | 0.857 ± 0.009 (K=4000, 3 seeds) | Same | `benchmarks/results/beir_scifact_*__lowrank__k4000__seed{0,1,2}__*.json` | 2026-07-21 |
| MLP adapter nDCG@10 retention | 0.727 ± 0.007 (K=4000, 3 seeds) | Same | `benchmarks/results/beir_scifact_*__mlp__k4000__seed{0,1,2}__*.json` | 2026-07-21 |
| K-sweep (ridge only) | K=500: 0.704±0.004, K=1000: 0.781±0.008, K=2000: 0.818±0.007, K=4000: 0.871±0.006 | Same | `benchmarks/results/beir_scifact_*__ridge__k{500,1000,2000,4000}__*.json` | 2026-07-21 |
| Floor nDCG@10 = 0.0 when dims differ | 0.0 | Raw cross-space vectors cannot be queried when dims differ | Same | 2026-07-21 |
| Ceiling (full re-embed) nDCG@10 | ≈ 0.735 | Quality upper bound for this model pair | Same | 2026-07-21 |
| Same-dim pair (bge→e5, 1024→1024) | floor=0.0, Isotrieve≈0.667, ceiling=0.722, retention=0.923±0.010 | Same with e5 prefixes | `benchmarks/results/beir_scifact_*__BAAI_bge-large-en-v1.5__to__intfloat_e5-large-v2__ridge__k2000__seed{0,1,2}__*.json` | 2026-07-21 |

### Gate Model

| Claim | Value | Methodology | Artifact | Date |
|-------|-------|-------------|----------|------|
| Gate model trained on local pairs only | No API model pairs | Gate model valid for local model pairs; API pair performance may differ | `src/isotrieve/quality/gate_model_v1.json` | — |

### Working Sessions (WS-A through WS-E)

| Claim | Value | Methodology | Artifact | Date |
|-------|-------|-------------|----------|------|
| WS-A: bge→e5 raw scores agree at 100% for τ≤0.75 | +4.7% agreement at τ=0.80 with recalibration | MAE=0.095, margin compression=0.83x | `benchmarks/results/ws_a_bge_to_e5_recall_tables.json` | — |
| WS-A: MiniLM→bge rectangular pair | Raw scores severely compressed (mean 0.157 vs ceiling 0.521, MAE=0.364) | Recalibration essential | `benchmarks/results/ws_a_minilm_to_bge_recall_tables.json` | — |
| WS-B: Confidence flags predictive | bge→e5 high=0.955/low=0.637; MiniLM→bge high=0.875/low=0.651 | 50-56ms/query latency | `benchmarks/results/ws_b_confidence_flags.json` | — |
| WS-B: Cross-encoder reranking NULL RESULT | -10.7pts bge→e5, -9.8pts MiniLM→bge | MS MARCO domain-mismatched for sci-text | `benchmarks/results/ws_b_cross_encoder.json` | — |
| WS-C: Independent inverse-α | +2.17pts (bge→e5), +2.23pts (MiniLM→bge) | Consistent across pairs | `benchmarks/results/ws_c_independent_inv_alpha.json` | — |
| WS-C: TSVD shrinkage NULL RESULT | rank=512 only -0.33pt | Not worth complexity | `benchmarks/results/ws_c_tsvd_shrinkage.json` | — |
| WS-E: Rectangular pair re-validation | MiniLM→bge: 86% retention, margin compression 0.85x | Independent inverse-α +2.23pts | `benchmarks/results/ws_e_minilm_to_bge_revalidation.json` | — |
| WS-D: Gate v3 margin compression | <0.85 widens prediction interval | `_predict_retention()` accepts `margin_compression` parameter | `src/isotrieve/quality/gate_model_v1.json` | — |

## Retired / deleted claims (must not reappear)

These appeared in prior docs without reproducible artifacts and were removed:

- "97% semantic fidelity" / "97.2%" / "97.35%"
- "<10ms" / "<1ms" transfer latency as a product claim
- "85% Top-1" abort threshold as a validated number
- "86% corpus fidelity" / "43% text baseline"
- "150x faster" / "200x"
- "Validated on 300k vocab, zero overfitting"
- Ridge ">90%" / future MLP ">99%" as stated facts
- "100% cross-domain retention" (was synthetic artifact, not real benchmark)

Replacement rule: only cite numbers produced by `benchmarks/run_benchmark.py`
and stored under `benchmarks/results/`.
