# AECP Design Decisions

Running log of design choices and open questions.
Bias toward the boring, verifiable choice.

## Postmortems

### 2026-07-19 — Number Discrepancy Postmortem (WS-0)

**Symptom:** Adapter sweep reported Ridge/K=4000/SciFact = 0.866 ± 0.008. K-sweep reported the same nominal config = 0.814 ± 0.068. The ±0.068 (8× larger std) flagged a discrepancy.

**Root cause:** The K-sweep summary (`run_benchmark.py` line 538) filtered by `k_cal` but **not by adapter**. When running `--adapter ridge lowrank mlp --k 500 1000 2000 4000`, the K-sweep summary averaged across ALL adapters for each K value:
- ridge K=4000: 0.866 ± 0.008
- lowrank K=4000: 0.857 ± 0.009
- mlp K=4000: 0.719 ± 0.008
- **All-adapter average at K=4000: 0.814 ± 0.068** ← this was the K-sweep number

The adapter sweep was adapter-specific (ridge only). Two valid but different experiments, presented in a way that implied they should match.

**Resolution:**
1. K-sweep summary now broken down by adapter (`run_benchmark.py` updated).
2. `protocol` field added to result JSON (`fixed_calibration_per_seed`).
3. Audit script `benchmarks/audit_configs.py` committed for field-by-field comparison.
4. CLAIMS.md and DECISIONS.md updated to distinguish adapter-specific vs cross-adapter numbers.

**Lesson:** Always declare which dimension (adapter, K, seed) a summary averages over. The `protocol` field in result JSONs makes this explicit.

## Locked decisions

### 2026-07-19 — Package strategy (Phase 1)

**Choice:** Rewrite `aecp-python` onto a `src/aecp/` layout as the migration toolkit.
The old agent-protocol API under the flat `aecp/` package is excluded from the
build and deferred to a later "advanced usage" docs page. TypeScript (`@aecp/core`)
is out of scope for v1.

**Why:** The product contract repositions AECP as embedding-space migration, not
agent handoff. One excellent Python package beats two mediocre packages. The PyPI
name `aecp` must not collide with a second package.

### 2026-07-19 — Version reset

**Choice:** Ship as `0.1.0` (alpha), not continue from marketed `1.0.0`.

**Why:** Public claims from the old package are unverified. Starting at 0.1.0
signals that numbers are earned via the benchmark harness.

### 2026-07-19 — License

**Choice:** Apache-2.0 (per product contract). Previous MIT license in this
directory is superseded for the new package.

### 2026-07-19 — Mapping defaults

**Choice:** `RidgeMapping` is primary; `OrthogonalProcrustesMapping` only when
`d_src == d_tgt`. All `transform` outputs are L2-normalized. Ridge uses an
augmented bias column and `alpha="auto"` via leave-one-out GCV over a log grid
(`sklearn.linear_model.RidgeCV`).

**Why:** Rectangular maps (1536→3072) are the normal case; Procrustes cannot
handle them. Normalization matches retrieval practice and makes cosine metrics
stable.

### 2026-07-19 — No wall-clock expiry on `.aecp` files

**Choice:** Header includes `expires_hint` for operator convenience but the
library never auto-expires. Drift detection is `aecp verify` (fresh probe embeds).

**Why:** Model weights do not drift on a calendar; silent provider updates do.

### 2026-07-19 — Phase 1 scope

**Choice:** Implement core math, quality metrics, numpy store, minimal file CLI,
offline tests, and a local-model SciFact harness before any API providers or
vector-DB adapters.

### 2026-07-19 — Phase 1 gate result (first honest number)

**Measured** on SciFact test (300 queries with qrels), in-domain K=4000,
`all-MiniLM-L6-v2` → `BAAI/bge-large-en-v1.5`, 3 seeds:

| | nDCG@10 |
|--|---------|
| Floor (raw cross-space; dims 384≠1024) | 0.000 |
| AECP (mapped) | ≈ 0.640 |
| Ceiling (full re-embed) | ≈ 0.735 |
| **Retention (AECP÷ceiling)** | **0.871 ± 0.006** |

Artifacts under `benchmarks/results/`. Rank warning: MiniLM source matrix was
slightly rank-deficient (382/384) at K=4000 — logged, not hidden.

**Gate status:** PASSED — AECP retention meaningfully above floor.

### 2026-07-19 — Phase 1 prerequisite fixes (before Phase 2)

Inferred from Phase 1 debt + contaminated artifacts (PM referenced "four fixes
above" without the prior message in-thread). Logged here for audit:

| Fix | Action |
|-----|--------|
| F1 | Purge `self_retrieval_fallback` result JSONs; harness **requires** real BEIR qrels for claimable runs — no silent fallback |
| F2 | Record **holdout-vs-corpus optimism gap** (fit holdout top-1 − retrieval retention) in every result + QualityGate |
| F3 | **CachedEmbedder mandatory** on all provider/CLI/benchmark embed paths — never embed the same text twice |
| F4 | Gate/corpus sample must be **disjoint from calibration**; rank-deficiency stays a warning (ridge handles it) |

### 2026-07-19 — Phase 2 adapter zoo

**Choice:** Add 4 adapters to the existing Ridge: Procrustes (square only), Procrustes+Diagonal, LowRankAffine, ResidualMLP. Ridge remains the default for rectangular mappings.

**Why:** Drift-Adapter (EMNLP 2025) validates Procrustes and MLP approaches. LowRankAffine provides compressed matrices. Each adapter has different compute/memory/quality tradeoffs.

**Measured (SciFact, all-MiniLM-L6-v2 → bge-large-en-v1.5, K=4000, 3 seeds):**

| Adapter | nDCG@10 retention | Notes |
|---------|------------------|-------|
| Ridge | 0.866 ± 0.008 | Default. Fast, stable. |
| LowRank | 0.857 ± 0.009 | Compressed matrix. ~1% worse than Ridge. |
| MLP | 0.719 ± 0.008 | No hyperparameter tuning. Significantly worse than linear. |

### 2026-07-19 — Phase 2 K-sweep (data-driven gate thresholds)

**Choice:** Benchmark across K=500,1000,2000,4000 to derive QualityGate thresholds from data.

**Measured (SciFact, all adapters averaged, 3 seeds):**

| K | nDCG@10 retention | Notes |
|---|------------------|-------|
| 500 | 0.667 ± 0.039 | Below recommended minimum. Rank-deficient. |
| 1000 | 0.732 ± 0.056 | Still below minimum. Better but high variance. |
| 2000 | 0.788 ± 0.054 | Approaching acceptable quality. |
| 4000 | 0.814 ± 0.068 | Full quality. Recommended minimum. |

**Gate thresholds derived:**
- PASS: retention ≥ 0.75 (K≥2000 typically achieves this)
- WARN: 0.55 ≤ retention < 0.75 (K=500-1000 range, or poor mapping)
- FAIL: retention < 0.55 (mapping fundamentally broken)

### 2026-07-19 — K minimum constraint downgraded to warning

**Choice:** Change adapter `fit()` from raising `ValueError` on small K to issuing `UserWarning`. The 10×min_dim rule is a guideline, not a hard error.

**Why:** Users need to see what happens with small K. The benchmark harness needs to sweep across K values for threshold calibration. Warning gives visibility without blocking experimentation.

### 2026-07-19 — MLP rectangular dims

**Choice:** ResidualMLP uses plain MLP (no residual skip) when d_src ≠ d_tgt. Residual skip only when dimensions match.

**Why:** The residual connection `x + net(x)` requires matching input/output dims. Rectangular projections need a feedforward architecture.

### 2026-07-19 — QualityGate v2: proxy-based prediction (not raw holdout)

**Choice:** Redefine gate thresholds from "requires ceiling measurement" to "predicts retention from holdout proxy using isotonic regression." Gate model ships with the package (`gate_model_v1.json`). Gate verdict uses prediction intervals.

**Why:** The old gate required ceiling quality (full re-embed) to measure retention — defeating the purpose of a migration tool. The new gate predicts retention from holdout proxies that are always available: `top1_retention`, `holdout_rank_corr`, `cosine_mean`. Trained on 48 benchmark results (4 adapter pairs), LOPO CV: MAE=0.0443, RMSE=0.0592, 80% interval half-width=0.0706.

**Artifacts:** `gate_model_v1.json`, `benchmarks/fit_gate_model.py`, `benchmarks/results/gate_lopo.json`

### 2026-07-19 — Same-dimension floor is 0.0

**Choice:** Measure bge-large→e5-large (1024→1024) raw cross-space retrieval. Floor = 0.0.

**Why:** "Same dimension" is not "same space." Different models produce geometrically incompatible embeddings even at the same dimension. Raw cross-space nDCG@10 = 0.0 — search runs fine, results are garbage. This is the core problem AECP solves. Retention = 0.908 with mapping (with e5 prefixes).

### 2026-07-19 — e5 prefix bug (postmortem)

**Symptom:** Same-dim pair (bge-large→e5-large) showed ceiling nDCG@10 = 0.355, retention = 95%. e5-large alone scores ~0.74 on SciFact — ceiling was clearly broken.

**Root cause:** e5 models require `"query: "` / `"passage: "` text prefixes for correct embeddings. The benchmark harness passed raw texts with no prefix handling. Both documents and queries were embedded without prefixes, degrading e5's retrieval quality by ~50%.

**Resolution:** Added `MODEL_PREFIXES` dict to `run_benchmark.py` with model-specific prefix mappings. Modified `embed_texts()` and `load_or_embed()` to apply correct prefixes. Cache invalidated via `prefix_version` field.

**Corrected results:**
- Before fix: ceiling=0.355, AECP=0.337, retention=0.950 (broken)
- After fix: ceiling=0.722, AECP=0.656, retention=0.908 (honest)

**Lesson:** Model-specific preprocessing (prefixes, tokenization quirks) must be part of the embedding pipeline, not assumed to be "just encode text." When a number looks too good (95% retention), verify the ceiling is realistic.

### 2026-07-19 — Bidirectional evaluation

**Choice:** Evaluate both directions: source→target (transform corpus) AND target→source (map queries). Report "query→legacy retention" = 86.2% of ceiling.

**Why:** Serve-mode quality is measured by mapping queries into legacy space (query→legacy), not by transforming corpus (legacy→target). Both directions matter for the "before/after" narrative.

## Open questions

None for Phase 1 gates. Questions that affect later phases:

| ID | Question | Affects | Status |
|----|----------|---------|--------|
| Q1 | Exact QualityGate PASS/WARN/FAIL thresholds | Phase 2 | **Resolved** — data-driven from K-sweep. See thresholds.json. |
| Q2 | Frozen calibration corpus composition (`aecp-calib-v1`) | Phase 2 | Open — need permissive sources + checksum |
| Q3 | Fate of legacy `aecp/` protocol modules and `aecp-npm` | Phase 4 | Open — archive vs delete |
| Q4 | PyPI publish account / GitHub org URLs | Phase 4 | Open — placeholders remain |
| Q5 | Procrustes adapters for square pairs | Phase 2 | Open — only works when d_src == d_tgt |
| Q6 | MLP hyperparameter tuning (epochs, lr, hidden_dim) | Phase 3 | Open — current defaults unoptimized |

## PM sign-off

- [ ] Results table reviewed
- [ ] "When NOT to use AECP" section reviewed
- [ ] Sign-off date: _
