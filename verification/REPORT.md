# Isotrieve Test Barrage — REPORT (2026-08-29)

## 1. Verdict

**Ship with named caveats.** The benchmark methodology reproduces independently (8/8 committed claim configs, Δ ≤ 0.013), no train/eval leakage was found in the sampling path, packaging is release-clean, and crash/resume is genuinely idempotent. Five real product defects were found and filed (#85 binary format TS→PY broken, #86 gate bootstrap CIs biased low by duplicate ties, #87 legacy artifacts lack declared config, #88 `device="auto"` crashes + margin-compression degenerates for good migrations, **#90 batched migration does not stream — peak RSS 6.8× corpus**). None blocks the v0.4.0 release mechanically; #86/#88 undercut the *fairness* of the gate's PASS/WARN boundaries, and #90 makes large-store migrations OOM-prone and must be fixed before anyone points Isotrieve at a production-size store.

## 2. Environment fingerprint

```
base_sha=51c3538 (verify/test-barrage-20260829; parent 7cc21c9 = development HEAD)
macOS (Apple M4 Pro, 24 GB RAM, 12 cores) · Python 3.13.5 + 3.10.19 lanes · node v24.2.0 · uv
Docker: ABSENT (daemon never started) → P7-01/02 BLOCKED
Disk: exhaustively constrained (~1–4 GB free throughout); package caches cleaned; 118 GB HF user cache untouched
Models warm from cache: all-MiniLM-L6-v2 (384d), bge-large-en-v1.5 (1024d), e5-large-v2 (1024d); HF_HUB_OFFLINE=1 during lanes
```

## 3. Scorecard

| Phase | Checks | PASS | FAIL (real) | FAIL (test-bug, fixed) | BLOCKED/SKIPPED |
|---|---|---|---|---|---|
| P0 env | — | ✅ | 0 | 0 | 3.11/3.12 lanes SKIPPED (disk; CI covers) |
| P1 matrix | 4 lanes + npm | ✅ | 4 (→ findings) | 3 (hypothesis missing, flag) | — |
| P2 packaging | build/twine/contents/clean-room | 4/4 | 0 | 0 | — |
| P3 parity | bidirectional + tamper + gate asset | 6/7 | 1 → #85 | 2 (API misuse) | — |
| P4 properties | 13 hypothesis tests + NCs | 13/13 | 0 | 3 (criteria: inverse semantics, fit_date, conditioning) | — |
| P5 bias audit | 11 items | 9/11 | 2 → #86, #87 | 4 (qrels API, dedent, classmethod, construction) | — |
| P6 claims repro | 8 configs, 21 runs | **8/8** | 0 | 0 | MLP time-boxed (ran 12 min, OK) |
| P7 E2E | 9 checks | 7/9 | 1 (04b → #88 evidence) | 5 (fixture/API) | 2 BLOCKED (Docker) |
| P8 memory | 2 sizes × 2 methods + delta isolation | measured | 1 → **#90** | 1 (time parsing) | — |
| P9 stretch | audits + fuzz | done | 0 | — | mutation CUT (budget) |

### Supplement (post-disk-free continuation, same day)

| Item | Result |
|---|---|
| P8 full run (space restored) | **FAIL → #90**: 200k×768 corpus 614 MB → migration peak RSS **4.57 GB** (delta 4.16 GB = 6.8× corpus, 17× the 40% streaming budget); 1M×384 → 7.73 GB. Methods agree <1%. Root cause: `write_vectors` per-batch full reload + float64 upcast + full rewrite |
| P7-01/02 Docker E2E | still BLOCKED — Docker Desktop daemon refuses to start **even with 126 GB free** (broken install; machine-level, not touched) |
| Python lanes 3.11/3.12 + 3.10 re-run | **244 passed each**, same 4 intended findings, 0 unexpected; property file 13/13 on every version |
| P9-02 binary-reader fuzz | **1000 hypothesis examples**: 0 corrupted inputs accepted, no crashes, all failures typed errors — PASS |
| P9-01 npm audit | informational: moderate advisories in transitive deps (`@aws-sdk/*` via @xenova/transformers) — recorded in artifacts |

## 4. Findings (severity order)

| # | Severity | Symptom | Issue | Bucket | Status |
|---|---|---|---|---|---|
| F1 | HIGH | Gate bootstrap CIs biased low (resample duplicate ties break self-match); empirical coverage 0% vs nominal 80% | #86 | product-bug | deferred |
| F2 | HIGH | Cross-runtime binary format broken TS→PY (camelCase vs snake_case header schema); "cross-compatible" claim is one-directional | #85 | product-bug | deferred |
| F3 | HIGH | `device="auto"` crashes (unreachable auto-detect branch); `margin_compression` degenerates → ~0 for excellent mappings → spurious WARN with [0.34, 1.0] interval — fires on real K=2000 migrations (top1=0.985 → WARN) | #88 | product-bug | deferred |
| F4 | MEDIUM | Legacy benchmark artifacts (K-sweep, adapter-sweep era) lack `protocol`/`config_hash`/`commit` — postmortem discipline is not retroactive | #87 | product-bug (process) | deferred |
| F5 | LOW | Gate does not enforce its own documented contract ("evaluation pairs must not overlap calibration") — silently yields pessimistic verdicts | (noted in #88 comment thread) | dx/product | deferred |
| F6 | LOW | npm root `lint` script misconfigured (no ESLint config anywhere); pre-existing | (report note) | env | — |

## 5. Phase 5 — bias/fairness audit (full detail)

| Item | Result | Evidence (observed) |
|---|---|---|
| 01 calibration/eval disjointness | PASS | cal sampling derives only from `doc_ids`+seed (source-quoted); deterministic per seed (exec-reproduced); measured calib-vs-qrels overlap **0/4→re-run 0/100 sampled** (scifact) — reported as methodology metric |
| 02 gate LOPO revalidation | PASS | re-run LOPO on committed results: per-fold \|pred−actual\| = 0.080/0.081/0.162/0.029; mean MAE **0.0882 ≤ 0.20**; fold-exclusion verified in source |
| 03 alpha-selection isolation | PASS | solver spy saw only calibration arrays (no eval sentinels); alphas across 5 data scales: 10 / 1000 / 0.1 / 1000 / 0.001 (not degenerate) |
| 04 recalibrator split | PASS (note) | `fit_from_holdout` is a classmethod fitting on derived pairs; report lacks explicit split counts (source-verified only) |
| 05 bootstrap CI coverage | **FAIL → #86** | coverage 0.0000 ± 0.0000 at r∈{0.5,0.7,0.9} (nominal 80%); duplicate-tie mechanism proven; negative control n/a (result itself is the finding) |
| 06 seed-sensitivity honesty | PASS (note) | deterministic at fixed seed; std == np.std(per_seed); spread min/max bracket mean; synthetic data saturated (std=0) — mechanism verified in source (`_run(s)` permutes by seed) |
| 07 margin compression | PASS | variance-ratio responds monotonically to injected noise (0.13 @ eps 0.2 → 0.56 @ eps 0.4); penalization widens interval; bug-#8 regression (None/0 on valid input) does not recur |
| 08 averaging declaration | **FAIL → #87** | all artifacts carry `commit`+`config_hash` except legacy K-sweep/adapter-sweep files missing `protocol` |
| 09 truncation fairness | PASS | floor/ceiling/mapped consume the same doc arrays inside `run_seed` (source-verified) |
| 10 fit-loop purity | PASS | no eval/qrels/early-stopping references in MLP/contrastive fit modules |
| 11 probe purity | PASS | probes embedded per-model, scored same-index (source-verified; `PROBE_VERSION` cache) |

## 6. Phase 6 — claims reproduction (report-only; CLAIMS.md untouched)

| Config | Committed | Reproduced | Δ | Verdict |
|---|---|---|---|---|
| SciFact MiniLM→bge ridge K=4000 (3 seeds) | 0.871 ± 0.006 | 0.8714 ± 0.0063 | +0.0004 | ✅ |
| same, LowRank | 0.857 ± 0.009 | 0.8573 ± 0.0046 | +0.0003 | ✅ |
| same, MLP | 0.727 ± 0.007 | 0.7141 ± 0.0137 | −0.013 | ✅ (within ±0.02) |
| bge→e5 same-dim ridge K=2000 | 0.923 ± 0.010 | 0.9230 ± 0.0103 | +0.0000 | ✅ |
| SciFact K=500 ridge / LowRank (max-docs 2000) | 0.813 / 0.810 | 0.8126 / 0.8103 | −0.0004/+0.0003 | ✅ |
| FiQA K=500 ridge / LowRank (max-docs 2000) | 0.773 / 0.768 | 0.7727 / 0.7678 | −0.0003/−0.0002 | ✅ |
| Probe corpora code / legal | 1.022 / 1.037 | 1.0222 / 1.0367 | +0.0002/−0.0003 | ✅ |
| K-sweep ridge 500/1000/2000 | 0.704/0.781/0.818 | 0.700/0.778/0.815 | −0.004 each | ✅ monotone |

**Orderings preserved:** ridge > LowRank > MLP; same-dim > rectangular; K↑ monotone. Hardware/library drift ≤ 0.013 everywhere.

## 7. What was NOT tested

- **claims-lint artifact gap** (WS-A/B/C/E rows): explicitly out of scope per owner decision — not examined, not fixed, not filed.
- **pgvector / Qdrant-server E2E** (P7-01/02): BLOCKED — Docker daemon never started on this machine (CI covers both).
- **1M×384 streaming memory measurement** (P8): BLOCKED — disk exhausted mid-write (ENOSPC); partial peak-RSS captured (1.97 GB @ 200k×768 incl. interpreter+model imports) but the <40%-of-corpus streaming criterion could not be validly measured.
- **Python lanes 3.11/3.12**: SKIPPED locally (disk); CI runs them on every push.
- **Mutation testing**: CUT (disk + time budget).
- **API-key providers** (OpenAI/Voyage/Cohere/Gemini/Pinecone cloud): mocked/skipped by design.
- **npm ESLint**: pre-existing misconfig noted, not fixed.

## 8. Threats to validity of this audit

1. **Pre-registration corrections were required mid-run** (all documented): P4-04's inverse invariant was mis-specified twice (L2-normalization discards scale by design; d_src>d_tgt maps have a √(d_tgt/d_src) projection bound) — final form is direction-cos with an explicit projective bound; P4-05's byte-identity was unattainable by design (`fit_date` in header) — corrected to payload-bit-identity + header-modulo-date; P5-07's construction was rebuilt against the metric's real semantics (variance ratio, not mean margins); P7-06 dropped cross-device weight parity (invalid for independently trained nets). Each correction tightened or scoped the claim; none relaxed a threshold to convert a real failure into a pass. Review the diff in `P*/preregistration.md` vs this report.
2. **The P5-05 coverage simulation** uses an idealized orthonormal stub; real embeddings produce the same exact-tie mechanism on resampled duplicates, but the magnitude on real data is inferred, not measured end-to-end.
3. **P6 reproduction ran on a warm embedding cache** generated partly on this machine; the committed artifacts were produced on CI Linux. Identical results to 3–4 decimals across that hardware gap is the strongest available evidence the harness is deterministic, but cache provenance means "same bytes in" is not strictly guaranteed.
4. **P7-04b never achieved a PASS exit-0 arm** at any scale (K=100 synthetic, K=2000 real) — the WARN-at-top1-0.985 behavior is attributed to #88, but a world where the gate simply never passes at defaults cannot be excluded without fixing #88 first.
5. **Single machine, single operator, no independent replication** of this audit itself.

## 9. Pre-registration diff

- P4-04: criterion rewritten (vector identity → direction-cos with projective bound). Justification: normalization + rectangular projection make the original criterion mathematically unsatisfiable; falsifier (perturbed inverse detected) unchanged.
- P4-05: byte-identity → payload-bit-identity + header-equality-modulo-`fit_date`. Justification: format embeds fit time by design; recorded as criteria-defect.
- P5-07: construction changed (mean-shrink → noise-injection) to match the metric's actual variance-ratio semantics. Falsifier unchanged.
- P7-06: cross-device parity → device resolution + same-device determinism. Justification: independent training runs are not expected to converge to identical weights.
- P8: 1M×384 marked BLOCKED (ENOSPC), not silently dropped.
- No threshold was changed after observing a failing result. Two criteria were *narrowed* (P4-04 scope, P4-05 semantics) — both before their final runs, both documented above.
