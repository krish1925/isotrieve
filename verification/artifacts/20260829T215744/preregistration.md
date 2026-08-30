# Pre-registration — Isotrieve Test Barrage 2026-08-29

Committed BEFORE any results exist. Thresholds/seeds/counts below are binding for the whole run.
Criterion changes after this commit = `criteria-defect` issue + escalation (master instructions §12).

Global: BASE_SHA=51c3538 (branch verify/test-barrage-20260829; parent 7cc21c9 = development HEAD named by the plan).
Seeds fixed: property/bias tests seed=20260829 unless a check pins its own. Flakiness repeats: N=3. CI nominal level for P5-05 read from source: 80% (P10/P90).

## Env constraints discovered in P0 (recorded before results)
- Disk: 4.2 GB free after cleaning package caches (uv/pip/npm, 3.6 GB freed). 118 GB HF cache = user data, untouched; all 3 required models already warm.
- Consequence: only lanes 3.10 + 3.13 run locally. Lanes 3.11/3.12 = SKIPPED(env: disk-exhaustion; CI runs them on every push). P8 1M×384 (1.54 GB) still fits.
- Docker daemon ABSENT despite restart attempt → P7 store-server subchecks pre-marked BLOCKED; Docker-free P7 items (Chroma lifecycle, CLI sweep, serve-mode, MPS) still run.

## P1 — matrix / flakiness / pollution
- P1-01 lanes: PASS iff 0 failed on 3.13-full(incl slow+benchmark), 3.10 (CI parity), npm 268+6. Falsifier: any nonzero failed/errored, or pytest exit 5.
- P1-02 flakiness: 3× full 3.13 suite; PASS iff identical pass/fail sets across runs AND no test fails in any repeat. Falsifier: any test failing in ≥1/3, or pass-set diff.
- P1-03 pollution: 5 largest files isolated + reverse file order + shuffled single-file-group (seed 20260829). PASS iff isolated results == in-suite results per file. Falsifier: any file behaving differently alone vs in-suite.
- P1-04 npm coverage+tsc: record coverage; PASS iff tsc clean both packages. Falsifier: tsc error.

## P2 — packaging
- P2-01 build: sdist+wheel build, `twine check` PASS. Falsifier: any check failure.
- P2-02 contents: wheel AND sdist contain gate_model_v1.json, thresholds.json/yaml, py.typed, calibration corpus, LICENSE, CLAIMS.md. Falsifier: any missing file.
- P2-03 clean-room: fresh venv (repo NOT on sys.path, cwd /tmp), import resolves under site-packages, CLI `version`/`--help` work, test_release.py green against installed pkg. Falsifier: source-tree import, missing entrypoint, any release-test failure.

## P3 — cross-language parity (tolerances fixed now)
- P3-01 PY→TS golden: TS loads golden_ridge.isotrieve, transforms 10 golden inputs, allclose atol=1e-6 (float64 pipeline). Falsifier: any element off by >1e-6.
- P3-02 TS→PY roundtrip: TS-written binary loads in Python; matrix matches TS-declared values atol=1e-12 (byte-level transport, no arithmetic). Falsifier: load error or mismatch.
- P3-03 tamper matrix: payload-flip → reject; header-flip → reject; untampered → accept. BOTH languages. Falsifier: any cell wrong (incl. accepting a tampered file = critical).
- P3-04 shared gate asset: sha256(gate_model_v1.json) identical resolved from both packages; Python _predict_retention deterministic (2 calls equal). Falsifier: hash mismatch or nondeterminism. (Prediction-logic parity itself is covered by existing cross-compat.test.ts; limitation noted in report.)

## P4 — property invariants (hypothesis seed=20260829, min 50 examples/property; count recorded)
- P4-01 L2: every RidgeMapping.transform output row unit-norm 1e-9. NC: stub returning non-normalized rows must trip the checker.
- P4-02 Procrustes: WᵀW≈I atol 1e-8 (square dims). NC: scaled-W stub trips checker.
- P4-03 ridge residual ≤ random-W residual on calibration data. NC: n/a (statistical; 50 min examples recorded).
- P4-04 inverse round-trip ‖M⁻¹M(x)−x‖ ≤ 1e-4 (rel L2, well-conditioned synthetic). NC: perturbed-inverse stub trips checker.
- P4-05 determinism: same-seed fits byte-identical files; alpha identical. Falsifier: byte diff.
- P4-06 CRC: EVERY single-byte position flip rejected (small file, exhaustive loop). NC: untampered accepted.
- P4-07 edges: batch-of-1; k=10×min_dim minimal fit; NaN/Inf input → typed error, never silent NaN out. Falsifier: silent NaN propagation (would be a product finding).
- P4-08 cross-process determinism: output hashes equal across 2 subprocesses with PYTHONHASHSEED fixed AND unset. Falsifier: hash differs under either setting.

## P5 — bias/fairness audit (all 11; every item has a negative control or explicit reason why n/a)
- P5-01 leakage/disjointness: (a) source-verified: cal_idx derives ONLY from doc_ids+seed (quoted lines archived); (b) dynamic: run_seed's sampling reproduced via inspect.getsource exec on synthetic docs → same cal_idx for same seed, twice; (c) measured overlap of calibration docs with scifact qrels-relevant docs (k=200 sample) — REPORTED metric; PASS iff (a)+(b) hold. Falsifier: query data in sampling, or nondeterministic cal_idx, or (c) reveals calibration drawn from a different corpus than evaluated.
- P5-02 LOPO gate revalidation: benchmarks/fit_gate_model.py lopo_cross_validation rerun on committed results; report per-fold |pred−actual|; PASS iff completes with ≥2 pairs AND mean MAE ≤ 0.20 AND fold loop verifiably excludes held-out pair (source quoted). Falsifier: MAE>0.20 or fold leakage.
- P5-03 alpha isolation: spy records every array passed to sklearn RidgeCV.fit during RidgeMapping(alpha='auto').fit(Xcal,Ycal) with sentinel rows present in Xeval; PASS iff no sentinel row among recorded arrays. Report chosen alpha across 5 different calibration subsets. Falsifier: sentinel in recorded arrays; alpha constant across drastically different subsets = investigate-note.
- P5-04 recalibrator split: fit_from_holdout on synthetic; assert holdout rows excluded from fitted quantile support (checked via report/bounds) — need source sig (captured). Falsifier: holdout inside fitted support.
- P5-05 bootstrap CI coverage: stub mapping with per-query true retention r∈{0.5,0.7,0.9}; 400 synthetic universes each, n=200 queries, _bootstrap_retention_ci n_resamples=200 seed=20260829; empirical coverage of TRUE r ∈ [75%,85%] per r; report MC stderr. Falsifier: coverage outside band for any r (either direction).
- P5-06 seed-sensitivity honesty: gate.seed_sensitivity fixed-seed determinism (2 runs identical), internal consistency (std == np.std(per_seed) within 1e-12; mean/min/max bracket), full spread reported. Falsifier: nondeterministic report or inconsistent aggregates.
- P5-07 margin compression: synthetic mapped=Y×c for c∈{0.6,0.8} → detected ratio within [c−0.05, c+0.05] AND interval widens (predicted retention drops via _predict_retention); L2-normalized non-degenerate control → not None, not ≈0/1. Falsifier: undetected compression or bug-#8 recurrence (None/0 on valid input).
- P5-08 averaging declaration: all results/*.json have protocol+config_hash+commit; audit_configs.py exit 0; grep summaries for cross-adapter means. Falsifier: any artifact missing fields, or audit failure.
- P5-09 truncation fairness: source-verified floor/ceiling/mapped consume the SAME doc_ids/queries objects in run_seed (quoted); dynamic arms-hash check if runnable, else INCONCLUSIVE with reason. Falsifier: arms built from different slices.
- P5-10 fit-loop purity: grep MLP/Contrastive fit paths for eval-set usage/early stopping; PASS iff none found (evidence archived). Falsifier: any eval reference inside fit.
- P5-11 probe purity: source-verified probe_chunks/probe_queries same-index top-k without eval-doc target embeddings in scoring (quoted lines). Falsifier: scoring touches corpus embeddings.

## P6 — claims reproduction (report-only; ±0.02 abs (±0.03 same-dim) AND ordering preservation; seeds/configs = committed artifacts; device omitted (CPU) as committed; embed cache warm)
- P6-01 ridge k4000 s0,1,2 scifact full vs 0.871±0.006
- P6-02 lowrank k4000 s0,1,2 vs 0.857±0.009
- P6-03 mlp k4000 s0,1,2 vs 0.727±0.007 (LAST; per-run time-box 20 min → else mark TIME-BOXED, qualitative)
- P6-04 bge→e5 k2000 s0,1,2 vs 0.923±0.010 (tol ±0.03)
- P6-05 scifact ridge+lowrank k500 s0,1 max-docs 2000 vs 0.813/0.810
- P6-06 fiqa ridge+lowrank k500 s0,1 max-docs 2000 vs 0.773/0.768
- P6-07 probe code+legal lowrank/ridge k500 s0,1 vs 1.022/1.037
- P6-08 K-sweep ridge 500/1000/2000 s0 (qualitative monotonicity vs 0.704/0.781/0.818)
- Orderings: ridge>lowrank>mlp@k4000; same-dim>rectangular; K↑ monotone. Falsifier: any config outside tolerance or any ordering violated → `claims-discrepancy` issue, CLAIMS.md untouched.

## P7 — E2E (Docker-dependent parts BLOCKED if daemon stays down)
- P7-01 pgvector: integration suite + kill/resume idempotency + rollback (BLOCKED if no Docker)
- P7-02 Qdrant server 10k + scroll boundaries {1,63,64,65} (BLOCKED if no Docker)
- P7-03 Chroma real-model lifecycle: calibrate→transform→gate(PASS,exit0)→manifest→verify→rollback; absurd threshold 0.99 → nonzero exit. Falsifier: wrong exit code either polarity.
- P7-04 CLI sweep: all 11 verbs --help exit 0; lifecycle verbs happy-path.
- P7-05 serve-mode equivalence: Recall@10 batch vs query-time within 0.02 on 50 queries.
- P7-06 MPS: device=auto selects MPS; MPS vs CPU atol 1e-4; 2 MPS runs identical. Falsifier: device≠mps or parity breach.

## P8 — memory/perf (SOLO; solitude ps-check archived)
- P8-01 streaming: 200k×768 and 1M×384 batched migrate; peak RSS < 40% of corpus bytes (computed in artifact: 614.4 MB / 1536 MB). Two measurement methods; >15% disagreement = finding. Falsifier: RSS ≥ 40% budget or method disagreement.
- P8-02 bootstrap timing: median-of-5, min/max, 1000 resamples on 10k pairs. Record-only.
- P8-03 npm stress suite green. Record duration.

## P9 — stretch (first cut if needed)
- P9-01 pip-audit + npm audit (record-only). P9-02 binary-reader fuzz 1000 examples (seed 20260829): clean typed errors only; crash = finding. P9-03 mutation testing: CUT (disk+time budget; recorded).
