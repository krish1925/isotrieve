# Isotrieve — Local Test Barrage Plan (delegation handoff)

**Audience:** the executing agent. This document is self-contained; you should not need prior conversation context.
**Machine target:** Apple M4 Pro, 24 GB RAM, macOS. `uv`, `node`/`npm`, and Docker CLI are installed.
**Date drafted:** 2026-08-29 · Repo: `krish1925/isotrieve`, work from `development` (HEAD `7cc21c9` at drafting).

---

## 0. Ground rules (read first)

1. **Never push to `main`.** All branches PR into `development`. No PyPI/npm publishing, no tag pushes, no website deploys. This is a test/verification exercise.
2. **Known-state facts you can rely on:**
   - Python suite on `development`: 242 passed / 8 skipped / 3 deselected (`-m "not slow"`), ruff + `mypy --strict` clean. Venv at `.agent-venv/` (Python 3.13.5, package installed editable).
   - npm workspace: `@isotrieve/core` 268 jest tests + `@isotrieve/demo-cli` 6 tests, `tsc --noEmit` clean on both. Run via `npm test` in `isotrieve-npm/`.
   - CI (`ci.yml`) runs: ruff, mypy strict, pytest matrix 3.10–3.13 (`-m "not slow"`), claims-lint (soft-fail), adapter-matrix check (soft-fail), pgvector integration (Docker service `pgvector/pgvector:pg16`, env `PGVECTOR_TEST_DSN`).
   - pytest markers defined: `integration`, `slow`, `benchmark`.
3. **Out of scope — do NOT do these:**
   - Fixing the claims-lint artifact gap (WS-A/B/C/E, `gate_lopo.json` rows in `isotrieve-python/CLAIMS.md`). Deliberately excluded by the owner. The claims-lint job is expected red; ignore it.
   - Editing any number in `CLAIMS.md` or READMEs. Reproduction runs in Phase 6 **report deltas only** — file issues, never patch claims.
   - API-key-dependent paths (OpenAI/Voyage/Cohere/Gemini/Pinecone real calls). Use fakes/mocks (`tests/fakes.py` exists).
   - Retuning hyperparameters, "fixing" benchmarks to make numbers match, or upgrading dependency versions mid-barrage.
4. **Failure protocol:** every failure gets (a) a saved log/artifact under `verification/barrage-artifacts/`, (b) a GitHub issue (`bug`/`test`/`claims` label, repo taxonomy) with the repro command, (c) NO in-place fix unless it's a broken test you wrote yourself in this barrage. Do not close existing issues.
5. **RAM discipline:** single-process arrays capped ≤ ~4 GB (e.g. 1M×384 float32 ≈ 1.5 GB is fine; 1M×1024 float32 = 4 GB is the ceiling). Stop and remove Docker containers between phases. Model downloads total ~3 GB (MiniLM-L6 ~90 MB, bge-large ~1.3 GB, e5-large ~1.3 GB).

---

## Phase P0 — Environment prep (~20 min)

**Objective:** everything later phases need is cached and services verified.

```bash
git checkout development && git pull --ff-only
git checkout -b verification/test-barrage
mkdir -p verification/barrage-artifacts

# Python matrix via uv (reuse .agent-venv for the 3.13 lane)
for v in 3.10 3.11 3.12; do
  uv venv .venvs/$v --python $v
  uv pip install --python .venvs/$v/bin/python -e "isotrieve-python[dev]"
done

# Docker services (skip + mark if daemon down)
docker run -d --name iso-pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
docker run -d --name iso-qdrant -p 6333:6333 qdrant/qdrant
sleep 15 && docker ps   # both healthy?

# Pre-warm models + datasets (network required; time-box 15 min)
.agent-venv/bin/python - <<'EOF'
from sentence_transformers import SentenceTransformer
for m in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5", "intfloat/e5-large-v2"]:
    print("caching", m); SentenceTransformer(m)
import ir_datasets
for d in ["beir/scifact", "beir/fiqa"]:
    print("caching", d); ir_datasets.load(d)
EOF
```

**Pass:** all venvs install; `docker ps` shows both containers; models/datasets cache without error. **Save:** prep log.

---

## Phase P1 — Full test matrix, all lanes (~40 min)

**Objective:** establish the true local baseline, including the tests CI deselects.

```bash
# 3.13 lane (.agent-venv) — full suite INCLUDING slow + benchmark markers
cd isotrieve-python && ../.agent-venv/bin/python -m pytest tests/ -q 2>&1 | tee ../verification/barrage-artifacts/p1_py313_full.log

# slow-only and benchmark-only counts, explicitly
../.agent-venv/bin/python -m pytest tests/ -q -m "slow" --no-header 2>&1 | tail -3
../.agent-venv/bin/python -m pytest tests/ -q -m "benchmark" --no-header 2>&1 | tail -3

# 3.10/3.11/3.12 lanes (mirrors CI)
for v in 3.10 3.11 3.12; do
  cd isotrieve-python && ../.venvs/$v/bin/python -m pytest tests/ -q -m "not slow" 2>&1 | tee ../verification/barrage-artifacts/p1_py$v.log | tail -2
done

# Flakiness/pollution check: run the full 3.13 suite twice more; pass sets must be IDENTICAL
# Isolation check: run the 5 largest test files individually (catches cross-file state leakage)
for f in test_gate test_quality test_cli test_migrate test_mapping; do
  ../.agent-venv/bin/python -m pytest tests/$f.py -q 2>&1 | tail -1
done

# npm: both suites + coverage informational
cd ../isotrieve-npm && npm test 2>&1 | tail -8
npm test --workspace @isotrieve/core -- --coverage 2>&1 | tail -20
npx tsc --noEmit -p packages/core/tsconfig.json && npx tsc --noEmit -p packages/isotrieve-demo-cli/tsconfig.json
```

**Pass criteria:** every Python lane green (0 failed); slow/benchmark lanes green or explicitly-skipped with recorded reason; npm 268+6 green; tsc clean; repeat runs identical pass sets.
**Known acceptable skips:** optional-dep skips (8) and API-key skips. Anything else gets an issue.

---

## Phase P2 — Packaging & publish-path integrity (~25 min)

**Objective:** what `release.yml` will publish actually works — caught locally, not on tag night.

```bash
cd isotrieve-python && ../.agent-venv/bin/python -m pip install build twine
../.agent-venv/bin/python -m build 2>&1 | tail -3
../.agent-venv/bin/python -m twine check dist/*

# Wheel + sdist contents: data files MUST be inside
unzip -l dist/*.whl | grep -E "gate_model_v1.json|thresholds|py.typed|calib"
tar -tzf dist/*.tar.gz | grep -E "gate_model_v1.json|LICENSE|CLAIMS"

# Clean-room install test (the point: import must resolve to site-packages, NOT the source tree)
uv venv /tmp/iso-wheel && uv pip install --python /tmp/iso-wheel/bin/python dist/*.whl pytest numpy
/tmp/iso-wheel/bin/python -c "import isotrieve, pathlib; print(isotrieve.__version__, pathlib.Path(isotrieve.__file__).parent)"
/tmp/iso-wheel/bin/isotrieve version && /tmp/iso-wheel/bin/isotrieve --help >/dev/null && echo CLI_OK
# run test_release.py against the INSTALLED package (cwd anywhere BUT inside isotrieve-python/src)
cd /tmp && /tmp/iso-wheel/bin/python -m pytest $OLDPWD/tests/test_release.py -q 2>&1 | tail -3

# npm pack dry-run (no publish): files field honored
cd $OLDPWD/../isotrieve-npm && npm pack --dry-run --workspace @isotrieve/core 2>&1 | tail -5
```

**Pass:** wheel/sdist contain `quality/gate_model_v1.json`, thresholds, `py.typed`, calibration corpus; clean-room import path is `site-packages/isotrieve/`; CLI works; `test_release.py` green against installed package (version-consistency tests included). **Failure here = P1 release blocker** (SKILLS.md §6 tag↔version↔notes agreement is next).

---

## Phase P3 — Cross-language parity, Python ↔ TypeScript (~40 min)

**Objective:** the `.isotrieve` binary v2 contract and shared numerics hold in BOTH implementations, not just each side's own tests.

Create `isotrieve-python/scripts/gen_cross_compat_fixtures.py`: with fixed seed 42, fit `RidgeMapping` on synthetic (200×8 → 200×12), save:
- `fixtures/golden_ridge.isotrieve` (binary v2)
- `fixtures/golden_expected.json` — mapping matrix + transform outputs on 10 held-back inputs, rounded to 1e-9
Write fixtures into `isotrieve-npm/packages/core/src/__tests__/fixtures/`.

New jest test(s) in `packages/core/src/__tests__/golden-parity.test.ts`:
1. Load golden binary → transform the 10 golden inputs → allclose to golden outputs (atol 1e-6).
2. Write a mapping from TS → load it back in Python (small script) → same matrix bytes/CRC.
3. Tamper tests both directions: flip one payload byte → reader must reject (CRC32); truncate file → reject; header claiming >1MB → reject.
4. Load `gate_model_v1.json` (shared asset) in both languages → identical predicted retention on a fixed score array (atol 1e-9).

Also a numeric-parity spot check: TS one-sided-Jacobi SVD vs numpy SVD on 5 random matrices (condition numbers 1..10³) — singular values match to 1e-8, and ridge-with-GCV alpha on golden data picks the same alpha within one grid step.

**Pass:** all parity tests green in both languages. **Save:** fixture generation log + jest output.

---

## Phase P4 — Numerical correctness & property-based invariants (~60 min)

**Objective:** laws that must always hold, beyond example-based tests.

Install `hypothesis` into `.agent-venv` (ad hoc; no pyproject change). New file `isotrieve-python/tests/test_property_invariants.py`:

- **L2 normalization:** every `Mapping.transform` output row has unit norm (atol 1e-9) — the exact invariant behind the old #8 bug.
- **Procrustes orthogonality:** `OrthogonalProcrustesMapping.WᵀW ≈ I` (1e-8) when d_src == d_tgt.
- **Ridge solves the fit:** residual of fitted W on calibration data ≤ random-W residual (sanity of the solver, not just shape).
- **Inverse round-trip:** for invertible maps, `‖M.inverse(M(x)) − x‖ < 1e-6`.
- **Determinism:** same seed → byte-identical `.isotrieve` file; `RidgeCV` alpha identical across two fresh fits.
- **Binary format:** save/load round-trip byte-identical; CRC catches any single-bit flip (flip every byte position in a small file, assert rejection).
- **Edges:** batch of exactly 1; k_cal < d (underdetermined); 384→1024 and 1024→384 rectangular; NaN/Inf input raises a typed error, never silent NaN propagation.

Then **process-level determinism**: run `isotrieve calibrate` twice in separate subprocesses on the same synthetic inputs → hash the two output files → identical.

**Pass:** 0 property violations; health check that the suite itself is sound (hypothesis finds at least the planted-tamper cases). **Save:** pytest output.

---

## Phase P5 — Bias, leakage & fairness audit (~90 min) — **core of this barrage**

**Objective:** prove the evaluation methodology itself is fair: no train-on-test, no hidden averaging tricks, honest uncertainty.

New file `isotrieve-python/tests/test_bias_audit.py` plus manual audit steps:

1. **Calibration/eval disjointness (leakage).** Instrument the benchmark sampler (monkeypatch or direct call of the sampling function in `benchmarks/run_benchmark.py`) on synthetic data: assert calibration indices ∩ evaluation indices = ∅. Repeat for `calibrate --queries-only` mode. If the harness ever evaluates on calibration rows, that's a P0 finding.
2. **Gate-model leave-one-pair-out revalidation.** `benchmarks/fit_gate_model.py` builds `quality/gate_model_v1.json` via LOPO. Re-run the LOPO procedure fresh; record MAE of predicted-vs-actual retention per held-out model pair. Pass: held-out MAE within the model's documented error budget — i.e., the shipped gate model isn't overfit to the pairs it scored. **Report only; do not replace the shipped JSON.**
3. **Alpha-selection isolation.** `RidgeMapping(alpha="auto")` uses RidgeCV-GCV. Assert (by recording arrays passed into `fit`) that only calibration X/Y ever reach the solver — evaluation embeddings must never touch alpha selection.
4. **Recalibrator train/held-out split.** `ScoreRecalibrator` (isotonic) must fit on calibration-split scores and score held-out queries. Assert the code path cannot fit on eval scores (construct the object the way the CLI does, then inspect fitted data bounds).
5. **Bootstrap CI coverage simulation.** Synthesize 500 small datasets with known true retention r ∈ {0.5, 0.7, 0.9}; compute the gate's 80% bootstrap CI on each; empirical coverage must land in [0.75, 0.85]. CIs that under-cover = overstated confidence = unfair assessment.
6. **Seed-sensitivity honesty.** Run `isotrieve gate --seed-sensitivity` (shipped in PR #67) on a synthetic corpus; then re-run the underlying fits with those same seeds manually; reported spread must match the manual spread (no understating variance).
7. **Margin-compression regression (bug #8 line).** Synthetic non-degenerate scores with known margin compression ratio ≠ 1.0 → gate must report it and widen the prediction interval (claim WS-D). With L2-normalized synthetic pairs where true variance is nonzero, compression must NOT read ~0/None.
8. **Averaging-dimension declaration (postmortem lesson, DECISIONS.md 2026-07-19).** Every JSON in `benchmarks/results/` carries a `protocol` field; run `python benchmarks/audit_configs.py` — must pass. Grep all summary-producing code for cross-adapter averages presented as single-adapter numbers.
9. **Truncation fairness (FiQA ceiling).** CLAIMS note says FiQA ceiling=0.032 is truncation-limited. Verify the harness computes floor/ceiling/mapped arms **on the same truncated corpus and same qrels+query set** for a given run (instrument one small run; assert identical doc-ids/queries across arms). A mapped arm evaluated on a different corpus than its ceiling is not a fair ratio.
10. **No eval-set peeking in fit loops.** Grep MLP/contrastive `fit` paths for any use of held-out or eval data (early stopping, best-epoch selection). Fixed epochs only = pass.
11. **Probe-retention purity.** `benchmarks/domain_probes.py` identifier probes must be same-index only (no target-model embeddings of eval docs involved). Read the implementation; confirm; spot-test.

**Pass:** all 11 green, each with a one-line evidence note in the report. **This phase is what "no training or validation bias, fair assessments" means concretely — treat any failure here as the highest-priority finding of the entire barrage.**

---

## Phase P6 — Independent reproduction of committed claims (~2 h)

**Objective:** the repo's own rule — claims are only as good as their artifacts — applied to *you*: reproduce committed numbers on different hardware (this M4 vs CI linux) without touching CLAIMS.md.

Read each committed artifact's embedded config first (`benchmarks/results/*.json` contain dataset/models/adapter/k/seed/protocol — and check whether `max_docs` was used; the domain-matrix runs used max-docs=2000).

Runs (via `benchmarks/run_benchmark.py`, exact flags from `--help` + the artifact's recorded config):

| # | Claim (from CLAIMS.md) | Committed | Runs |
|---|---|---|---|
| 1 | SciFact MiniLM→bge ridge K=4000, 3 seeds | 0.871 ± 0.006 | seeds 0,1,2 |
| 2 | same, lowrank K=4000 | 0.857 ± 0.009 | seeds 0,1,2 |
| 3 | same, MLP K=4000 (slowest — do last, optional) | 0.727 ± 0.007 | seeds 0,1,2 |
| 4 | bge→e5 same-dim ridge K=2000, 3 seeds (e5 prefixes ON) | 0.923 ± 0.010 | seeds 0,1,2 |
| 5 | Domain matrix: scifact ridge/lowrank K=500 seeds 0,1 (max-docs 2000) | 0.813 / 0.810 | 4 runs |
| 6 | Domain matrix: fiqa ridge/lowrank K=500 seeds 0,1 (max-docs 2000) | 0.773 / 0.768 | 4 runs |
| 7 | Probe corpora code/legal K=500 seeds 0,1 (offline, fast) | 1.022 / 1.037 | 4 runs |
| 8 | K-monotonicity mini-sweep: ridge K=500→1000→2000 (truncated) | monotone ↑ | 3 runs |

**Pass criteria (fair, not exact):**
- Each reproduced mean within **±0.02** of committed (library/hardware drift tolerance). Same-dim pair tolerance ±0.03 (prefix handling is version-sensitive).
- **Orderings preserved:** ridge > lowrank > MLP at K=4000; same-dim retention > rectangular; K-sweep monotone.
- Any miss → issue with `claims` label + both numbers + config diff. Do NOT edit CLAIMS.md.

**Save:** every result JSON + a comparison table in the report. Time-box: if a single run exceeds 20 min, truncate with max-docs and mark the comparison qualitative.

---

## Phase P7 — Real-store end-to-end integration (~90 min)

**Objective:** the full lifecycle works against real stores on this machine, not just fakes.

1. **pgvector (Docker, mirrors CI):**
   ```bash
   cd isotrieve-python && PGVECTOR_TEST_DSN=postgresql://postgres:postgres@localhost:5432/postgres \
     ../.agent-venv/bin/python -m pytest tests/test_pgvector_adapter.py -q -m integration
   ```
   Then a manual E2E: create table with 1k random vectors + metadata → `migrate` (shadow-column path) → mid-migration SIGKILL → resume (idempotent, no dupes) → `verify` → `rollback --dry-run` → `rollback` → assert original column restored.
2. **Qdrant:** in-memory suite (already in tests) + against the Docker server on :6333: seed 10k points (the Chroma-parity pattern from PR #72), migrate with small batches, exercise scroll boundary sizes {1, 63, 64, 65, batch_size}, rollback via target-collection drop.
3. **Chroma in-process, real models, small corpus (500 docs, MiniLM→bge):** full lifecycle — `calibrate` → `transform` → `gate` (expect PASS verdict + exit 0) → `manifest show` → `verify` → `rollback`. Then rerun `gate` with an absurd `--threshold 0.99` → expect non-zero exit + failure report (gate actually gates).
4. **CLI surface sweep:** every verb (`plan calibrate transform inspect gate doctor report manifest rollback resume verify`) via subprocess with `--help` at minimum, happy path for the lifecycle set. Exit codes recorded.
5. **Serve-mode equivalence (QueryAdapter):** query-time transform must retrieve ~identically to pre-transformed corpus: Recall@10 on 50 queries within 0.02 of the migrated-index arm.
6. **MPS device path:** `ResidualMLPMapping(device="auto")` on this Mac selects MPS; outputs vs CPU within atol 1e-4; two MPS runs deterministic.
7. **Wrappers with fakes (no API keys):** LangChain `IsotrieveEmbeddings`, OpenAI shim, LlamaIndex wrapper — happy path + telemetry off.

**Pass:** all green; crash-resume proven idempotent on two stores; gate exit codes correct both polarities.

---

## Phase P8 — Performance & memory ceiling on 24 GB (~45 min)

**Objective:** streaming claims hold; record numbers as the perf baseline for this machine.

1. **Peak RSS during batched migration (streaming proof):** `NumpyFileStore` with 200k×768 float32 (~600 MB) and 1M×384 (~1.5 GB), batch 5k → measure peak RSS via `resource.getrusage(RUSAGE_SELF).ru_maxrss`. Pass: peak < 40% of corpus size (batching actually streams; a materialize-everything regression like old bug #9 would blow past this).
2. **Gate bootstrap timing:** 1000 resamples on 10k pairs — record wall time (target: seconds, not minutes).
3. **10k-point Qdrant migration wall time** (Docker server) — record.
4. **npm stress suite** (`stress.test.ts` already exists) — green, record duration.
5. Optional if time: 1M×1024 float32 (4 GB) transform in batches — the 24 GB ceiling case; just record peak RSS and pass/fail vs OOM.

**Save:** all numbers into the report's perf table.

---

## Phase P9 — Stretch (only if P1–P8 are green and time remains)

1. **Mutation testing (suite self-assessment):** `mutmut run --paths-to-mutate isotrieve-python/src/isotrieve/mapping/linear.py` plus `quality/gate.py`, time-boxed 30 min. Record kill rate; <70% kill on those two modules → `test` issue.
2. **Dependency audit:** `pip-audit` on the dev env + `npm audit` in `isotrieve-npm` — informational, record only.
3. **Fuzz the binary reader:** hypothesis, 1000 examples — random truncations/corruptions of a valid `.isotrieve` file; reader must always raise a typed error, never crash the interpreter or allocate the claimed 1MB+ header.
4. **npm root `lint` script:** currently misconfigured (ESLint finds no config) — pre-existing; note it, optionally fix config in a small separate PR.

---

## Reporting & definition of done

Write **`verification/BARRAGE_REPORT.md`** (commit on the barrage branch, PR → `development`, `dx`/`test` labels, no milestone or `v0.4.1` at most) containing:

1. Environment block (macOS version, chip, RAM, python/node versions, docker versions, model/dataset cache dates).
2. Per-phase verdict table: `PASS / PASS-with-notes / FAIL / SKIPPED(reason)` + artifact links.
3. P5 bias audit: 11 one-line evidence notes.
4. P6 reproduction table: committed vs reproduced, delta, verdict per claim row.
5. P8 perf numbers.
6. Issues filed (numbers + links).
7. Overall verdict + recommended next actions.

**Done =** phases P1–P8 all executed with recorded verdicts, every failure has an issue, report PR open against `development`, no changes to `main`, no claims edited, nothing published.

---

## Appendix A — Quick command cheat sheet

```bash
# Python (3.13 lane)
cd isotrieve-python && ../.agent-venv/bin/python -m pytest tests/ -q          # full (incl slow)
../.agent-venv/bin/python -m pytest tests/ -q -m "not slow"                   # CI parity
PGVECTOR_TEST_DSN=postgresql://postgres:postgres@localhost:5432/postgres \
  ../.agent-venv/bin/python -m pytest tests/test_pgvector_adapter.py -m integration
../.agent-venv/bin/ruff check src/ tests/ && ../.agent-venv/bin/ruff format --check src/ tests/
../.agent-venv/bin/mypy src/isotrieve/ --strict --ignore-missing-imports

# Other Python lanes
cd isotrieve-python && ../.venvs/3.1X/bin/python -m pytest tests/ -q -m "not slow"

# npm
cd isotrieve-npm && npm test && npx tsc --noEmit -p packages/core/tsconfig.json

# Benchmarks
cd isotrieve-python && ../.agent-venv/bin/python benchmarks/run_benchmark.py --help   # discover exact flags
../.agent-venv/bin/python benchmarks/audit_configs.py

# Docker teardown
docker rm -f iso-pg iso-qdrant
```

## Appendix B — Pass-criteria summary

| Phase | Hard pass condition |
|---|---|
| P1 | 0 failed on every lane; repeat runs identical |
| P2 | clean-room wheel import + CLI + release tests green |
| P3 | golden parity both directions, tamper rejection both languages |
| P4 | 0 property violations; determinism byte-identical |
| P5 | all 11 audit items green — any failure is top-priority |
| P6 | means within ±0.02 (±0.03 same-dim); orderings preserved |
| P7 | lifecycle green on pgvector+Qdrant+Chroma; crash-resume idempotent; gate exit codes both polarities |
| P8 | peak RSS < 40% of corpus on streaming migration; no OOM |
| P9 | optional; record-only |
