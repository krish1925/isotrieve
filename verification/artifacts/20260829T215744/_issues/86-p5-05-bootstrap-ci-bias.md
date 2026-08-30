## Check
check_id: P5-05-bootstrap-coverage — pre-registered: nominal-80% bootstrap CI must achieve empirical coverage in [75%, 85%] on synthetic universes with known true retention r ∈ {0.5, 0.7, 0.9} (n=200 queries, 400 universes per r, 200 resamples, seed 20260829).

## Observed
coverage = 0.0000 ± 0.0000 for ALL THREE r values. The CI never contains the true retention.

## Mechanism (why this affects real embeddings, not just the synthetic stub)
`_bootstrap_retention_ci` (isotrieve-python/src/isotrieve/cli_gate.py:304) resamples rows **with replacement**, then calls `topk_retention(m_sub, y_sub, k=1)`, which scores each row by "is my true match at **the same row index**". When an index is drawn ≥2 times, `y_sub` contains **exact duplicate vectors** (bit-identical embeddings → cosine similarity exactly 1.0 to the query's own mapped row). All duplicate positions tie at sim=1.0; `np.argpartition` hands the "top-1" to one of them, so the query row is only credited if it wins that tie — probability ≈ 1/k for k duplicates. With n=200, E[fraction of rows drawn ≥2×] ≈ 26% → resampled retention collapses to ≈ 0.58 × true, dragging every percentile interval far below the true value. Exact ties from resampled duplicates occur for **any real embedding input** — the synthetic stub only makes the failure total.

## Reproduce

    cd isotrieve-python && ../.agent-venv/bin/python -m pytest tests/test_bias_audit.py::TestP505BootstrapCoverage -q -s
    # → P5-05 r=0.5: coverage=0.0000 ± 0.0000 (nominal 80%)

Base SHA: 28cb6f1 · Python 3.13.5 · deterministic: yes (3/3 runs)

## Evidence
- verification/artifacts/20260829T215744/P5/p5_bias4.log
- verification/artifacts/20260829T215744/P5/p5.xml

## Triage
bucket: product-bug (confidence: high). The gate's reported 80% CIs (CLI gate output, HTML report) are systematically anti-conservative — intervals shifted/contracted below nominal coverage.

## Impact
Every `isotrieve gate` report ships CIs that do not cover the true retention at their nominal rate. PASS/WARN verdicts near the threshold boundary can be overconfident; CLAIMS.md rows quoting CI bounds inherit the bias. Report-only — CLAIMS.md not modified.

## Suggested fix
Make self-match deterministic under duplicates: deduplicate resample indices before scoring (score each drawn index once, weight by multiplicity), or compare `sims[i, i] >= max(sims[i, :]) - 1e-12` instead of argpartition position. The P5-05 coverage sim is ready to flip green as the regression test.

## Status
[ ] Fixed in this run   [x] Deferred to owner (product code — not fixable under barrage guardrails)
