## Check
check_id: P5-08-averaging-declaration — pre-registered: every `benchmarks/results/*.json` carries `protocol`, `config_hash`, and `commit`; `benchmarks/audit_configs.py` passes.

## Observed
audit_configs.py passes, but legacy artifacts are missing declaration fields. First offenders:
- `beir_scifact_..._lowrank__k1000__seed0__7671d810e251.json: missing protocol`
- same-family k1000/k2000/k4000 ridge/lowrank files (the 2026-07-21 K-sweep and adapter-sweep era)

These are exactly the artifacts backing CLAIMS.md rows 2–5 (LowRank retention, K-sweep numbers).

## Reproduce

    python - <<'EOF'
    import json, glob
    missing = [(f, [k for k in ("protocol","config_hash","commit") if k not in json.load(open(f))])
               for f in glob.glob("benchmarks/results/*.json")]
    print([(f.split("/")[-1][:60], m) for f, m in missing if m])
    EOF

Base SHA: 28cb6f1 · deterministic: yes

## Evidence
- verification/artifacts/20260829T215744/P5/p5_bias4.log (P5-08 assertion lists all offenders)

## Triage
bucket: product-bug — process-level: the DECISIONS.md 2026-07-19 postmortem mandated declared averaging dimensions (`protocol` field), but only runs created after the fix carry it. The repo's own rule (only cite numbers from run_benchmark.py with declared protocol) is met by newer artifacts and silently unmet by legacy ones still backing published claims.

## Impact
CLAIMS.md rows citing legacy artifacts cannot be audited for averaging dimension — the exact defect class that produced the 0.814-vs-0.866 postmortem.

## Suggested fix
One-time backfill: re-run the affected configs (K-sweep + adapter sweep, seeds 0–2) with the current harness to regenerate artifacts with full declarations, or annotate CLAIMS.md rows as "pre-declaration artifact" until re-run. `audit_configs.py` could warn on missing fields.

## Status
[ ] Fixed in this run   [x] Deferred to owner (report-only; regeneration is owner's call)
