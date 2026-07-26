# Competitive Strategy: Migration CI for Vector Stores

**Last updated:** 2026-07-25

## Positioning

Isotrieve is **Migration CI** — not another adapter library. EmbeddingAdapters and similar tools produce transforms. Isotrieve tells you whether to ship them. "Bring your own transform — the gate decides."

This is infrastructure, not competition. Their users are our users.

## The five plays

### Play 1 — Gate credibility (v0.3.0)
Ship the gate as the product. Every feature decision points back to: does this make the gate more trustworthy?

- CI workflow with retention gates (#13, #14)
- Claims linter enforcement
- `--queries-only` calibration (#21)
- **#58 BYOT gate** — accept external/pretrained transforms, not just Isotrieve-fitted ones

### Play 2 — Bring-your-own-transform (v0.3.0)
**#58**: `ExternalMapping` wrapping any callable `(np.ndarray) -> np.ndarray` or torch/onnx module. Positions Isotrieve as the quality gate for *all* embedding migrations, including those started with other tools. This is the structural moat.

### Play 3 — Prove it with benchmarks (v0.4.0)
- **#37**: Domain matrix (general/legal/medical/code BEIR subsets)
- **#59**: Head-to-head — corpus-calibrated Ridge vs pretrained general adapters, same eval set
- If we win → README chart + CLAIMS.md. If we lose → document the boundary condition honestly.
- No claim ships without a committed artifact.

### Play 4 — Production infrastructure (v0.4.0)
- **#22**: pgvector adapter (resolves #15 advertised-but-missing)
- **#23**: Post-migration revalidation (`isotrieve verify`)
- **#24**: Manifest/rollback CLI
- **#20**: GitHub Action (`isotrieve gate` in CI — "Migration CI" without a CI action is a naming problem)

### Play 5 — Consistency moat (v0.3.0+)
- Tagged release every 4-6 weeks, even if small
- CHANGELOG.md in Keep a Changelog format, backfilled from v0.2.x
- Release checklist derived from pre-release audit plan
- **#61**: Release cadence + changelog discipline

## Milestone roadmap

### v0.3.0 — "The gate is the product" (target: 2026-08-31)
Plays 1, 2, and 5.

| Issue | What | Status |
|-------|------|--------|
| #13 | CI workflow | Done |
| #14 | Claims linter | Done |
| #15 | pgvector honesty fix | Open |
| #17 | Qdrant tests | Done |
| #18 | CONTRIBUTING.md rewrite | Done |
| #19 | Docs split-brain resolution | Open |
| #20 | GitHub Action for gate | Open |
| #21 | `calibrate --queries-only` | Done |
| #58 | BYOT gate (ExternalMapping) | Open |
| #60 | "Why a gate" comparison page | Open |
| #61 | Release cadence + changelog | Open |

### v0.4.0 — "Prove it" (target: 2026-09-30)
Plays 3 and 4.

| Issue | What | Status |
|-------|------|--------|
| #22 | pgvector adapter | Open |
| #23 | Post-migration revalidation | Open |
| #24 | Manifest/rollback CLI | Open |
| #35 | Hubness correction (P0) | Open |
| #36 | Seed-sensitivity analysis | Open |
| #37 | Domain matrix benchmark | Open |
| #59 | Head-to-head benchmark | Open |

### v0.4.1 — Opportunistic
Timed to deprecation events, not calendar.

| Issue | What |
|-------|------|
| #26 | Gate threshold presets |
| #27 | Cohere/Voyage playbooks |

### v0.5.0 — Differentiation (target: 2026-11-30)
Multi-tenant, confidence hooks, ensemble transforms.

### Icebox (not touched)
#45, #39, #38, #32, #46 — job-search-brain features, not competitive features.

## Sequencing constraints

1. **#58 before #59** — head-to-head needs BYOT path to load pretrained adapters
2. **P0 bugs first** — #9, #10, #12 are data-integrity issues; competitive strategy on top of open P0s is building on sand
3. **#43 falsification review** — 60 days after v0.4.0, judge whether the migration-CI framing held

## Anti-patterns

- Don't compete on adapter breadth — that's their game. Compete on "did you measure retention?"
- Don't ship claims without artifacts. If results are mixed, show the mixed result.
- Don't touch Icebox issues — they're the wrong product track.
