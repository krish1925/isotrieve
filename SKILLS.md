# isotrieve — Contributor & Workflow Guide (SKILLS.md)

This file is the operating manual for making changes to `krish1925/isotrieve`. It exists because the repo has real branch protection, a two-package version skew (Python + npm), a messy release history, and CI gates that will silently block merges if ignored. Follow this exactly.

> **Where things live:** the canonical docs below are not all at the repo root. `CLAIMS.md`, `DECISIONS.md` and `CHANGELOG.md` live under `isotrieve-python/` unless noted. The one-paragraph quick start lives in `isotrieve-python/CONTRIBUTING.md`.

---

## 1. Golden Rules

1. **Never push to `main`.** It's protected (enforce_admins: true, required checks: lint / typecheck / test-3.10 / test-3.11 / test-3.12; no force-push, no deletion). You cannot bypass this even as an admin. All work lands on `development` first.
2. **`development` is the integration branch.** All feature/fix branches merge into `development` via PR, not directly.
3. **`main` only advances via a batch "Release vX.Y.Z" PR** from `development` → `main` (see §6), followed by a version tag.
4. Every change must leave the repo **more consistent than it found it** — close/link the issue it fixes, update `isotrieve-python/CLAIMS.md` if it touches benchmarks, update the changelog, and don't leave stale branches/PRs behind.
5. If in doubt about scope, check the issue's milestone (§5) before starting work — don't silently pull in v0.5.0 work while meeting a v0.4.0 deadline.

---

## 2. Branch Workflow

```
feat/<short-name>   or   fix/<short-name>   or   docs/<short-name>
        │
        ▼  PR into development
development
        │
        ▼  batch "Release vX.Y.Z" PR (see §6)
main  ──▶ tag v* ──▶ release.yml ──▶ PyPI publish
```

**Rules:**
- Branch names: `feat/…`, `fix/…`, `docs/…`, `infra/…` — matches label taxonomy in §5.
- Never open a PR directly from a feature branch into `main`. If you catch yourself doing this, retarget to `development`.
- Delete your branch after merge. The repo has historically left stale `dependabot/*` and closed-unmerged branches behind — don't add to the mess.
- If a PR sits open against `main` waiting on review, don't stack further `main`-targeted PRs behind it — queue against `development` instead.

---

## 3. Before You Start Coding

- [ ] Find or file the GitHub issue. No untracked work.
- [ ] Confirm the issue's milestone matches what you intend to ship (v0.4.0 is next; don't quietly do v0.4.1/v0.5.0/Icebox work under a v0.4.0 PR).
- [ ] Check `isotrieve-python/CLAIMS.md` if your change touches retrieval-retention numbers, benchmarks, or adapter support claims — every benchmark claim must be artifact-backed (a script/output that regenerates it), not asserted from memory.
- [ ] Check `isotrieve-python/DECISIONS.md` for prior architectural rulings before re-litigating something (e.g. "gate mechanism is the core differentiator," "support third-party adapters rather than compete").
- [ ] If working across both packages, check version skew: Python `isotrieve` is the main product, npm `@isotrieve/core` is a TS port that lags behind and is **not actively maintained** (see §7). Don't assume feature parity — check before porting.

---

## 4. Making the Change

### Code hygiene
- Keep `.agent-venv/`, the `agent/` loop scaffolding, and any local agent-loop artifacts **out of tracked files** — verify `.gitignore` covers them before committing (it currently does **not**; add `.agent-venv/` and `agent/` if missing). Audit `git status` for stray artifacts (build output, `.DS_Store`, notebook checkpoints, `__pycache__/`) before every commit.
- No dead code, no commented-out blocks, no TODO-without-issue-link.
- CLI additions must extend the existing subcommand set (`plan / calibrate / transform / inspect / gate / doctor / report`) consistently — new verbs need a `feat:cli` labeled issue first.
- Adapter work (Chroma/Qdrant/Pinecone/pgvector) must match the `adapter:*` label taxonomy; if you're building the pgvector adapter, it's docs-advertised-but-unshipped (issue #22) — closing that gap is real, tracked work, not a side effect of another PR.

### Local checks (must pass before opening a PR — these mirror `ci.yml` exactly)
```bash
cd isotrieve-python
ruff check .                     # lint
mypy .                           # typecheck
pytest                           # full suite; CI runs a matrix on 3.10 / 3.11 / 3.12
python scripts/lint_claims.py    # soft-fail in CI, treat as hard fail locally
```
Required checks (`lint`, `typecheck`, `test (3.10)`, `test (3.11)`, `test (3.12)`) must all be green — `main` enforces them, and `development` PRs should meet the same bar since they feed the release batch.

### Docs & changelog
- Update `isotrieve-python/CHANGELOG.md` (or the website changelog page) in the **same PR** as the code change — not deferred to release time.
- If the change affects the public API surface, docs pages under `isotrieve-website/` (docs, playground, protocol, performance, integrations, npm) need a matching update in the same PR or a linked follow-up issue.
- README/homepage claims (currently: "~87–91% retrieval retention, BEIR-benchmarked") only change alongside a CLAIMS.md artifact update — never edit the number without the backing evidence.

---

## 5. Issues & Labels

Use the existing taxonomy — don't invent new labels ad hoc.

| Category | Labels |
|---|---|
| Priority | `P0`, `P1` |
| Adapters | `adapter:chroma`, `adapter:qdrant`, `adapter:pgvector`, `adapter:pinecone` |
| Areas | `bug`, `feat`, `docs`, `test`, `infra`, `dx`, `cli`, `gate`, `calibration`, `safety`, `claims`, `benchmarks`, `strategy`, `growth`, `research`, `tenants`, `wrappers`, `experimental`, `playbooks`, `ci-product`, `design-review`, `out-of-scope-core` |
| Meta | `duplicate`, `invalid`, `wontfix`, `good first issue`, `help wanted`, `question` |

**Rules:**
- Every PR description must reference the issue it closes (`Closes #NN`) so merge auto-closes it. GitHub won't do this for you if the phrasing is off — use `Closes`, `Fixes`, or `Resolves` literally.
- Bug reports and feature requests use the existing templates (`bug_report`, `feature_request` — has a Milestone field, fill it in). Playbook requests use the `playbook_request` model-pair form.
- Before closing an issue as done, verify it's not a duplicate of an already-closed one. If a PR only *partially* resolves an issue, don't close it — comment progress and leave it open, or split into a tracked follow-up issue and link both directions.
- Milestones in play: `v0.4.0` (next release — where active work should land), `v0.4.1`, `v0.5.0`, `Icebox`.

---

## 6. Release Process

Releases are **batched**, not per-PR.

1. Confirm all issues under the target milestone (e.g. `v0.4.0`) are done/closed.
2. Open one PR: `development` → `main`, titled `Release vX.Y.Z`. Summary lists every merged issue/PR since the last tag.
3. This PR must pass the 5 required checks and get 1 approving review (stale reviews are dismissed on new pushes — don't push after approval without re-requesting review).
4. Merge to `main`.
5. Tag `vX.Y.Z` on `main` **matching the version actually bumped in `pyproject.toml`/`package.json`** — the repo has a history of version-label mismatches (v0.2.0 release object mistakenly titled "0.2.1"; the real v0.2.1 tag has no release object). Double check tag ↔ version ↔ release-notes-title all agree before pushing the tag.
6. Pushing the tag triggers `release.yml`: full test run, build, then PyPI publish via trusted publishing (no manual token).
7. `deploy-pages.yml` fires separately on push to `main` — verify the live site actually matches what shipped (it has drifted ahead of branches before via out-of-band deploy branches; don't let that recur).
8. After tagging, close the milestone in GitHub — `v0.3.0` was released but never formally closed; don't repeat that.
9. Keep `release.sh` correct — it historically hardcoded the wrong version and a wrong-case repo path. The tag, `pyproject.toml`, and release-notes title must agree before you push.

---

## 7. Packages & Environment (two-package skew)

- **Python `isotrieve` (PyPI)** — the actively maintained, benchmark-validated product. On the daily in `isotrieve-python/`. Bumped in `pyproject.toml`.
- **npm `@isotrieve/core`** (in `isotrieve-npm/`) — a TS port that lags behind and is **not actively maintained**. Do not assume parity; only touch it if the issue explicitly calls for npm work.
- Touch `main`-level environments cautiously:
  - `github-pages`: whitelisted branches currently include `development`, `gh-pages`, `main`, and a leftover deploy branch. Clean up workaround branches once their PRs merge.
  - `pypi`: trusted publishing, unprotected — only `release.yml` on a `v*` tag should publish. Never publish manually.

---

## 8. Definition of Done (checklist for every PR)

- [ ] Branch targets `development` (or is a release-batch PR targeting `main`)
- [ ] Linked to an issue with `Closes #NN`
- [ ] `ruff`, `mypy`, full `pytest` matrix, and claims-lint all pass locally
- [ ] No stray/gitignore-worthy files committed (`.agent-venv/`, `agent/`, unrepo caches, build artifacts)
- [ ] CHANGELOG updated
- [ ] Docs/website updated if public surface changed
- [ ] CLAIMS.md updated with a regenerable artifact if any benchmark number changed
- [ ] Issue milestone matches what's actually being shipped
- [ ] Branch deleted after merge