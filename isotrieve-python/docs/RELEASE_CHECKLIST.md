# Release Checklist

Releases are **batched**, not per-PR. One PR advances `main`, then one tag
publishes to PyPI and GitHub Releases. Follow this checklist in order.

## Phase 1 — Confirm scope

- [ ] All issues under the target milestone (e.g. `v0.4.0`) are closed, or
      explicitly deferred to a later milestone with a comment on the issue.
- [ ] `pyproject.toml` version and `package.json` version (if npm changed) are
      bumped to the release version. They must match the tag exactly.
- [ ] `CHANGELOG.md` has a `## [X.Y.Z] - <date>` section for this release.
- [ ] No open PRs or stale branches accidentally target `main` (check the
      pull request list). Anything still open should be re-targeted to
      `development`.

## Phase 2 — Release PR

- [ ] Open one PR: `development` → `main`, titled `Release vX.Y.Z`.
- [ ] PR summary lists every merged issue/PR since the last tag.
- [ ] Wait for the 5 required checks: `lint`, `typecheck`, `test (3.10)`,
      `test (3.11)`, `test (3.12)`.
- [ ] Get 1 approving review. Stale reviews are dismissed on new pushes —
      don't push after approval without re-requesting review.
- [ ] Merge to `main`. Never push to `main` directly; branch protection
      blocks it.

## Phase 3 — Tag & publish

- [ ] Tag `vX.Y.Z` on `main`. The tag must match the version actually bumped
      in `pyproject.toml`/`package.json`. Verify tag ↔ version ↔
      release-notes-title all agree before pushing.
- [ ] Pushing the tag triggers `release.yml`:
      1. Release tests + full pytest (3.12)
      2. Build package
      3. Publish to PyPI (trusted publishing, no manual token)
      4. Create GitHub Release from the `CHANGELOG.md` section for this tag
- [ ] Watch the workflow: `gh run watch`.
- [ ] Verify PyPI: install into a fresh venv and run `isotrieve version`.

## Phase 4 — Post-release

- [ ] `deploy-pages.yml` fires separately on push to `main`. Verify the live
      site at `https://krish1925.github.io/isotrieve/` matches what shipped.
      The site has drifted ahead of both branches before via out-of-band
      deploy branches — don't let that recur.
- [ ] Close the milestone in GitHub (mark done/closed). Don't leave a shipped
      release with an open milestone, as happened with `v0.3.0`.
- [ ] Update `release.sh` if the process changed; it must not carry stale
      versions or wrong-case repo paths.

## Version consistency check

History has produced mismatched artifacts (a `v0.2.0` release object titled
"0.2.1", and a `v0.2.1` tag with no release object). Before pushing the tag,
confirm all four agree:

| Artifact | Value |
|---|---|
| `pyproject.toml` version | `X.Y.Z` |
| Git tag | `vX.Y.Z` |
| GitHub Release title | `isotrieve X.Y.Z` |
| `CHANGELOG.md` section | `## [X.Y.Z]` |
