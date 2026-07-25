# Submission: awesome-vector-search

**Status:** PR OPEN
**PR URL:** https://github.com/currentslab/awesome-vector-search/pull/62
**Opened:** 2026-07-25

## Target Repo

- **Repo:** https://github.com/currentslab/awesome-vector-search
- **Stars:** ~1,600
- **Maintainer:** theblackcat102
- **Last activity:** Jul 6, 2026 (batch-merged 8 PRs)
- **Open PRs:** 9 (at time of submission)

## Entry Added

```diff
 - [Moss - Sub-10ms semantic search engine for Voice & Conversational AI, built in Rust/WebAssembly for on-device / in-browser retrieval](https://github.com/usemoss/moss)
+- [isotrieve - Migrate a vector database to a new embedding model without re-embedding the corpus; fits a linear map from ~2K calibration texts](https://github.com/krish1925/isotrieve)
```

## PR Details

- **Title:** `Add isotrieve to Library`
- **Branch:** `add-isotrieve`
- **Description:** One-liner + brief technical summary + PyPI install

## Commands Used

```bash
gh repo fork currentslab/awesome-vector-search --clone=false
gh repo clone krish1925/awesome-vector-search /tmp/awesome-vector-search-fork
cd /tmp/awesome-vector-search-fork
git checkout -b add-isotrieve
# Edit README.md: add line after Moss entry in Library section
git add README.md && git commit -m "Add isotrieve to Library"
git push origin add-isotrieve
gh pr create --title "Add isotrieve to Library" --body "..." --head krish1925:add-isotrieve --base main
```

## Judgment Calls

- **Format:** Used `- [Name - Description](URL)` matching the exact format of surrounding entries (Moss, CocoIndex, chromem-go). No trailing period (consistent with recent entries).
- **Placement:** Appended to end of Library section (after Moss), not alphabetically inserted. Recent additions (Jul 6 batch) were all appended to end, so this matches maintainer's preferred merge pattern.
- **Description:** Plain technical — "Migrate a vector database to a new embedding model without re-embedding the corpus; fits a linear map from ~2K calibration texts". No marketing adjectives.
- **Link:** Points to GitHub repo (not PyPI), per POLICY.md requirement.
- **No CONTRIBUTING.md:** Repo has POLICY.md instead, which only requires open-source GitHub link. Met.
