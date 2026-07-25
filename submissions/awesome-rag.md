# Submission: awesome-rag

**Status:** PR OPEN
**PR URL:** https://github.com/Danielskry/Awesome-RAG/pull/135
**Opened:** 2026-07-25
**Fork:** https://github.com/krish1925/Awesome-RAG
**Branch:** `add-isotrieve`

## Target Repo

- **Repo:** https://github.com/Danielskry/Awesome-RAG
- **Stars:** ~1,300
- **Maintainer:** Daniel Skryseth
- **Last activity:** Jul 9, 2026
- **Open PRs:** 74 (slow merge cadence)

## Entry Added

```diff
 - [FAISS](https://github.com/facebookresearch/faiss): A library for efficient similarity search and clustering of dense vectors, designed to handle large-scale datasets and optimized for fast retrieval of nearest neighbors.
+- [isotrieve](https://github.com/krish1925/isotrieve): A tool for migrating vector databases to new embedding models without re-embedding the corpus.
```

## PR Details (ready to execute)

- **Title:** `Add isotrieve to Vector Search Libraries and Tools`
- **Branch:** `add-isotrieve` (already pushed)
- **Base:** `main`

## Commands to Open PR

```bash
# Run this 4-12 hours after the awesome-vector-search PR was opened
cd /tmp/awesome-rag-fork
gh pr create \
  --title "Add isotrieve to Vector Search Libraries and Tools" \
  --body "Adds [isotrieve](https://github.com/krish1925/isotrieve) to the Vector Search Libraries and Tools subsection.

Migrates stored vectors to a new embedding model via a learned linear map (~2K calibration texts). Gates on measured retrieval retention. PyPI: \`pip install isotrieve\`." \
  --head krish1925:add-isotrieve \
  --base main
```

## Judgment Calls

- **Format:** Used `- [Name](URL): Description.` matching the exact format of the FAISS entry in the same subsection (colon separator, trailing period). This differs from the awesome-vector-search format — each list has its own convention.
- **Placement:** Added to "Vector Search Libraries and Tools" subsection under Databases. Currently only contains FAISS. This is the most defensible category — isotrieve operates on vector databases but is a migration utility, not a database itself.
- **Description:** "A tool for migrating vector databases to new embedding models without re-embedding the corpus." Plain technical, matches the colon+period format of FAISS entry.
- **Why this subsection:** The README already acknowledges the need for "embedding model upgrades and migration strategies" under Best Practices > Iterative Improvement (line 480). isotrieve directly addresses this.
- **Note:** This repo has 74 open PRs — the PR may sit for weeks. That's normal for the list.
