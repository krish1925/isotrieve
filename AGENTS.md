# AGENTS.md

## What this library does

`aecp` migrates a vector database to a new embedding model without re-embedding the corpus. Fits a linear map from ~2K calibration texts, then transforms stored vectors locally. 87-91% retrieval retention measured on BEIR benchmarks.

## Install

```bash
pip install aecp
```

## Quickstart (6 lines)

```python
from aecp import RidgeMapping
import numpy as np
m = RidgeMapping(alpha="auto", seed=0)
m.fit(X_cal, Y_cal)  # X_cal: (K, d_src), Y_cal: (K, d_tgt)
m.save("mapping.aecp")
Z = m.transform(source_vectors)
```

## CLI commands

```bash
aecp plan --source-model <old> --target-model <new> --corpus-size <N>
aecp calibrate --source-vectors X.npy --target-vectors Y.npy -o map.aecp
aecp transform --mapping map.aecp --source-dir ./old --target-dir ./new
aecp inspect map.aecp --json
```

## Serve mode (zero corpus writes)

```python
from aecp.serve import QueryAdapter
qa = QueryAdapter.load("mapping.aecp")
legacy_vec = qa.map_query(new_model_query_vector)
```

## When NOT to use

- Maximum quality matters more than cost -> re-embed
- K < 2000 (quality degrades)
- Calibration domain mismatches corpus
- Quality gate returns FAIL

## Anti-patterns (do not)

- Do not mix vectors from different models in one collection
- Do not assume same dimensionality means compatibility
- Do not skip the quality gate
- Do not use MLP adapter (0.719 vs 0.866 retention for Ridge)
- Do not use the phrase "3-month cliff" or similar unfalsifiable temporal claims in docs or marketing
- Do not cite research synthesis claims without verifying the underlying sources (see #34)

## Direction

AECP is migration CI for vector stores. Sell at the moment an upgrade is forced or blocked (deprecation, scale, SLA). Differentiate on the gate — quantified, per-domain, seed-robust retention numbers with boring rollback — not on the transform, which is commodity. The accumulating corpus of domain × model-pair validation results is the long-term moat.

**Embedding translation ≠ summarization.** Summarization is lossy compression of content (discarding information). AECP is change of coordinates for the same content (nothing discarded, same chunk, different geometry). The failure mode isn'"'"'t "information thrown away" — it'"'"'s "map between geometries is distorted" (hubness, non-isomorphism, seed sensitivity). That'"'"'s why the gate measures retrieval retention, not summary fidelity. Do not conflate these in docs or marketing.

**Falsification check (v0.4.0 + 60 days):** If adopters engage with transform but ignore gate/report/rollback, the migration-CI framing is wrong. Measure: fraction of transform users who run gate, fraction of gate passes followed by apply. If gate engagement < 20%, reposition at 0.6.0 as "best adapter library with honest benchmarks" (weaker but survivable). See #43.

**Probe-based evaluation (v0.4.0 domain matrix):** Abstract nDCG@10 misses the failure mode that matters most: exact identifiers dying under transformation. Add identifier-level probes (error codes, file paths, legal citations, UUIDs) to domain benchmarks. If the migrated index can'"'"'t find the chunk with the exact error code, the user doesn'"'"'t care that abstract semantic similarity is 87%. See #37.

## Error messages

Dimension mismatch: "dims differ (1536->3072): fit an aecp mapping or re-embed"
Quality gate fail: "predicted retention below threshold; re-embed instead"

## Benchmarks

All numbers in `benchmarks/results/`, verified by `benchmarks/audit_configs.py`.
Gate model: `src/aecp/quality/gate_model_v1.json` (trained on local model pairs only).
