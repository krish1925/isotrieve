# Isotrieve

**Migration CI for vector stores.**

Switch embedding models without re-embedding your corpus — and know, before you cut over, exactly what it costs you in retrieval quality.

[![PyPI](https://img.shields.io/pypi/v/isotrieve)](https://pypi.org/project/isotrieve/)
[![CI](https://github.com/krish1925/isotrieve/actions/workflows/ci.yml/badge.svg)](https://github.com/krish1925/isotrieve/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/isotrieve)](https://pypi.org/project/isotrieve/)

```bash
pip install isotrieve
```

---

## Nobody upgrades their embedding model

Not because they don't want to. Because the upgrade path is a migration project.

A better model ships. You want it. Then you price it out: re-embed the entire corpus, re-index, re-tune every downstream threshold, and pray retrieval quality survives. For a corpus of any real size, that's a quarter of engineering time and a compute bill, with no way to know whether it was worth it until it's already done.

So the model in production is the model you picked when you started. Everyone knows it's not the best one anymore. Nobody has a safe way to change it.

**The problem was never the transform. It was the risk.**

Techniques for mapping one embedding space onto another have existed in the literature for years. What hasn't existed is the thing that makes a migration shippable: a way to measure what you're about to lose, and a way to stop the migration if the answer is "too much."

That's Isotrieve.

## How it works

**1. Calibrate** — Embed a small sample of *your* corpus with both models. Hundreds of documents, not millions. No full-corpus embedding bill.

**2. Fit** — Learn a transform between the two spaces: Orthogonal Procrustes, ridge/affine, or a small residual model. Cross-dimension is supported (384 → 1536 works).

**3. Transform** — Apply the mapping to your stored vectors, in place, in your store. Your source text is never touched. The new model is never called on the full corpus.

**4. Gate** — Measure retrieval retention (Recall@k, MRR) against a held-out query set. Clears your threshold, it ships. Doesn't, it fails loudly with a report explaining why — and your index is never left half-migrated.

Step 4 is the product. Steps 1–3 are how it earns the right to exist.

## Quickstart

```bash
# Fit a mapping from calibration vectors
isotrieve calibrate \
  --source old_vectors.npy \
  --target new_vectors.npy \
  --method ridge \
  --out mapping.isotrieve

# Check retention before you touch anything
isotrieve gate \
  --mapping mapping.isotrieve \
  --queries held_out_queries.npy \
  --threshold 0.95 \
  --format json

# Exit code 0 = safe to migrate. Exit code 1 = don't.
```

Wire that gate into CI and an embedding upgrade becomes a pull request instead of a project.

Full walkthroughs live in [`notebooks/`](notebooks/) — six self-contained demos, Colab-ready, including cross-architecture dimension changes and drift recalibration.

## Vector store support

Isotrieve separates the transform (store-agnostic) from the adapter layer (store-specific), so one learned mapping applies to whatever you're actually running.

| Store | Query | Migrate in place | Dry run |
|---|---|---|---|
| ChromaDB | ✅ `IsotrieveChromaFunction` | ✅ `migrate_collection()` | ✅ |
| Qdrant | ✅ `QdrantAdapter.query()` | ✅ `QdrantAdapter.migrate()` | ✅ |
| Pinecone | ✅ `PineconeAdapter.query()` | ✅ `PineconeAdapter.migrate()` | ✅ |
| LangChain | ✅ `IsotrieveEmbeddings` | — | — |
| LlamaIndex | — | ✅ `migrate_llamaindex_store()` | ✅ |
| pgvector | planned | planned | — |
| Weaviate | hook only | — | — |
| FAISS | example only | — | — |

## How this differs from embedding adapter libraries

Several libraries translate between embedding spaces using pretrained, general-purpose adapters for popular model pairs. They're good at what they do. They solve a different problem.

| | Adapter libraries | Isotrieve |
|---|---|---|
| Mapping source | Pretrained on general data | Fit on **your** corpus |
| Where vectors live | Translated at query time | Migrated **in place**, in your store |
| Failure mode | You find out in production | Gate fails the build |
| Domain corpora | General-purpose fit | Calibrated to your distribution |

If you want to query an existing index with a different model today, use an adapter library. If you want to *permanently move* your index to a new model and be able to prove it didn't degrade, that's this.

## Claims and benchmarks

Every quantitative claim in this repo links to a committed artifact under `benchmarks/results/` and is listed in [`CLAIMS.md`](isotrieve-python/CLAIMS.md). CI enforces it — a claim without a resolvable artifact fails the build.

Retention numbers are corpus- and model-pair specific. Yours will differ from ours. That's what the gate is for.

### Adapter comparison (SciFact, MiniLM→bge-large, K=4000, 3 seeds)

| Adapter | nDCG@10 retention | Notes |
|---------|------------------|-------|
| Ridge | 0.871 ± 0.006 | Default. Fast, stable. |
| LowRank | 0.857 ± 0.009 | Compressed matrix. ~1% worse. |
| MLP | 0.727 ± 0.007 | No tuning. Linear wins. |

### K-sweep (all adapters averaged, SciFact, 3 seeds)

| K | nDCG@10 retention | Gate |
|---|------------------|------|
| 500 | 0.671 ± 0.041 | WARN |
| 1000 | 0.735 ± 0.058 | WARN |
| 2000 | 0.785 ± 0.052 | PASS |
| 4000 | 0.832 ± 0.061 | PASS |

### Same-dim pair (bge-large→e5-large, 1024→1024)

| Metric | Value |
|--------|-------|
| Floor (raw cross-space) | 0.0 |
| Isotrieve (mapped) | 0.667 |
| Ceiling (full re-embed) | 0.722 |
| Retention | 0.923 ± 0.010 |

Same dimension ≠ same space. e5 models require "query: "/"passage: " prefixes; without them ceiling drops to 0.36.

## When NOT to use Isotrieve

- Maximum retrieval quality matters more than cost → re-embed
- Calibration domain mismatches corpus (e.g., code index calibrated on prose)
- Quality gate returns FAIL → do not migrate; re-embed
- You need unsupervised migration (Isotrieve requires paired calibration)
- K < 2000 (quality degrades significantly below this)

## Anti-patterns

- Do not mix vectors from different models in one collection
- Do not assume same dimensionality means compatibility
- Do not skip the quality gate
- Do not use MLP adapter (0.727 vs 0.871 for Ridge, same cost)

## Mapping types

| Type | Use case | Inverse | Save/Load |
|------|----------|---------|-----------|
| **RidgeMapping** | Default. Unequal dims, noise robust. | No | ✅ |
| **OrthogonalProcrustesMapping** | Square dims. Best when source/target are similar spaces. | Yes | ✅ |
| **ProcrustesDiagMapping** | Square dims. Axis-aligned transform. | Yes | ✅ |
| **LowRankAffineMapping** | High-dim with limited calibration data. | No | ✅ |
| **ResidualMLPMapping** | Non-linear transforms (requires torch). Beta. | Yes | ✅ |

## CLI commands

| Command | What it does |
|---------|-------------|
| `isotrieve plan` | Estimate cost: API calls, storage, time |
| `isotrieve calibrate` | Fit mapping from calibration vectors |
| `isotrieve calibrate --queries-only` | Fit mapping from query-side calibration only |
| `isotrieve transform` | Transform stored vectors to new space |
| `isotrieve gate` | Evaluate retrieval quality (PASS/WARN/FAIL) |
| `isotrieve inspect` | Show mapping metadata and validation report |
| `isotrieve report` | Render migration report as markdown |
| `isotrieve doctor` | Check environment and dependencies |
| `isotrieve version` | Show version |

## Prior art

Isotrieve builds on [vec2vec](https://arxiv.org/abs/2505.12540), mini-vec2vec, Drift-Adapter, and the Platonic Representation Hypothesis. The contribution here is engineering — library, CLI, quality gate, store adapters, reproducible benchmarks — not algorithmic novelty. Citing the technique? Cite that work. Citing the tool? Cite this repo.

## Project structure

```
isotrieve/
├── isotrieve-python/     # Maintained Python package (PyPI: isotrieve)
├── notebooks/            # Six self-contained demos
├── benchmarks/           # Harness and committed results
├── verification/         # Audit reports, coverage, test baselines
├── assets/               # Diagrams and visual assets
└── .github/              # CI, issue templates, gate action
```

## Status

Actively developed, pre-1.0. APIs may change between minor versions until 1.0. Beta-marked adapters are functional but not yet load-tested.

Issues and PRs welcome — see [CONTRIBUTING.md](isotrieve-python/CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](isotrieve-python/LICENSE).
