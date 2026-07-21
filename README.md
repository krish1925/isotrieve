# AECP — Embedding migration without re-embedding

[![PyPI](https://img.shields.io/pypi/v/aecp)](https://pypi.org/project/aecp/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](aecp-python/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](aecp-python/)

Open-source toolkit to switch embedding models **without re-embedding your corpus**.
Learn a mapping from a small calibration sample, transform stored vectors in place,
and gate the migration on measured retrieval retention.

## The problem

Upgrading from `ada-002` to `text-embedding-3`, or swapping any embedding provider,
normally means re-embedding the entire corpus, re-indexing, and hoping nothing
breaks downstream. For any non-trivial vector store, that's an engineering project
nobody wants to own — not a config change. Most teams just... don't upgrade.

AECP learns a lightweight transform between the old and new embedding spaces from
a small calibration sample, applies it to your existing vectors, and reports the
retrieval retention (Recall@k, MRR) *before* you commit to the migration. If
retention doesn't clear your threshold, the gate fails and nothing ships.

## How it works

```mermaid
flowchart LR
    A[Calibration sample<br/><i>small subset of corpus</i>] --> B1[Embed with<br/>old model]
    A --> B2[Embed with<br/>new model]
    B1 --> C[Fit transform<br/><i>Procrustes / ridge / residual</i>]
    B2 --> C
    C --> D[Apply transform to<br/>full stored vector set]
    D --> E{Retrieval gate<br/>Recall@k, MRR vs.<br/>held-out queries}
    E -- "≥ threshold" --> F[✅ Commit migration<br/>vectors updated in place]
    E -- "below threshold" --> G[⛔ Abort<br/>report gap + fallback]

    classDef stage fill:#1f2937,stroke:#4b5563,color:#f9fafb;
    classDef gate fill:#78350f,stroke:#d97706,color:#fef3c7;
    classDef pass fill:#14532d,stroke:#22c55e,color:#dcfce7;
    classDef fail fill:#7f1d1d,stroke:#ef4444,color:#fee2e2;
    class A,B1,B2,C,D stage;
    class E gate;
    class F pass;
    class G fail;
```

1. **Calibrate** — embed a small sample of your corpus (or a provided default set)
   with both the old and new models. No full-corpus embedding calls required.
2. **Fit** — learn a transform (Orthogonal Procrustes, ridge/affine, or a small
   residual model) mapping new-model space onto old-model space, or vice versa.
3. **Transform** — apply the learned map to your stored vectors in place, without
   touching the source text or calling the new embedding model on the full corpus.
4. **Gate** — measure retrieval retention against a held-out query set. The
   migration only proceeds if retention clears a configurable threshold; otherwise
   AECP reports why and falls back safely — your index is never left in a
   half-migrated state.

## Package

[`aecp-python/`](aecp-python/) — pip-installable as `aecp`.

```bash
pip install aecp
```

See [`aecp-python/README.md`](aecp-python/README.md) for the full quickstart,
CLI usage, and adapter-specific guides.

## Vector store adapters

AECP separates the transform logic (store-agnostic) from the adapter layer
(store-specific read/write), so the same learned mapping can be applied to
whatever you're actually running in production:

```mermaid
flowchart TB
    subgraph Core["aecp core (store-agnostic)"]
        direction LR
        Calib[Calibration + fit]
        Xform[Transform]
        Gate[Retrieval gate]
        Calib --> Xform --> Gate
    end

    Core --> Chroma[(ChromaDB<br/>supported)]
    Core --> PGV[(pgvector<br/>supported)]
    Core --> LC[LangChain<br/>AECPEmbeddings<br/>query-time wrapper]
    Core --> LI[(LlamaIndex<br/>in progress)]
    Core -.-> QD[(Qdrant<br/>hook-only)]
    Core -.-> PC[(Pinecone<br/>hook-only)]
    Core -.-> WV[(Weaviate<br/>hook-only)]
    Core -.-> FA[FAISS<br/>notebook example only]

    classDef core fill:#1e3a5f,stroke:#3b82f6,color:#eff6ff;
    classDef ready fill:#14532d,stroke:#22c55e,color:#dcfce7;
    classDef partial fill:#78350f,stroke:#d97706,color:#fef3c7;
    classDef planned fill:#374151,stroke:#6b7280,color:#e5e7eb,stroke-dasharray: 4 3;
    class Calib,Xform,Gate core;
    class Chroma,PGV,LC ready;
    class LI partial;
    class QD,PC,WV,FA planned;
```

Solid arrows mean an in-place migration path exists today; dashed arrows mean
only a hook or example is available. See the status table for detail:

| Store | Status |
|---|---|
| ChromaDB | Supported |
| pgvector | Supported |
| LangChain (`AECPEmbeddings`) | Supported (query-time wrapper, store-agnostic) |
| LlamaIndex | In progress |
| Qdrant | Hook-only |
| Pinecone | Hook-only |
| Weaviate | Hook-only |
| FAISS | Notebook example only (no persistence layer to migrate against) |

## On claims and benchmarks

Quantitative performance claims in this repo appear **only** when backed by
committed artifacts under [`benchmarks/results/`](benchmarks/results/) and listed
in [`aecp-python/CLAIMS.md`](aecp-python/CLAIMS.md). If a number isn't in `CLAIMS.md` with a linked artifact, treat it
as unverified — that also means: don't take our word for it, rerun the gate on
your own corpus and model pair before you migrate.

## Prior art

AECP builds on [vec2vec](https://arxiv.org/abs/2505.12540), mini-vec2vec,
Drift-Adapter, and the Platonic Representation Hypothesis. Our contribution is
engineering — library, CLI, quality gate, adapters, and benchmarks — not
algorithmic novelty. If you're citing the underlying technique, cite that prior
work; if you're citing the tool, cite this repo.

## Status

Early-stage, actively developed. APIs may change between minor versions until
1.0. Issues and PRs welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Apache-2.0 (Python package). See [`aecp-python/LICENSE`](aecp-python/LICENSE).
