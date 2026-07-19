# AECP — Embedding migration without re-embedding

Open-source toolkit to switch embedding models **without re-embedding your corpus**.
Learn a mapping from a small calibration sample, transform stored vectors in place,
and gate the migration on measured retrieval retention.

Package: [`aecp-python/`](aecp-python/) (`pip`-installable as `aecp`).

> Quantitative performance claims appear only when backed by committed artifacts
> under [`benchmarks/results/`](benchmarks/results/) and listed in
> [`aecp-python/CLAIMS.md`](aecp-python/CLAIMS.md). Older marketing numbers
> (97% fidelity, &lt;10ms, etc.) have been removed — they were not reproducible.

## Quickstart

See [`aecp-python/README.md`](aecp-python/README.md).

## Prior art

Builds on vec2vec, mini-vec2vec, Drift-Adapter, and the Platonic Representation
Hypothesis. Our contribution is engineering (library + CLI + quality gate +
benchmarks), not algorithmic novelty.

## License

Apache-2.0 (Python package). See `aecp-python/LICENSE`.
