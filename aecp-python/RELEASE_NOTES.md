# aecp 0.1.0

Initial release. Embedding migration without re-embedding.

## What it does

Fits a linear map between embedding spaces from a small calibration sample (~2K texts), then transforms stored vectors locally. 87-91% retrieval retention measured on BEIR benchmarks.

## Install

```bash
pip install aecp
```

## Highlights

- **RidgeMapping** with auto alpha selection — handles rectangular dims (e.g., 1536→3072)
- **QueryAdapter serve mode** — map queries into legacy space, zero corpus writes, instant rollback
- **QualityGate v2** — predicts retrieval retention from holdout proxies using isotonic regression
- **CLI** — `aecp plan`, `aecp calibrate`, `aecp transform`, `aecp inspect`
- **Store adapters** — NumpyFileStore, QdrantStore with resumable migration
- **5 adapters** — Ridge, Procrustes, ProcrustesDiag, LowRankAffine, ResidualMLP

## Benchmarks

| Adapter | nDCG@10 retention (SciFact, K=4000, 3 seeds) |
|---------|----------------------------------------------|
| Ridge | 0.866 ± 0.008 |
| LowRank | 0.857 ± 0.009 |
| MLP | 0.719 ± 0.008 |

Same-dim pair (bge-large→e5-large): 90.8% retention.

All numbers from `benchmarks/results/`, verified by `benchmarks/audit_configs.py`.

## What's next

- API model pair benchmarks (ada-002→te3-large)
- Chroma store adapter
- MCP wrapper for agent frameworks
- conda-forge package

## License

Apache-2.0
