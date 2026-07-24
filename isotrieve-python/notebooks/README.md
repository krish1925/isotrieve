# isotrieve Notebooks

Interactive demos showing how isotrieve migrates vector databases to new embedding models without re-embedding.

| # | Notebook | Use Case | Runtime |
|---|----------|----------|---------|
| 1 | [Quickstart: Ridge Mapping](01_quickstart_ridge_mapping.ipynb) | Fit a mapping, transform vectors, run the quality gate | ~1 min |
| 2 | [ChromaDB Migration](02_chromadb_migration.ipynb) | Migrate a ChromaDB collection to a new model | ~2 min |
| 3 | [Qdrant Migration](03_qdrant_migration.ipynb) | Same migration on Qdrant in-memory | ~2 min |
| 4 | [Adapter as Intermediary](04_adapter_as_intermediary.ipynb) | isotrieve as a transformation shim between embedder and store | ~3 min |
| 5 | [Cross-Architecture Dimension Change](05_cross_architecture_dimension_change.ipynb) | Map between different-dimensional spaces (384 -> 1536) | ~2 min |
| 6 | [Recalibration and Drift](06_recalibration_and_drift.ipynb) | Correct score drift after mapping | ~2 min |

## Running locally

```bash
cd isotrieve-python
pip install -e ".[dev]"
pip install jupyter chromadb qdrant-client
jupyter notebook notebooks/
```

## Running on Colab

Click the "Open in Colab" badge at the top of any notebook. The first cell installs dependencies automatically.
