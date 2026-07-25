# Contributing to Isotrieve

Thank you for contributing to isotrieve — migration CI for vector stores. This guide covers development setup, testing, code style, and PR expectations.

## Quick Start

```bash
git clone https://github.com/krish1925/isotrieve.git
cd isotrieve/isotrieve-python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Verify everything works:

```bash
ruff check src/ tests/            # lint
ruff format --check src/ tests/   # format
mypy src/isotrieve                # types
python -m pytest tests/ -q        # tests
```

All four must pass before opening a PR.

## Project Structure

```
isotrieve-python/
  src/isotrieve/
    __init__.py         __version__, public exports
    mapping/            Embedding-space transformations
      base.py             Mapping ABC, ValidationReport
      linear.py           Ridge, Procrustes, LowRankAffine
      mlp.py              ResidualMLPMapping (optional, requires torch)
      contrastive.py      ContrastiveMapping (cross-modal, feature branch)
      registry.py         load_mapping(), register_mapping()
    providers/          Embedding providers
      base.py             Embedder ABC
      sentence_transformers.py  SentenceTransformerEmbedder
      cached.py           Disk-cached wrapper
      factory.py          create_embedder() from model name
    adapters/           Vector store adapters
      base.py             VectorStoreAdapter ABC
      chroma.py           ChromaDB adapter + migrate_collection()
      qdrant.py           QdrantAdapter (scroll + upsert)
      pinecone.py         PineconeAdapter (shadow namespace)
      langchain.py        IsotrieveEmbeddings shim
      llamaindex_store.py LlamaIndex migration helpers
    calibration/        Calibration pipelines
      corpus.py           Built-in calibration texts
      calib_v1.py         Frozen generic calibration corpus
      planner.py          Cost estimation, K recommendation
    quality/            Quality gate
      gate.py             QualityGate, retention prediction
      metrics.py          Recall, MRR, bootstrap CIs
    serve.py            QueryAdapter for zero-corpus-write serving
    cli.py              Typer CLI (plan/calibrate/transform/gate/inspect)
    cli_gate.py         Gate CLI command
    cli_doctor.py       Doctor CLI command
    cli_report.py       Report CLI command
    migrate.py          Migration orchestration
    recalibration.py    Score recalibration (isotonic regression)
    reranking.py        Cross-encoder reranking (deprecated — NULL RESULT)
    stores/             Vector store backends
      base.py             VectorRecord, Store ABC
      numpy_files.py      NumpyFileStore
      qdrant_store.py     QdrantStore
  tests/
    test_mapping.py     Core mapping tests (Ridge, Procrustes, etc.)
    test_gate.py        Quality gate tests
    test_chroma_adapter.py  ChromaDB adapter tests
    test_qdrant_adapter.py  Qdrant adapter tests
    test_release.py     Release validation
    test_cli.py         CLI integration tests
    fakes.py            Shared test helpers
    ...                 (19 test files, ~158 tests total)
  scripts/
    lint_claims.py      CLAIMS.md artifact linter
```

## Code Style

### Tools (all mandatory)

| Tool | Purpose | Command |
|------|---------|---------|
| **ruff** | Lint + format | `ruff check src/ tests/` / `ruff format src/ tests/` |
| **mypy** | Type checking | `mypy src/isotrieve --ignore-missing-imports` |

No black, isort, or flake8. Ruff replaces all three.

### ruff Configuration (pyproject.toml)

```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["B008", "B904", "B905", "E501", "E402", "SIM105", "SIM108"]

[tool.ruff.lint.per-file-ignores]
"src/isotrieve/mapping/__init__.py" = ["F401"]
"tests/*" = ["E731"]
```

### Code Rules

1. **No `type: ignore` without comment.** Every suppression must explain why.
2. **No bare except.** Always catch specific exceptions.
3. **No hardcoded paths.** Use `pathlib.Path`.
4. **No global mutable state.** All state in `self` attributes.
5. **No model downloads in unit tests.** Use synthetic data. Model tests go in `@pytest.mark.slow`.
6. **Deterministic tests.** Seed RNGs: `np.random.default_rng(42)`.
7. **Type hints on all public methods.** Input and output types.
8. **Docstrings on all public classes/methods.** NumPy-style.
9. **No `print()` in library code.** Use `logging`.
10. **No unused imports.** Ruff catches this.

## Testing

### Test Categories

| Category | Marker | When | What |
|----------|--------|------|------|
| **Unit** | (none) | Always | Shapes, dtypes, math, save/load |
| **Slow** | `@pytest.mark.slow` | Opt-in | Real model download + inference |
| **Integration** | (none) | Always | CLI, end-to-end pipelines |

### Running Tests

```bash
pytest -q                              # all fast tests
pytest -m slow -v                      # only slow tests (downloads models)
pytest tests/test_mapping.py -v        # specific file
pytest --cov=isotrieve --cov-report=html  # coverage
```

### What Must Be Tested

**For new mapping types:**
- `test_basic_fit_transform` — fit on synthetic data, verify shape + finiteness
- `test_inverse_transform` — round-trip X -> Y -> X
- `test_save_load_roundtrip` — save -> load -> identical output
- `test_different_dims` — d_src != d_tgt
- `test_validation_report` — fields populated

**For new adapters:**
- `test_query` — round-trip query returns results
- `test_migrate` — vectors transfer correctly
- `test_dry_run` — no writes when dry_run=True
- Mock external services in unit tests

**For CLI:**
- Test via `test_cli.py` — runner, --help, --json output

### Writing Good Tests

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(100, 64))
Y = rng.normal(size=(100, 64))

Z = m.transform(X)
assert Z.shape == (100, d_tgt)
assert np.all(np.isfinite(Z))
```

## Branch Strategy

```
main              <- stable, tagged releases
  development     <- integration branch, CI runs on every push
    feat/branch   <- feature branches
```

### Workflow

1. Branch from `development`
2. Commit with conventional messages (`feat:`, `fix:`, `docs:`, `chore:`)
3. Open PR into `development`
4. CI must pass: lint, typecheck, test (3.10, 3.11, 3.12)
5. Squash merge into `development`
6. Periodically merge `development` -> `main` for releases

## PR Guidelines

### Size Limit

**< 600 lines of diff.** Split larger PRs.

### PR Checklist

- [ ] `ruff check src/ tests/` passes
- [ ] `ruff format --check src/ tests/` passes
- [ ] `mypy src/isotrieve --ignore-missing-imports` passes
- [ ] `pytest -q` passes
- [ ] New code has tests
- [ ] CHANGELOG.md updated (for releases)
- [ ] No secrets or API keys committed
- [ ] Type hints on all public functions

### Escalation

Discuss in a GitHub issue first for:
- New mapping types or provider pipelines
- Breaking API changes
- New CLI commands
- Changes to quality gate logic

## Reporting Issues

Include:
- Python version, isotrieve version, OS
- Minimal reproducible example
- Full error traceback

## License

Contributions are licensed under Apache-2.0.
