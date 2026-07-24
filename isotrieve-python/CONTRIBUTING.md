# Contributing to Isotrieve

Thank you for contributing to isotrieve — migration CI for vector stores. This guide is the single source of truth for development setup, testing, code style, and PR expectations.

## Quick Start

```bash
git clone https://github.com/krish1925/isotrieve.git
cd isotrieve/isotrieve-python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # core + dev tools
pip install -e ".[sentence-transformers]"  # optional: CLIP / SigLIP models
```

Verify:

```bash
ruff check src/ tests/            # lint
ruff format --check src/ tests/   # format
mypy src/isotrieve                # types
python -m pytest tests/ -q        # expect ~123 pass, 1 skip
```

All four must be green before opening a PR.

## Project Structure

```
isotrieve-python/
  src/isotrieve/
    providers/          Embedding providers (text + image)
      base.py             Embedder ABC
      sentence_transformers.py  SentenceTransformerEmbedder (text)
      image_providers.py  CLIPEmbedder, SigLIPEmbedder (image)
      cached.py           Disk-cached wrapper
      factory.py          create_embedder() from model name
    mapping/            Embedding-space transformations
      base.py             Mapping ABC, ValidationReport, header format
      linear.py           Ridge, OrthogonalProcrustes, ProcrustesDiag, LowRankAffine
      contrastive.py      ContrastiveMapping (InfoNCE-based, cross-modal)
      mlp.py              ResidualMLPMapping (optional, requires torch)
      registry.py         load_mapping(), register_mapping()
    calibration/        Calibration pipelines
      corpus.py           Built-in calibration texts, checksum, sampling
      image_calibration.py  Image loading, paired embedding, corpus embedding
    quality/            Quality gate
    serve.py            QueryAdapter for zero-corpus-write serving
    cli.py              Typer CLI (isotrieve plan/calibrate/transform/inspect)
    __init__.py         __version__, public exports
  tests/
    test_matrix.py, test_calibration.py, test_mapping.py, test_providers.py, ...
    test_image_migration.py  Image/multimodal consumer tests
    test_release.py     Release validation (16 checks)
    test_cli.py         CLI integration tests
```

## Architecture: Two Provider Pipelines

isotrieve supports two embedding pipelines that share the same mapping layer:

| Pipeline | Provider | Input | Interface |
|----------|----------|-------|-----------|
| **Text** | `SentenceTransformerEmbedder` | `str` | `embed(texts)` |
| **Image** | `CLIPEmbedder`, `SigLIPEmbedder` | `Path` | `embed_images(images)` |

Both implement the `Embedder` ABC. The mapping layer is pipeline-agnostic — `RidgeMapping` works on any `np.ndarray` regardless of whether the vectors came from text or images.

### Adding a New Provider

1. Create `src/isotrieve/providers/your_provider.py`
2. Implement `Embedder` ABC with `embed(texts)` and/or `embed_images(images)`
3. If it's a new pipeline type, update the ABC or create a sibling ABC
4. Export in `providers/__init__.py`
5. Write unit tests (mock the model, test shapes + dtypes)
6. Write integration test with `@pytest.mark.slow` if it downloads models

```python
from isotrieve.providers.base import Embedder

class YourProvider(Embedder):
    @property
    def model_id(self) -> str: ...
    @property
    def dims(self) -> int: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_images(self, images: list[Path]) -> np.ndarray: ...  # if applicable
```

## Testing

### Test Categories

| Category | Marker | When it runs | What it tests |
|----------|--------|-------------|---------------|
| **Unit** | (none) | Always | Shapes, dtypes, math, save/load roundtrips |
| **Slow** | `@pytest.mark.slow` | Opt-in | Real model download + inference |
| **Integration** | (none) | Always | CLI, end-to-end pipelines |

### Running Tests

```bash
# All tests (fast, no model download)
pytest -q

# Only slow tests (downloads CLIP models)
pytest -m slow -v

# Specific test file
pytest tests/test_image_migration.py -v

# Specific test class
pytest tests/test_image_migration.py::TestContrastiveMapping -v

# With coverage
pytest --cov=isotrieve --cov-report=html
```

### Test Markers

- **`@pytest.mark.slow`** — requires model download (sentence-transformers models). Only run in CI with cache, or locally if you have the models.
- **`SKIP_REASON`** — every test that downloads models must define `SKIP_REASON` and call `pytest.skip(SKIP_REASON)` on import failure. Tests must never fail due to missing models.

### What Must Be Tested

**For new mapping types:**
- `test_basic_fit_transform` — fit on synthetic data, verify shape + finiteness
- `test_inverse_transform` — round-trip X → Y → X preserves structure
- `test_save_load_roundtrip` — save → load → transform produces identical output
- `test_different_dims` — d_src ≠ d_tgt works correctly
- `test_validation_report` — report fields are populated and sensible

**For new providers:**
- `test_embed_shapes` — output shape matches `(N, dims)`
- `test_embed_dtypes` — float32 or float64, finite
- `test_embed_images_shapes` — image pipeline produces correct shapes
- `test_batch_consistency` — batch=1 vs batch=32 produces same output

**For calibration pipelines:**
- `test_load_images_from_directory` — finds images, respects extensions
- `test_paired_images` — matching filenames pair correctly
- `test_paired_images_no_match` — raises ValueError on mismatch
- `test_embed_image_pairs` — produces correct shapes

**For CLI:**
- Test via `test_cli.py` — runner, --help, --json output
- New commands need CLI tests

### Writing Good Tests

```python
# 1. Use synthetic data for unit tests — no model download
rng = np.random.default_rng(42)
X = rng.normal(size=(100, 64))
Y = rng.normal(size=(100, 64))

# 2. Verify shapes AND finiteness
Z = m.transform(X)
assert Z.shape == (100, d_tgt)
assert np.all(np.isfinite(Z))

# 3. Round-trip test
loaded = MappingClass.load(path)
np.testing.assert_allclose(m.transform(X), loaded.transform(X), atol=1e-10)

# 4. Consumer scenario test — realistic workflow
# Create DB, embed with Model A, fit mapping, transform, verify retention
```

## Code Style

### Tools (all mandatory)

| Tool | Purpose | Command |
|------|---------|---------|
| **ruff** | Lint + format | `ruff check src/ tests/` / `ruff format src/ tests/` |
| **mypy** | Type checking | `mypy src/isotrieve` |

No black, isort, or flake8. Ruff replaces all three.

### ruff Configuration (pyproject.toml)

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
ignore = ["B008", "B904", "B905", "E501", "E402", "SIM105", "SIM108"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
"examples/*" = ["E731"]
```

### mypy Configuration

```toml
[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
```

### Code Rules

1. **No type: ignore without comment.** Every `# type: ignore` must explain what's being suppressed and why.

2. **No bare except.** Always catch specific exceptions.

3. **No hardcoded paths.** Use `pathlib.Path` and pass paths as parameters.

4. **No global mutable state.** Mappings must be stateless after fit — all state in `self` attributes, not module-level variables.

5. **No model downloads in unit tests.** Use synthetic data. Model downloads belong in `@pytest.mark.slow` tests only.

6. **No magic numbers in tests.** Use named constants:
   ```python
   k = 100      # number of samples
   d = 64       # embedding dimension
   threshold = 0.5  # expected minimum retention
   ```

7. **Tests must be deterministic.** Always seed RNGs: `np.random.default_rng(42)`. Never rely on filesystem state or external services.

8. **All public methods must have type hints.** Input and output types. Use `numpy.ndarray` for arrays, `Path` for files, `str | Path` for flexible paths.

9. **Docstrings on all public classes and methods.** NumPy-style. Parameters, Returns, Raises. No fluff.

10. **No print() in library code.** Use `logging` if you need to output. Print only in examples and CLI output.

11. **No unused imports.** Ruff catches this — keep it clean.

12. **Commit messages must match the repo style:**
    - `feat:` new feature
    - `fix:` bug fix
    - `chore:` maintenance
    - `docs:` documentation only
    - `test:` test additions/changes
    - `refactor:` code change that neither fixes a bug nor adds a feature

## Branch Strategy

```
main              ← stable, tagged releases, GitHub Pages
  └── development ← integration branch, CI runs on every push
        └── feat/your-branch   ← feature branches
```

### Branch Naming

| Prefix | Use |
|--------|-----|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `chore/` | Maintenance, deps, cleanup |
| `docs/` | Documentation only |

### Workflow

1. Branch from `development`
2. Make changes, commit with conventional messages
3. Open PR into `development`
4. CI must pass: lint, typecheck, test (3.10, 3.11, 3.12, 3.13)
5. At least 1 approval required
6. Squash merge into `development`
7. Periodically merge `development` → `main` for releases

### Branch Protection (main)

- 6 CI checks required: lint, typecheck, test × 4 Python versions
- 1 PR approval required
- Admin enforcement enabled
- No force pushes, no direct commits

## PR Guidelines

### Size Limit

**< 600 lines of diff.** Larger PRs get rejected — split into smaller pieces.

### PR Checklist

- [ ] `ruff check src/ tests/` passes
- [ ] `ruff format --check src/ tests/` passes
- [ ] `mypy src/isotrieve` passes
- [ ] `pytest -q` passes (123+ pass, 1 skip)
- [ ] New code has tests (unit + integration where applicable)
- [ ] CHANGELOG.md updated (for releases)
- [ ] No secrets or API keys committed
- [ ] Type hints on all public functions
- [ ] Docstrings on all public classes/methods

### What Belongs in a PR

- One logical change per PR
- Tests alongside the code they test
- No drive-by refactors
- No formatting-only changes mixed with feature work

### Escalation Protocol

Before building significant new infrastructure, discuss in a GitHub issue first:

- New mapping type
- New provider pipeline (e.g., audio, video)
- Breaking changes to public API
- New CLI commands
- Changes to quality gate logic

## Image / Multimodal Support (In Progress)

The image pipeline is being developed on `feat/image-multimodal-support`. Here's what's implemented, tested, and needs work.

### What's Done

| Component | File | Status |
|-----------|------|--------|
| `CLIPEmbedder` | `providers/image_providers.py` | Implemented, unit tested |
| `SigLIPEmbedder` | `providers/image_providers.py` | Implemented, unit tested |
| `ContrastiveMapping` | `mapping/contrastive.py` | Implemented, unit tested |
| Image calibration pipeline | `calibration/image_calibration.py` | Implemented |
| Consumer test (unit) | `tests/test_image_migration.py` | Implemented |
| Consumer test (integration) | `tests/test_image_migration.py` | Implemented, `@slow` |
| Registry registration | `mapping/registry.py` | Done |
| Exports | `mapping/__init__.py`, `providers/__init__.py` | Done |

### What Needs Testing / Validation

| Gap | Why It Matters | How to Validate |
|-----|---------------|-----------------|
| SigLIP ↔ CLIP cross-model migration | Real-world consumer scenario | Integration test: embed with both, fit ContrastiveMapping, verify recall@k |
| Large image corpus (1000+ images) | Memory + batch processing | Stress test: embed 1000 images, verify OOM doesn't happen |
| Different image sizes / formats | Robustness | Test with PNG, JPEG, WebP, different resolutions |
| CLI integration for image calibration | User-facing workflow | Add `--images` flag to `calibrate` command, test via CLI |
| Save/load with image mappings | Persistence | Round-trip test with ContrastiveMapping saved to `.isotrieve` |
| Cross-modal (text ↔ image) | CLIP's native capability | Test: text queries finding images, image queries finding text |
| Performance benchmarks | Comparison with re-embedding | Time: fit + transform vs re-embed, measure speedup |

### How to Run Image Tests

```bash
# Unit tests only (no model download)
pytest tests/test_image_migration.py -m "not slow" -v

# Integration tests (downloads CLIP models, ~2min first time)
pytest tests/test_image_migration.py -m slow -v

# All image tests
pytest tests/test_image_migration.py -v
```

### Key Design Decisions

- **ContrastiveMapping uses InfoNCE loss** — differentiates from Ridge (which assumes linear correspondence). InfoNCE learns contrastive alignment, better for cross-modal pairs.
- **Temperature parameter** — controls sharpness of alignment. 0.07 is CLIP default, 0.1 is reasonable starting point.
- **Alternating optimization** — Ridge init for warm start, then Nelder-Mead fine-tune. Avoids gradient computation.
- **Rank constraint** — optional `rank` parameter prevents overfitting on small calibration sets.
- **Retrieval-native validation** — `ValidationReport` uses recall@k instead of raw cosine similarity, which is what actually matters for vector search.

### Coding Rules for Image Work

1. **Image providers must implement both `embed()` and `embed_images()`** — CLIP models naturally do both.
2. **Never assume same dimensionality** — source and target models almost always have different dims.
3. **Test with synthetic images first** — use `_make_test_images()` helper, not real photos.
4. **Integration tests must be `@pytest.mark.slow`** — they download models.
5. **Image paths must be `str | Path`** — accept both, convert internally.
6. **Batch embedding must handle odd batch sizes** — don't assume N is divisible by batch_size.

## Reporting Issues

Include:
- Python version (`python --version`)
- isotrieve version (`python -c "import isotrieve; print(isotrieve.__version__)"`)
- OS
- Minimal reproducible example
- Full error traceback
- Whether you're using text or image pipeline

## License

By contributing, you agree your contributions will be licensed under the MIT License.
