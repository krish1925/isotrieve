# Contributing to Isotrieve

**Canonical contributing guide:** [`isotrieve-python/CONTRIBUTING.md`](isotrieve-python/CONTRIBUTING.md)

That file covers development setup, testing, code style, PR expectations, and the escalation protocol for design decisions.

## Quick reference

```bash
# Clone and setup
git clone https://github.com/krish1925/Isotrieve.git
cd Isotrieve/isotrieve-python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Before submitting a PR
ruff check src/ tests/           # lint
ruff format --check src/ tests/  # format
python -m pytest tests/ -q       # tests (97 pass, 6 skip expected)
```

## Project structure

- `isotrieve-python/` — the maintained, benchmark-validated Python package ([PyPI](https://pypi.org/project/isotrieve/))
- `isotrieve-npm/` — historical/experimental NPM protocol package (not actively maintained)
- `isotrieve-website/` — GitHub Pages site
- `benchmarks/` — benchmark harness and results
- `spec/` — protocol specification (RFC-001)

## Which package is current?

**`isotrieve` on PyPI** is the actively maintained, benchmark-validated package. The NPM package (`isotrieve-npm/`) is historical/experimental — the maintained project is `isotrieve-python/`.
