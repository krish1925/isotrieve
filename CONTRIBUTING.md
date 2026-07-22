# Contributing to AECP

**Canonical contributing guide:** [`aecp-python/CONTRIBUTING.md`](aecp-python/CONTRIBUTING.md)

That file covers development setup, testing, code style, PR expectations, and the escalation protocol for design decisions.

## Quick reference

```bash
# Clone and setup
git clone https://github.com/krish1925/AECP.git
cd AECP/aecp-python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Before submitting a PR
ruff check src/ tests/           # lint
ruff format --check src/ tests/  # format
python -m pytest tests/ -q       # tests (97 pass, 6 skip expected)
```

## Project structure

- `aecp-python/` — the maintained, benchmark-validated Python package ([PyPI](https://pypi.org/project/aecp/))
- `aecp-npm/` — historical/experimental NPM protocol package (not actively maintained)
- `aecp-website/` — GitHub Pages site
- `benchmarks/` — benchmark harness and results
- `spec/` — protocol specification (RFC-001)

## Which package is current?

**`aecp` on PyPI** is the actively maintained, benchmark-validated package. The NPM package (`aecp-npm/`) is historical/experimental — the maintained project is `aecp-python/`.
