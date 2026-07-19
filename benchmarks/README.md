# Benchmarks

Credibility engine for AECP. Results live in [`results/`](results/) as one JSON
per run. The README results table must be generated from those files — never
hand-edited.

## Phase 1 local pair

```bash
# from repo root (needs network once for SciFact + model weights)
pip install -e "aecp-python/[benchmarks]"
python benchmarks/run_benchmark.py \
  --source-model sentence-transformers/all-MiniLM-L6-v2 \
  --target-model BAAI/bge-large-en-v1.5 \
  --k 4000 \
  --seeds 0 1 2
```

## Smoke (not for CLAIMS)

```bash
python benchmarks/run_benchmark.py --smoke
```

Writes a JSON labeled as smoke/synthetic; do not cite in CLAIMS.md.
