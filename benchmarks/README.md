# Benchmarks

Credibility engine for Isotrieve. Results live in [`results/`](results/) as one JSON
per run. The README results table must be generated from those files — never
hand-edited.

## Phase 1 local pair

```bash
# from repo root (needs network once for SciFact + model weights)
pip install -e "isotrieve-python/[benchmarks]"
python benchmarks/run_benchmark.py \
  --source-model sentence-transformers/all-MiniLM-L6-v2 \
  --target-model BAAI/bge-large-en-v1.5 \
  --k 4000 \
  --seeds 0 1 2
```

## Domain matrix (issue #37)

The harness maps each dataset to a domain regime — SciFact/nfcorpus = `medical`,
FiQA = `general` — and additionally supports offline, synthetic identifier-probe
corpora for `code` and `legal` (no BEIR download needed). Run any of:

```bash
python benchmarks/run_benchmark.py --dataset scifact --adapter ridge lowrank --k 500 --seeds 0 1 --max-docs 2000
python benchmarks/run_benchmark.py --dataset fiqa   --adapter ridge lowrank --k 500 --seeds 0 1 --max-docs 2000
python benchmarks/run_benchmark.py --dataset code   --adapter ridge lowrank --k 500 --seeds 0 1
python benchmarks/run_benchmark.py --dataset legal  --adapter ridge lowrank --k 500 --seeds 0 1
```

Every run reports `probe_retention` alongside nDCG@10: exact identifiers
(ICD-10 codes, case citations, file paths, error codes, UUIDs, dates, version
strings) embedded as a small probe corpus and measured for same-index true-match
retention after the mapping transform (top-1/top-10). Probe definitions live in
[`domain_probes.py`](domain_probes.py); they are deterministic and offline.

### Medical (SciFact, MiniLM→bge-large, K=500, max-docs=2000, seeds 0-1)

| Adapter | nDCG@10 retention | Probe top-1 | Probe top-10 |
|---------|-------------------|-------------|--------------|
| Ridge | 0.813 ± 0.020 | 0.658 | 0.973 |
| LowRank | 0.810 ± 0.022 | 0.685 | 0.982 |

### General (FiQA, MiniLM→bge-large, K=500, max-docs=2000, seeds 0-1)

| Adapter | nDCG@10 retention | Probe top-1 | Probe top-10 |
|---------|-------------------|-------------|--------------|
| Ridge | 0.773 ± 0.005 | 0.693 | 1.000 |
| LowRank | 0.768 ± 0.012 | 0.693 | 0.991 |

Note: with max-docs=2000 the FiQA ceiling is near-zero (0.032) because most
qrels-relevant docs lie beyond the first 2k of the corpus; treat the FiQA
nDCG@10 retention as high-variance. Probe retention is corpus-truncation
independent.

### Code / Legal (offline synthetic probe corpora, 520 docs)

| Dataset | Adapter | nDCG@10 retention | Probe top-1 | Probe top-10 |
|---------|---------|-------------------|-------------|--------------|
| code | Ridge/LowRank | 1.022 ± 0.114 | 0.390 | 1.000 |
| legal | Ridge/LowRank | 1.037 ± 0.126 | 0.469 | 0.946 |

Synthetic corpora are small; retention can exceed 1.0 because the mapped
retrieval can beat the target model's own ceiling on such a small index.

### Cross-domain probe top-1 (identifier retention after mapping, Ridge)

| Mapping fit on → | general | legal | medical | code |
|------------------|---------|-------|---------|------|
| FiQA (general) | 0.650 | 0.667 | 0.600 | 0.857 |
| SciFact (medical) | 0.550 | 0.611 | 0.650 | 0.821 |
| code (synthetic) | 0.200 | 0.111 | 0.250 | 1.000 |
| legal (synthetic) | 0.400 | 0.889 | 0.300 | 0.286 |

In-domain probe retention is consistently highest; cross-domain identifier
retention drops sharply for code/legal-fit mappings, evidence that a mapping fit
on one domain transfers imperfectly to identifiers from another domain.

## Smoke (not for CLAIMS)

```bash
python benchmarks/run_benchmark.py --smoke
```

Writes a JSON labeled as smoke/synthetic; do not cite in CLAIMS.md.
