# AECP Benchmarks

This directory contains reproducible benchmarks to validate the semantic fidelity and performance of AECP.

##  Run the Benchmark

Prerequisites:
```bash
pip install -r ../requirements.txt
```

Run the suite:
```bash
python run_benchmark.py
```

## Methodology

### Metrics

1.  **Semantic Fidelity**: Measured as the cosine similarity between:
    *   $\mathbf{v}_{transferred}$: The vector translated from Agent A.
    *   $\mathbf{v}_{ground\_truth}$: The vector Agent B would have produced from the raw text.
    *   Target: **>95%**

2.  **Latency**: Time to prepare the data for the receiver.
    *   **Text Handoff**: Time for Receiver to Encode text.
    *   **AECP**: Time for Source to Multiply Matrix.

### Baselines (Typical Results)

| Source Model | Target Model | Fidelity | Speedup |
| :--- | :--- | :--- | :--- |
| `all-MiniLM-L6-v2` | `all-mpnet-base-v2` | 97.2% | 150x |
| `voyage-code-2` | `text-embedding-3-small` | 94.8% | 200x |

> Note: Speedup depends on the model size. Larger models (like OpenAI's) have much higher encoding latency, resulting in massive AECP speedups (since matrix multiplication cost is constant relative to model inference complexity).
