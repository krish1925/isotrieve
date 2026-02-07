# Contributing to AECP

We are building the standard for Agent-to-Agent communication. We welcome your help!

## Getting Started

1.  **Fork and Clone**
    ```bash
    git clone https://github.com/yourusername/aecp.git
    cd aecp
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Benchmarks**
    Before submitting a PR, ensure your changes don't degrade fidelity.
    ```bash
    python benchmarks/run_benchmark.py
    ```

## Definition of Done

*   Code follows the existing style (clean, typed Python).
*   Benchmarks pass (>95% fidelity).
*   Any new feature has an example in `examples/`.
*   Any protocol change is reflected in `spec/`.

## Roadmap

*   [ ] Add support for Cohere Embeddings
*   [ ] Optimize Matrix Multiplication with quantization
*   [ ] Create a Rust implementation for maximum speed

## License

By contributing, you agree that your code will be licensed under the MIT License.
