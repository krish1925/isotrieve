# Contributing to Isotrieve

Thank you for your interest in contributing to Isotrieve! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all backgrounds and experience levels.

## Getting Started

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/isotrieve.git
cd isotrieve/isotrieve-python
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Run tests to verify setup:
```bash
pytest
```

## Development Workflow

### Code Style

We use the following tools for code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks:
```bash
black isotrieve tests
isort isotrieve tests
flake8 isotrieve tests
mypy isotrieve
```

### Testing

Write tests for all new functionality:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=isotrieve --cov-report=html

# Run specific test file
pytest tests/test_matrix.py

# Run with verbose output
pytest -v
```

### Type Hints

All public functions should have type hints:

```python
def transfer_embedding(
    embedding: np.ndarray,
    transfer_matrix: np.ndarray
) -> np.ndarray:
    """
    Transfer an embedding to a different space.
    
    Args:
        embedding: Input embedding vector
        transfer_matrix: Transfer matrix
        
    Returns:
        Transformed embedding
    """
    ...
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/my-feature`
3. **Make your changes** with tests
4. **Run all checks**: `black`, `isort`, `flake8`, `mypy`, `pytest`
5. **Commit** with clear messages
6. **Push** to your fork
7. **Open a Pull Request**

### PR Guidelines

- Keep PRs focused on a single feature or fix
- Include tests for new functionality
- Update documentation as needed
- Ensure all CI checks pass

## Adding a New Adapter

To add support for a new embedding provider:

1. Create a new file in `isotrieve/adapters/`:

```python
# isotrieve/adapters/newprovider.py
from typing import List, Optional
from .base import BaseAdapter

class NewProviderAdapter(BaseAdapter):
    def __init__(self, api_key: str, model: str = "default-model", **kwargs):
        super().__init__(**kwargs)
        # Initialize client
        
    def _embed_impl(self, text: str) -> List[float]:
        # Implementation
        pass
    
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        # Implementation
        pass
    
    def get_dimensions(self) -> int:
        return self._dimensions
    
    def get_model_id(self) -> str:
        return f"newprovider:{self.model}"
```

2. Add to `isotrieve/adapters/__init__.py`:

```python
from .newprovider import NewProviderAdapter
__all__ = [..., "NewProviderAdapter"]
```

3. Add optional dependency in `pyproject.toml`:

```toml
[project.optional-dependencies]
newprovider = ["newprovider-sdk>=1.0.0"]
```

4. Write tests in `tests/test_adapters.py`

5. Update README with new provider

## Reporting Issues

When reporting issues, please include:

- Python version
- Isotrieve version
- Operating system
- Minimal reproducible example
- Full error traceback

## Questions?

Open a GitHub issue with the "question" label.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
