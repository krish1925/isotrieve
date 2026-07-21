"""AECP — Embedding migration without re-embedding.

Learn a linear (or optional shallow non-linear) mapping between source and
target embedding spaces from a small calibration sample, then transform stored
vectors in place instead of re-embedding an entire corpus.

This package does **not** claim algorithmic novelty. It productizes known
linear-mapping techniques with a quality gate, providers, store adapters, and
a reproducible benchmark harness. See README "Prior Art & Research Basis".
"""

from aecp.mapping.linear import OrthogonalProcrustesMapping, RidgeMapping
from aecp.mapping.base import Mapping, ValidationReport

__version__ = "0.2.0"
__all__ = [
    "Mapping",
    "RidgeMapping",
    "OrthogonalProcrustesMapping",
    "ValidationReport",
    "__version__",
]
