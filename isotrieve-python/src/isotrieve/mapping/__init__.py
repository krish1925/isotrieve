"""Embedding-space mappings."""

from isotrieve.mapping.base import Mapping, ValidationReport, read_isotrieve_header
from isotrieve.mapping.linear import (
    LowRankAffineMapping,
    OrthogonalProcrustesMapping,
    ProcrustesDiagMapping,
    RidgeMapping,
)
from isotrieve.mapping.registry import load_mapping

__all__ = [
    "Mapping",
    "ValidationReport",
    "RidgeMapping",
    "OrthogonalProcrustesMapping",
    "ProcrustesDiagMapping",
    "LowRankAffineMapping",
    "load_mapping",
    "read_isotrieve_header",
]

# Optional: ResidualMLP (only importable with torch)
try:
    from isotrieve.mapping.mlp import ResidualMLPMapping

    __all__.append("ResidualMLPMapping")
except ImportError:
    pass
