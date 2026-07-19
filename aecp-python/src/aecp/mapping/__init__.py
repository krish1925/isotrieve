"""Embedding-space mappings."""

from aecp.mapping.base import Mapping, ValidationReport, read_aecp_header
from aecp.mapping.linear import (
    LowRankAffineMapping,
    OrthogonalProcrustesMapping,
    ProcrustesDiagMapping,
    RidgeMapping,
)
from aecp.mapping.registry import load_mapping

__all__ = [
    "Mapping",
    "ValidationReport",
    "RidgeMapping",
    "OrthogonalProcrustesMapping",
    "ProcrustesDiagMapping",
    "LowRankAffineMapping",
    "load_mapping",
    "read_aecp_header",
]

# Optional: ResidualMLP (only importable with torch)
try:
    from aecp.mapping.mlp import ResidualMLPMapping

    __all__.append("ResidualMLPMapping")
except ImportError:
    pass
