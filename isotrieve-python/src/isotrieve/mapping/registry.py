"""Mapping type registry for ``.isotrieve`` load dispatch."""

from __future__ import annotations

from pathlib import Path

from isotrieve.mapping.base import Mapping, load_isotrieve_payload

_REGISTRY: dict[str, type[Mapping]] = {}


def register_mapping(cls: type[Mapping]) -> type[Mapping]:
    """Register a Mapping subclass by its ``mapping_type`` attribute."""
    _REGISTRY[cls.mapping_type] = cls
    return cls


def get_mapping_class(mapping_type: str) -> type[Mapping]:
    if mapping_type not in _REGISTRY:
        _ensure_builtins()
    if mapping_type not in _REGISTRY:
        raise KeyError(f"Unknown mapping_type: {mapping_type!r}")
    return _REGISTRY[mapping_type]


def _ensure_builtins() -> None:
    from isotrieve.mapping.linear import (
        LowRankAffineMapping,
        OrthogonalProcrustesMapping,
        ProcrustesDiagMapping,
        RidgeMapping,
    )

    register_mapping(RidgeMapping)
    register_mapping(OrthogonalProcrustesMapping)
    register_mapping(ProcrustesDiagMapping)
    register_mapping(LowRankAffineMapping)

    # Optional: ResidualMLP (only if torch available)
    try:
        from isotrieve.mapping.mlp import ResidualMLPMapping

        register_mapping(ResidualMLPMapping)
    except ImportError:
        pass


def load_mapping(path: str | Path) -> Mapping:
    """Load any registered mapping from a ``.isotrieve`` file."""
    _ensure_builtins()
    header, W, W_inv, extra = load_isotrieve_payload(path)
    mapping_type = header["mapping_type"]

    # Special handling for ResidualMLP (torch-based, state in separate file)
    if mapping_type == "residual_mlp":
        return _load_mlp_mapping(header)

    cls = get_mapping_class(mapping_type)
    obj = cls.__new__(cls)
    Mapping.__init__(obj)
    obj._W = W
    obj._W_inv = W_inv
    obj._d_src = int(header["d_src"])
    obj._d_tgt = int(header["d_tgt"])
    obj._bias = bool(header.get("bias", False))
    obj._seed = int(header.get("seed", 0))
    obj._fitted = True
    obj._meta = dict(header.get("meta") or {})
    val = header.get("validation")
    if val is not None:
        from isotrieve.mapping.base import ValidationReport

        obj._validation_report = ValidationReport(**val)
    object.__setattr__(obj, "_normalize_output", True)
    object.__setattr__(obj, "_holdout_fraction", 0.1)

    # Load score recalibrator if present
    recal_data = header.get("score_recal_v1")
    if recal_data is not None:
        from isotrieve.recalibration import ScoreRecalibrator

        object.__setattr__(
            obj, "_recalibrator", ScoreRecalibrator.from_dict(recal_data)
        )
    else:
        object.__setattr__(obj, "_recalibrator", None)

    if mapping_type == "ridge" or mapping_type == "lowrank_affine":
        alpha = (val or {}).get("alpha") if val else None
        object.__setattr__(obj, "alpha", alpha if alpha is not None else "auto")
        object.__setattr__(obj, "_chosen_alpha", alpha)
        object.__setattr__(obj, "_chosen_inv_alpha", alpha)
        object.__setattr__(obj, "_rank", None)
    elif mapping_type == "procrustes_diag":
        object.__setattr__(obj, "_D", None)
        object.__setattr__(obj, "_D_inv", None)
    return obj


def _load_mlp_mapping(header: dict) -> Mapping:
    """Load ResidualMLPMapping from header + torch state file."""
    try:
        import torch
    except ImportError:
        raise ImportError("ResidualMLPMapping requires torch to load")

    from isotrieve.mapping.mlp import ResidualMLPMapping

    torch_state_path = header.get("torch_state_file")
    if not torch_state_path:
        raise ValueError("ResidualMLP mapping missing torch_state_file in header")

    state = torch.load(torch_state_path, map_location="cpu", weights_only=True)
    obj = ResidualMLPMapping.__new__(ResidualMLPMapping)
    Mapping.__init__(obj)
    obj._d_src = int(header["d_src"])
    obj._d_tgt = int(header["d_tgt"])
    obj._d_src_int = obj._d_src
    obj._d_tgt_int = obj._d_tgt
    obj._bias = False
    obj._seed = int(header.get("seed", 0))
    obj._fitted = True
    obj._meta = dict(header.get("meta") or {})
    obj._hidden_dim = state.get("hidden_dim")
    object.__setattr__(obj, "_normalize_output", True)
    object.__setattr__(obj, "_holdout_fraction", 0.1)

    val = header.get("validation")
    if val is not None:
        from isotrieve.mapping.base import ValidationReport

        obj._validation_report = ValidationReport(**val)

    # Reconstruct models
    try:
        import torch.nn as nn
    except ImportError:
        raise ImportError("torch.nn required to load ResidualMLPMapping")

    d_src = obj._d_src
    d_tgt = obj._d_tgt
    hidden = state.get("hidden_dim") or min(256, d_src)

    class ResidualMLP(nn.Module):
        def __init__(self, d_in, d_out, h):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, h),
                nn.GELU(),
                nn.Linear(h, d_out),
            )

        def forward(self, x):
            return x + self.net(x)

    model = ResidualMLP(d_src, d_tgt, hidden)
    model.load_state_dict(state["model"])
    obj._model = model

    if state.get("inv_model"):
        inv_model = ResidualMLP(d_tgt, d_src, hidden)
        inv_model.load_state_dict(state["inv_model"])
        obj._inv_model = inv_model

    return obj
