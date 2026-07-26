"""External mapping: wrap any pretrained callable for gating."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from isotrieve.mapping.base import Mapping, l2_normalize


class ExternalMapping(Mapping):
    """Wrap an external callable as an Isotrieve mapping.

    This allows gating pretrained transforms from other libraries
    (EmbeddingAdapters, sentence-transformers, etc.) without
    retraining. The callable must accept ``(np.ndarray) -> np.ndarray``
    with correct input/output shapes.

    Parameters
    ----------
    fn : callable
        A function that takes a 2-D float64 array ``(n, d_in)`` and
        returns a 2-D float64 array ``(n, d_out)``.
    d_src : int, optional
        Source embedding dimensionality. If *None*, inferred on first
        call to ``transform()``.
    d_tgt : int, optional
        Target embedding dimensionality. If *None*, inferred on first
        call to ``transform()``.
    inverse_fn : callable, optional
        Inverse transform ``(n, d_out) -> (n, d_src)``. If *None*,
        ``inverse_transform()`` raises ``NotImplementedError``.
    """

    mapping_type = "external"

    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        *,
        d_src: int | None = None,
        d_tgt: int | None = None,
        inverse_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        self._fn = fn
        self._inverse_fn = inverse_fn
        self._d_src = d_src
        self._d_tgt = d_tgt
        # ExternalMapping is considered fitted from construction
        self._fitted = True
        self._meta["source"] = "external"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> ExternalMapping:
        """No-op: external mappings are pre-trained.

        Dimensions are inferred from the calibration pair and stored
        for downstream validation.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2-D arrays")
        self._d_src = X.shape[1]
        self._d_tgt = Y.shape[1]
        return self

    def transform(self, V: np.ndarray) -> np.ndarray:
        """Apply the external callable to source-space vectors."""
        V = np.asarray(V, dtype=np.float64)
        if V.ndim == 1:
            V = V.reshape(1, -1)
        if V.ndim != 2:
            raise ValueError("Input must be 1-D or 2-D")

        # Infer dimensions on first call
        if self._d_src is None:
            self._d_src = V.shape[1]
        elif V.shape[1] != self._d_src:
            raise ValueError(f"Expected {self._d_src}-D input, got {V.shape[1]}-D")

        result = self._fn(V)

        result = np.asarray(result, dtype=np.float64)
        if result.ndim == 1:
            result = result.reshape(1, -1)
        if result.ndim != 2:
            raise ValueError("Transform must return 1-D or 2-D array")

        # Infer target dim on first call
        if self._d_tgt is None:
            self._d_tgt = result.shape[1]
        elif result.shape[1] != self._d_tgt:
            raise ValueError(
                f"Transform returned {result.shape[1]}-D, expected {self._d_tgt}-D"
            )

        return l2_normalize(result)

    def inverse_transform(self, V: np.ndarray) -> np.ndarray:
        """Apply the inverse external callable, if provided."""
        if self._inverse_fn is None:
            raise NotImplementedError(
                "No inverse_fn provided; cannot inverse-transform"
            )
        V = np.asarray(V, dtype=np.float64)
        if V.ndim == 1:
            V = V.reshape(1, -1)

        result = self._inverse_fn(V)
        result = np.asarray(result, dtype=np.float64)
        if result.ndim == 1:
            result = result.reshape(1, -1)

        return l2_normalize(result)

    def save(self, path: str | Path) -> None:
        """External callables cannot be serialized to .isotrieve format.

        Use ``ExternalMapping`` directly in Python instead of saving/loading.
        """
        raise NotImplementedError(
            "ExternalMapping wraps a live callable and cannot be saved "
            "to .isotrieve format. Use the callable directly."
        )


def load_external_callable(
    spec: str,
) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Load a callable from a ``module:attr`` spec.

    Parameters
    ----------
    spec : str
        Dotted path like ``"my_package.transform:my_fn"``.

    Returns
    -------
    tuple[callable, str]
        The callable and its qualified name.
    """
    import importlib

    if ":" not in spec:
        raise ValueError(f"Invalid external spec '{spec}'; expected 'module:callable'")
    module_path, attr_name = spec.rsplit(":", 1)

    mod = importlib.import_module(module_path)
    parts = attr_name.split(".")
    obj: Any = mod
    for part in parts:
        obj = getattr(obj, part)

    if not callable(obj):
        raise TypeError(f"'{spec}' resolved to {type(obj).__name__}, not callable")

    return obj, spec
