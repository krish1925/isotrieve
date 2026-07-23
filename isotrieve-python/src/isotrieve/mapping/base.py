"""Mapping abstract base and shared helpers."""

from __future__ import annotations

import json
import struct
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# .isotrieve binary format: magic + header_len + utf-8 JSON header + float64 matrix payload
_ISOTRIEVE_MAGIC = b"ISTR"
_ISOTRIEVE_FORMAT_VERSION = 1
_HEADER_LEN_STRUCT = struct.Struct("<I")


@dataclass
class ValidationReport:
    """Holdout validation metrics produced by :meth:`Mapping.fit`."""

    holdout_cosine_mean: float
    holdout_cosine_median: float
    holdout_cosine_p5: float
    top1_retention: float
    top10_retention: float
    n_train: int
    n_holdout: int
    seed: int
    alpha: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize. Zero rows stay zero."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim == 1:
        norm = float(np.linalg.norm(vectors))
        if norm < eps:
            return vectors.astype(np.float64, copy=True)
        return vectors / norm
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms


def _check_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf values")


def _augment_bias(X: np.ndarray) -> np.ndarray:
    """Append a column of ones for an affine (bias) term."""
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.hstack([X, ones])


class Mapping(ABC):
    """Abstract embedding-space mapping.

    Subclasses implement :meth:`fit`, :meth:`transform`, and
    :meth:`inverse_transform`. Persistence uses the versioned ``.isotrieve`` format.
    """

    mapping_type: str = "base"

    def __init__(self) -> None:
        self._fitted: bool = False
        self._W: np.ndarray | None = None  # (d_src_aug, d_tgt) or (d_src, d_tgt)
        self._W_inv: np.ndarray | None = None
        self._d_src: int | None = None
        self._d_tgt: int | None = None
        self._bias: bool = True
        self._seed: int = 0
        self._validation_report: ValidationReport | None = None
        self._meta: dict[str, Any] = {}
        self._recalibrator: Any = None  # ScoreRecalibrator | None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def d_src(self) -> int:
        if self._d_src is None:
            raise RuntimeError("Mapping is not fitted")
        return self._d_src

    @property
    def d_tgt(self) -> int:
        if self._d_tgt is None:
            raise RuntimeError("Mapping is not fitted")
        return self._d_tgt

    def validation_report(self) -> ValidationReport:
        if self._validation_report is None:
            raise RuntimeError("No validation report; call fit() first")
        return self._validation_report

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> Mapping:
        """Fit mapping from source embeddings ``X`` to target embeddings ``Y``."""

    @abstractmethod
    def transform(self, V: np.ndarray) -> np.ndarray:
        """Map source-space vectors to target space (L2-normalized)."""

    @abstractmethod
    def inverse_transform(self, V: np.ndarray) -> np.ndarray:
        """Map target-space vectors back toward source space (L2-normalized)."""

    def transform_batches(
        self,
        batches: Iterator[np.ndarray] | Sequence[np.ndarray],
    ) -> Iterator[np.ndarray]:
        """Stream transform: yield one transformed batch at a time.

        Accepts a generator/iterator so large corpora never need to fit in RAM.
        """
        for batch in batches:
            yield self.transform(batch)

    def save(self, path: str | Path) -> None:
        """Persist mapping to a single ``.isotrieve`` file."""
        if not self._fitted or self._W is None:
            raise RuntimeError("Cannot save an unfitted mapping")

        path = Path(path)
        header = {
            "format_version": _ISOTRIEVE_FORMAT_VERSION,
            "isotrieve_version": _pkg_version(),
            "mapping_type": self.mapping_type,
            "d_src": self._d_src,
            "d_tgt": self._d_tgt,
            "bias": self._bias,
            "seed": self._seed,
            "fit_date": datetime.now(timezone.utc).isoformat(),
            "expires_hint": None,
            "validation": (
                self._validation_report.to_dict()
                if self._validation_report is not None
                else None
            ),
            "meta": self._meta,
            "matrix_shape": list(self._W.shape),
            "has_inverse": self._W_inv is not None,
        }
        if self._W_inv is not None:
            header["inverse_matrix_shape"] = list(self._W_inv.shape)

        # Extra matrices (centering means for Procrustes variants)
        extra_matrices = {}
        if hasattr(self, "_mean_X") and self._mean_X is not None:
            extra_matrices["mean_X"] = list(self._mean_X.shape)
        if hasattr(self, "_mean_Y") and self._mean_Y is not None:
            extra_matrices["mean_Y"] = list(self._mean_Y.shape)
        if extra_matrices:
            header["extra_matrices"] = extra_matrices

        # Score recalibrator (optional section)
        if self._recalibrator is not None and self._recalibrator.is_fitted:
            header["score_recal_v1"] = self._recalibrator.save_dict()

        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        payload = self._W.astype(np.float64, copy=False).tobytes(order="C")
        if self._W_inv is not None:
            payload += self._W_inv.astype(np.float64, copy=False).tobytes(order="C")
        # Append extra matrices (centering means for Procrustes)
        if hasattr(self, "_mean_X") and self._mean_X is not None:
            payload += self._mean_X.astype(np.float64, copy=False).tobytes(order="C")
        if hasattr(self, "_mean_Y") and self._mean_Y is not None:
            payload += self._mean_Y.astype(np.float64, copy=False).tobytes(order="C")

        with path.open("wb") as f:
            f.write(_ISOTRIEVE_MAGIC)
            f.write(_HEADER_LEN_STRUCT.pack(len(header_bytes)))
            f.write(header_bytes)
            f.write(payload)

    @classmethod
    def load(cls, path: str | Path) -> Mapping:
        """Load a mapping from a ``.isotrieve`` file.

        Dispatches to the concrete class registered for ``mapping_type``.
        """
        from isotrieve.mapping.registry import load_mapping

        return load_mapping(path)

    def _require_fitted(self) -> None:
        if not self._fitted or self._W is None:
            raise RuntimeError("Mapping is not fitted; call fit() first")

    def _apply_mapping(
        self,
        V: np.ndarray,
        matrix: np.ndarray,
        expected_dim: int,
        *,
        direction: str,
        bias: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Shared transform/inverse_transform logic.

        Handles: single-vector reshape, dimension check, finiteness, bias
        augmentation, matrix multiply, L2 normalization.

        Parameters
        ----------
        V:
            Input vectors (n, d) or (d,).
        matrix:
            Weight matrix to multiply by.
        expected_dim:
            Expected input dimensionality.
        direction:
            ``"forward"`` or ``"inverse"`` — used in error messages.
        bias:
            If True, augment input with a column of ones before multiply.
        normalize:
            If True, L2-normalize output rows.
        """
        V = np.asarray(V, dtype=np.float64)
        single = V.ndim == 1
        if single:
            V = V.reshape(1, -1)
        if V.shape[1] != expected_dim:
            dim_name = "source" if direction == "forward" else "target"
            raise ValueError(
                f"Dimension mismatch: expected {expected_dim} ({dim_name} model dim), got {V.shape[1]}. "
                f"Vectors must be from the {'source' if direction == 'forward' else 'target'} embedding model. "
                f"If dims differ between models, fit an isotrieve mapping first: "
                f"RidgeMapping(alpha='auto').fit(X_cal, Y_cal)"
            )
        _check_finite("V", V)
        if bias:
            V = _augment_bias(V)
        out = V @ matrix
        if normalize:
            out = l2_normalize(out)
        return out.ravel() if single else out

    def set_meta(self, **kwargs: Any) -> None:
        """Attach metadata stored in the ``.isotrieve`` header (model ids, corpus id, …)."""
        self._meta.update(kwargs)

    @property
    def has_recalibrator(self) -> bool:
        return self._recalibrator is not None and self._recalibrator.is_fitted

    def recalibrate_scores(self, scores: np.ndarray) -> np.ndarray:
        """Map post-migration similarity scores to ceiling-equivalent scores.

        Requires that a ScoreRecalibrator was fitted during calibration.
        Returns scores unchanged if no recalibrator is present.
        """
        if self._recalibrator is None or not self._recalibrator.is_fitted:
            return np.asarray(scores, dtype=np.float64)
        return self._recalibrator.transform(scores)


def _pkg_version() -> str:
    try:
        from isotrieve import __version__

        return __version__
    except Exception:
        return "0.0.0"


def read_isotrieve_header(path: str | Path) -> dict[str, Any]:
    """Read and return the JSON header of a ``.isotrieve`` file without loading matrices."""
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != _ISOTRIEVE_MAGIC:
            raise ValueError(f"Not an .isotrieve file (bad magic): {path}")
        (header_len,) = _HEADER_LEN_STRUCT.unpack(f.read(4))
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header


def load_isotrieve_payload(
    path: str | Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    """Load header, forward matrix, and optional inverse matrix from ``.isotrieve``."""
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != _ISOTRIEVE_MAGIC:
            raise ValueError(f"Not an .isotrieve file (bad magic): {path}")
        (header_len,) = _HEADER_LEN_STRUCT.unpack(f.read(4))
        header = json.loads(f.read(header_len).decode("utf-8"))
        rest = f.read()

    shape = tuple(header["matrix_shape"])
    n_fwd = int(np.prod(shape))
    fwd = np.frombuffer(rest, dtype=np.float64, count=n_fwd).reshape(shape).copy()
    inv: np.ndarray | None = None
    offset = n_fwd * 8
    if header.get("has_inverse") and "inverse_matrix_shape" in header:
        inv_shape = tuple(header["inverse_matrix_shape"])
        n_inv = int(np.prod(inv_shape))
        inv = (
            np.frombuffer(rest, dtype=np.float64, count=n_inv, offset=offset)
            .reshape(inv_shape)
            .copy()
        )
        offset += n_inv * 8
    # Extra matrices (centering means for Procrustes)
    extra = header.get("extra_matrices", {})
    extra_arrays: dict[str, np.ndarray] = {}
    for name in ("mean_X", "mean_Y"):
        if name in extra:
            eshape = tuple(extra[name])
            n_extra = int(np.prod(eshape))
            extra_arrays[name] = (
                np.frombuffer(rest, dtype=np.float64, count=n_extra, offset=offset)
                .reshape(eshape)
                .copy()
            )
            offset += n_extra * 8
    return header, fwd, inv, extra_arrays
