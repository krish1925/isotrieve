"""Linear embedding-space mappings: Ridge and Orthogonal Procrustes."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split

from aecp.mapping.base import Mapping, ValidationReport, _check_finite, l2_normalize
from aecp.quality.metrics import pairwise_cosine_stats, topk_retention

# Default GCV alpha grid (log-spaced)
_ALPHA_GRID = np.logspace(-3, 3, 25)


def _augment_bias(X: np.ndarray) -> np.ndarray:
    """Append a column of ones for an affine (bias) term."""
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    return np.hstack([X, ones])


def _coef_to_W(
    coef: np.ndarray, n_features: int, n_targets: int
) -> np.ndarray:
    """Convert sklearn ``coef_`` to a right-multiply matrix ``(n_features, n_targets)``."""
    coef = np.asarray(coef, dtype=np.float64)
    if coef.ndim == 1:
        # Single target: (n_features,)
        W = coef.reshape(-1, 1)
    else:
        # Multi-target: (n_targets, n_features) → transpose
        W = coef.T
    if W.shape != (n_features, n_targets):
        raise RuntimeError(
            f"Unexpected coef shape {coef.shape} → W {W.shape}; "
            f"expected ({n_features}, {n_targets})"
        )
    return W


def _validate_xy(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-D arrays of shape (K, d)")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Sample counts must match: X has {X.shape[0]}, Y has {Y.shape[0]}"
        )
    if X.shape[0] == 0:
        raise ValueError("Cannot fit on zero samples")
    _check_finite("X", X)
    _check_finite("Y", Y)

    min_dim = min(X.shape[1], Y.shape[1])
    k_min = 10 * min_dim
    if X.shape[0] < k_min:
        import warnings as _warnings
        _warnings.warn(
            f"K={X.shape[0]} is below the recommended minimum "
            f"10×min(d_src,d_tgt)={k_min}. Mapping may be rank-deficient. "
            f"(Suggested K ≥ {k_min}.)",
            UserWarning, stacklevel=3,
        )

    rank = int(np.linalg.matrix_rank(X, tol=1e-8))
    if rank < min(X.shape):
        warnings.warn(
            f"Source matrix X is rank-deficient (rank={rank} < "
            f"min(shape)={min(X.shape)}). Mapping quality may suffer; "
            "prefer more diverse calibration texts.",
            UserWarning,
            stacklevel=3,
        )
    return X, Y


def _holdout_metrics(
    mapping: Mapping,
    X_hold: np.ndarray,
    Y_hold: np.ndarray,
    *,
    n_train: int,
    seed: int,
    alpha: float | None,
) -> ValidationReport:
    mapped = mapping.transform(X_hold)
    cos = pairwise_cosine_stats(mapped, Y_hold)
    # Retention vs identity retrieval in true target space
    t1 = topk_retention(mapped, Y_hold, k=1)
    t10 = topk_retention(mapped, Y_hold, k=min(10, len(Y_hold)))
    return ValidationReport(
        holdout_cosine_mean=cos["mean"],
        holdout_cosine_median=cos["median"],
        holdout_cosine_p5=cos["p5"],
        top1_retention=t1,
        top10_retention=t10,
        n_train=n_train,
        n_holdout=len(X_hold),
        seed=seed,
        alpha=alpha,
    )


class RidgeMapping(Mapping):
    """Ridge regression mapping with optional bias (handles rectangular dims).

    Parameters
    ----------
    alpha:
        Ridge regularization. ``"auto"`` selects via generalized cross-validation
        over a log-spaced grid. A float uses that fixed value.
    bias:
        If True (default), fit an affine map ``Y ≈ [X | 1] W``.
    seed:
        RNG seed for the internal 10% holdout split.
    normalize_output:
        If True (default), L2-normalize every transformed row.
    """

    mapping_type = "ridge"

    def __init__(
        self,
        alpha: float | Literal["auto"] = "auto",
        *,
        bias: bool = True,
        seed: int = 0,
        normalize_output: bool = True,
        holdout_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha: float | Literal["auto"] = alpha
        self._bias = bias
        self._seed = seed
        self._normalize_output = normalize_output
        self._holdout_fraction = holdout_fraction
        self._chosen_alpha: float | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> RidgeMapping:
        X, Y = _validate_xy(X, Y)
        self._d_src = int(X.shape[1])
        self._d_tgt = int(Y.shape[1])

        X_train, X_hold, Y_train, Y_hold = train_test_split(
            X,
            Y,
            test_size=self._holdout_fraction,
            random_state=self._seed,
        )
        if len(X_hold) < 1:
            # Tiny K edge case after split — use last row as holdout
            X_hold, Y_hold = X_train[-1:], Y_train[-1:]
            X_train, Y_train = X_train[:-1], Y_train[:-1]

        X_fit = _augment_bias(X_train) if self._bias else X_train

        if self.alpha == "auto":
            model = RidgeCV(alphas=_ALPHA_GRID, fit_intercept=False)
            model.fit(X_fit, Y_train)
            self._chosen_alpha = float(model.alpha_)
        else:
            self._chosen_alpha = float(self.alpha)
            model = Ridge(alpha=self._chosen_alpha, fit_intercept=False)
            model.fit(X_fit, Y_train)

        # sklearn multi-output coef_ is (n_targets, n_features); we want (feat, tgt)
        self._W = _coef_to_W(model.coef_, X_fit.shape[1], Y_train.shape[1])

        # Inverse map: ridge Y -> X
        Y_fit = _augment_bias(Y_train) if self._bias else Y_train
        inv_alpha = self._chosen_alpha if self._chosen_alpha is not None else 1.0
        inv_model = Ridge(alpha=inv_alpha, fit_intercept=False)
        inv_model.fit(Y_fit, X_train)
        self._W_inv = _coef_to_W(
            inv_model.coef_, Y_fit.shape[1], X_train.shape[1]
        )

        self._fitted = True
        self._validation_report = _holdout_metrics(
            self,
            X_hold,
            Y_hold,
            n_train=len(X_train),
            seed=self._seed,
            alpha=self._chosen_alpha,
        )
        return self

    def transform(self, V: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self._W is not None
        V = np.asarray(V, dtype=np.float64)
        single = V.ndim == 1
        if single:
            V = V.reshape(1, -1)
        if V.shape[1] != self._d_src:
            raise ValueError(
                f"Dimension mismatch: expected {self._d_src} (source model dim), got {V.shape[1]}. "
                f"Vectors must be from the source embedding model. "
                f"If dims differ between models, fit an aecp mapping first: "
                f"RidgeMapping(alpha='auto').fit(X_cal, Y_cal)"
            )
        _check_finite("V", V)
        V_in = _augment_bias(V) if self._bias else V
        out = V_in @ self._W
        if self._normalize_output:
            out = l2_normalize(out)
        return out.ravel() if single else out

    def inverse_transform(self, V: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._W_inv is None:
            raise RuntimeError(
                "Inverse mapping not available. "
                "Fit both directions for serve mode: "
                "m.fit(X_cal, Y_cal) trains forward; inverse is computed automatically."
            )
        V = np.asarray(V, dtype=np.float64)
        single = V.ndim == 1
        if single:
            V = V.reshape(1, -1)
        if V.shape[1] != self._d_tgt:
            raise ValueError(
                f"Dimension mismatch: expected {self._d_tgt} (target model dim), got {V.shape[1]}. "
                f"Vectors must be from the target embedding model."
            )
        _check_finite("V", V)
        V_in = _augment_bias(V) if self._bias else V
        out = V_in @ self._W_inv
        if self._normalize_output:
            out = l2_normalize(out)
        return out.ravel() if single else out


class OrthogonalProcrustesMapping(Mapping):
    """Orthogonal Procrustes mapping (square dims only).

    Preserves pairwise geometry exactly under the orthogonal constraint, but
    often fits worse than ridge on real embedding pairs. Only available when
    ``d_src == d_tgt``. Prefer :class:`RidgeMapping` for migration workloads.
    """

    mapping_type = "orthogonal_procrustes"

    def __init__(
        self,
        *,
        seed: int = 0,
        normalize_output: bool = True,
        holdout_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self._bias = False
        self._seed = seed
        self._normalize_output = normalize_output
        self._holdout_fraction = holdout_fraction

    def fit(self, X: np.ndarray, Y: np.ndarray) -> OrthogonalProcrustesMapping:
        X, Y = _validate_xy(X, Y)
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"Procrustes requires d_src == d_tgt; got {X.shape[1]} vs {Y.shape[1]}. "
                f"Use RidgeMapping for rectangular mappings (d_src != d_tgt)."
            )
        self._d_src = int(X.shape[1])
        self._d_tgt = int(Y.shape[1])

        X_train, X_hold, Y_train, Y_hold = train_test_split(
            X,
            Y,
            test_size=self._holdout_fraction,
            random_state=self._seed,
        )

        # Solve min ||X R - Y||_F s.t. R^T R = I via SVD of X^T Y
        M = X_train.T @ Y_train
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt
        self._W = R
        self._W_inv = R.T.copy()
        self._fitted = True
        self._validation_report = _holdout_metrics(
            self,
            X_hold,
            Y_hold,
            n_train=len(X_train),
            seed=self._seed,
            alpha=None,
        )
        return self

    def transform(self, V: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self._W is not None
        V = np.asarray(V, dtype=np.float64)
        single = V.ndim == 1
        if single:
            V = V.reshape(1, -1)
        if V.shape[1] != self._d_src:
            raise ValueError(
                f"Dimension mismatch: expected {self._d_src} (source model dim), got {V.shape[1]}. "
                f"Vectors must be from the source embedding model."
            )
        out = V @ self._W
        if self._normalize_output:
            out = l2_normalize(out)
        return out.ravel() if single else out

    def inverse_transform(self, V: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self._W_inv is not None
        V = np.asarray(V, dtype=np.float64)
        single = V.ndim == 1
        if single:
            V = V.reshape(1, -1)
        out = V @ self._W_inv
        if self._normalize_output:
            out = l2_normalize(out)
        return out.ravel() if single else out


class ProcrustesDiagMapping(Mapping):
    """Orthogonal Procrustes + Diagonal Scaling (square dims only).

    Drift-Adapter (EMNLP 2025) found this parameterization remarkably effective:
    first fit an orthogonal rotation via Procrustes, then learn a per-dimension
    diagonal scaling to absorb magnitude differences. More expressive than pure
    Procrustes while preserving pairwise geometry better than ridge.

    ``Y ≈ diag(s) · X · R`` where R is orthogonal.
    """

    mapping_type = "procrustes_diag"

    def __init__(
        self,
        *,
        seed: int = 0,
        normalize_output: bool = True,
        holdout_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self._bias = False
        self._seed = seed
        self._normalize_output = normalize_output
        self._holdout_fraction = holdout_fraction
        self._D: np.ndarray | None = None
        self._D_inv: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> ProcrustesDiagMapping:
        X, Y = _validate_xy(X, Y)
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"ProcrustesDiag requires d_src == d_tgt; got {X.shape[1]} vs {Y.shape[1]}. "
                f"Use RidgeMapping for rectangular mappings (d_src != d_tgt)."
            )
        self._d_src = int(X.shape[1])
        self._d_tgt = int(Y.shape[1])

        X_train, X_hold, Y_train, Y_hold = train_test_split(
            X, Y, test_size=self._holdout_fraction, random_state=self._seed,
        )

        # Step 1: Procrustes rotation
        M = X_train.T @ Y_train
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt  # orthogonal rotation

        # Step 2: Diagonal scaling after rotation
        # After rotation: Y ≈ X · R · diag(s)
        # Solve for s: minimize ||X · R · diag(s) - Y||^2
        XR = X_train @ R  # (n, d)
        # For each dimension j, best s_j = sum(XR[:,j] * Y[:,j]) / sum(XR[:,j]^2)
        denom = np.sum(XR * XR, axis=0)  # (d,)
        numer = np.sum(XR * Y_train, axis=0)  # (d,)
        # Regularize to avoid division by zero
        s = numer / np.maximum(denom, 1e-8)

        # Combined: W = R · diag(s)  =>  v_out = v_in @ R @ diag(s)
        self._W = R * s[np.newaxis, :]  # (d, d) * (1, d) broadcast
        self._D = s

        # Inverse: diag(s)^{-1} · R^T
        s_inv = np.where(np.abs(s) > 1e-8, 1.0 / s, 0.0)
        self._W_inv = R.T * s_inv[np.newaxis, :]
        self._D_inv = s_inv

        self._fitted = True
        self._validation_report = _holdout_metrics(
            self, X_hold, Y_hold, n_train=len(X_train),
            seed=self._seed, alpha=None,
        )
        return self

    def transform(self, V: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self._W is not None
        V = np.asarray(V, dtype=np.float64)
        single = V.ndim == 1
        if single:
            V = V.reshape(1, -1)
        if V.shape[1] != self._d_src:
            raise ValueError(
                f"Dimension mismatch: expected {self._d_src} (source model dim), got {V.shape[1]}."
            )
        out = V @ self._W
        if self._normalize_output:
            out = l2_normalize(out)
        return out.ravel() if single else out

    def inverse_transform(self, V: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self._W_inv is not None
        V = np.asarray(V, dtype=np.float64)
        single = V.ndim == 1
        if single:
            V = V.reshape(1, -1)
        out = V @ self._W_inv
        if self._normalize_output:
            out = l2_normalize(out)
        return out.ravel() if single else out


class LowRankAffineMapping(Mapping):
    """Low-rank affine mapping via truncated SVD on the ridge residual.

    Drift-Adapter variant: fit a ridge mapping, then compress the learned
    weight matrix to rank ``r`` via truncated SVD. Useful when d_src/d_tgt
    are large and you want a more compact mapping (smaller .aecp file,
    faster transform).

    When ``rank >= min(d_src, d_tgt)``, behaves identically to RidgeMapping.
    """

    mapping_type = "lowrank_affine"

    def __init__(
        self,
        alpha: float | Literal["auto"] = "auto",
        rank: int | None = None,
        *,
        bias: bool = True,
        seed: int = 0,
        normalize_output: bool = True,
        holdout_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self._rank = rank
        self._bias = bias
        self._seed = seed
        self._normalize_output = normalize_output
        self._holdout_fraction = holdout_fraction
        self._chosen_alpha: float | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> LowRankAffineMapping:
        X_val, Y_val = _validate_xy(X, Y)
        self._d_src = int(X_val.shape[1])
        self._d_tgt = int(Y_val.shape[1])

        X_train, X_hold, Y_train, Y_hold = train_test_split(
            X_val, Y_val, test_size=self._holdout_fraction, random_state=self._seed,
        )
        if len(X_hold) < 1:
            X_hold, Y_hold = X_train[-1:], Y_train[-1:]
            X_train, Y_train = X_train[:-1], Y_train[:-1]

        X_fit = _augment_bias(X_train) if self._bias else X_train

        # Fit ridge first
        if self.alpha == "auto":
            model = RidgeCV(alphas=_ALPHA_GRID, fit_intercept=False)
            model.fit(X_fit, Y_train)
            self._chosen_alpha = float(model.alpha_)
        else:
            self._chosen_alpha = float(self.alpha)
            model = Ridge(alpha=self._chosen_alpha, fit_intercept=False)
            model.fit(X_fit, Y_train)

        W_full = _coef_to_W(model.coef_, X_fit.shape[1], Y_train.shape[1])

        # Truncated SVD for low-rank approximation
        rank = self._rank or min(W_full.shape)
        rank = min(rank, min(W_full.shape))
        if rank < min(W_full.shape):
            U, sigma, Vt = np.linalg.svd(W_full, full_matrices=False)
            W_lr = U[:, :rank] @ np.diag(sigma[:rank]) @ Vt[:rank, :]
        else:
            W_lr = W_full

        self._W = W_lr

        # Inverse via ridge on Y -> X
        Y_fit = _augment_bias(Y_train) if self._bias else Y_train
        inv_alpha = self._chosen_alpha if self._chosen_alpha is not None else 1.0
        inv_model = Ridge(alpha=inv_alpha, fit_intercept=False)
        inv_model.fit(Y_fit, X_train)
        W_inv_full = _coef_to_W(inv_model.coef_, Y_fit.shape[1], X_train.shape[1])
        if rank < min(W_inv_full.shape):
            U2, s2, Vt2 = np.linalg.svd(W_inv_full, full_matrices=False)
            self._W_inv = U2[:, :rank] @ np.diag(s2[:rank]) @ Vt2[:rank, :]
        else:
            self._W_inv = W_inv_full

        self._fitted = True
        self._validation_report = _holdout_metrics(
            self, X_hold, Y_hold, n_train=len(X_train),
            seed=self._seed, alpha=self._chosen_alpha,
        )
        return self

    def transform(self, V: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self._W is not None
        V = np.asarray(V, dtype=np.float64)
        single = V.ndim == 1
        if single:
            V = V.reshape(1, -1)
        if V.shape[1] != self._d_src:
            raise ValueError(
                f"Dimension mismatch: expected {self._d_src} (source model dim), got {V.shape[1]}."
            )
        _check_finite("V", V)
        V_in = _augment_bias(V) if self._bias else V
        out = V_in @ self._W
        if self._normalize_output:
            out = l2_normalize(out)
        return out.ravel() if single else out

    def inverse_transform(self, V: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._W_inv is None:
            raise RuntimeError("Inverse mapping unavailable")
        V = np.asarray(V, dtype=np.float64)
        single = V.ndim == 1
        if single:
            V = V.reshape(1, -1)
        if V.shape[1] != self._d_tgt:
            raise ValueError(
                f"Dimension mismatch: expected {self._d_tgt} (target model dim), got {V.shape[1]}."
            )
        _check_finite("V", V)
        V_in = _augment_bias(V) if self._bias else V
        out = V_in @ self._W_inv
        if self._normalize_output:
            out = l2_normalize(out)
        return out.ravel() if single else out
