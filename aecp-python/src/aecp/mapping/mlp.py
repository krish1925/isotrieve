"""Optional ResidualMLP mapping (requires torch)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from aecp.mapping.base import Mapping, _check_finite, l2_normalize
from aecp.mapping.linear import _holdout_metrics


class ResidualMLPMapping(Mapping):
    """Compact residual MLP for embedding-space alignment.

    Drift-Adapter (EMNLP 2025) variant: a small 2-layer residual network
    that learns a non-linear mapping between embedding spaces. The residual
    connection ensures the model starts from the identity and learns only
    the necessary correction.

    Architecture: ``output = input + MLP(input)``
    where MLP is ``Linear(d, h) -> GELU -> Linear(h, d)``.

    Requires ``torch`` (install with ``pip install aecp[mlp]``).
    Marked experimental — benchmark against ridge before trusting.

    Parameters
    ----------
    hidden_dim:
        Hidden layer dimension. Default: min(256, d_src).
    learning_rate:
        Adam optimizer learning rate.
    n_epochs:
        Training epochs.
    rank:
        If set, project to lower rank before MLP (compression).
    """

    mapping_type = "residual_mlp"

    def __init__(
        self,
        hidden_dim: int | None = None,
        learning_rate: float = 1e-3,
        n_epochs: int = 200,
        *,
        seed: int = 0,
        normalize_output: bool = True,
        holdout_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._lr = learning_rate
        self._n_epochs = n_epochs
        self._bias = False
        self._seed = seed
        self._normalize_output = normalize_output
        self._holdout_fraction = holdout_fraction
        self._model = None
        self._d_src_int: int = 0
        self._d_tgt_int: int = 0

    def fit(self, X: np.ndarray, Y: np.ndarray) -> ResidualMLPMapping:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError(
                "ResidualMLPMapping requires torch. Install with: pip install aecp[mlp]"
            )

        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2-D arrays")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Sample counts must match")
        min_dim = min(X.shape[1], Y.shape[1])
        if X.shape[0] < 10 * min_dim:
            import warnings as _warnings

            _warnings.warn(
                f"K={X.shape[0]} below recommended minimum 10×min_dim={10 * min_dim}. "
                f"Mapping may be underdetermined.",
                UserWarning,
                stacklevel=3,
            )
        _check_finite("X", X)
        _check_finite("Y", Y)

        self._d_src = int(X.shape[1])
        self._d_tgt = int(Y.shape[1])
        self._d_src_int = self._d_src
        self._d_tgt_int = self._d_tgt

        X_train, X_hold, Y_train, Y_hold = train_test_split(
            X,
            Y,
            test_size=self._holdout_fraction,
            random_state=self._seed,
        )

        d_src = X_train.shape[1]
        d_tgt = Y_train.shape[1]
        hidden = self._hidden_dim or min(256, d_src)

        device = torch.device("cpu")

        class ResidualMLP(nn.Module):
            def __init__(self, d_in: int, d_out: int, h: int):
                super().__init__()
                self.is_residual = d_in == d_out
                if self.is_residual:
                    self.net = nn.Sequential(
                        nn.Linear(d_in, h),
                        nn.GELU(),
                        nn.Linear(h, d_out),
                    )
                    nn.init.zeros_(self.net[-1].weight)
                    nn.init.zeros_(self.net[-1].bias)
                else:
                    # Rectangular: project in -> hidden -> out (no residual)
                    self.net = nn.Sequential(
                        nn.Linear(d_in, h),
                        nn.GELU(),
                        nn.Linear(h, d_out),
                    )

            def forward(self, x):
                if self.is_residual:
                    return x + self.net(x)
                return self.net(x)

        model = ResidualMLP(d_src, d_tgt, hidden).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self._lr)
        criterion = nn.MSELoss()

        X_t = torch.from_numpy(X_train).to(device)
        Y_t = torch.from_numpy(Y_train).to(device)

        model.train()
        for _ in range(self._n_epochs):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = criterion(pred, Y_t)
            loss.backward()
            optimizer.step()

        self._model = model
        self._fitted = True
        # MLP doesn't use _W, but base class _require_fitted checks it
        self._W = np.zeros((self._d_src, self._d_tgt), dtype=np.float64)

        # Compute inverse via separate MLP
        class InverseMLP(nn.Module):
            def __init__(self, d_in: int, d_out: int, h: int):
                super().__init__()
                self.is_residual = d_in == d_out
                if self.is_residual:
                    self.net = nn.Sequential(
                        nn.Linear(d_in, h),
                        nn.GELU(),
                        nn.Linear(h, d_out),
                    )
                    nn.init.zeros_(self.net[-1].weight)
                    nn.init.zeros_(self.net[-1].bias)
                else:
                    self.net = nn.Sequential(
                        nn.Linear(d_in, h),
                        nn.GELU(),
                        nn.Linear(h, d_out),
                    )

            def forward(self, x):
                if self.is_residual:
                    return x + self.net(x)
                return self.net(x)

        inv_model = InverseMLP(d_tgt, d_src, hidden).to(device)
        inv_optimizer = optim.Adam(inv_model.parameters(), lr=self._lr)

        Y_t_inv = torch.from_numpy(Y_train).to(device)
        X_t_inv = torch.from_numpy(X_train).to(device)

        inv_model.train()
        for _ in range(self._n_epochs):
            inv_optimizer.zero_grad()
            pred = inv_model(Y_t_inv)
            loss = criterion(pred, X_t_inv)
            loss.backward()
            inv_optimizer.step()

        self._inv_model = inv_model
        self._W_inv = np.zeros((self._d_tgt, self._d_src), dtype=np.float64)

        self._validation_report = _holdout_metrics(
            self,
            X_hold,
            Y_hold,
            n_train=len(X_train),
            seed=self._seed,
            alpha=None,
        )
        return self

    def _validate_input(
        self, V: np.ndarray, expected_dim: int, direction: str
    ) -> tuple[np.ndarray, bool]:
        """Shared preprocessing: reshape, dimension check, finiteness."""
        V = np.asarray(V, dtype=np.float32)
        single = V.ndim == 1
        if single:
            V = V.reshape(1, -1)
        if V.shape[1] != expected_dim:
            dim_name = "source" if direction == "forward" else "target"
            raise ValueError(
                f"Dimension mismatch: expected {expected_dim} ({dim_name} model dim), got {V.shape[1]}. "
                f"Vectors must be from the {'source' if direction == 'forward' else 'target'} embedding model. "
                f"If dims differ between models, fit an aecp mapping first: "
                f"RidgeMapping(alpha='auto').fit(X_cal, Y_cal)"
            )
        _check_finite("V", V)
        return V, single

    def transform(self, V: np.ndarray) -> np.ndarray:
        try:
            import torch
        except ImportError:
            raise ImportError("ResidualMLPMapping requires torch")

        self._require_fitted()
        V, single = self._validate_input(V, self._d_src, "forward")

        self._model.eval()
        with torch.no_grad():
            V_t = torch.from_numpy(V)
            out = self._model(V_t).numpy().astype(np.float64)
        if self._normalize_output:
            out = l2_normalize(out)
        return out.ravel() if single else out

    def inverse_transform(self, V: np.ndarray) -> np.ndarray:
        try:
            import torch
        except ImportError:
            raise ImportError("ResidualMLPMapping requires torch")

        self._require_fitted()
        inv_model = getattr(self, "_inv_model", None)
        if inv_model is None:
            raise RuntimeError(
                "Inverse mapping not available. "
                "Fit both directions for serve mode: "
                "m.fit(X_cal, Y_cal) trains forward; inverse is computed automatically."
            )

        V, single = self._validate_input(V, self._d_tgt, "inverse")

        inv_model.eval()
        with torch.no_grad():
            V_t = torch.from_numpy(V)
            out = inv_model(V_t).numpy().astype(np.float64)
        if self._normalize_output:
            out = l2_normalize(out)
        return out.ravel() if single else out

    def save(self, path) -> None:
        """Save with torch model state dict."""
        try:
            import torch
        except ImportError:
            raise ImportError("ResidualMLPMapping requires torch for save")

        import json
        from pathlib import Path

        path = Path(path)
        state = {
            "model": self._model.state_dict() if self._model else None,
            "inv_model": self._inv_model.state_dict()
            if hasattr(self, "_inv_model") and self._inv_model
            else None,
            "d_src": self._d_src,
            "d_tgt": self._d_tgt,
            "hidden_dim": self._hidden_dim,
        }
        # Save state dict as .pt and header as .json
        torch.save(state, str(path) + ".pt")
        # Write minimal header
        from aecp.mapping.base import (
            _AECP_FORMAT_VERSION,
            _AECP_MAGIC,
            _HEADER_LEN_STRUCT,
            _pkg_version,
        )

        header = {
            "format_version": _AECP_FORMAT_VERSION,
            "aecp_version": _pkg_version(),
            "mapping_type": self.mapping_type,
            "d_src": self._d_src,
            "d_tgt": self._d_tgt,
            "bias": False,
            "seed": self._seed,
            "has_inverse": True,
            "meta": self._meta,
            "torch_state_file": str(path) + ".pt",
            "validation": self._validation_report.to_dict()
            if self._validation_report
            else None,
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with path.open("wb") as f:
            f.write(_AECP_MAGIC)
            f.write(_HEADER_LEN_STRUCT.pack(len(header_bytes)))
            f.write(header_bytes)
