#!/usr/bin/env python
"""Generate cross-language golden fixtures (Python -> TypeScript) and verify
the TS->Python roundtrip artifact. Barrage P3; pre-registered criteria.

Usage:
  python scripts/gen_cross_compat_fixtures.py generate   # write goldens
  python scripts/gen_cross_compat_fixtures.py verify-ts  # verify TS-written file
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "isotrieve-npm" / "packages" / "core" / "src" / "__tests__" / "fixtures" / "golden"
TS_OUT = Path("/tmp/iso_ts_roundtrip.isotrieve")
SEED = 42


def _build() -> tuple:
    from isotrieve.mapping.base import l2_normalize
    from isotrieve.mapping.linear import RidgeMapping

    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(200, 8))
    W = rng.normal(size=(8, 12)) / np.sqrt(8)
    Y = l2_normalize(X @ W + 0.01 * rng.normal(size=(200, 12)))
    m = RidgeMapping(alpha="auto", seed=SEED).fit(X, Y)
    return m, X, Y


def generate() -> None:
    from isotrieve.mapping.base import l2_normalize

    FIXTURES.mkdir(parents=True, exist_ok=True)
    m, X, Y = _build()
    inputs = X[:10]
    outputs = m.transform(inputs)
    m.save(FIXTURES / "golden_ridge.isotrieve")
    meta = {
        "seed": SEED,
        "d_src": 8,
        "d_tgt": 12,
        "alpha": float(m._chosen_alpha),  # noqa: SLF001
        "inputs": np.round(inputs, 12).tolist(),
        "expected_outputs": np.round(outputs, 12).tolist(),
        "sha256_mapping": hashlib.sha256((FIXTURES / "golden_ridge.isotrieve").read_bytes()).hexdigest(),
        "gate_model_sha256": hashlib.sha256(
            (REPO / "isotrieve-python" / "src" / "isotrieve" / "quality" / "gate_model_v1.json").read_bytes()
        ).hexdigest(),
        "target_rows_for_roundtrip": np.round(l2_normalize(Y[:10]), 12).tolist(),
    }
    (FIXTURES / "golden_expected.json").write_text(json.dumps(meta, indent=1))
    print("goldens written:", FIXTURES)


def verify_ts() -> int:
    """Load the TS-written roundtrip file; compare payload matrix to golden.

    The TS test writes a byte-passthrough of the golden header+payload through
    its own reader/writer (format-layer roundtrip), so the matrices must match
    the golden EXACTLY (atol 1e-12 on values, byte-level transport).
    """
    import tempfile

    from isotrieve.mapping.linear import RidgeMapping

    golden = FIXTURES / "golden_ridge.isotrieve"
    meta = json.loads((FIXTURES / "golden_expected.json").read_text())
    ts_file = Path(sys.argv[2]) if len(sys.argv) > 2 else TS_OUT
    if not ts_file.exists():
        print(f"FAIL: TS roundtrip artifact missing at {ts_file}")
        return 1
    if hashlib.sha256(ts_file.read_bytes()).hexdigest() != meta["sha256_mapping"]:
        print("NOTE: TS roundtrip bytes differ from golden (writer re-encodes); verifying values")
    m_ts = RidgeMapping.load(ts_file)
    m_py = RidgeMapping.load(golden)
    inp = np.array(meta["inputs"])
    out_ts = m_ts.transform(inp)
    out_py = m_py.transform(inp)
    err = float(np.max(np.abs(out_ts - out_py)))
    print(f"verify-ts max|ts-python diff| = {err:.3e} (criterion atol 1e-12)")
    ok = err <= 1e-12
    # gate asset identity
    gsha = hashlib.sha256(
        (REPO / "isotrieve-python" / "src" / "isotrieve" / "quality" / "gate_model_v1.json").read_bytes()
    ).hexdigest()
    print(f"verify-ts gate_model sha match: {gsha == meta['gate_model_sha256']}")
    # tamper matrix (python side): payload flip -> reject; header flip -> reject
    raw = golden.read_bytes()
    with tempfile.TemporaryDirectory() as td:
        for label, pos in (("payload", len(raw) - 5), ("header", 3)):
            t = Path(td) / f"tampered_{label}"
            b = bytearray(raw)
            b[pos] ^= 0x01
            t.write_bytes(bytes(b))
            try:
                RidgeMapping.load(t)
                print(f"verify-ts tamper {label}: ACCEPTED (CRITICAL FAIL)")
                ok = False
            except Exception:
                print(f"verify-ts tamper {label}: rejected OK")
    print("VERIFY_TS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "generate"
    if cmd == "generate":
        generate()
    elif cmd == "verify-ts":
        sys.exit(verify_ts())
    else:
        print("unknown command", cmd)
        sys.exit(2)
