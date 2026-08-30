## Check
check_id: P7-06-mps (+ follow-on from P7-04b evidence)

## Observed 1 — device="auto" crashes
`ResidualMLPMapping(device="auto")` raises:
`RuntimeError: Expected one of cpu, cuda, ... at start of device string: auto`

Root cause (`src/isotrieve/mapping/mlp.py:68-73`): `_resolve_device()` returns `torch.device(self._device_str)` for ANY non-None string, so "auto" is passed to torch verbatim; the auto-detection branch below is unreachable for the documented value. The changelog claims "ResidualMLPMapping(device=) for MPS/CUDA auto-detection" and the unit tests only ever use `device="cpu"`, so this path is untested.

## Observed 2 — margin_compression degenerates for high-fidelity mappings
`isotrieve gate` on a synthetic pair where the mapping generalizes perfectly (top1_retention = 1.0, cosine_mean = 0.99999986, disjoint holdout per the evaluate() contract):

    verdict: WARN
    predicted_retention: 0.8378   (from a gate model fed top1=1.0)
    margin_compression: 3.6e-07
    prediction_interval: [0.342, 1.0]
    rationale: "Score margins are compressed (ratio=0.00)..."

`_compute_margin_compression` = var(paired cosine) / var(random target-pair cosine). A mapping that is *excell* has tightly clustered paired cosines → numerator → 0 → the metric reads "extreme compression" and the gate model penalizes a perfect migration, widening the interval to [0.34, 1.0]. The same signature appeared with calibration-overlapping pairs AND with disjoint holdout pairs, at noise scales from ~0 to 0.15.

## Reproduce

    python - <<'EOF'
    import numpy as np
    from isotrieve.mapping.linear import RidgeMapping
    from isotrieve.quality.gate import QualityGate
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 64)); q,_ = np.linalg.qr(rng.normal(size=(64,64)))
    W = q @ rng.normal(size=(64,128))*0.2
    Y = X @ W + 0.001*rng.normal(size=(200,128))
    m = RidgeMapping(alpha="auto", seed=0).fit(X[:100], Y[:100])
    print(QualityGate().evaluate(m, X[100:], Y[100:]).to_dict()["margin_compression"])
    # → ~0 despite near-perfect retention
    EOF

Base SHA: 28cb6f1 · deterministic

## Evidence
- verification/artifacts/20260829T215744/P7/p7_final2.log (P7-04b, P7-06 lines)
- gate JSON: verdict WARN / top1 1.0 / ratio 3.6e-07 / interval [0.342, 1.0]

## Triage
bucket: product-bug (confidence: high for device="auto"; medium for margin_compression — on real BEIR pairs the ratio is not degenerate, but the metric's semantics invert quality exactly when the transform is best, and nothing clamps or floors it).

## Impact
- MPS users cannot use the documented auto-detection at all.
- Gates of very good migrations can emit WARN with a near-useless [0.34, 1.0] interval and a "score margins compressed" rationale at ratio 0.00 — misleading operator guidance in the failure report.

## Suggested fix
- `_resolve_device`: check `== "auto"` BEFORE the generic branch (resolve to mps/cuda/cpu by availability).
- `margin_compression`: floor the ratio (e.g. max(ratio, epsilon) with paired-variance floor relative to numerical noise) or gate the penalty on the paired-cosine MEAN being low, not just variance being low; add a high-fidelity regression case.

## Status
[ ] Fixed in this run   [x] Deferred to owner (product code)
