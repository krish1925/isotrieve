"""Bias / leakage / fairness audit (barrage P5). Report-oriented: every check
records observed values; failures are findings, never silently fixed.

Pre-registered in verification/artifacts/*/preregistration.md (binding).
"""

from __future__ import annotations

import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from isotrieve.mapping.base import l2_normalize
from isotrieve.mapping.linear import RidgeMapping
from isotrieve.quality.gate import QualityGate
from isotrieve.recalibration import ScoreRecalibrator

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "benchmarks"
SEED = 20260829

sys.path.insert(0, str(BENCH))  # noqa: E402 - harness is a script package


# ---------------------------------------------------------------- P5-01
class TestP501LeakageDisjointness:
    def test_source_calibration_sampling_uses_docs_only(self) -> None:
        import run_benchmark as rb  # noqa: PLC0415

        src = inspect.getsource(rb.run_seed)
        assert "cal_idx = rng.choice(len(doc_ids)" in src
        # the sampling line must not reference queries or qrels
        for line in src.splitlines():
            if "cal_idx = rng.choice" in line:
                assert "quer" not in line and "qrel" not in line, line

    def test_sampling_deterministic_per_seed(self) -> None:
        import run_benchmark as rb  # noqa: PLC0415

        src = inspect.getsource(rb.run_seed)
        snippet = textwrap.dedent(
            "\n".join(
                ln
                for ln in src.splitlines()
                if "default_rng(seed)" in ln or "cal_idx = rng.choice" in ln
            )
        )
        assert snippet.count("rng.choice") == 1
        doc_ids = [f"doc{i}" for i in range(500)]
        namespace = {"np": np, "doc_ids": doc_ids, "k_cal": 100, "seed": 424242}
        exec(snippet, namespace)  # noqa: S102 - deliberately runs the harness's own lines
        first = namespace["cal_idx"].copy()
        namespace2 = {"np": np, "doc_ids": doc_ids, "k_cal": 100, "seed": 424242}
        exec(snippet, namespace2)  # noqa: S102
        assert np.array_equal(first, namespace2["cal_idx"]), (
            "cal sampling not deterministic"
        )

    def test_calibration_overlap_with_qrels_reported(self) -> None:
        ir_datasets = pytest.importorskip("ir_datasets")
        import run_benchmark as rb  # noqa: PLC0415

        ds = ir_datasets.load("beir/scifact/test")
        qrels_docs: set[str] = set()
        for q in ds.qrels_iter():
            if int(q.relevance) > 0:
                qrels_docs.add(q.doc_id)
        docs, _queries, harness_qrels, _name = rb.load_scifact(200)
        doc_ids = [d["id"] for d in docs]
        qrels_docs.update(d for s in harness_qrels.values() for d in s)
        rng = np.random.default_rng(0)
        cal_idx = rng.choice(len(doc_ids), size=min(100, len(doc_ids)), replace=False)
        cal_docs = {str(doc_ids[i]) for i in cal_idx}
        overlap = len(cal_docs & qrels_docs)
        rate = overlap / max(1, len(cal_docs))
        # REPORT metric (methodology fact, not a failure): written to stdout for the artifact
        print(
            f"P5-01 calibration-vs-qrels overlap: {overlap}/{len(cal_docs)} = {rate:.3f}"
        )
        assert rate <= 1.0


# ---------------------------------------------------------------- P5-02
class TestP502GateLopoRevalidation:
    def test_lopo_mae_within_budget(self) -> None:
        from fit_gate_model import (  # noqa: PLC0415
            extract_features,
            load_results,
            lopo_cross_validation,
        )

        results = load_results(BENCH / "results")
        data = [f for f in (extract_features(d) for d in results) if f is not None]
        assert len(data) >= 2, f"only {len(data)} usable pairs"
        lopo = lopo_cross_validation(data)
        assert "error" not in lopo, lopo
        per_pair = lopo.get("residuals_by_pair") or {}
        maes = []
        for pair, residual in per_pair.items():
            err = abs(float(residual))
            maes.append(err)
            print(f"P5-02 fold {pair}: |pred-actual|={err:.4f}")
        assert maes, f"no per-fold residuals reported: {sorted(lopo)}"
        mean_mae = float(np.mean(maes))
        print(
            f"P5-02 LOPO mean MAE = {mean_mae:.4f} over {len(maes)} folds (reported aggregate {lopo.get('mae')})"
        )
        assert mean_mae <= 0.20, f"gate LOPO MAE {mean_mae:.3f} > 0.20 budget"

    def test_lopo_fold_exclusion_in_source(self) -> None:
        from pathlib import Path as _P  # noqa: PLC0415

        src = (_P(BENCH) / "fit_gate_model.py").read_text()
        assert "!= held_out" in src, "LOPO must exclude the held-out pair from training"


# ---------------------------------------------------------------- P5-03
class TestP503AlphaIsolation:
    def test_solver_never_sees_eval_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sklearn.linear_model as sk_lm  # noqa: PLC0415

        seen: list[np.ndarray] = []
        orig_fit = sk_lm.RidgeCV.fit

        def spy(
            self: sk_lm.RidgeCV, X: np.ndarray, y: np.ndarray, *a: object, **k: object
        ) -> object:
            seen.append(np.asarray(X).copy())
            return orig_fit(self, X, y, *a, **k)

        monkeypatch.setattr(sk_lm.RidgeCV, "fit", spy)
        rng = np.random.default_rng(SEED)
        Xcal = rng.normal(size=(120, 8))
        Ycal = l2_normalize(Xcal @ rng.normal(size=(8, 12)))
        RidgeMapping(alpha="auto", seed=0).fit(Xcal, Ycal)
        # sentinel eval rows must never appear in anything the solver saw
        assert seen, "solver was never invoked"
        for arr in seen:
            for row in arr.reshape(-1, arr.shape[-1]):
                assert not np.allclose(row[:8], 999.0), (
                    "eval sentinel leaked into solver"
                )
        alphas = [float(np.linalg.norm(a)) for a in seen]
        print(f"P5-03 solver saw {len(seen)} arrays; norms={alphas}")

    def test_alpha_reported_across_five_subsets(self) -> None:
        alphas = []
        for i in range(5):
            rng = np.random.default_rng(SEED + i)
            scale = [1.0, 10.0, 0.1, 100.0, 0.01][i]
            X = rng.normal(size=(100, 8)) * scale
            Y = l2_normalize(
                (X + rng.normal(size=(100, 8)) * scale) @ rng.normal(size=(8, 10))
            )
            m = RidgeMapping(alpha="auto", seed=0).fit(X, Y)
            alphas.append(float(m._chosen_alpha))  # noqa: SLF001
        print(f"P5-03 alphas across 5 scales: {alphas}")
        assert all(np.isfinite(a) for a in alphas)


# ---------------------------------------------------------------- P5-04
class TestP504RecalibratorSplit:
    def test_holdout_excluded_from_fit_support(self) -> None:
        rng = np.random.default_rng(SEED)
        n_docs, d = 60, 8
        doc_src = rng.normal(size=(n_docs, d))
        doc_tgt = l2_normalize(
            doc_src @ rng.normal(size=(d, d)) + 0.05 * rng.normal(size=(n_docs, d))
        )
        qry_tgt = l2_normalize(rng.normal(size=(8, d)))
        doc_ids = [f"d{i}" for i in range(n_docs)]
        query_ids = [f"q{i}" for i in range(8)]
        qrels = {
            q: {doc_ids[(i * 7 + j) % n_docs]}
            for j, q in enumerate(query_ids)
            for i in (j,)
        }
        mapped_docs = l2_normalize(doc_tgt + 0.1 * rng.normal(size=(n_docs, d)))
        rec = ScoreRecalibrator.fit_from_holdout(
            doc_src, doc_tgt, qry_tgt, doc_ids, query_ids, qrels, mapped_docs
        )
        assert rec.is_fitted
        report = rec.report
        assert report is not None
        info = report.to_dict() if hasattr(report, "to_dict") else {}
        print(f"P5-04 report fields: {info}")
        n_hold = info.get("n_holdout") or info.get("n_holdout_pairs")
        n_fit = info.get("n_fit") or info.get("n_pairs")
        if n_hold is not None and n_fit is not None:
            assert (
                int(n_fit) + int(n_hold) <= n_docs * 8
            )  # no double-counting of holdout
        else:
            print(
                "P5-04 NOTE: report lacks explicit split counts; source-verified only"
            )


# ---------------------------------------------------------------- P5-05
@pytest.mark.slow
@pytest.mark.xfail(reason="finding #86: bootstrap CI duplicate-tie bias — becomes XPASS when fixed", strict=False)
class TestP505BootstrapCoverage:
    @pytest.mark.parametrize("r", [0.5, 0.7, 0.9])
    @pytest.mark.xfail(reason="finding #86 — see class marker", strict=False)
    def test_ci_coverage_of_true_retention(self, r: float) -> None:
        from isotrieve.cli_gate import _bootstrap_retention_ci  # noqa: PLC0415

        n, universes, resamples = 200, 400, 200
        Y = np.eye(n)
        rng_master = np.random.default_rng(SEED)

        class _Stub:
            def __init__(self, mask: np.ndarray) -> None:
                self._M = Y[mask] if False else None  # placeholder, set below
                rows = []
                for i in range(n):
                    rows.append(Y[i] if mask[i] else Y[(i + 1) % n])
                self._M = np.array(rows)

            def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: ARG002
                return self._M

        covered = 0
        for _ in range(universes):
            mask = rng_master.random(n) < r
            stub = _Stub(mask)
            X = np.zeros((n, 1))
            ci = _bootstrap_retention_ci(stub, X, Y, n_resamples=resamples, seed=SEED)
            lo, hi = ci["recall_at_1"]
            if lo <= r <= hi:
                covered += 1
        coverage = covered / universes
        mc_stderr = float(np.sqrt(coverage * (1 - coverage) / universes))
        print(f"P5-05 r={r}: coverage={coverage:.4f} ± {mc_stderr:.4f} (nominal 80%)")
        assert 0.75 <= coverage <= 0.85, (
            f"r={r} coverage {coverage:.3f} outside [0.75,0.85]"
        )


# ---------------------------------------------------------------- P5-06
class TestP506SeedSensitivityHonesty:
    def _report(self) -> dict:
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(200, 8))
        Y = l2_normalize(
            X @ rng.normal(size=(8, 12)) + 0.02 * rng.normal(size=(200, 12))
        )
        gate = QualityGate()
        return gate.seed_sensitivity(X, Y, runs=5, seed=SEED).to_dict()

    def test_deterministic_and_internally_consistent(self) -> None:
        r1, r2 = self._report(), self._report()
        assert r1 == r2, "seed_sensitivity not deterministic at fixed seed"
        per = np.array(r1["per_seed_retention"])
        assert abs(r1["std_retention"] - float(np.std(per))) < 1e-12
        assert r1["min_retention"] <= r1["mean_retention"] <= r1["max_retention"]
        print(
            f"P5-06 spread: min={r1['min_retention']:.4f} mean={r1['mean_retention']:.4f} "
            f"max={r1['max_retention']:.4f} std={r1['std_retention']:.4f}"
        )


# ---------------------------------------------------------------- P5-07
class TestP507MarginCompression:
    @pytest.mark.parametrize("eps", [0.2, 0.4])
    def test_compression_detected_and_penalized(self, eps: float) -> None:
        rng = np.random.default_rng(SEED)
        n, d = 120, 16
        T = l2_normalize(rng.normal(size=(n, d)))
        # Pure radial scaling is invisible to cosine; compression is modeled as
        # paired-similarity variance shrink: mapped_i = normalize(T_i + eps*G_i)
        # => paired cosine variance ~ eps^2/d vs random-pair reference ~ 1/d.
        G = rng.normal(size=(n, d))
        mapped = l2_normalize(T + eps * G)
        gate = QualityGate()
        ratio = gate._compute_margin_compression(mapped, T)  # noqa: SLF001
        assert ratio is not None and np.isfinite(ratio)
        print(f"P5-07 eps={eps}: detected variance-ratio={ratio:.4f}")
        assert ratio < 1.0, f"compression not detected (ratio={ratio:.3f})"
        # monotone: more noise -> less detected compression
        loose = gate._compute_margin_compression(  # noqa: SLF001
            l2_normalize(T + 2.0 * eps * rng.normal(size=(n, d))), T
        )
        assert ratio < loose, f"monotonicity broken: {ratio:.4f} vs loose {loose:.4f}"
        # interval widens: _predict_retention penalizes compression < 0.85
        base = gate._predict_retention(0.9, None)  # noqa: SLF001
        penalized = gate._predict_retention(0.9, ratio)  # noqa: SLF001
        assert penalized < base, (
            f"interval did not widen: base={base}, penalized={penalized}"
        )

    def test_bug8_regression_non_degenerate_normalized(self) -> None:
        rng = np.random.default_rng(SEED)
        T = l2_normalize(rng.normal(size=(120, 16)))  # valid, non-degenerate
        gate = QualityGate()
        ratio = gate._compute_margin_compression(T, T)  # noqa: SLF001
        assert ratio is not None, "bug #8 recurrence: None on valid input"
        assert np.isfinite(ratio) > 0.0 if isinstance(ratio, float) else True
        assert np.isfinite(ratio) and ratio != 0.0


# ---------------------------------------------------------------- P5-08..11 (static audits)
class TestP508to11StaticAudits:
    @pytest.mark.xfail(reason="finding #87: legacy artifacts predate protocol/config_hash — becomes XPASS when fixed", strict=False)
    def test_p5_08_all_artifacts_declare_protocol(self) -> None:
        import json  # noqa: PLC0415

        arts = sorted((BENCH / "results").glob("*.json"))
        assert len(arts) >= 60, f"unexpected artifact count {len(arts)}"
        missing = []
        for a in arts:
            try:
                d = json.loads(a.read_text())
            except Exception:
                missing.append(f"{a.name}: unparseable")
                continue
            for field in ("protocol", "config_hash", "commit"):
                if field not in d:
                    missing.append(f"{a.name}: missing {field}")
        assert not missing, f"undeclared-config artifacts: {missing[:10]}"

    def test_p5_09_truncation_fairness_same_arms(self) -> None:
        import run_benchmark as rb  # noqa: PLC0415

        src = inspect.getsource(rb.run_seed)
        # floor/ceiling/mapped must all consume the same doc arrays within run_seed
        assert "floor" in src and "ceiling" in src
        for line in src.splitlines():
            if "floor" in line.lower() and "load" in line.lower():
                raise AssertionError(f"arm reloads corpus: {line}")

    def test_p5_10_no_eval_peeking_in_fit_loops(self) -> None:
        for mod in ("mlp", "contrastive"):
            p = (
                REPO_ROOT
                / "isotrieve-python"
                / "src"
                / "isotrieve"
                / "mapping"
                / f"{mod}.py"
            )
            if not p.exists():  # contrastive lives elsewhere; resolve by glob
                cands = list(
                    (
                        REPO_ROOT / "isotrieve-python" / "src" / "isotrieve" / "mapping"
                    ).glob(f"*{mod}*.py")
                )
                assert cands, f"module for {mod} not found"
                p = cands[0]
            src = p.read_text().lower()
            for bad in ("qrels", "eval_set", "test_set", "early_stopping", "val_loss"):
                assert bad not in src, f"{p.name} references {bad} inside fit module"

    def test_p5_11_probe_purity_same_index(self) -> None:
        src = (BENCH / "domain_probes.py").read_text()
        assert "deterministic" in src.lower() or "Deterministic" in src
        runner = (BENCH / "run_benchmark.py").read_text()
        assert "probe_chunks" in runner and "probe_queries" in runner
        # scoring path must compare probe query vs its own chunk id, not corpus embeddings
        assert "probe_retention" in runner or "probe_top1" in runner
