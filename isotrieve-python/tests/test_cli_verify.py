"""Tests for ``isotrieve verify`` — post-migration drift revalidation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tests.fakes import save_mapping
from typer.testing import CliRunner

from isotrieve.cli import app
from isotrieve.mapping.linear import RidgeMapping
from isotrieve.quality.gate import QualityGate
from isotrieve.stores.numpy_files import NumpyFileStore

runner = CliRunner()

D_SRC, D_TGT = 8, 12


def _write_manifest(manifest_dir: Path, data: dict[str, object]) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{data['id']}.json"
    path.write_text(json.dumps(data, default=str), encoding="utf-8")
    return path


def _migration_manifest(
    mapping_path: str, gate_result: dict[str, object]
) -> dict[str, object]:
    return {
        "id": "mig_verify",
        "source_collection": "source",
        "target_collection": "target",
        "source_model": "src-model",
        "target_model": "tgt-model",
        "total_vectors": 40,
        "migrated_vectors": 40,
        "batch_start": 0,
        "batch_end": 1,
        "last_batch_hash": "abc",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:01:00+00:00",
        "batches": [],
        "mapping_path": mapping_path,
        "transform_invertible": True,
        "rollback_strategy": "inverse",
        "gate_result": gate_result,
        "source_store": {},
        "target_store": {},
    }


def _strong_mapping(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a mapping with ~perfect retention on a fresh query set.

    Fit ``Y = X @ W`` exactly on 500 samples, then draw a disjoint query set
    from the same generative process: the ridge mapping recovers W, so
    gate retention is ~1.0 (predicted ~0.838). Degraded corpora collapse to
    the model floor (~0.683), guaranteeing a detectable drop >= 0.10.
    """
    rng = np.random.default_rng(123)
    X_fit = rng.normal(size=(500, D_SRC))
    W = rng.normal(size=(D_SRC, D_TGT))
    Y_fit = X_fit @ W
    mapping = RidgeMapping(alpha=0.01, seed=0)
    mapping.fit(X_fit, Y_fit)
    map_path = Path(save_mapping(mapping, tmp_path, "map.isotrieve"))

    rng2 = np.random.default_rng(1234)
    queries = rng2.normal(size=(40, D_SRC))
    corpus = queries @ W
    return map_path, queries, corpus, W


def _migration_gate_result(mapping_path: Path, queries: np.ndarray, corpus: np.ndarray):
    from isotrieve.mapping.registry import load_mapping

    mapping = load_mapping(mapping_path)
    return QualityGate().evaluate(mapping, queries, corpus).to_dict()


def _invoke_verify(
    manifest_path_or_id: str,
    manifest_dir: Path,
    queries: Path,
    corpus: Path | None = None,
    live_store: Path | None = None,
    *,
    json_out: bool = False,
    **extra: str,
) -> object:
    args = [
        "verify",
        "--manifest",
        manifest_path_or_id,
        "--dir",
        str(manifest_dir),
        "--queries",
        str(queries),
    ]
    if corpus is not None:
        args += ["--corpus", str(corpus)]
    if live_store is not None:
        args += ["--live-store", str(live_store)]
    if json_out:
        args.append("--json")
    for flag, value in extra.items():
        args += [f"--{flag.replace('_', '-')}", value]
    return runner.invoke(app, args)


class TestVerifyNoDrift:
    def test_no_drift_when_live_equals_migration_baseline(self, tmp_path):
        map_path, queries, corpus, _ = _strong_mapping(tmp_path)
        gate_result = _migration_gate_result(map_path, queries, corpus)
        manifest_dir = tmp_path / "m"
        _write_manifest(manifest_dir, _migration_manifest(str(map_path), gate_result))

        q_path = tmp_path / "q.npy"
        c_path = tmp_path / "c.npy"
        np.save(q_path, queries)
        np.save(c_path, corpus)

        r = _invoke_verify(
            "mig_verify", manifest_dir, q_path, corpus=c_path, json_out=True
        )
        assert r.exit_code == 0, r.stdout + str(r.exception)
        data = json.loads(r.output)
        assert data["drifted"] is False
        assert data["verdict"] == "PASS"
        assert data["retention_drop"] == 0.0

    def test_verify_accepts_direct_manifest_path(self, tmp_path):
        map_path, queries, corpus, _ = _strong_mapping(tmp_path)
        gate_result = _migration_gate_result(map_path, queries, corpus)
        manifest_file = _write_manifest(
            tmp_path / "m", _migration_manifest(str(map_path), gate_result)
        )
        q_path = tmp_path / "q.npy"
        c_path = tmp_path / "c.npy"
        np.save(q_path, queries)
        np.save(c_path, corpus)

        r = _invoke_verify(
            str(manifest_file), tmp_path / "m", q_path, corpus=c_path, json_out=True
        )
        assert r.exit_code == 0, r.stdout + str(r.exception)
        assert json.loads(r.output)["verdict"] == "PASS"

    def test_verify_with_live_store(self, tmp_path):
        """--live-store re-reads the migrated corpus from a numpy dir."""
        map_path, queries, corpus, _ = _strong_mapping(tmp_path)
        gate_result = _migration_gate_result(map_path, queries, corpus)
        manifest_dir = tmp_path / "m"
        _write_manifest(manifest_dir, _migration_manifest(str(map_path), gate_result))

        live = NumpyFileStore.from_arrays(tmp_path / "live", corpus.astype(np.float32))
        q_path = tmp_path / "q.npy"
        np.save(q_path, queries)

        r = _invoke_verify(
            "mig_verify", manifest_dir, q_path, live_store=live.path, json_out=True
        )
        assert r.exit_code == 0, r.stdout + str(r.exception)
        data = json.loads(r.output)
        assert data["drifted"] is False


class TestVerifyDrift:
    def test_drift_warning_when_retention_drops(self, tmp_path):
        map_path, queries, corpus, _ = _strong_mapping(tmp_path)
        gate_result = _migration_gate_result(map_path, queries, corpus)
        manifest_dir = tmp_path / "m"
        _write_manifest(manifest_dir, _migration_manifest(str(map_path), gate_result))

        rng = np.random.default_rng(99)
        noisy_corpus = rng.normal(size=(40, D_TGT))
        q_path = tmp_path / "q.npy"
        c_path = tmp_path / "c.npy"
        np.save(q_path, rng.normal(size=(40, D_SRC)))
        np.save(c_path, noisy_corpus)

        r = _invoke_verify(
            "mig_verify", manifest_dir, q_path, corpus=c_path, json_out=True
        )
        assert r.exit_code == 1, r.stdout + str(r.exception)
        data = json.loads(r.output)
        assert data["drifted"] is True
        assert data["verdict"] == "DRIFT"
        assert data["retention_drop"] >= 0.10

    def test_warn_threshold_flag_triggers_absolute_drift(self, tmp_path):
        """--warn-threshold flags drift even when the drop is small."""
        map_path, queries, corpus, _ = _strong_mapping(tmp_path)
        gate_result = _migration_gate_result(map_path, queries, corpus)
        manifest_dir = tmp_path / "m"
        _write_manifest(manifest_dir, _migration_manifest(str(map_path), gate_result))

        q_path = tmp_path / "q.npy"
        c_path = tmp_path / "c.npy"
        np.save(q_path, queries)
        np.save(c_path, corpus)

        r = _invoke_verify(
            "mig_verify",
            manifest_dir,
            q_path,
            corpus=c_path,
            warn_threshold="0.99",
            json_out=True,
        )
        assert r.exit_code == 1, r.stdout + str(r.exception)
        data = json.loads(r.output)
        assert data["drifted"] is True
        assert "warn threshold" in data["reason"]

    def test_verify_no_baseline_uses_absolute_threshold(self, tmp_path):
        """Without a gate_result baseline, drift is judged on absolute value."""
        map_path, queries, corpus, _ = _strong_mapping(tmp_path)
        manifest_dir = tmp_path / "m"
        _write_manifest(manifest_dir, _migration_manifest(str(map_path), {}))

        q_path = tmp_path / "q.npy"
        c_path = tmp_path / "c.npy"
        np.save(q_path, queries)
        np.save(c_path, corpus)

        r = _invoke_verify(
            "mig_verify", manifest_dir, q_path, corpus=c_path, json_out=True
        )
        assert r.exit_code == 0, r.stdout + str(r.exception)
        assert json.loads(r.output)["drifted"] is False


class TestVerifyExitCodes:
    def test_usage_errors_exit_2(self, tmp_path):
        map_path, queries, corpus, _ = _strong_mapping(tmp_path)
        manifest_dir = tmp_path / "m"
        _write_manifest(manifest_dir, _migration_manifest(str(map_path), {}))

        q_path = tmp_path / "q.npy"
        c_path = tmp_path / "c.npy"
        np.save(q_path, queries)
        np.save(c_path, corpus)

        # no --queries
        r = runner.invoke(
            app,
            [
                "verify",
                "--manifest",
                "mig_verify",
                "--dir",
                str(manifest_dir),
                "--corpus",
                str(c_path),
            ],
        )
        assert r.exit_code == 2

        # neither --corpus nor --live-store
        r = runner.invoke(
            app,
            [
                "verify",
                "--manifest",
                "mig_verify",
                "--dir",
                str(manifest_dir),
                "--queries",
                str(q_path),
            ],
        )
        assert r.exit_code == 2

        # both --corpus and --live-store
        live = NumpyFileStore.from_arrays(tmp_path / "live", corpus.astype(np.float32))
        r = runner.invoke(
            app,
            [
                "verify",
                "--manifest",
                "mig_verify",
                "--dir",
                str(manifest_dir),
                "--queries",
                str(q_path),
                "--corpus",
                str(c_path),
                "--live-store",
                str(live.path),
            ],
        )
        assert r.exit_code == 2

    def test_missing_mapping_path_exits_1(self, tmp_path):
        manifest_dir = tmp_path / "m"
        _write_manifest(
            manifest_dir, _migration_manifest("/does/not/exist.isotrieve", {})
        )
        rng = np.random.default_rng(1)
        q_path = tmp_path / "q.npy"
        c_path = tmp_path / "c.npy"
        np.save(q_path, rng.normal(size=(5, D_SRC)))
        np.save(c_path, rng.normal(size=(5, D_TGT)))

        r = _invoke_verify(
            "mig_verify", manifest_dir, q_path, corpus=c_path, json_out=True
        )
        assert r.exit_code == 1
        assert "no usable mapping_path" in r.stdout


class TestVerifySchedule:
    def test_schedule_flag_acknowledged(self, tmp_path):
        map_path, queries, corpus, _ = _strong_mapping(tmp_path)
        gate_result = _migration_gate_result(map_path, queries, corpus)
        manifest_dir = tmp_path / "m"
        _write_manifest(manifest_dir, _migration_manifest(str(map_path), gate_result))

        q_path = tmp_path / "q.npy"
        c_path = tmp_path / "c.npy"
        np.save(q_path, queries)
        np.save(c_path, corpus)

        r = _invoke_verify(
            "mig_verify",
            manifest_dir,
            q_path,
            corpus=c_path,
            schedule="0 2 * * *",
            json_out=True,
        )
        assert r.exit_code == 0, r.stdout + str(r.exception)
        assert "Scheduled (0 2 * * *)" in r.output
