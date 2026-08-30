"""Tests for the manifest lifecycle CLI (list/show, rollback, resume)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tests.fakes import make_mapping, save_mapping
from typer.testing import CliRunner

from isotrieve.cli import app
from isotrieve.migrate import migrate_store
from isotrieve.stores.numpy_files import NumpyFileStore

runner = CliRunner()


def _write_manifest(manifest_dir: Path, data: dict[str, object]) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{data['id']}.json"
    path.write_text(json.dumps(data, default=str), encoding="utf-8")
    return path


def _base_manifest(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "mig_test",
        "source_collection": "source",
        "target_collection": "target",
        "source_model": "src-model",
        "target_model": "tgt-model",
        "total_vectors": 30,
        "migrated_vectors": 0,
        "batch_start": 0,
        "batch_end": 0,
        "last_batch_hash": "",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "",
        "batches": [],
        "mapping_path": "",
        "transform_invertible": True,
        "rollback_strategy": "inverse",
        "gate_result": {},
        "source_store": {},
        "target_store": {},
    }
    data.update(overrides)
    return data


class TestManifestListShow:
    def test_list_show_roundtrip(self, tmp_path):
        """A manifest written by migrate_store round-trips through list/show."""
        m = make_mapping(d_src=8, d_tgt=12, k=200)
        map_path = Path(save_mapping(m, tmp_path, "map.isotrieve"))
        rng = np.random.default_rng(42)
        source = NumpyFileStore.from_arrays(
            tmp_path / "src", rng.normal(size=(30, 8)).astype(np.float32)
        )
        target = NumpyFileStore(tmp_path / "tgt", create=True)
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()

        manifest = migrate_store(
            source,
            target,
            m,
            batch_size=10,
            manifest_path=manifest_dir / "migration.json",
            mapping_path=map_path,
            rollback_strategy="inverse",
        )

        listed = runner.invoke(
            app,
            ["manifest", "list", "--dir", str(manifest_dir), "--json"],
        )
        assert listed.exit_code == 0, listed.stdout + str(listed.exception)
        assert manifest.id in listed.output

        shown = runner.invoke(
            app,
            ["manifest", "show", manifest.id, "--dir", str(manifest_dir), "--json"],
        )
        assert shown.exit_code == 0, shown.stdout + str(shown.exception)
        data = json.loads(shown.output)
        assert data["id"] == manifest.id
        assert data["mapping_path"] == str(map_path)
        assert data["transform_invertible"] is True
        assert data["rollback_strategy"] == "inverse"
        assert data["source_store"] == {"type": "numpy", "uri": str(source.path)}
        assert data["target_store"] == {"type": "numpy", "uri": str(target.path)}
        assert data["migrated_vectors"] == 30
        assert data["completed_at"] != ""

    def test_show_missing_manifest(self, tmp_path):
        r = runner.invoke(
            app, ["manifest", "show", "does-not-exist", "--dir", str(tmp_path)]
        )
        assert r.exit_code == 1
        assert "Manifest not found" in r.stdout


class TestRollback:
    def test_rollback_dry_run_invertible(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12, k=200)
        map_path = Path(save_mapping(m, tmp_path, "map.isotrieve"))
        manifest_dir = tmp_path / "m"
        _write_manifest(manifest_dir, _base_manifest(mapping_path=str(map_path)))

        r = runner.invoke(
            app,
            [
                "rollback",
                "--manifest",
                "mig_test",
                "--dir",
                str(manifest_dir),
                "--dry-run",
            ],
        )
        assert r.exit_code == 0, r.stdout + str(r.exception)
        assert "Rollback plan" in r.stdout
        assert "inverse-transform target store" in r.stdout

    def test_rollback_refuses_non_invertible_without_shadow(self, tmp_path):
        """Non-invertible transform + no shadow copy → refused with exit 2."""
        manifest_dir = tmp_path / "m"
        _write_manifest(
            manifest_dir,
            _base_manifest(transform_invertible=False, rollback_strategy="none"),
        )

        r = runner.invoke(
            app,
            [
                "rollback",
                "--manifest",
                "mig_test",
                "--dir",
                str(manifest_dir),
                "--dry-run",
            ],
        )
        assert r.exit_code == 2
        assert "Rollback refused" in r.stdout
        assert "no analytic inverse" in r.stdout

    def test_rollback_non_invertible_allowed_with_shadow(self, tmp_path):
        """Non-invertible but shadow copy preserved → rollback allowed."""
        rng = np.random.default_rng(1)
        source = NumpyFileStore.from_arrays(
            tmp_path / "src", rng.normal(size=(10, 8)).astype(np.float32)
        )
        manifest_dir = tmp_path / "m"
        _write_manifest(
            manifest_dir,
            _base_manifest(
                transform_invertible=False,
                rollback_strategy="shadow",
                source_store={"type": "numpy", "uri": str(source.path)},
            ),
        )

        r = runner.invoke(
            app,
            [
                "rollback",
                "--manifest",
                "mig_test",
                "--dir",
                str(manifest_dir),
                "--dry-run",
            ],
        )
        assert r.exit_code == 0, r.stdout + str(r.exception)
        assert "restore preserved source" in r.stdout

    def test_rollback_inverse_execution(self, tmp_path):
        """Inverse rollback writes source-space vectors to the recovery dir."""
        m = make_mapping(d_src=8, d_tgt=12, k=200)
        map_path = Path(save_mapping(m, tmp_path, "map.isotrieve"))
        rng = np.random.default_rng(42)
        source = NumpyFileStore.from_arrays(
            tmp_path / "src", rng.normal(size=(20, 8)).astype(np.float32)
        )
        target = NumpyFileStore(tmp_path / "tgt", create=True)
        manifest_dir = tmp_path / "m"
        manifest_dir.mkdir()

        manifest = migrate_store(
            source,
            target,
            m,
            batch_size=10,
            manifest_path=manifest_dir / "m.json",
            mapping_path=map_path,
            rollback_strategy="inverse",
        )

        recovery = tmp_path / "recovery"
        r = runner.invoke(
            app,
            [
                "rollback",
                "--manifest",
                manifest.id,
                "--dir",
                str(manifest_dir),
                "--output-dir",
                str(recovery),
            ],
        )
        assert r.exit_code == 0, r.stdout + str(r.exception)
        assert "Rollback complete" in r.stdout

        recovered = NumpyFileStore(recovery)
        assert recovered.count() == 20
        for batch in recovered.iter_vectors(batch_size=20):
            for record in batch:
                assert record.vector.shape == (8,)

    def test_rollback_shadow_execution(self, tmp_path):
        """Shadow rollback restores the preserved source store."""
        rng = np.random.default_rng(3)
        source = NumpyFileStore.from_arrays(
            tmp_path / "src", rng.normal(size=(10, 8)).astype(np.float32)
        )
        manifest_dir = tmp_path / "m"
        _write_manifest(
            manifest_dir,
            _base_manifest(
                id="shadow1",
                transform_invertible=False,
                rollback_strategy="shadow",
                source_store={"type": "numpy", "uri": str(source.path)},
            ),
        )

        recovery = tmp_path / "recovery"
        r = runner.invoke(
            app,
            [
                "rollback",
                "--manifest",
                "shadow1",
                "--dir",
                str(manifest_dir),
                "--output-dir",
                str(recovery),
            ],
        )
        assert r.exit_code == 0, r.stdout + str(r.exception)
        assert NumpyFileStore(recovery).count() == 10


class TestResume:
    def test_resume_continues_partial_migration(self, tmp_path):
        """Resume skips completed batches and finishes the migration."""
        m = make_mapping(d_src=8, d_tgt=12, k=200)
        map_path = Path(save_mapping(m, tmp_path, "map.isotrieve"))
        rng = np.random.default_rng(42)
        source = NumpyFileStore.from_arrays(
            tmp_path / "src", rng.normal(size=(30, 8)).astype(np.float32)
        )
        target = NumpyFileStore(tmp_path / "tgt", create=True)

        manifest_dir = tmp_path / "m"
        _write_manifest(
            manifest_dir,
            _base_manifest(
                id="partial",
                total_vectors=30,
                migrated_vectors=10,
                batch_end=1,
                mapping_path=str(map_path),
                source_store={"type": "numpy", "uri": str(source.path)},
                target_store={"type": "numpy", "uri": str(target.path)},
            ),
        )

        r = runner.invoke(
            app,
            [
                "resume",
                "--manifest",
                "partial",
                "--dir",
                str(manifest_dir),
                "--batch-size",
                "10",
            ],
        )
        assert r.exit_code == 0, r.stdout + str(r.exception)
        assert "Resumed partial" in r.stdout

        reloaded = json.loads(
            (manifest_dir / "partial.json").read_text(encoding="utf-8")
        )
        assert reloaded["migrated_vectors"] == 30
        assert reloaded["batch_end"] == 3
        assert reloaded["completed_at"] != ""

        assert NumpyFileStore(tmp_path / "tgt").count() == 20

    def test_resume_completed_manifest_is_noop(self, tmp_path):
        manifest_dir = tmp_path / "m"
        _write_manifest(
            manifest_dir,
            _base_manifest(id="done", completed_at="2026-01-02T00:00:00+00:00"),
        )

        r = runner.invoke(
            app,
            ["resume", "--manifest", "done", "--dir", str(manifest_dir)],
        )
        assert r.exit_code == 0, r.stdout + str(r.exception)
        assert "already completed" in r.stdout

    def test_resume_requires_mapping_path(self, tmp_path):
        manifest_dir = tmp_path / "m"
        _write_manifest(manifest_dir, _base_manifest(id="nomap", mapping_path=""))

        r = runner.invoke(
            app,
            ["resume", "--manifest", "nomap", "--dir", str(manifest_dir)],
        )
        assert r.exit_code == 1
        assert "no mapping_path" in r.stdout
