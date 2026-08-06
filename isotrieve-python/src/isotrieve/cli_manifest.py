"""Manifest lifecycle CLI — ``manifest list|show``, ``rollback``, ``resume``.

Reversibility and resumability are driven entirely by the ``MigrationManifest``
record written by :func:`isotrieve.migrate.migrate_store`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from isotrieve.mapping.base import Mapping
from isotrieve.migrate import MigrationManifest, migrate_store
from isotrieve.stores.base import VectorRecord, VectorStore

console = Console()

DEFAULT_MANIFEST_DIR = "isotrieve-manifests"


def register_manifest_command(app: typer.Typer) -> None:
    """Register the ``manifest`` group, ``rollback`` and ``resume`` commands."""
    _register_manifest_group(app)
    _register_rollback(app)
    _register_resume(app)


def _print_json(data: object) -> None:
    console.print_json(json.dumps(data, default=str))


def _manifest_directory(manifest_dir: Path | None) -> Path:
    return manifest_dir or Path.cwd() / DEFAULT_MANIFEST_DIR


def _load_manifest_data(
    manifest_id: str, manifest_dir: Path
) -> tuple[dict[str, Any], Path]:
    """Resolve a manifest id to (data, file path).

    Accepts either a manifest id (looked up as ``<id>.json`` in the manifest
    directory) or a direct path to a manifest JSON file.
    """
    direct = Path(manifest_id)
    if direct.is_file():
        try:
            return json.loads(direct.read_text(encoding="utf-8")), direct
        except (json.JSONDecodeError, OSError) as exc:
            console.print(f"[red]Failed to read manifest file {direct}: {exc}[/red]")
            raise typer.Exit(1) from exc

    manifest_file = manifest_dir / f"{manifest_id}.json"
    if not manifest_file.exists() and manifest_dir.exists():
        for candidate in manifest_dir.glob("*.json"):
            try:
                if (
                    json.loads(candidate.read_text(encoding="utf-8")).get("id")
                    == manifest_id
                ):
                    manifest_file = candidate
                    break
            except (json.JSONDecodeError, OSError):
                continue
    if not manifest_file.exists():
        console.print(f"[red]Manifest not found: {manifest_id}[/red]")
        console.print(
            f"  Looked for {manifest_file}. "
            "Run [bold]isotrieve manifest list[/bold] to see known manifests."
        )
        raise typer.Exit(1)
    try:
        return json.loads(manifest_file.read_text(encoding="utf-8")), manifest_file
    except (json.JSONDecodeError, OSError) as exc:
        console.print(f"[red]Failed to read manifest {manifest_file}: {exc}[/red]")
        raise typer.Exit(1) from exc


def _load_mapping(mapping_path: str) -> Mapping:
    """Load the ``.isotrieve`` mapping recorded in a manifest."""
    if not mapping_path:
        console.print(
            "[red]Manifest has no mapping_path — cannot load the transform.[/red]"
        )
        console.print(
            "  Re-record the manifest with the mapping file path to enable "
            "rollback/resume."
        )
        raise typer.Exit(1)
    path = Path(mapping_path)
    if not path.exists():
        console.print(f"[red]Mapping file not found: {path}[/red]")
        raise typer.Exit(1)
    try:
        from isotrieve.mapping.registry import load_mapping

        return load_mapping(path)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Failed to load mapping: {exc}[/red]")
        raise typer.Exit(1) from exc


def _reopen_store(spec: dict[str, str], *, create: bool, label: str) -> VectorStore:
    """Reopen a file-backed store from a manifest store spec."""
    stype = spec.get("type", "")
    uri = spec.get("uri", "")
    if stype == "numpy" and uri:
        from isotrieve.stores.numpy_files import NumpyFileStore

        return NumpyFileStore(uri, create=create)
    console.print(
        f"[red]Cannot reopen {label} store from manifest "
        f"(type={stype!r}, uri={uri!r}).[/red]"
    )
    console.print(
        "  The manifest does not record a file-backed store. "
        "Pass --source-dir/--target-dir explicitly."
    )
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# isotrieve manifest list|show
# ---------------------------------------------------------------------------


def _register_manifest_group(app: typer.Typer) -> None:
    manifest_app = typer.Typer(
        help="Inspect migration manifests (list / show).", no_args_is_help=True
    )
    app.add_typer(manifest_app, name="manifest")

    @manifest_app.command("list")
    def manifest_list_cmd(
        manifest_dir: Annotated[
            Path | None, typer.Option("--dir", help="Manifest directory")
        ] = None,
        as_json: Annotated[bool, typer.Option("--json", help="Emit JSON")] = False,
    ) -> None:
        """List all migration manifests in the manifest directory."""
        directory = _manifest_directory(manifest_dir)
        entries: list[dict[str, Any]] = []
        if directory.exists():
            for manifest_file in sorted(directory.glob("*.json")):
                try:
                    data = json.loads(manifest_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                entries.append(
                    {
                        "id": data.get("id", manifest_file.stem),
                        "source_collection": data.get("source_collection", ""),
                        "target_collection": data.get("target_collection", ""),
                        "source_model": data.get("source_model", ""),
                        "target_model": data.get("target_model", ""),
                        "migrated_vectors": data.get("migrated_vectors", 0),
                        "total_vectors": data.get("total_vectors", 0),
                        "completed_at": data.get("completed_at", ""),
                    }
                )

        if as_json:
            _print_json({"directory": str(directory), "manifests": entries})
            return

        if not entries:
            console.print(f"[yellow]No manifests found in {directory}[/yellow]")
            return

        table = Table(title=f"Migration manifests in {directory}")
        table.add_column("id", overflow="fold")
        table.add_column("migration")
        table.add_column("progress", justify="right")
        table.add_column("status")
        for entry in entries:
            status = (
                "[green]completed[/green]"
                if entry["completed_at"]
                else "[yellow]interrupted[/yellow]"
            )
            table.add_row(
                entry["id"],
                f"{entry['source_collection']} → {entry['target_collection']}",
                f"{entry['migrated_vectors']}/{entry['total_vectors']}",
                status,
            )
        console.print(table)

    @manifest_app.command("show")
    def manifest_show_cmd(
        manifest_id: Annotated[str, typer.Argument(help="Manifest id to show")],
        manifest_dir: Annotated[
            Path | None, typer.Option("--dir", help="Manifest directory")
        ] = None,
        as_json: Annotated[bool, typer.Option("--json", help="Emit JSON")] = False,
    ) -> None:
        """Show one migration manifest in detail."""
        directory = _manifest_directory(manifest_dir)
        data, manifest_file = _load_manifest_data(manifest_id, directory)

        if as_json:
            _print_json(data)
            return

        try:
            manifest = MigrationManifest.from_dict(data)
        except TypeError:
            console.print(
                f"[yellow]Manifest {manifest_file.name} is missing required "
                "fields — showing raw record.[/yellow]"
            )
            manifest = None
        if manifest is None:
            table = Table(title=f"Manifest {manifest_file.stem} (raw)")
            table.add_column("Field")
            table.add_column("Value")
            for key in sorted(data):
                table.add_row(key, str(data[key]))
            console.print(table)
            return
        status = (
            "[green]completed[/green]"
            if manifest.completed_at
            else "[yellow]interrupted[/yellow]"
        )
        table = Table(title=f"Manifest {manifest.id}")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("File", str(manifest_file))
        table.add_row("Status", status)
        table.add_row(
            "Migration", f"{manifest.source_collection} → {manifest.target_collection}"
        )
        if manifest.source_model or manifest.target_model:
            table.add_row(
                "Models", f"{manifest.source_model} → {manifest.target_model}"
            )
        table.add_row(
            "Progress",
            f"{manifest.migrated_vectors}/{manifest.total_vectors} "
            f"(batch_end={manifest.batch_end})",
        )
        table.add_row("Started", manifest.started_at)
        table.add_row("Completed", manifest.completed_at or "—")
        table.add_row("Last batch hash", manifest.last_batch_hash or "—")
        table.add_row("Mapping", manifest.mapping_path or "—")
        table.add_row(
            "Transform invertible",
            "[green]yes[/green]" if manifest.transform_invertible else "[red]no[/red]",
        )
        table.add_row("Rollback strategy", manifest.rollback_strategy)
        table.add_row("Batches", str(len(manifest.batches)))
        console.print(table)

        gate = manifest.gate_result or {}
        if gate:
            console.print(
                f"\nGate (migration time): verdict={gate.get('verdict', '—')}, "
                f"predicted_retention={gate.get('predicted_retention', '—')}"
            )


# ---------------------------------------------------------------------------
# isotrieve rollback
# ---------------------------------------------------------------------------


def _register_rollback(app: typer.Typer) -> None:
    @app.command("rollback")
    def rollback_cmd(
        manifest_id: Annotated[
            str, typer.Option("--manifest", help="Manifest id to roll back")
        ],
        manifest_dir: Annotated[
            Path | None, typer.Option("--dir", help="Manifest directory")
        ] = None,
        output_dir: Annotated[
            Path | None,
            typer.Option(
                "--output-dir",
                help="Where recovered vectors are written "
                "(default: ./isotrieve-rollback-<id>/).",
            ),
        ] = None,
        dry_run: Annotated[
            bool, typer.Option("--dry-run", help="Print the plan without writing")
        ] = False,
        as_json: Annotated[bool, typer.Option("--json", help="Emit JSON")] = False,
    ) -> None:
        """Reverse a migration using its manifest.

        Rollback is refused for non-invertible transforms unless a shadow copy
        of the source was preserved (rollback_strategy="shadow"). For invertible
        mappings, target vectors are inverse-transformed back to source space;
        for shadow migrations the preserved source store is restored. Output is
        always written to a new directory — the live stores are never mutated.
        """
        directory = _manifest_directory(manifest_dir)
        data, _ = _load_manifest_data(manifest_id, directory)
        manifest = MigrationManifest.from_dict(data)

        invertible = manifest.transform_invertible
        strategy = manifest.rollback_strategy

        if not invertible and strategy != "shadow":
            console.print(
                "[red]Rollback refused: this migration cannot be rolled back.[/red]"
            )
            console.print(
                f"  transform_invertible={invertible}, rollback_strategy={strategy!r}."
            )
            console.print(
                "  The mapping has no analytic inverse and no shadow copy of "
                "the source was preserved."
            )
            console.print(
                "  Re-run the migration with an invertible mapping or with a "
                "shadow-copy strategy (e.g. Pinecone shadow namespace)."
            )
            raise typer.Exit(2)

        mapping: Mapping | None = None
        if invertible:
            mapping = _load_mapping(manifest.mapping_path)

        output = output_dir or Path.cwd() / f"isotrieve-rollback-{manifest.id}"
        plan = {
            "manifest_id": manifest.id,
            "transform_invertible": invertible,
            "rollback_strategy": strategy,
            "action": (
                "inverse-transform target store → recovery store"
                if invertible
                else "restore preserved source (shadow) store → recovery store"
            ),
            "output_dir": str(output),
        }

        if dry_run:
            if as_json:
                _print_json(plan)
                return
            console.print("[bold]Rollback plan (dry run — nothing written)[/bold]")
            table = Table()
            table.add_column("Field")
            table.add_column("Value")
            for key, value in plan.items():
                table.add_row(key, str(value))
            console.print(table)
            return

        written = _execute_rollback(
            manifest,
            strategy=strategy,
            invertible=invertible,
            mapping=mapping,
            output_dir=output,
        )

        if as_json:
            plan["vectors_written"] = written
            plan["completed"] = True
            _print_json(plan)
            return
        console.print(
            f"[green]Rollback complete: {written:,} vectors → {output}[/green]"
        )
        console.print(
            "Point your application back at the recovery store to finish the rollback."
        )


def _execute_rollback(
    manifest: MigrationManifest,
    *,
    strategy: str,
    invertible: bool,
    mapping: Mapping | None,
    output_dir: Path,
) -> int:
    """Write source-space embeddings to ``output_dir`` for recovery."""
    from isotrieve.stores.numpy_files import NumpyFileStore

    recovery = NumpyFileStore(output_dir, create=True)

    if invertible:
        assert mapping is not None
        target = _reopen_store(manifest.target_store, create=False, label="target")
        records: list[VectorRecord] = []
        for batch in target.iter_vectors(batch_size=1024):
            vectors = np.stack([record.vector for record in batch], axis=0)
            recovered = mapping.inverse_transform(vectors)
            records.extend(
                VectorRecord(
                    id=record.id,
                    vector=recovered[i],
                    text=record.text,
                    payload=record.payload,
                )
                for i, record in enumerate(batch)
            )
        return recovery.write_vectors(records, batch_size=1024)

    # shadow strategy: the source store is the preserved original
    assert strategy == "shadow"
    source = _reopen_store(manifest.source_store, create=False, label="source")
    return recovery.write_vectors(source.iter_vectors(), batch_size=1024)


# ---------------------------------------------------------------------------
# isotrieve resume
# ---------------------------------------------------------------------------


def _register_resume(app: typer.Typer) -> None:
    @app.command("resume")
    def resume_cmd(
        manifest_id: Annotated[
            str, typer.Option("--manifest", help="Manifest id to resume")
        ],
        manifest_dir: Annotated[
            Path | None, typer.Option("--dir", help="Manifest directory")
        ] = None,
        source_dir: Annotated[
            Path | None,
            typer.Option(
                "--source-dir",
                help="Override the source store (numpy dir) recorded in the manifest",
            ),
        ] = None,
        target_dir: Annotated[
            Path | None,
            typer.Option(
                "--target-dir",
                help="Override the target store (numpy dir) recorded in the manifest",
            ),
        ] = None,
        batch_size: Annotated[
            int, typer.Option("--batch-size", help="Batch size for streaming")
        ] = 1024,
        as_json: Annotated[bool, typer.Option("--json", help="Emit JSON")] = False,
    ) -> None:
        """Resume an interrupted migration from its manifest checkpoint."""
        directory = _manifest_directory(manifest_dir)
        data, manifest_file = _load_manifest_data(manifest_id, directory)
        manifest = MigrationManifest.from_dict(data)

        if manifest.completed_at:
            console.print(
                f"[yellow]Manifest {manifest.id} already completed at "
                f"{manifest.completed_at} — nothing to resume.[/yellow]"
            )
            return

        mapping = _load_mapping(manifest.mapping_path)

        source_spec = (
            {"type": "numpy", "uri": str(source_dir)}
            if source_dir is not None
            else manifest.source_store
        )
        target_spec = (
            {"type": "numpy", "uri": str(target_dir)}
            if target_dir is not None
            else manifest.target_store
        )
        source = _reopen_store(source_spec, create=False, label="source")
        target = _reopen_store(target_spec, create=True, label="target")

        result = migrate_store(
            source,
            target,
            mapping,
            batch_size=batch_size,
            manifest_path=manifest_file,
            resume=True,
            mapping_path=Path(manifest.mapping_path) if manifest.mapping_path else None,
            rollback_strategy=manifest.rollback_strategy,
            gate_result=manifest.gate_result,
        )

        if as_json:
            _print_json(result.to_dict())
            return
        console.print(
            f"[green]Resumed {manifest.id}: migrated "
            f"{result.migrated_vectors}/{result.total_vectors} vectors "
            f"(batch_end={result.batch_end}).[/green]"
        )
