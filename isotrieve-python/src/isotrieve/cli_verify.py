"""Verify CLI command — post-migration drift & gate revalidation.

Re-runs retrieval retention against the live (migrated) store and compares it
to the migration-time gate result recorded in the manifest, warning on drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from isotrieve.cli_manifest import _load_manifest_data, _manifest_directory

console = Console()


def register_verify_command(app: typer.Typer) -> None:
    """Register the ``isotrieve verify`` command on the Typer app."""

    @app.command("verify")
    def verify_cmd(
        manifest_id: Annotated[
            str,
            typer.Option(
                "--manifest",
                help="Manifest id (or direct path to a manifest JSON file)",
            ),
        ],
        manifest_dir: Annotated[
            Path | None, typer.Option("--dir", help="Manifest directory")
        ] = None,
        queries: Annotated[
            Path | None,
            typer.Option(
                "--queries",
                help="NPY of live query embeddings in source space (K, d_src)",
            ),
        ] = None,
        corpus: Annotated[
            Path | None,
            typer.Option(
                "--corpus",
                help="NPY of live corpus embeddings in target space (K, d_tgt)",
            ),
        ] = None,
        live_store: Annotated[
            Path | None,
            typer.Option(
                "--live-store",
                help="Numpy dir of the migrated (target) store to re-read the "
                "corpus from, instead of --corpus",
            ),
        ] = None,
        k: Annotated[int, typer.Option("--k", help="Top-k for live retention")] = 10,
        drift_threshold: Annotated[
            float,
            typer.Option(
                "--drift-threshold",
                help="WARN when live retention drops this much below the "
                "migration-time gate value (absolute) [default: 0.10]",
            ),
        ] = 0.10,
        warn_threshold: Annotated[
            float,
            typer.Option(
                "--warn-threshold",
                help="WARN when live retention falls below this absolute value "
                "[default: 0.55]",
            ),
        ] = 0.55,
        seed: Annotated[int, typer.Option("--seed", help="RNG seed")] = 0,
        schedule: Annotated[
            str | None,
            typer.Option(
                "--schedule",
                help="Cron/CI schedule example to run verify automatically. "
                "e.g. --schedule '0 2 * * *' (daily 02:00). "
                "Pair with a CI step: `cron: '0 2 * * *'` then "
                "`isotrieve verify --manifest <id> --queries q.npy --corpus c.npy`",
            ),
        ] = None,
        as_json: Annotated[bool, typer.Option("--json", help="Emit JSON")] = False,
    ) -> None:
        """Revalidate post-migration retention against the live store.

        Re-runs the quality gate on live query/corpus embeddings and compares
        against the migration-time gate result stored in the manifest. Emits a
        drift WARNING when live retention drops by --drift-threshold or more
        relative to the migration baseline, or falls below --warn-threshold.

        Exit codes: 0 = no drift (PASS), 1 = drift warning, 2 = usage error.

        Example (cron / CI):

            # every day at 02:00, check the migrated store for drift
            # cron: '0 2 * * *'
            isotrieve verify --manifest mig_20260805T020000_ab12cd \
                --queries live_queries.npy --corpus live_corpus.npy
        """
        from isotrieve.quality.gate import QualityGate

        directory = _manifest_directory(manifest_dir)
        data, _ = _load_manifest_data(manifest_id, directory)

        mapping_path = data.get("mapping_path") or ""
        if not mapping_path or not Path(mapping_path).exists():
            console.print(
                f"[red]Manifest has no usable mapping_path: {mapping_path!r}[/red]"
            )
            console.print(
                "  verify re-runs the gate through the mapping; re-record the "
                "manifest with the .isotrieve file path."
            )
            raise typer.Exit(1)
        mapping = _load_mapping(mapping_path)

        if queries is None:
            console.print(
                "[red]verify requires --queries (live source-space query "
                "embeddings as NPY).[/red]"
            )
            raise typer.Exit(2)
        if (corpus is None) == (live_store is None):
            console.print(
                "[red]verify requires exactly one of --corpus (NPY) or "
                "--live-store (numpy dir of the migrated store).[/red]"
            )
            raise typer.Exit(2)

        X_live = _load_npy(queries, "queries")
        if live_store is not None:
            Y_live = _read_store_corpus(live_store)
        else:
            assert corpus is not None
            Y_live = _load_npy(corpus, "corpus")

        if len(X_live) == 0 or len(Y_live) == 0:
            console.print(
                "[red]Query/corpus data is empty — need at least one vector.[/red]"
            )
            raise typer.Exit(1)
        if X_live.shape[1] != mapping.d_src:
            console.print(
                f"[red]Query dim mismatch: mapping expects {mapping.d_src}, "
                f"got {X_live.shape[1]}.[/red]"
            )
            raise typer.Exit(1)

        if schedule:
            console.print(
                f"[dim]Scheduled ({schedule}): verify will run on this "
                f"cron/CI trigger.[/dim]"
            )

        gate = QualityGate()
        live_report = gate.evaluate(mapping, X_live, Y_live)
        mapped = mapping.transform(X_live)

        from isotrieve.quality.metrics import topk_retention

        migration_gate = data.get("gate_result") or {}
        baseline = migration_gate.get("predicted_retention")

        live_retention = float(live_report.predicted_retention)
        retention_drop: float | None = None
        if baseline is not None:
            retention_drop = float(baseline) - live_retention

        drifted, reason = _evaluate_drift(
            baseline=baseline,
            live_retention=live_retention,
            drift_threshold=drift_threshold,
            warn_threshold=warn_threshold,
        )

        report: dict[str, Any] = {
            "manifest_id": data.get("id", Path(manifest_id).stem),
            "mapping_path": mapping_path,
            "migration_time": {
                "predicted_retention": baseline,
                "gate_verdict": migration_gate.get("verdict"),
            },
            "live": {
                "predicted_retention": live_retention,
                "top1_retention": float(live_report.top1_retention),
                "top10_retention": float(live_report.top10_retention),
                f"top{k}_retention": float(
                    topk_retention(mapped, Y_live, k=min(k, len(Y_live)))
                ),
                "gate_verdict": live_report.verdict.value,
            },
            "retention_drop": retention_drop,
            "drift_threshold": drift_threshold,
            "warn_threshold": warn_threshold,
            "drifted": drifted,
            "reason": reason,
            "verdict": "DRIFT" if drifted else "PASS",
        }

        if as_json:
            _print_json(report)
        else:
            _print_table(report)

        raise typer.Exit(1 if drifted else 0)


def _load_mapping(mapping_path: str) -> Any:
    from isotrieve.mapping.registry import load_mapping

    try:
        return load_mapping(mapping_path)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Failed to load mapping: {exc}[/red]")
        raise typer.Exit(1) from exc


def _load_npy(path: Path, label: str) -> np.ndarray:
    if not path.exists():
        console.print(f"[red]{label}: file not found: {path}[/red]")
        raise typer.Exit(1)
    try:
        return np.asarray(np.load(path))
    except Exception as exc:
        console.print(f"[red]{label}: failed to load {path}: {exc}[/red]")
        raise typer.Exit(1) from exc


def _read_store_corpus(path: Path) -> np.ndarray:
    """Read all vectors from a NumpyFileStore directory as a corpus array."""
    from isotrieve.stores.numpy_files import NumpyFileStore

    try:
        store = NumpyFileStore(path)
    except FileNotFoundError as exc:
        console.print(f"[red]--live-store invalid: {exc}[/red]")
        raise typer.Exit(1) from exc
    batches = list(store.iter_vectors(batch_size=1024))
    if not batches:
        console.print(f"[red]--live-store is empty: {path}[/red]")
        raise typer.Exit(1)
    return np.stack([record.vector for batch in batches for record in batch])


def _evaluate_drift(
    *,
    baseline: float | None,
    live_retention: float,
    drift_threshold: float,
    warn_threshold: float,
) -> tuple[bool, str]:
    """Decide whether live retention has drifted from the migration baseline.

    Drift when the absolute drop from the migration-time baseline reaches
    ``drift_threshold``, or when live retention falls below ``warn_threshold``.
    """
    if baseline is not None:
        drop = baseline - live_retention
        if drop >= drift_threshold:
            return (
                True,
                f"live retention {live_retention:.3f} is {drop:.3f} below the "
                f"migration-time value {baseline:.3f} (threshold {drift_threshold})",
            )
    if live_retention < warn_threshold:
        return (
            True,
            f"live retention {live_retention:.3f} is below the warn threshold "
            f"{warn_threshold}",
        )
    if baseline is None:
        return False, "no migration-time gate result recorded; absolute value OK"
    return (
        False,
        f"live retention {live_retention:.3f} within {drift_threshold:.2f} of "
        f"migration-time value {baseline:.3f}",
    )


def _print_json(data: object) -> None:
    console.print_json(json.dumps(data, default=str))


def _print_table(report: dict[str, Any]) -> None:
    verdict_color = "red" if report["drifted"] else "green"
    console.print(
        f"[bold][{verdict_color}]{report['verdict']}[/{verdict_color}][/bold] "
        f"manifest {report['manifest_id']}"
    )

    table = Table(title="Post-migration drift revalidation")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row(
        "Migration-time retention",
        _fmt(report["migration_time"]["predicted_retention"]),
    )
    table.add_row("Live retention", f"{report['live']['predicted_retention']:.4f}")
    table.add_row("Live top-1 retention", f"{report['live']['top1_retention']:.4f}")
    table.add_row("Live top-10 retention", f"{report['live']['top10_retention']:.4f}")
    table.add_row("Retention drop", _fmt(report["retention_drop"]))
    table.add_row("Drift threshold", f"{report['drift_threshold']:.3f}")
    table.add_row("Warn threshold", f"{report['warn_threshold']:.3f}")
    table.add_row("Live gate verdict", report["live"]["gate_verdict"])
    console.print(table)

    if report["drifted"]:
        console.print(f"[yellow]Drift warning: {report['reason']}[/yellow]")
    else:
        console.print(f"[dim]{report['reason']}[/dim]")


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)
