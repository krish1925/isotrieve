"""Gate CLI command — retention table, bootstrap CIs, exit codes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

console = Console()


def register_gate_command(app: typer.Typer) -> None:
    """Register the ``isotrieve gate`` command on the Typer app."""

    @app.command("gate")
    def gate_cmd(
        mapping_path: Path | None = typer.Option(
            None, "--mapping", help="Path to .isotrieve file"
        ),
        mapping_external: str | None = typer.Option(
            None,
            "--mapping-external",
            help="External callable as module:callable (e.g. my_lib:my_transform)",
        ),
        source_vectors: Path | None = typer.Option(
            None, "--source-vectors", help="NPY of source embeddings (K, d_src)"
        ),
        target_vectors: Path | None = typer.Option(
            None, "--target-vectors", help="NPY of target embeddings (K, d_tgt)"
        ),
        queries: Path | None = typer.Option(
            None, "--queries", help="NPY of query embeddings in source space"
        ),
        corpus: Path | None = typer.Option(
            None, "--corpus", help="NPY of corpus embeddings in target space"
        ),
        output_format: str = typer.Option(
            "md", "--format", help="Output format: json, md, html"
        ),
        output_file: Path | None = typer.Option(
            None, "-o", "--output", help="Write report to file"
        ),
        bootstrap_resamples: int = typer.Option(
            1000, "--bootstrap-resamples", help="Number of bootstrap resamples"
        ),
        seed: int = typer.Option(0, "--seed"),
        seed_sensitivity: bool = typer.Option(
            False,
            "--seed-sensitivity",
            help="Refit the transform on subsamples and report retention stability "
            "across seeds (requires paired --source-vectors/--target-vectors)",
        ),
        seed_sensitivity_runs: int = typer.Option(
            5,
            "--seed-sensitivity-runs",
            min=2,
            help="Number of refit-and-holdout runs for --seed-sensitivity",
        ),
        seed_sensitivity_threshold: float = typer.Option(
            0.05,
            "--seed-sensitivity-threshold",
            help="WARN when seed-sensitivity std exceeds this threshold",
        ),
        seed_sensitivity_seed: int = typer.Option(
            0, "--seed-sensitivity-seed", help="Base seed for --seed-sensitivity runs"
        ),
    ) -> None:
        """Evaluate a mapping against sample data and report retention.

        Exit code 0 = PASS, 1 = WARN or FAIL.

        Use --mapping for .isotrieve files or --mapping-external for
        external callables (module:callable).
        """
        from isotrieve.quality.gate import QualityGate

        # Resolve mapping from one of three sources
        mapping = None
        if mapping_path is not None:
            from isotrieve.mapping.registry import load_mapping

            if not mapping_path.exists():
                console.print(
                    f"[red]Mapping file not found: {mapping_path}[/red]\n"
                    f"  Run [bold]isotrieve calibrate[/bold] first to create a mapping."
                )
                raise typer.Exit(1)
            try:
                mapping = load_mapping(mapping_path)
            except (ValueError, FileNotFoundError) as exc:
                console.print(f"[red]Failed to load mapping: {exc}[/red]")
                raise typer.Exit(1) from exc

        elif mapping_external is not None:
            from isotrieve.mapping.external import (
                ExternalMapping,
                load_external_callable,
            )

            try:
                fn, spec = load_external_callable(mapping_external)
            except (ValueError, ImportError, AttributeError) as exc:
                console.print(
                    f"[red]Failed to load external callable: {exc}[/red]\n"
                    f"  Expected format: module:callable (e.g. my_lib:my_transform)"
                )
                raise typer.Exit(1) from exc
            console.print(f"[dim]Loaded external callable: {spec}[/dim]")
            mapping = ExternalMapping(fn)

        else:
            if not seed_sensitivity:
                console.print(
                    "[red]Provide --mapping (for .isotrieve files) or "
                    "--mapping-external (for external callables).[/red]"
                )
                raise typer.Exit(2)

        # Resolve gate inputs
        if source_vectors is not None and target_vectors is not None:
            if not source_vectors.exists():
                console.print(f"[red]Source vectors not found: {source_vectors}[/red]")
                raise typer.Exit(1)
            if not target_vectors.exists():
                console.print(f"[red]Target vectors not found: {target_vectors}[/red]")
                raise typer.Exit(1)
            try:
                X_sample = np.load(source_vectors)
            except Exception as exc:
                console.print(
                    f"[red]Failed to load source vectors: {exc}[/red]\n"
                    f"  Ensure {source_vectors} is a valid .npy file."
                )
                raise typer.Exit(1) from exc
            try:
                Y_sample = np.load(target_vectors)
            except Exception as exc:
                console.print(
                    f"[red]Failed to load target vectors: {exc}[/red]\n"
                    f"  Ensure {target_vectors} is a valid .npy file."
                )
                raise typer.Exit(1) from exc
        elif queries is not None and corpus is not None:
            if seed_sensitivity:
                console.print(
                    "[red]--seed-sensitivity requires paired "
                    "--source-vectors/--target-vectors (queries/corpus mode "
                    "has no paired calibration vectors to refit on).[/red]"
                )
                raise typer.Exit(2)
            if not queries.exists():
                console.print(f"[red]Queries file not found: {queries}[/red]")
                raise typer.Exit(1)
            if not corpus.exists():
                console.print(f"[red]Corpus file not found: {corpus}[/red]")
                raise typer.Exit(1)
            # Queries-only mode: use query embeddings as source,
            # corpus embeddings as target
            try:
                X_sample = np.load(queries)
            except Exception as exc:
                console.print(
                    f"[red]Failed to load queries file: {exc}[/red]\n"
                    f"  Ensure {queries} is a valid .npy file."
                )
                raise typer.Exit(1) from exc
            try:
                Y_sample = np.load(corpus)
            except Exception as exc:
                console.print(
                    f"[red]Failed to load corpus file: {exc}[/red]\n"
                    f"  Ensure {corpus} is a valid .npy file."
                )
                raise typer.Exit(1) from exc
        else:
            console.print(
                "[red]Provide --source-vectors/--target-vectors OR "
                "--queries/--corpus[/red]"
            )
            raise typer.Exit(2)

        # Validate vector dimensions and emptiness
        if len(X_sample) == 0 or len(Y_sample) == 0:
            console.print("[red]Vector file is empty — need at least one vector.[/red]")
            raise typer.Exit(1)
        # d_src may be None for ExternalMapping (inferred on first transform)
        expected_src_dim = None
        if mapping is not None:
            try:
                expected_src_dim = mapping.d_src
            except RuntimeError:
                expected_src_dim = None
        if expected_src_dim is not None and X_sample.shape[1] != expected_src_dim:
            console.print(
                f"[red]Source vector dim mismatch: expected {expected_src_dim} "
                f"(source model dim), got {X_sample.shape[1]}.[/red]\n"
                f"  Vectors must be from the source embedding model."
            )
            raise typer.Exit(1)

        # Run gate (point estimate) when a mapping was provided
        gate = QualityGate()
        report = None
        if mapping is not None:
            try:
                report = gate.evaluate(mapping, X_sample, Y_sample)
            except ValueError as exc:
                msg = str(exc)
                if "NaN" in msg or "Inf" in msg:
                    console.print(
                        "[red]Vectors contain NaN or Inf values.[/red]\n"
                        "  Check your source/target vector files for corrupt data."
                    )
                elif "Dimension" in msg or "dim" in msg.lower():
                    console.print(f"[red]{msg}[/red]")
                else:
                    console.print(f"[red]Gate evaluation failed: {exc}[/red]")
                raise typer.Exit(1) from exc
            except Exception as exc:
                console.print(f"[red]Gate evaluation failed: {exc}[/red]")
                raise typer.Exit(1) from exc

        # Bootstrap confidence intervals on retention metrics
        ci: dict[str, tuple[float, float]] = {}
        if report is not None:
            try:
                ci = _bootstrap_retention_ci(
                    mapping,
                    X_sample,
                    Y_sample,
                    n_resamples=bootstrap_resamples,
                    seed=seed,
                )
            except Exception as exc:
                console.print(
                    f"[yellow]Warning: bootstrap CI failed ({exc}). "
                    f"Showing point estimates only.[/yellow]"
                )
                ci = {}

        # Seed sensitivity: refit on subsamples and measure retention stability
        ss_report = None
        if seed_sensitivity:
            try:
                ss_report = gate.seed_sensitivity(
                    X_sample,
                    Y_sample,
                    runs=seed_sensitivity_runs,
                    threshold=seed_sensitivity_threshold,
                    seed=seed_sensitivity_seed,
                )
            except ValueError as exc:
                console.print(f"[red]Seed-sensitivity failed: {exc}[/red]")
                raise typer.Exit(1) from exc

        # Format output
        if output_format == "json":
            _output_json(report, ci, ss_report, output_file)
        elif output_format == "html":
            _output_html(report, ci, ss_report, output_file)
        else:
            _output_md(report, ci, ss_report, output_file)

        # Exit code: PASS=0, WARN/FAIL=1, seed instability=1
        exit_code = 0
        if report is not None and report.verdict.value != "PASS":
            exit_code = 1
        if ss_report is not None and ss_report.unstable:
            console.print(
                f"[yellow]WARN: seed-sensitivity std "
                f"{ss_report.std_retention:.3f} >= threshold "
                f"{ss_report.threshold:.3f} — gate result may be an artifact "
                f"of one calibration split.[/yellow]"
            )
            exit_code = 1
        raise typer.Exit(exit_code)


def _bootstrap_retention_ci(
    mapping: Any,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Bootstrap CIs for retention metrics over the query set."""
    from isotrieve.quality.metrics import mrr_delta, topk_retention

    rng = np.random.default_rng(seed)
    n = len(X)
    retained_k1, retained_k5, retained_k10, mrr_vals = [], [], [], []

    mapped = mapping.transform(X)

    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        m_sub = mapped[idx]
        y_sub = Y[idx]

        retained_k1.append(topk_retention(m_sub, y_sub, k=1))
        retained_k5.append(topk_retention(m_sub, y_sub, k=5))
        retained_k10.append(topk_retention(m_sub, y_sub, k=10))

        mrr = mrr_delta(m_sub, y_sub, m_sub)
        mrr_vals.append(mrr.get("mrr_mapped", 0.0))

    def _ci(vals: list[float]) -> tuple[float, float]:
        arr = np.array(vals)
        return (float(np.percentile(arr, 10)), float(np.percentile(arr, 90)))

    return {
        "recall_at_1": _ci(retained_k1),
        "recall_at_5": _ci(retained_k5),
        "recall_at_10": _ci(retained_k10),
        "mrr": _ci(mrr_vals),
    }


def _output_json(
    report: Any,
    ci: dict,
    ss_report: Any | None,
    output_file: Path | None,
) -> None:
    if report is None:
        data: dict[str, Any] = {}
    else:
        data = report.to_dict()
    data["confidence_intervals"] = {
        k: {"lower": v[0], "upper": v[1]} for k, v in ci.items()
    }
    if ss_report is not None:
        data["seed_sensitivity"] = ss_report.to_dict()
    text = json.dumps(data, indent=2, default=str)
    if output_file:
        output_file.write_text(text, encoding="utf-8")
        console.print(f"Written to {output_file}")
    else:
        console.print_json(text)


def _output_md(
    report: Any, ci: dict, ss_report: Any | None, output_file: Path | None
) -> None:
    lines: list[str] = []
    if report is not None:
        table = Table(title=f"Gate: {report.verdict.value}")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_column("90% CI", justify="right")

        table.add_row(
            "Predicted retention",
            f"{report.predicted_retention:.3f}",
            f"[{report.prediction_interval[0]:.3f}, {report.prediction_interval[1]:.3f}]",
        )

        for metric, key in [
            ("Recall@1", "recall_at_1"),
            ("Recall@5", "recall_at_5"),
            ("Recall@10", "recall_at_10"),
            ("MRR", "mrr"),
        ]:
            if key in ci:
                lower, upper = ci[key]
                mid = (lower + upper) / 2
                table.add_row(metric, f"{mid:.3f}", f"[{lower:.3f}, {upper:.3f}]")

        table.add_row("Verdict", f"[bold]{report.verdict.value}[/bold]", "")

        lines.append(_table_to_text(table))

    if ss_report is not None:
        ss_table = Table(title=f"Seed sensitivity ({ss_report.runs} refit runs)")
        ss_table.add_column("Metric")
        ss_table.add_column("Value", justify="right")
        ss_table.add_row(
            "Retention mean ± std",
            f"{ss_report.mean_retention:.3f} ± {ss_report.std_retention:.3f}",
        )
        ss_table.add_row(
            "Range",
            f"[{ss_report.min_retention:.3f}, {ss_report.max_retention:.3f}]",
        )
        ss_table.add_row(
            "Per-seed", ", ".join(f"{v:.3f}" for v in ss_report.per_seed_retention)
        )
        ss_table.add_row(
            "Stability",
            "[red]UNSTABLE[/red]" if ss_report.unstable else "[green]STABLE[/green]",
        )
        lines.append(_table_to_text(ss_table))

    text_content = "\n".join(lines).strip()
    if output_file:
        output_file.write_text(text_content, encoding="utf-8")
        console.print(f"Written to {output_file}")
    else:
        for line in lines:
            console.print(line)


def _table_to_text(table: Table) -> str:
    """Convert a Rich table to plain text."""
    from io import StringIO

    buf = StringIO()
    tmp_console = Console(file=buf, force_terminal=False)
    tmp_console.print(table)
    return buf.getvalue()


def _output_html(
    report: Any, ci: dict, ss_report: Any | None, output_file: Path | None
) -> None:
    from isotrieve.reporting.html_report import generate_gate_html

    html = generate_gate_html(report, ci, ss_report)
    if output_file:
        output_file.write_text(html, encoding="utf-8")
        console.print(f"Written to {output_file}")
    else:
        console.print(html)
