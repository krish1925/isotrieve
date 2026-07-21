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
    """Register the ``aecp gate`` command on the Typer app."""

    @app.command("gate")
    def gate_cmd(
        mapping_path: Path = typer.Option(..., "--mapping", help="Path to .aecp file"),
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
    ) -> None:
        """Evaluate a mapping against sample data and report retention.

        Exit code 0 = PASS, 1 = WARN or FAIL.
        """
        from aecp.mapping.registry import load_mapping
        from aecp.quality.gate import QualityGate

        mapping = load_mapping(mapping_path)

        # Resolve gate inputs
        if source_vectors is not None and target_vectors is not None:
            X_sample = np.load(source_vectors)
            Y_sample = np.load(target_vectors)
        elif queries is not None and corpus is not None:
            # Queries-only mode: use query embeddings as source,
            # corpus embeddings as target
            X_sample = np.load(queries)
            Y_sample = np.load(corpus)
        else:
            console.print(
                "[red]Provide --source-vectors/--target-vectors OR "
                "--queries/--corpus[/red]"
            )
            raise typer.Exit(2)

        # Run gate
        gate = QualityGate()
        report = gate.evaluate(mapping, X_sample, Y_sample)

        # Bootstrap confidence intervals on retention metrics
        ci = _bootstrap_retention_ci(
            mapping,
            X_sample,
            Y_sample,
            n_resamples=bootstrap_resamples,
            seed=seed,
        )

        # Format output
        if output_format == "json":
            _output_json(report, ci, output_file)
        elif output_format == "html":
            _output_html(report, ci, output_file)
        else:
            _output_md(report, ci, output_file)

        # Exit code
        if report.verdict.value == "PASS":
            raise typer.Exit(0)
        raise typer.Exit(1)


def _bootstrap_retention_ci(
    mapping: Any,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Bootstrap CIs for retention metrics over the query set."""
    from aecp.quality.metrics import mrr_delta, topk_retention

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


def _output_json(report: Any, ci: dict, output_file: Path | None) -> None:
    data = report.to_dict()
    data["confidence_intervals"] = {
        k: {"lower": v[0], "upper": v[1]} for k, v in ci.items()
    }
    text = json.dumps(data, indent=2, default=str)
    if output_file:
        output_file.write_text(text, encoding="utf-8")
        console.print(f"Written to {output_file}")
    else:
        console.print_json(text)


def _output_md(report: Any, ci: dict, output_file: Path | None) -> None:
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

    text_content = _table_to_text(table)
    if output_file:
        output_file.write_text(text_content, encoding="utf-8")
        console.print(f"Written to {output_file}")
    else:
        console.print(table)


def _table_to_text(table: Table) -> str:
    """Convert a Rich table to plain text."""
    from io import StringIO

    buf = StringIO()
    tmp_console = Console(file=buf, force_terminal=False)
    tmp_console.print(table)
    return buf.getvalue()


def _output_html(report: Any, ci: dict, output_file: Path | None) -> None:
    from aecp.reporting.html_report import generate_gate_html

    html = generate_gate_html(report, ci)
    if output_file:
        output_file.write_text(html, encoding="utf-8")
        console.print(f"Written to {output_file}")
    else:
        console.print(html)
