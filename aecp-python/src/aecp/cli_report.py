"""Report CLI command — summarize wrapper usage and graduate to migration."""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()


def register_report_command(app: typer.Typer) -> None:
    """Register the ``aecp report`` command on the Typer app."""

    @app.command("report")
    def report_cmd() -> None:
        """Summarize wrapper usage and print graduation commands.

        Reads local telemetry (AECP_TELEMETRY=local) and prints
        the exact ``aecp gate`` / ``aecp migrate`` commands to
        graduate from query-time wrapper to full migration.
        """
        from aecp.wrappers.telemetry import summary

        s = summary()

        if s["total_queries"] == 0:
            console.print(
                "[yellow]No wrapper telemetry found.[/yellow]\n"
                "Enable with: export AECP_TELEMETRY=local\n"
                "Then use a wrapper (LlamaIndex, OpenAI shim, or LangChain) "
                "to log queries."
            )
            return

        console.print("[bold]Wrapper usage summary[/bold]")
        console.print(f"Total queries: {s['total_queries']}")
        for wrapper, count in s["per_wrapper"].items():
            console.print(f"  {wrapper}: {count}")
        console.print(f"\nTelemetry path: {s['telemetry_path']}")

        console.print("\n[bold]Graduation steps[/bold]")
        console.print(
            "1. Run the quality gate on a sample:"
        )
        console.print(
            "   aecp gate --mapping mapping.aecp "
            "--source-vectors X.npy --target-vectors Y.npy"
        )
        console.print("2. If PASS, migrate the corpus:")
        console.print(
            "   aecp transform --mapping mapping.aecp "
            "--source-dir ./old --target-dir ./new"
        )
        console.print("3. Verify post-migration retention with another gate run.")
