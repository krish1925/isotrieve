import typer
import time
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from sentence_transformers import SentenceTransformer

app = typer.Typer(help="AECP Zero-Friction Demo CLI")
console = Console()

@app.command()
def basic():
    """Run a basic semantic transfer demo comparing Text vs Vector handoff."""
    console.print("\n[bold blue]🚀 AECP Zero-Friction Demo: Basic Transfer[/bold blue]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task1 = progress.add_task(description="Initializing local models (MiniLM & MPNet)...", total=None)
        
        # Simulate Agent A (MiniLM)
        # Note: In a real app we'd load these once globally, but for demo we load here
        model_a = SentenceTransformer('all-MiniLM-L6-v2')
        model_b = SentenceTransformer('all-mpnet-base-v2')
        
        progress.update(task1, completed=100)
    
    console.print("[green]✔ Agents initialized locally.[/green]")

    query = "How do I optimize database queries?"
    console.print(f"\n[yellow]📝 Query: \"{query}\"[/yellow]")

    # 1. Text Handoff Simulation
    start_text = time.time()
    # In text handoff, B re-encodes the text
    _ = model_b.encode(query)
    end_text = time.time()
    text_latency_ms = (end_text - start_text) * 1000
    console.print(f"[red]✖ Text Handoff: Took {text_latency_ms:.2f}ms (Re-encoding cost)[/red]")

    # 2. AECP Vector Handoff Simulation
    start_aecp = time.time()
    
    # Agent A encodes once (already done in source)
    vec_a = model_a.encode(query)
    
    # Simulate Matrix Mult (O(1))
    # In real AECP: vec_b = vec_a @ W
    # We simulate the cost of a 384x768 matrix multiplication
    W_dummy = np.random.rand(384, 768).astype(np.float32)
    _ = vec_a @ W_dummy
    
    end_aecp = time.time()
    aecp_latency_ms = (end_aecp - start_aecp) * 1000
    console.print(f"[green]✔ AECP Handoff: Took {aecp_latency_ms:.2f}ms (Matrix Mult)[/green]")

    # 3. Results Table
    table = Table(title="Performance Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Text Handoff", style="red")
    table.add_column("AECP Vector Handoff", style="green")

    table.add_row("Latency", f"{text_latency_ms:.2f}ms", f"{aecp_latency_ms:.2f}ms")
    table.add_row("Privacy", "Text Exposed", "Vectors Only")
    table.add_row("Cost", "$$$ (Re-encode)", "FREE")

    console.print("\n", table)
    console.print("\n[dim]Note: First run downloads models from HuggingFace. Subsequent runs are instant.[/dim]\n")


def main():
    app()

if __name__ == "__main__":
    main()
