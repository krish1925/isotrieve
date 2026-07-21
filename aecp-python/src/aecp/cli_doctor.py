"""Doctor CLI command — inspect a target store and suggest next steps."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def register_doctor_command(app: typer.Typer) -> None:
    """Register the ``aecp doctor`` command on the Typer app."""

    @app.command("doctor")
    def doctor_cmd(
        store_type: str = typer.Option(
            ..., "--store", help="Store type: chroma, qdrant, numpy"
        ),
        store_url: str | None = typer.Option(
            None, "--url", help="Store connection URL"
        ),
        collection: str | None = typer.Option(
            None, "--collection", help="Collection/namespace name"
        ),
        source_model: str | None = typer.Option(
            None, "--source-model", help="Expected source model ID"
        ),
        as_json: bool = typer.Option(False, "--json"),
    ) -> None:
        """Inspect a vector store and suggest migration steps.

        Read-only: no writes, no mutations.
        """
        info = _inspect_store(store_type, store_url, collection, source_model)

        if as_json:
            console.print_json(
                __import__("json").dumps(info, indent=2, default=str)
            )
            return

        table = Table(title=f"Doctor: {store_type} store")
        table.add_column("Property")
        table.add_column("Value")
        for k, v in info.items():
            table.add_row(k, str(v))
        console.print(table)

        # Suggest playbook
        playbook = _suggest_playbook(info)
        if playbook:
            console.print(f"\n[bold]Suggested playbook:[/bold] {playbook}")
        else:
            console.print(
                "\n[yellow]No matching playbook found. "
                "Run 'aecp gate' with calibration data to evaluate migration.[/yellow]"
            )

        # Print scenario-calibrated expectations
        console.print(SCENARIO_GUIDANCE)


def _inspect_store(
    store_type: str,
    url: str | None,
    collection: str | None,
    source_model: str | None,
) -> dict:
    """Inspect store and return metadata dict."""
    info: dict = {
        "store_type": store_type,
        "url": url or "default",
        "collection": collection or "default",
        "vector_count": None,
        "dimension": None,
        "inferred_model": source_model or "unknown",
        "has_aecp_metadata": False,
    }

    if store_type == "chroma":
        info.update(_inspect_chroma(url, collection))
    elif store_type == "qdrant":
        info.update(_inspect_qdrant(url, collection))
    elif store_type == "numpy":
        info.update(_inspect_numpy(url))
    else:
        console.print(f"[yellow]Unknown store type: {store_type}[/yellow]")

    return info


def _inspect_chroma(url: str | None, collection: str | None) -> dict:
    """Inspect a ChromaDB collection."""
    try:
        import chromadb  # type: ignore[import-untyped]

        client = chromadb.Client() if not url else chromadb.HttpClient(host=url)
        col = client.get_collection(collection or "default")
        count = col.count()
        sample = col.get(limit=1, include=["embeddings", "metadatas"])
        dim = len(sample["embeddings"][0]) if sample.get("embeddings") else None
        has_aecp = False
        if sample.get("metadatas") and sample["metadatas"]:
            has_aecp = "aecp_mapping_id" in sample["metadatas"][0]
        return {"vector_count": count, "dimension": dim, "has_aecp_metadata": has_aecp}
    except Exception as e:
        return {"vector_count": f"error: {e}", "dimension": None}


def _inspect_qdrant(url: str | None, collection: str | None) -> dict:
    """Inspect a Qdrant collection."""
    try:
        from qdrant_client import QdrantClient  # type: ignore[import-untyped]

        client = QdrantClient(url=url or "http://localhost:6333")
        info = client.get_collection(collection or "default")
        return {
            "vector_count": info.points_count,
            "dimension": (
                info.config.params.vectors.size
                if info.config.params.vectors
                else None
            ),
        }
    except Exception as e:
        return {"vector_count": f"error: {e}", "dimension": None}


def _inspect_numpy(path: str | None) -> dict:
    """Inspect a NumpyFileStore directory."""
    if not path:
        return {"vector_count": "no path", "dimension": None}
    try:
        from aecp.stores.numpy_files import NumpyFileStore

        store = NumpyFileStore(Path(path))
        count = store.count()
        # Peek at first batch for dimension
        for batch in store.iter_vectors(batch_size=1):
            dim = batch[0].vector.shape[0] if batch else None
            return {"vector_count": count, "dimension": dim}
        return {"vector_count": count, "dimension": None}
    except Exception as e:
        return {"vector_count": f"error: {e}", "dimension": None}


def _suggest_playbook(info: dict) -> str | None:
    """Suggest a playbook based on store metadata."""
    model = (info.get("inferred_model") or "").lower()
    if "ada-002" in model or "ada" in model:
        return "ada-002 → text-embedding-3-small (docs/playbooks/ada-002-to-te3.md)"
    if "embed-v3" in model:
        return "cohere embed-v3 → embed-v4 (docs/playbooks/cohere-v3-to-v4.md)"
    if "voyage-2" in model:
        return "voyage-2 → voyage-3 (docs/playbooks/voyage-2-to-v3.md)"
    return None


# Scenario-calibrated retention expectations (from benchmarks).
# Same-family = same provider, similar architecture (e.g., ada-002 → te3-small).
# Cross-family = different provider/architecture (e.g., MiniLM → bge-large).
SCENARIO_GUIDANCE = """
[bold]Expected retention by scenario (SciFact benchmarks, 3 seeds):[/bold]

  Same-family pairs (e.g., ada-002 → te3-small, bge → e5):
    K ≥ 2000: 0.85–0.93 nDCG@10 retention (PASS)
    K = 1000: 0.73–0.80 (WARN — usable for recall-tolerant workloads)

  Cross-family pairs (e.g., MiniLM → bge-large):
    K ≥ 2000: 0.78–0.87 nDCG@10 retention (PASS)
    K = 1000: 0.67–0.78 (WARN — consider more calibration)

  Same-dim pairs (e.g., bge-large → e5-large, 1024→1024):
    K ≥ 2000: 0.90–0.95 retention (high confidence)

  Gate thresholds: PASS ≥ 0.75, WARN ≥ 0.55, FAIL < 0.55
  These are conservative defaults calibrated to real benchmarks.
  If your gate returns WARN, it does NOT mean the tool is broken —
  it means your specific pair needs more calibration or is cross-family.
"""
