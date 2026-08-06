"""Doctor CLI command — inspect a target store and suggest next steps."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.table import Table

from isotrieve.quality.domain import infer_domain

console = Console()


def register_doctor_command(app: typer.Typer) -> None:
    """Register the ``isotrieve doctor`` command on the Typer app."""

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
            console.print_json(__import__("json").dumps(info, indent=2, default=str))
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
                "Run 'isotrieve gate' with calibration data to evaluate migration.[/yellow]"
            )

        # Domain-regime guidance (issue #37)
        domain = str(info.get("domain_regime") or "general")
        guidance = _domain_guidance(domain)
        console.print(guidance)

        # Print scenario-calibrated expectations
        console.print(SCENARIO_GUIDANCE)


def _inspect_store(
    store_type: str,
    url: str | None,
    collection: str | None,
    source_model: str | None,
) -> dict[str, object]:
    """Inspect store and return metadata dict."""
    info: dict[str, object] = {
        "store_type": store_type,
        "url": url or "default",
        "collection": collection or "default",
        "vector_count": None,
        "dimension": None,
        "inferred_model": source_model or "unknown",
        "has_isotrieve_metadata": False,
    }

    if store_type == "chroma":
        info.update(_inspect_chroma(url, collection))
    elif store_type == "qdrant":
        info.update(_inspect_qdrant(url, collection))
    elif store_type == "numpy":
        info.update(_inspect_numpy(url))
    else:
        console.print(f"[yellow]Unknown store type: {store_type}[/yellow]")

    # Infer the domain regime from sampled collection text (issue #37).
    raw_texts = cast(list[str], info.pop("sample_texts", []))
    sample_texts = [t for t in raw_texts if isinstance(t, str)]
    info["n_sampled_texts"] = len(sample_texts)
    info["domain_regime"] = infer_domain(sample_texts)

    return info


def _inspect_chroma(url: str | None, collection: str | None) -> dict[str, object]:
    """Inspect a ChromaDB collection."""
    try:
        import chromadb

        client = chromadb.Client() if not url else chromadb.HttpClient(host=url)
        col = client.get_collection(collection or "default")
        count = col.count()
        sample = col.get(limit=20, include=["embeddings", "metadatas", "documents"])
        dim = len(sample["embeddings"][0]) if sample.get("embeddings") else None
        has_isotrieve = False
        sample_texts: list[str] = []
        if sample.get("documents"):
            sample_texts.extend(d for d in sample["documents"] if d)
        if sample.get("metadatas"):
            for meta in sample["metadatas"]:
                if isinstance(meta, dict):
                    for v in meta.values():
                        if isinstance(v, str) and v not in sample_texts:
                            sample_texts.append(v)
            if sample["metadatas"] and "isotrieve_mapping_id" in sample["metadatas"][0]:
                has_isotrieve = True
        return {
            "vector_count": count,
            "dimension": dim,
            "has_isotrieve_metadata": has_isotrieve,
            "sample_texts": sample_texts[:20],
        }
    except Exception as e:
        return {"vector_count": f"error: {e}", "dimension": None}


def _inspect_qdrant(url: str | None, collection: str | None) -> dict[str, object]:
    """Inspect a Qdrant collection."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=url or "http://localhost:6333")
        info = client.get_collection(collection or "default")
        vectors = info.config.params.vectors
        size = None
        if isinstance(vectors, (dict,)):
            size = None
        elif vectors is not None:
            size = vectors.size
        return {
            "vector_count": info.points_count,
            "dimension": size,
        }
    except Exception as e:
        return {"vector_count": f"error: {e}", "dimension": None}


def _inspect_numpy(path: str | None) -> dict[str, object]:
    """Inspect a NumpyFileStore directory."""
    if not path:
        return {"vector_count": "no path", "dimension": None}
    try:
        from isotrieve.stores.numpy_files import NumpyFileStore

        store = NumpyFileStore(Path(path))
        count = store.count()
        dim = None
        sample_texts: list[str] = []
        for batch in store.iter_vectors(batch_size=64):
            for rec in batch:
                if dim is None:
                    dim = rec.vector.shape[0]
                if rec.text and len(sample_texts) < 20:
                    sample_texts.append(rec.text)
            if dim is not None and len(sample_texts) >= 20:
                break
        return {
            "vector_count": count,
            "dimension": dim,
            "sample_texts": sample_texts,
        }
    except Exception as e:
        return {"vector_count": f"error: {e}", "dimension": None}


def _suggest_playbook(info: dict[str, object]) -> str | None:
    """Suggest a playbook based on store metadata."""
    model = str(info.get("inferred_model") or "").lower()
    if "ada-002" in model or "ada" in model:
        return "ada-002 → text-embedding-3-small (docs/playbooks/ada-002-to-te3.md)"
    if "embed-v3" in model:
        return "cohere embed-v3 → embed-v4 (docs/playbooks/cohere-v3-to-v4.md)"
    if "voyage-2" in model:
        return "voyage-2 → voyage-3 (docs/playbooks/voyage-2-to-v3.md)"
    return None


# Published domain-matrix benchmarks (issue #37) keyed by domain regime.
DOMAIN_BENCHMARKS: dict[str, str] = {
    "general": "FiQA (BEIR) — `--dataset fiqa`",
    "medical": "SciFact (BEIR) — `--dataset scifact`",
    "code": "offline code probe corpus — `--dataset code`",
    "legal": "offline legal probe corpus — `--dataset legal`",
}


def _domain_guidance(domain: str) -> str:
    """Print which published domain benchmarks are most relevant."""
    bench = DOMAIN_BENCHMARKS.get(domain, DOMAIN_BENCHMARKS["general"])
    return (
        f"[bold]Domain regime:[/bold] {domain}. Most relevant published domain "
        f"benchmark: {bench}. See benchmarks/results/ for per-domain retention."
    )


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
