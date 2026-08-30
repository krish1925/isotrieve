"""BEIR dataset loaders for the benchmark harness."""

from __future__ import annotations

# Map each benchmark dataset name to its domain regime. BEIR-backed where a
# suitable public corpus exists; ``code`` and ``legal`` use the offline
# identifier-probe corpora in domain_probes.py (issue #37).
DOMAIN_OF_DATASET: dict[str, str] = {
    "scifact": "medical",
    "nfcorpus": "medical",
    "fiqa": "general",
    "code": "code",
    "legal": "legal",
}

# Datasets backed by a built-in synthetic probe corpus (no network).
_PROBE_DATASETS: frozenset[str] = frozenset({"code", "legal"})


def domain_of(name: str) -> str:
    """Return the domain regime associated with a benchmark dataset name."""
    if name not in DOMAIN_OF_DATASET:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DOMAIN_OF_DATASET)}")
    return DOMAIN_OF_DATASET[name]


def load_beir_dataset(
    name: str,
    *,
    max_docs: int | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, set[str]], str]:
    """Load a benchmark dataset.

    Returns ``(docs, queries, qrels, dataset_id)``.
    Raises if qrels are unavailable — claimable runs must never fall back to
    self-retrieval.
    """
    if name in _PROBE_DATASETS:
        from domain_probes import build_probe_corpus

        return build_probe_corpus(name)

    import ir_datasets

    specs = {
        "scifact": ("beir/scifact", "beir/scifact/test"),
        "nfcorpus": ("beir/nfcorpus", "beir/nfcorpus/test"),
        "fiqa": ("beir/fiqa", "beir/fiqa/test"),
    }
    if name not in specs:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(specs)}")

    corpus_id, eval_id = specs[name]
    corpus_ds = ir_datasets.load(corpus_id)
    eval_ds = ir_datasets.load(eval_id)

    if not hasattr(eval_ds, "qrels_iter"):
        raise RuntimeError(
            f"{eval_id} has no qrels — refusing self-retrieval fallback (Fix F1)"
        )

    docs: list[dict[str, str]] = []
    for doc in corpus_ds.docs_iter():
        title = getattr(doc, "title", "") or ""
        text = getattr(doc, "text", "") or ""
        docs.append({"id": doc.doc_id, "text": f"{title} {text}".strip()})
        if max_docs and len(docs) >= max_docs:
            break

    queries = [{"id": q.query_id, "text": q.text} for q in eval_ds.queries_iter()]
    qrels: dict[str, set[str]] = {}
    for qrel in eval_ds.qrels_iter():
        if int(qrel.relevance) > 0:
            qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)

    if not qrels:
        raise RuntimeError(
            f"No positive qrels for {eval_id} — refusing claimable run (Fix F1)"
        )

    # Keep only queries that have qrels
    qids = set(qrels)
    queries = [q for q in queries if q["id"] in qids]
    return docs, queries, qrels, f"{eval_id}@ir_datasets"
