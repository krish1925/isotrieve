"""BEIR dataset loaders for the benchmark harness."""

from __future__ import annotations

from typing import Any


def load_beir_dataset(
    name: str,
    *,
    max_docs: int | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, set[str]], str]:
    """Load a BEIR dataset via ir_datasets.

    Returns ``(docs, queries, qrels, dataset_id)``.
    Raises if qrels are unavailable — claimable runs must never fall back to
    self-retrieval.
    """
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
