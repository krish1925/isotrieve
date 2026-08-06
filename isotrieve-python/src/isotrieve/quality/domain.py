"""Lightweight corpus-domain inference (issue #37).

Deterministic keyword-overlap heuristic that maps a sample of corpus texts to
a domain regime (``general`` / ``legal`` / ``medical`` / ``code``). Used by the
gate report and the doctor CLI to point users at the most relevant published
domain benchmark. No model, no network.
"""

from __future__ import annotations

from collections.abc import Sequence

# Substring keywords per domain. Substring matching keeps the heuristic robust
# to morphology (e.g. "patient", "patients", "patently") and to identifiers.
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "medical": (
        "diagnos",
        "patient",
        "disease",
        "symptom",
        "clinical",
        "medication",
        "dosage",
        "physician",
        "surgery",
        "hospital",
        "infection",
        "diabetes",
        "cardiac",
        "icd-10",
        "cpt",
        "screening",
        "prognos",
        "syndrome",
    ),
    "legal": (
        "court",
        "statute",
        "plaintiff",
        "defendant",
        "appeal",
        "judge",
        "contract",
        "trial",
        "motion",
        "counsel",
        "damages",
        "jurisdiction",
        "opinion",
        "brief",
        "liability",
        "witness",
        "hearing",
        "statutory",
        "summary judgment",
        "u.s.c.",
        "f.3d",
    ),
    "code": (
        "function",
        "class",
        "exception",
        "error",
        "import",
        "endpoint",
        "api",
        "return",
        "raise",
        "module",
        "http",
        "request",
        "config",
        "compile",
        "runtime",
        "client",
        "server",
        "database",
        "stack trace",
        "query",
        "payload",
        "retry",
        "async",
        "def ",
    ),
    "general": (),
}

DEFAULT_DOMAIN = "general"


def infer_domain(texts: Sequence[str]) -> str:
    """Return the domain whose keywords occur most often in ``texts``.

    Zero-hit and tied corpora default to ``general``. Deterministic.
    """
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for text in texts:
        norm = text.lower()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            scores[domain] += sum(1 for keyword in keywords if keyword in norm)

    best = max(scores, key=lambda d: scores[d])
    if scores[best] == 0:
        return DEFAULT_DOMAIN
    return best
