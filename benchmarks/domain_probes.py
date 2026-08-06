"""Identifier-level domain probes for the domain-matrix benchmark (issue #37).

Each probe is a pair ``(identifier_query, chunk_text)``: an exact identifier
(ICD-10 code, case citation, file path, error code, UUID, version string,
date, ...) and the chunk of text that contains it. Probe retrieval is
*same-index true-match* (like :func:`isotrieve.quality.metrics.topk_retention`):
a probe query "hits" when the chunk at the same index is in the top-k nearest
neighbors of the probe corpus after the mapping transform.

Probes are deterministic and offline — no network calls, no model downloads
beyond the harness models that are already loaded.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

DOMAINS: tuple[str, ...] = ("general", "legal", "medical", "code")

# Synthetic probe corpora are padded with generated distractors to this size so
# that a K=500 calibration run is meaningful (rank-wise) without any download.
SYNTHETIC_CORPUS_SIZE = 520

# domain -> list of (identifier_query_text, chunk_containing_text)
PROBE_DOMAINS: dict[str, list[tuple[str, str]]] = {
    "general": [
        ("INV-2026-0042",
         "Invoice INV-2026-0042 dated 2026-03-14 totals $1,240.50 and is due on 2026-04-14."),
        ("TRK-9371028845",
         "Your package tracking number TRK-9371028845 shipped on 2026-01-30 via express courier."),
        ("v2.14.0",
         "The desktop app updated to version v2.14.0 on 2026-02-11 with a new settings panel."),
        ("a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
         "Session a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d expired after 30 minutes of inactivity."),
        ("2026-05-01T09:30:00Z",
         "The webinar starts at 2026-05-01T09:30:00Z UTC; join early for the keynote."),
        ("PWR-SVC-8891",
         "Replacement part PWR-SVC-8891 was dispatched from the regional warehouse on Friday."),
        ("856-23-1049",
         "Member 856-23-1049 renewed their annual subscription on 2026-02-20."),
        ("CUS-7023",
         "Customer CUS-7023 reported a billing discrepancy in their September statement."),
        ("BK-11424",
         "Booking reference BK-11424 confirms two seats on flight 482 departing 2026-06-18."),
        ("Q4-2026",
         "The Q4-2026 earnings call is scheduled for the first Tuesday of the quarter."),
    ],
    "legal": [
        ("123 F.3d 456",
         "In Brown v. Board, 123 F.3d 456 (10th Cir. 2024), the panel held the fee award was "
         "not an abuse of discretion."),
        ("29 U.S.C. § 201",
         "The Fair Labor Standards Act, 29 U.S.C. § 201 et seq., establishes minimum wage and "
         "overtime protections."),
        ("41 C.F.R. § 60-1.4",
         "Contractors must comply with 41 C.F.R. § 60-1.4, which requires affirmative action "
         "postings."),
        ("No. 24-1042",
         "Appeal No. 24-1042, United States v. Reyes, is set for oral argument in January."),
        ("611 F. Supp. 3d 88",
         "SEC v. Martinson, 611 F. Supp. 3d 88 (S.D.N.Y. 2025), applied the Morrison test."),
        ("§ 12(b)(6)",
         "The court granted dismissal under § 12(b)(6) for failure to state a claim upon which "
         "relief can be granted."),
        ("18 U.S.C. § 1030",
         "Computer-fraud charges under 18 U.S.C. § 1030 require unauthorized access to a "
         "protected computer."),
        ("Pub. L. No. 116-260",
         "Appropriations were enacted by Pub. L. No. 116-260 on December 27, 2020."),
        ("Cal. Civ. Code § 1798.140",
         "The CCPA defines personal information in Cal. Civ. Code § 1798.140."),
    ],
    "medical": [
        ("E11.9",
         "Type 2 diabetes mellitus without complications is coded E11.9 in the discharge summary."),
        ("I21.4",
         "Non-ST elevation myocardial infarction maps to ICD-10 code I21.4."),
        ("J45.909",
         "Unspecified asthma without complication is classified as J45.909."),
        ("99385",
         "CPT code 99385 covers an initial comprehensive preventive medicine visit for a patient "
         "aged 18-39."),
        ("NDC 00074-9152-01",
         "Dispensed amoxicillin 500 mg capsules (NDC 00074-9152-01), one capsule three times "
         "daily for ten days."),
        ("R10.9",
         "Unspecified abdominal pain is documented under R10.9 in the emergency record."),
        ("M54.5",
         "Low back pain is coded M54.5; consider imaging only for red-flag symptoms."),
        ("F32.9",
         "Major depressive disorder, single episode, unspecified severity is coded F32.9."),
        ("Z79.899",
         "Long-term use of other medications is indicated by Z79.899 on the medication list."),
        ("U07.1",
         "COVID-19 was assigned emergency code U07.1 in early 2020."),
    ],
    "code": [
        ("FileNotFoundError",
         "FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml' at "
         "src/main.py:42"),
        ("ValueError",
         "ValueError: invalid literal for int() with base 10: 'abc' raised in parse_input at "
         "utils/parsers.py:88"),
        ("requests.exceptions.Timeout",
         "requests.exceptions.Timeout was raised when the retry policy exhausted backoff in "
         "clients/http.py:210"),
        ("src/utils/validate.py",
         "Import validate from src/utils/validate.py and call validate.schema(payload) before "
         "persisting."),
        ("GET /v1/users/{id}",
         "The API endpoint GET /v1/users/{id} returns a 200 with the user profile."),
        ("POST /api/v2/ingest",
         "Send raw events to POST /api/v2/ingest with an HMAC signature header."),
        ("AttributeError",
         "AttributeError: 'NoneType' object has no attribute 'items' surfaced in build_payload "
         "at core/events.py:156"),
        ("node_modules/.cache",
         "Clear the node_modules/.cache directory before re-running the install step in CI."),
        ("ETIMEDOUT",
         "ETIMEDOUT after 30s connecting to the metrics endpoint; check the load balancer "
         "health checks."),
        ("KeyError",
         "KeyError: 'session_id' thrown by the middleware when the cookie jar is empty."),
        ("RuntimeError",
         "RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB in train_loop at "
         "model/trainer.py:77."),
        ("def _chunk_text",
         "def _chunk_text(text, size): yields overlapping spans; used by the ingestion "
         "pipeline."),
        ("requirements.txt",
         "Pin exact versions in requirements.txt so builds are reproducible across "
         "environments."),
        ("e2e/smoke.spec.ts",
         "Run e2e/smoke.spec.ts after every deploy to validate the checkout flow."),
    ],
}

# Distractor chunks pad the synthetic corpora so retrieval is not trivially
# self-match-only while keeping the runs fully offline.
PROBE_DISTRACTORS: dict[str, list[str]] = {
    "general": [
        "Please allow five business days for the refund to appear on your statement.",
        "Our support team is available Monday through Friday from 9am to 6pm.",
        "You can manage notification preferences from the account settings page.",
        "Prices do not include applicable taxes or delivery fees at checkout.",
    ],
    "legal": [
        "Discovery closed on the date set by the scheduling order.",
        "The parties are ordered to participate in a settlement conference.",
        "Jurisdiction is proper because the events giving rise to the claim occurred here.",
        "The clerk shall serve a copy of this order on all counsel of record.",
    ],
    "medical": [
        "Vital signs were stable and the patient tolerated the procedure well.",
        "Instruct the patient to follow up with the primary care clinic in two weeks.",
        "A chest x-ray was obtained and showed no acute abnormalities.",
        "Medication reconciliation should be completed at every care transition.",
    ],
    "code": [
        "Compile the project in release mode before running the integration suite.",
        "The service degrades gracefully when the upstream connection is lost.",
        "Log structured events with correlation ids to trace requests across services.",
        "Run the linter and type checker before opening a pull request.",
    ],
}


# Template pools for the deterministic distractor generator. Each template
# references ``{i}`` (doc index) so every generated chunk is unique while
# remaining fully offline and reproducible.
_CODE_TEMPLATES: list[str] = [
    "def {fn}({args}):\n    \"\"\"{doc}\"\"\"\n    {body}",
    "async def {fn}({args}):\n    \"\"\"{doc}\"\"\"\n    {body}",
    "class {cls}:\n    def __init__(self):\n        self.{attr} = {value}",
    "try:\n    {body}\nexcept {exc}:\n    logger.exception('{msg}')",
    "import {mod}\nfrom {pkg} import {name}",
    "if {cond}:\n    return {val}\nelse:\n    raise {exc}('{msg}')",
    "def {fn}({args}):\n    return {val}  # reference {i}",
    "const {name} = require('{pkg}');  // reference {i}",
    "GET /api/{res}/{i} -> 200 OK",
    "let {name} = await client.query('{sql}');  // reference {i}",
]

_LEGAL_TEMPLATES: list[str] = [
    "The {party} filed a {doctype} on {date} seeking {relief}.",
    "Pursuant to {statute}, the court {action} the motion in reference {i}.",
    "Section {sec} of the agreement provides that {provision}.",
    "The {court} held that {holding}, citing prior precedent in reference {i}.",
    "Neither party shall {prohibited} without prior written consent.",
    "Discovery in reference {i} shall close on {date}, subject to extension for good cause.",
    "The {party} objected on the ground that the {item} was privileged.",
    "For the reasons stated in reference {i}, the motion is granted in part and denied in part.",
    "The court retains jurisdiction to enforce the terms of the {agreement}.",
    "Any dispute arising under this agreement shall be resolved in {venue}.",
]

_LEGAL_WORDS = {
    "party": ("plaintiff", "defendant", "appellant", "respondent"),
    "doctype": ("motion to dismiss", "summary-judgment motion", "complaint", "answer", "brief"),
    "date": ("2026-01-15", "2026-02-03", "2026-03-19", "2026-04-22", "2026-05-07"),
    "relief": ("dismissal", "injunctive relief", "damages", "attorney's fees", "declaratory judgment"),
    "statute": ("42 U.S.C. § 1983", "29 U.S.C. § 621", "15 U.S.C. § 78j", "11 U.S.C. § 362"),
    "action": ("granted", "denied", "stayed", "reconsidered", "vacated"),
    "sec": ("7.4", "12.1", "9.3", "3.8", "15.2"),
    "provision": ("either party may terminate on sixty days notice", "the forum-selection clause governs",
                  "confidential information remains the discloser's property", "indemnification is mutual"),
    "court": ("district court", "court of appeals", "bankruptcy court", "state trial court"),
    "holding": ("the fee-shifting provision is enforceable", "the statute of limitations had run",
                "qualified immunity applies", "the contract was unconscionable"),
    "prohibited": ("assign its rights", "commence arbitration", "use the marks", "disclose the terms"),
    "item": ("letter", "memorandum", "draft report", "communication"),
    "agreement": ("settlement agreement", "licensing agreement", "employment agreement", "non-disclosure agreement"),
    "venue": ("the Southern District of New York", "arbitration in Geneva", "the courts of Delaware"),
}

_CODE_WORDS = {
    "fn": ("parse_payload", "validate_schema", "retry_with_backoff", "build_index", "chunk_text",
           "normalize_embedding", "stream_events", "load_config", "emit_metric", "render_row"),
    "args": ("payload: dict", "text: str, size: int", "retries: int = 3", "events: list[dict]",
             "paths: Sequence[Path]", "raw: bytes", "rows: Iterator[Row]", "cfg: Config"),
    "doc": ("Parse the incoming payload into a normalized record.", "Split text into overlapping spans.",
            "Apply exponential backoff on transient failures.", "Build an inverted index over tokens.",
            "Emit a metric for every processed batch."),
    "body": ("return normalize(payload)", "return _chunk_text(text, size)",
             "raise RetryExhausted() if attempts > retries", "yield from transform(events)",
             "return paths and validate each one", "return cfg.reload(force=True)"),
    "cls": ("PayloadParser", "VectorIndex", "RetryPolicy", "EventStream", "ConfigLoader"),
    "attr": ("_buffer", "_dim", "_timeout", "_registry", "_cursor"),
    "value": ("[]", "{}", "0", "None", "128", "3.0"),
    "exc": ("ValueError", "KeyError", "TimeoutError", "RuntimeError", "TypeError"),
    "msg": ("invalid input", "missing key", "connection timed out", "unexpected state", "bad type"),
    "mod": ("numpy", "pandas", "asyncio", "httpx", "sqlalchemy", "pydantic"),
    "pkg": ("numpy as np", "pandas as pd", "os", "sys", "json", "logging"),
    "name": ("np", "pd", "client", "logger", "config", "store", "index"),
    "cond": ("result is None", "not cfg.valid", "attempts >= max_retries", "payload.get('id') is None"),
    "val": ("None", "True", "{}", "[]", "0", "records"),
    "sql": ("SELECT id FROM docs WHERE status = 'ready'", "INSERT INTO events (id, ts) VALUES (?, ?)",
            "UPDATE jobs SET state = 'done' WHERE id = ?"),
    "res": ("users", "orders", "events", "jobs", "search", "ingest"),
    "attr2": ("name", "id", "status", "version", "count"),
}

_DOMAIN_TEMPLATES: dict[str, list[str]] = {"code": _CODE_TEMPLATES, "legal": _LEGAL_TEMPLATES}
_DOMAIN_WORDS: dict[str, dict[str, tuple[str, ...]]] = {"code": _CODE_WORDS, "legal": _LEGAL_WORDS}


def _generate_distractors(domain: str, target_total: int, seed: int = 7) -> list[str]:
    """Deterministically generate ``target_total`` distractor chunks for a domain."""
    templates = _DOMAIN_TEMPLATES[domain]
    words = _DOMAIN_WORDS[domain]
    rng = random.Random(seed)
    out: list[str] = []
    i = 0
    while len(out) < target_total:
        template = templates[i % len(templates)]
        kwargs: dict[str, str] = {"i": i}
        for key, options in words.items():
            kwargs[key] = options[i % len(options)] if isinstance(options, tuple) else options
        chunk = template.format(**kwargs)
        if rng.random() < 0.2:
            chunk = chunk + f" #{i:05d}"
        out.append(chunk)
        i += 1
    return out


def probe_chunks(domain: str) -> list[str]:
    """Return the chunk texts that contain each identifier for ``domain``."""
    return [chunk for _, chunk in PROBE_DOMAINS[domain]]


def probe_queries(domain: str) -> list[str]:
    """Return the exact identifier query texts for ``domain``."""
    return [query for query, _ in PROBE_DOMAINS[domain]]


def build_probe_corpus(
    domain: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, set[str]], str]:
    """Build a synthetic, self-contained retrieval corpus for ``domain``.

    Returns ``(docs, queries, qrels, dataset_id)`` matching the BEIR loader
    contract. Every query maps to exactly one doc (the chunk containing its
    identifier). No network access.
    """
    if domain not in PROBE_DOMAINS:
        raise ValueError(f"Unknown probe domain {domain!r}; choose from {sorted(PROBE_DOMAINS)}")

    docs: list[dict[str, str]] = []
    for i, (_, chunk) in enumerate(PROBE_DOMAINS[domain]):
        docs.append({"id": f"{domain}_p{i:02d}", "text": chunk})
    for i, distractor in enumerate(PROBE_DISTRACTORS[domain]):
        docs.append({"id": f"{domain}_d{i:02d}", "text": distractor})
    if domain in _DOMAIN_TEMPLATES:
        n_distractors = max(0, SYNTHETIC_CORPUS_SIZE - len(docs))
        for i, chunk in enumerate(_generate_distractors(domain, n_distractors)):
            docs.append({"id": f"{domain}_g{i:04d}", "text": chunk})

    n_probes = len(PROBE_DOMAINS[domain])
    queries = [
        {"id": f"{domain}_q{i:02d}", "text": q}
        for i, (q, _) in enumerate(PROBE_DOMAINS[domain])
    ]
    qrels = {f"{domain}_q{i:02d}": {f"{domain}_p{i:02d}"} for i in range(n_probes)}
    return docs, queries, qrels, f"probe/{domain}"


def probe_specs(domains: Iterable[str]) -> dict[str, dict[str, list[str]]]:
    """Map selected domains to their probe chunk/query texts."""
    return {
        domain: {"chunks": probe_chunks(domain), "queries": probe_queries(domain)}
        for domain in domains
    }
