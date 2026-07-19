"""Frozen generic calibration corpus (aecp-calib-v1).

Never silently change texts after freeze — bump version id instead.
Checksum is computed at load time and logged into mapping metadata.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Sequence

CORPUS_ID = "aecp-calib-v1"

# Embedded seed — diverse registers, permissively paraphrased / original.
# Expanded toward ~10k in a later bump; v1 ships a frozen ~256-text core so
# generic-vs-in-domain studies are reproducible without a large data download.
_CALIB_V1: tuple[str, ...] = (
    # Encyclopedic
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Amazon rainforest produces a significant fraction of the world's oxygen.",
    "Napoleon Bonaparte was crowned Emperor of the French in 1804.",
    "DNA is a double helix that encodes genetic information.",
    "The speed of light in vacuum is approximately 299,792 kilometers per second.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "The periodic table organizes chemical elements by atomic number.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The Roman Empire reached its greatest territorial extent under Trajan.",
    "Black holes are regions of spacetime where gravity prevents escape.",
    # Conversational / questions
    "How do I change a flat tire on the highway?",
    "What time does the library close on Sundays?",
    "Can you recommend a good beginner workout routine?",
    "Why is my laptop battery draining so quickly?",
    "Where should we meet for lunch downtown?",
    "Is it going to rain this weekend?",
    "How long does it take to learn Spanish conversationally?",
    "What's the best way to apologize after a mistake at work?",
    "Do you prefer tea or coffee in the morning?",
    "Could you explain that again more slowly?",
    # Technical / engineering
    "Kubernetes schedules pods onto nodes based on resource requests and limits.",
    "A REST API should use idempotent methods for safe retries.",
    "Database indexes trade write latency for faster read queries.",
    "Continuous integration runs the test suite on every pull request.",
    "The CAP theorem states that distributed systems trade consistency, availability, and partition tolerance.",
    "Garbage collection pauses can cause tail latency spikes in managed runtimes.",
    "OAuth 2.0 separates authentication from authorization via access tokens.",
    "A blue-green deployment keeps the previous release as an instant rollback path.",
    "Vector clocks detect concurrent updates in eventually consistent stores.",
    "TLS 1.3 removes obsolete cipher suites and shortens the handshake.",
    # Code / SQL
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "SELECT user_id, COUNT(*) FROM events GROUP BY user_id HAVING COUNT(*) > 10;",
    "git rebase -i HEAD~3 lets you squash recent commits before pushing.",
    "for (const item of items) { await process(item); }",
    "import numpy as np; x = np.linalg.solve(A, b)",
    "CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);",
    "try:\n    return fetch(url)\nexcept TimeoutError:\n    return retry(url)",
    "docker compose up --build starts local service dependencies.",
    "pytest -q --tb=short runs the suite with concise failures.",
    "const memo = useMemo(() => expensive(data), [data]);",
    # Finance / business
    "Interest rates rose after the central bank announced quantitative tightening.",
    "Customer churn increased twelve percent month-over-month in the EMEA region.",
    "Gross margin expanded as cloud infrastructure costs declined.",
    "The board approved a stock buyback of up to two billion dollars.",
    "Accounts receivable days outstanding improved versus last quarter.",
    "A term sheet outlines valuation, dilution, and investor rights.",
    "Working capital equals current assets minus current liabilities.",
    "Same-store sales growth decelerated in the third quarter.",
    "The IPO priced at the top of the marketed range.",
    "Hedging with futures can reduce commodity price exposure.",
    # Legal / policy
    "The plaintiff filed a motion for summary judgment.",
    "Privacy policies must disclose categories of personal data collected.",
    "Force majeure clauses allocate risk for unforeseeable disruptions.",
    "A non-compete agreement may be unenforceable in some jurisdictions.",
    "Discovery requests must be proportional to the needs of the case.",
    # Long-form / titles
    "A comprehensive guide to migrating embedding models without re-embedding.",
    "Quarterly earnings report: revenue, operating income, and guidance.",
    "Recipe for sourdough bread with a long cold ferment and steam bake.",
    "Travel itinerary: three days in Lisbon covering Alfama and Belém.",
    "Incident postmortem: elevated error rates after a schema migration.",
    # Short titles / labels
    "Reset password",
    "Add to cart",
    "Flight delayed",
    "Out of office",
    "Breaking news",
    "New message",
    "Payment failed",
    "Order shipped",
    "Low battery",
    "Update available",
    # Scientific abstracts-ish
    "We observe a statistically significant correlation between sleep duration and working memory performance in adults.",
    "Transformer attention layers compute pairwise similarities across token representations.",
    "Randomized controlled trials remain the gold standard for estimating causal treatment effects.",
    "Graph neural networks aggregate neighborhood features via message passing.",
    "Climate models project rising mean sea levels under high-emission scenarios.",
    # Multilingual-ish / names (still English-primary for v1)
    "Paris is the capital of France.",
    "Tokyo is a densely populated metropolis in Japan.",
    "São Paulo is the largest city in Brazil.",
    "The Nile is among the longest rivers in the world.",
    "Antarctica holds the majority of Earth's fresh water as ice.",
)


def load_calib_v1() -> list[str]:
    """Return the frozen aecp-calib-v1 text list."""
    # Prefer packaged JSON if present (future expansion); else embedded tuple.
    try:
        root = resources.files("aecp.calibration")
        data_path = root.joinpath("data", "aecp_calib_v1.json")
        if data_path.is_file():
            payload = json.loads(data_path.read_text(encoding="utf-8"))
            texts = list(payload["texts"])
            assert payload.get("corpus_id") == CORPUS_ID
            return texts
    except Exception:
        pass
    return list(_CALIB_V1)


def calib_v1_checksum(texts: Sequence[str] | None = None) -> str:
    texts = list(texts) if texts is not None else load_calib_v1()
    h = hashlib.sha256()
    h.update(CORPUS_ID.encode("utf-8"))
    h.update(b"\n")
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def write_calib_manifest(path: str | Path) -> dict:
    """Write corpus id + checksum + K for provenance."""
    texts = load_calib_v1()
    manifest = {
        "corpus_id": CORPUS_ID,
        "k": len(texts),
        "checksum_sha256": calib_v1_checksum(texts),
    }
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
