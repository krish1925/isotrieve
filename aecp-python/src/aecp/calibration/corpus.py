"""Calibration corpus helpers (Phase 1: minimal built-in + from-texts)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

# Tiny frozen seed set for offline demos/tests — NOT aecp-calib-v1.
# Full versioned corpus lands in Phase 2 (DECISIONS.md Q2).
_BUILTIN_SEED_TEXTS: tuple[str, ...] = (
    "The mitochondria is the powerhouse of the cell.",
    "How do I reset my password?",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Interest rates rose after the central bank announcement.",
    "A recipe for sourdough bread with a long cold ferment.",
    "What is the capital of France?",
    "Kubernetes schedules pods onto nodes based on resource requests.",
    "The plaintiff filed a motion for summary judgment.",
    "Photosynthesis converts light energy into chemical energy.",
    "Can you summarize this quarterly earnings report?",
    "SELECT user_id, COUNT(*) FROM events GROUP BY user_id;",
    "The quick brown fox jumps over the lazy dog.",
    "Climate models project rising sea levels this century.",
    "Push the latest commits and open a pull request.",
    "Customer churn increased 12% month-over-month in EMEA.",
    "Explain quantum entanglement in simple terms.",
    "The museum opens at 10am on weekdays.",
    "Error: connection refused on port 5432.",
    "She whispered a secret across the crowded room.",
    "Vector databases index high-dimensional embeddings for ANN search.",
)


def builtin_calibration_texts(k: int = 20, *, seed: int = 0) -> list[str]:
    """Return up to ``k`` built-in seed texts (deterministic, for demos/tests).

    This is **not** the frozen ``aecp-calib-v1`` corpus. Do not cite quality
    numbers from this set as product claims.
    """
    texts = list(_BUILTIN_SEED_TEXTS)
    if k <= len(texts):
        # Deterministic subset: rotate by seed
        start = seed % len(texts)
        rotated = texts[start:] + texts[:start]
        return rotated[:k]
    # Repeat with index suffixes to reach k (synthetic diversity for unit tests)
    out: list[str] = []
    i = 0
    while len(out) < k:
        out.append(f"{texts[i % len(texts)]} [{i}]")
        i += 1
    return out


def sample_from_texts(
    texts: Sequence[str],
    k: int,
    *,
    seed: int = 0,
) -> list[str]:
    """Sample ``k`` texts without replacement (or with if k > len)."""
    import numpy as np

    arr = list(texts)
    if not arr:
        raise ValueError("No texts to sample from")
    rng = np.random.default_rng(seed)
    if k <= len(arr):
        idx = rng.choice(len(arr), size=k, replace=False)
        return [arr[i] for i in idx]
    idx = rng.choice(len(arr), size=k, replace=True)
    return [arr[i] for i in idx]


def corpus_checksum(texts: Sequence[str]) -> str:
    """SHA256 checksum of joined texts for provenance logging."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()
