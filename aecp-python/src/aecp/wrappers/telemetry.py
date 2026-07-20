"""Wrapper telemetry — opt-in, local-only JSONL query counts.

Never sends data off-machine.  Never logs query text.
Env ``AECP_TELEMETRY=local`` enables writing to ``~/.cache/aecp/telemetry.jsonl``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _telemetry_enabled() -> bool:
    """Check if local telemetry is enabled."""
    return os.environ.get("AECP_TELEMETRY", "off").lower() == "local"


def _telemetry_path() -> Path:
    """Return the local telemetry JSONL path."""
    return Path.home() / ".cache" / "aecp" / "telemetry.jsonl"


def log_query(wrapper_type: str, **extra: Any) -> None:
    """Append a query-count event to the local telemetry JSONL.

    Parameters
    ----------
    wrapper_type:
        ``"llamaindex"``, ``"openai_shim"``, or ``"langchain"``.
    **extra:
        Arbitrary metadata (e.g. ``mapping_type``, ``d_src``).  Never
        includes query text.
    """
    if not _telemetry_enabled():
        return

    path = _telemetry_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "ts": time.time(),
        "event": "query",
        "wrapper": wrapper_type,
        **extra,
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")


def read_telemetry() -> list[dict[str, Any]]:
    """Read all events from the local telemetry JSONL."""
    path = _telemetry_path()
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def summary() -> dict[str, Any]:
    """Summarize local telemetry: total queries, per-wrapper counts."""
    events = read_telemetry()
    queries = [e for e in events if e.get("event") == "query"]
    per_wrapper: dict[str, int] = {}
    for q in queries:
        w = q.get("wrapper", "unknown")
        per_wrapper[w] = per_wrapper.get(w, 0) + 1
    return {
        "total_queries": len(queries),
        "per_wrapper": per_wrapper,
        "telemetry_path": str(_telemetry_path()),
    }
