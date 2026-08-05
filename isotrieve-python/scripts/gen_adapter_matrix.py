#!/usr/bin/env python3
"""Generate the adapter capability parity matrix from adapter class attributes.

Run from repo root:
    python isotrieve-python/scripts/gen_adapter_matrix.py [--check]

--check exits 1 if the generated docs/adapters.md is stale vs the code, so CI
can fail on drift. Otherwise it rewrites docs/adapters.md in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isotrieve.adapters.base import VectorStoreAdapter  # noqa: E402
from isotrieve.adapters.chroma import IsotrieveChromaFunction  # noqa: E402
from isotrieve.adapters.pinecone import PineconeAdapter  # noqa: E402
from isotrieve.adapters.qdrant import QdrantAdapter  # noqa: E402

DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "adapters.md"

# adapter-label -> (class, kind)
CAPS = [
    ("serve_mode", "Serve mode"),
    ("offline_migration", "Offline migration"),
    ("idempotency_guard", "Idempotency guard"),
    ("resume", "Resume"),
    ("rollback_strategy", "Rollback strategy"),
    ("tested_in_ci", "Tested in CI"),
]

ADAPTERS = [
    ("VectorStoreAdapter (base)", VectorStoreAdapter, "Base"),
    ("QdrantAdapter", QdrantAdapter, "Query"),
    ("PineconeAdapter", PineconeAdapter, "Query"),
    ("IsotrieveChromaFunction", IsotrieveChromaFunction, "Serve"),
]


def _value(attr: str, value: object) -> str:
    if attr == "rollback_strategy":
        return f"`{value}`"
    if isinstance(value, bool):
        return "Yes" if value else "—"
    return str(value)


def render() -> str:
    header = """# Adapter capability matrix

Automatically generated from adapter class attributes by
`python isotrieve-python/scripts/gen_adapter_matrix.py`. **Do not edit by hand.**

| Adapter | Kind | Serve mode | Offline migration | Idempotency guard | Resume | Rollback strategy | Tested in CI |
|---|---|---|---|---|---|---|---|
"""
    lines: list[str] = [header]
    for name, cls, kind in ADAPTERS:
        cells = [f"`{name}`", kind]
        for attr, _label in CAPS:
            cells.append(_value(attr, getattr(cls, attr)))
        lines.append("| " + " | ".join(cells) + " |\n")
    return "".join(lines)


def main() -> int:
    doc = render()
    if "--check" in sys.argv:
        if DOC_PATH.exists() and DOC_PATH.read_text() == doc:
            print("OK: docs/adapters.md is up to date")
            return 0
        print("STALE: docs/adapters.md is out of date — run "
              "isotrieve-python/scripts/gen_adapter_matrix.py")
        return 1
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(doc)
    print(f"Wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
