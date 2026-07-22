"""Migration tool for vector stores (WS-4).

End-to-end migration with resumability, non-destructive safety, and progress tracking.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aecp.mapping.base import Mapping
from aecp.stores.base import VectorRecord, VectorStore


@dataclass
class MigrationManifest:
    """Track migration progress for resumability."""

    source_collection: str
    target_collection: str
    source_model: str
    target_model: str
    total_vectors: int
    migrated_vectors: int = 0
    batch_start: int = 0
    batch_end: int = 0
    last_batch_hash: str = ""
    started_at: str = ""
    completed_at: str = ""
    batches: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_collection": self.source_collection,
            "target_collection": self.target_collection,
            "source_model": self.source_model,
            "target_model": self.target_model,
            "total_vectors": self.total_vectors,
            "migrated_vectors": self.migrated_vectors,
            "batch_start": self.batch_start,
            "batch_end": self.batch_end,
            "last_batch_hash": self.last_batch_hash,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "batches": self.batches,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MigrationManifest:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def compute_batch_hash(vectors: np.ndarray) -> str:
    """Compute hash of a batch for integrity verification."""
    return hashlib.sha256(vectors.tobytes()).hexdigest()[:16]


def migrate_store(
    source: VectorStore,
    target: VectorStore,
    mapping: Mapping,
    *,
    batch_size: int = 1024,
    manifest_path: Path | None = None,
    resume: bool = False,
) -> MigrationManifest:
    """Migrate vectors from source to target with transformation.

    Args:
        source: Source vector store.
        target: Target vector store.
        mapping: Fitted mapping to transform vectors.
        batch_size: Batch size for streaming.
        manifest_path: Path to save/load manifest for resumability.
        resume: If True, resume from last checkpoint.

    Returns:
        MigrationManifest with progress info.
    """
    from datetime import datetime, timezone

    # Load existing manifest if resuming
    manifest = None
    start_idx = 0
    if resume and manifest_path and manifest_path.exists():
        try:
            manifest = MigrationManifest.from_dict(
                json.loads(manifest_path.read_text())
            )
            start_idx = manifest.batch_end
            print(f"Resuming from batch {start_idx}")
        except Exception:
            pass

    if manifest is None:
        total = source.count()
        manifest = MigrationManifest(
            source_collection="source",
            target_collection="target",
            source_model="",
            target_model="",
            total_vectors=total,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    # Migrate in batches
    written = 0
    batch_num = 0
    t_start = time.perf_counter()

    for batch_records in source.iter_vectors(batch_size=batch_size):
        # Skip already-migrated batches
        if batch_num < start_idx:
            batch_num += 1
            continue

        # Extract vectors
        vectors = np.array([r.vector for r in batch_records])

        # Transform
        transformed = mapping.transform(vectors)

        # Create new records
        new_records = []
        for i, r in enumerate(batch_records):
            new_records.append(
                VectorRecord(
                    id=r.id,
                    vector=transformed[i],
                    text=r.text,
                    payload=r.payload,
                )
            )

        # Write to target
        target.write_vectors(new_records, batch_size=batch_size)

        # Update manifest
        batch_hash = compute_batch_hash(transformed)
        manifest.batches.append(
            {
                "batch_num": batch_num,
                "start_idx": batch_num * batch_size,
                "count": len(batch_records),
                "hash": batch_hash,
            }
        )
        manifest.migrated_vectors += len(batch_records)
        manifest.batch_end = batch_num + 1
        manifest.last_batch_hash = batch_hash

        # Save manifest
        if manifest_path:
            manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        written += len(batch_records)
        batch_num += 1

        # Progress
        elapsed = time.perf_counter() - t_start
        rate = written / elapsed if elapsed > 0 else 0
        pct = (
            (written / manifest.total_vectors * 100)
            if manifest.total_vectors > 0
            else 0
        )
        print(
            f"\r  Migrated {written}/{manifest.total_vectors} "
            f"({pct:.1f}%) at {rate:.0f} vec/s",
            end="",
            flush=True,
        )

    print()  # newline after progress
    manifest.completed_at = datetime.now(timezone.utc).isoformat()
    if manifest_path:
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

    return manifest
