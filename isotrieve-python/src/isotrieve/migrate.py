"""Migration tool for vector stores (WS-4).

End-to-end migration with resumability, non-destructive safety, and progress tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from isotrieve.mapping.base import Mapping
from isotrieve.stores.base import VectorRecord, VectorStore


def _new_manifest_id() -> str:
    """Short, sortable, collision-resistant manifest id."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"mig_{ts}_{uuid4().hex[:6]}"


def _store_spec(store: VectorStore) -> dict[str, str]:
    """Record enough store info to reopen the store from a manifest.

    The ``VectorStore`` ABC has no URI accessor, so we introspect: file-backed
    stores (``NumpyFileStore``) expose ``.path`` and can be reopened by URI;
    anything else is recorded by class name for display only.
    """
    path = getattr(store, "path", None)
    if path is not None:
        return {"type": "numpy", "uri": str(path)}
    return {"type": type(store).__name__, "uri": ""}


@dataclass
class MigrationManifest:
    """Track migration progress for resumability.

    New fields are backward-compatible: ``from_dict`` filters unknown keys, so
    manifests written by older versions load cleanly (missing fields fall back
    to their defaults).
    """

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
    id: str = field(default_factory=_new_manifest_id)
    mapping_path: str = ""
    transform_invertible: bool = False
    rollback_strategy: str = "none"
    gate_result: dict[str, Any] = field(default_factory=dict)
    source_store: dict[str, str] = field(default_factory=dict)
    target_store: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
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
            "mapping_path": self.mapping_path,
            "transform_invertible": self.transform_invertible,
            "rollback_strategy": self.rollback_strategy,
            "gate_result": self.gate_result,
            "source_store": self.source_store,
            "target_store": self.target_store,
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
    mapping_path: Path | None = None,
    rollback_strategy: str = "none",
    gate_result: dict[str, Any] | None = None,
) -> MigrationManifest:
    """Migrate vectors from source to target with transformation.

    Args:
        source: Source vector store.
        target: Target vector store.
        mapping: Fitted mapping to transform vectors.
        batch_size: Batch size for streaming.
        manifest_path: Path to save/load manifest for resumability.
        resume: If True, resume from last checkpoint.
        mapping_path: Path to the ``.isotrieve`` mapping file, recorded in the
            manifest so post-migration tooling (rollback/verify) can reload it.
        rollback_strategy: How the migration can be rolled back: one of
            ``"shadow"`` (original source preserved untouched), ``"inverse"``
            (transform has an analytic inverse), ``"snapshot"`` (store-level
            snapshot), or ``"none"``.
        gate_result: Optional ``GateReport.to_dict()`` captured at migration
            time; used later by ``isotrieve verify`` for drift detection.

    Returns:
        MigrationManifest with progress info.
    """
    # Load existing manifest if resuming
    manifest = None
    start_idx = 0
    if resume and manifest_path and manifest_path.exists():
        try:
            manifest = MigrationManifest.from_dict(
                json.loads(manifest_path.read_text())
            )
            start_idx = manifest.batch_end
            logging.info("Resuming from batch %d", start_idx)
        except Exception:
            pass

    if manifest is None:
        total = source.count()
        meta = mapping.meta
        manifest = MigrationManifest(
            source_collection="source",
            target_collection="target",
            source_model=str(meta.get("source_model_id", "")),
            target_model=str(meta.get("target_model_id", "")),
            total_vectors=total,
            started_at=datetime.now(timezone.utc).isoformat(),
            mapping_path=str(mapping_path) if mapping_path else "",
            transform_invertible=mapping.has_inverse,
            rollback_strategy=rollback_strategy,
            gate_result=dict(gate_result or {}),
            source_store=_store_spec(source),
            target_store=_store_spec(target),
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
