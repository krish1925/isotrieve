"""Numpy / Parquet file-backed vector store (Phase 1 primary path)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from isotrieve.stores.base import VectorRecord, VectorStore


class NumpyFileStore(VectorStore):
    """Store vectors as ``.npy`` plus optional sidecar JSONL for ids/text.

    Layout::

        directory/
          vectors.npy      # float32/float64, shape (N, D)
          meta.jsonl       # optional: {"id": "...", "text": "..."} per line
          manifest.json    # optional resume cursor

    Writing always targets this directory; callers should point at a *new*
    output directory for migrations (dual-collection safety).
    """

    def __init__(self, path: str | Path, *, create: bool = False) -> None:
        self.path = Path(path)
        self.vectors_path = self.path / "vectors.npy"
        self.meta_path = self.path / "meta.jsonl"
        self.manifest_path = self.path / "manifest.json"
        if create:
            self.path.mkdir(parents=True, exist_ok=True)
        elif not self.vectors_path.exists():
            raise FileNotFoundError(f"No vectors.npy at {self.path}")

    @classmethod
    def from_arrays(
        cls,
        path: str | Path,
        vectors: np.ndarray,
        *,
        ids: list[str] | None = None,
        texts: list[str] | None = None,
    ) -> NumpyFileStore:
        """Create a store directory from in-memory arrays."""
        store = cls(path, create=True)
        vectors = np.asarray(vectors)
        np.save(store.vectors_path, vectors)
        n = vectors.shape[0]
        ids = ids or [str(i) for i in range(n)]
        with store.meta_path.open("w", encoding="utf-8") as f:
            for i in range(n):
                row: dict[str, Any] = {"id": ids[i]}
                if texts is not None:
                    row["text"] = texts[i]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return store

    def count(self) -> int:
        if not self.vectors_path.exists():
            return 0
        arr = np.load(self.vectors_path, mmap_mode="r")
        return int(arr.shape[0])

    def _load_meta(self) -> list[dict[str, Any]]:
        if not self.meta_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.meta_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def iter_vectors(self, batch_size: int = 1024) -> Iterator[list[VectorRecord]]:
        vectors = np.load(self.vectors_path, mmap_mode="r")
        meta = self._load_meta()
        n = vectors.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch: list[VectorRecord] = []
            for i in range(start, end):
                m = meta[i] if i < len(meta) else {}
                batch.append(
                    VectorRecord(
                        id=str(m.get("id", i)),
                        vector=np.asarray(vectors[i], dtype=np.float64),
                        text=m.get("text"),
                        payload=m if m else None,
                    )
                )
            yield batch

    def write_vectors(
        self,
        records: Iterator[list[VectorRecord]] | list[VectorRecord],
        *,
        batch_size: int = 1024,
    ) -> int:
        self.path.mkdir(parents=True, exist_ok=True)
        if isinstance(records, list) and records:
            # Convert dicts to VectorRecords
            first = records[0]
            if isinstance(first, dict):
                records = [
                    VectorRecord(
                        id=r["id"],
                        vector=np.asarray(r["vector"]),
                        text=r.get("text"),
                        payload=r.get("payload"),
                    )
                    for r in records
                ]
                first = records[0]

            if isinstance(first, VectorRecord):
                batches: list[list[VectorRecord]] = []
                buf: list[VectorRecord] = []
                for r in records:  # type: ignore[assignment]
                    assert isinstance(r, VectorRecord)
                    buf.append(r)
                    if len(buf) >= batch_size:
                        batches.append(buf)
                        buf = []
                if buf:
                    batches.append(buf)
                record_iter: Iterator[list[VectorRecord]] = iter(batches)
            else:
                record_iter = records  # type: ignore[assignment]
        else:
            record_iter = records  # type: ignore[assignment]

        batch_files: list[Path] = []
        meta_lines: list[str] = []
        written = 0
        last_id: str | None = None
        tmp_dir = self.path / ".write_tmp"
        tmp_dir.mkdir(exist_ok=True)

        try:
            for batch in record_iter:
                vecs = [np.asarray(rec.vector, dtype=np.float64) for rec in batch]
                arr = np.stack(vecs, axis=0)
                batch_file = tmp_dir / f"batch_{len(batch_files)}.npy"
                np.save(batch_file, arr)
                batch_files.append(batch_file)

                for rec in batch:
                    row: dict[str, Any] = {"id": rec.id}
                    if rec.text is not None:
                        row["text"] = rec.text
                    if rec.payload:
                        for k, v in rec.payload.items():
                            if k not in row:
                                row[k] = v
                    meta_lines.append(json.dumps(row, ensure_ascii=False))
                    last_id = rec.id
                    written += 1

            if not batch_files:
                return 0

            # Concatenate all batch files into final array
            arrays = [np.load(f) for f in batch_files]
            arr = np.concatenate(arrays, axis=0)
            np.save(self.vectors_path, arr)
        finally:
            # Clean up temp files
            for f in batch_files:
                f.unlink(missing_ok=True)
            tmp_dir.rmdir()

        with self.meta_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(meta_lines) + "\n")
        manifest = {"last_written_id": last_id, "count": written}
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return written
