"""Postgres/pgvector adapter — transactional in-place migration.

Safety model (issue #22): instead of copying the table, we add a shadow
column ``embedding_isotrieve_new vector(D)``, transform stored vectors into it in
idempotent batches (keyed by primary key), then swap the columns atomically
in a single transaction:

    BEGIN;
    ALTER TABLE items RENAME COLUMN embedding TO embedding_isotrieve_old;
    ALTER TABLE items RENAME COLUMN embedding_isotrieve_new TO embedding;
    ALTER TABLE items DROP COLUMN IF EXISTS embedding_isotrieve_old;
    COMMIT;

Rollback before the swap = drop the shadow column (source untouched).
Resume = re-run batches; UPDATE-by-id is idempotent.

Requires: ``pip install isotrieve[pgvector]`` and the ``vector`` extension
(``CREATE EXTENSION IF NOT EXISTS vector``) enabled in the target database.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from isotrieve.adapters.base import MigrationReport, VectorStoreAdapter
from isotrieve.mapping.base import Mapping

#: Shadow column holding transformed vectors before the atomic swap.
SHADOW_COLUMN = "embedding_isotrieve_new"
#: Column the swapped-out original is parked in mid-transaction.
LEGACY_COLUMN = "embedding_isotrieve_old"
#: Optional JSONB payload column copied through migration untouched.
PAYLOAD_COLUMN = "payload"


def _require_psycopg() -> Any:
    """Import and return the psycopg module, or raise."""
    try:
        import psycopg

        return psycopg
    except ImportError:
        raise ImportError(
            "Postgres/pgvector adapter requires psycopg. "
            "Install it with: pip install isotrieve[pgvector]"
        ) from None


def _parse_vector(value: Any) -> np.ndarray:
    """Coerce a pgvector column value into a float array.

    With ``pgvector.psycopg.register_vector`` (the ``pgvector`` package)
    columns come back as ``pgvector.vector.Vector`` objects (expose
    ``.to_numpy()``); without registration psycopg returns the text form
    ``[a,b,c,...]``. Both are handled here.
    """
    if isinstance(value, str):
        inner = value.strip().strip("[]")
        if not inner:
            return np.empty(0, dtype=np.float64)
        return np.array([float(x) for x in inner.split(",")], dtype=np.float64)
    to_numpy = getattr(value, "to_numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


class PgvectorAdapter(VectorStoreAdapter):
    """pgvector vector store adapter with shadow-column atomic-swap migration.

    Parameters
    ----------
    mapping:
        A fitted Isotrieve Mapping.
    dsn:
        libpq connection string (overrides individual host/port/dbname/...).
    host, port, dbname, user, password:
        Connection parameters used when ``dsn`` is not provided.
    table:
        Name of the table holding the vectors.
    id_column:
        Primary-key column (used for idempotent resume/rollback).
    vector_column:
        Column of type ``vector(D)`` holding the stored vectors.
    mode:
        ``"serve"`` = map queries on-the-fly.
        ``"migrated"`` = corpus already transformed.
    """

    # pgvector: idempotent batch writes to a shadow column, atomic column swap,
    # drop-shadow-column rollback. Tested in CI (dockerized postgres).
    resume: bool = True
    rollback_strategy: str = "shadow"
    tested_in_ci: bool = True

    def __init__(
        self,
        mapping: Mapping,
        *,
        dsn: str | None = None,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "postgres",
        user: str = "postgres",
        password: str | None = None,
        table: str = "items",
        id_column: str = "id",
        vector_column: str = "embedding",
        mode: str = "serve",
    ) -> None:
        super().__init__(mapping, mode=mode)  # type: ignore[arg-type]
        self._psycopg = _require_psycopg()
        self._connect_kwargs: dict[str, Any]
        if dsn:
            self._connect_kwargs = {"dsn": dsn}
        else:
            self._connect_kwargs = {
                "host": host,
                "port": port,
                "dbname": dbname,
                "user": user,
                "password": password,
            }
        self._table = table
        self._id_column = id_column
        self._vector_column = vector_column
        self._shadow = SHADOW_COLUMN
        self._legacy = LEGACY_COLUMN

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> Any:
        """Open a new psycopg connection (context manager commits/rolls back)."""
        conn = self._psycopg.connect(**self._connect_kwargs)
        try:
            from pgvector.psycopg import register_vector

            register_vector(conn)
        except Exception:
            pass
        return conn

    def _has_payload_column(self) -> bool:
        """Return True if a JSONB ``payload`` column exists on the table."""
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(f"SELECT {PAYLOAD_COLUMN} FROM {self._table} LIMIT 1")
                cur.fetchone()
            return True
        except Exception:
            return False

    def _ensure_shadow_column(self, dim: int) -> None:
        """Add the shadow column if missing. Idempotent."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"ALTER TABLE {self._table} "
                f"ADD COLUMN IF NOT EXISTS {self._shadow} vector({dim})"
            )

    # ------------------------------------------------------------------
    # Contract: count / dims / read / query
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Number of rows in the table."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def dimensions(self) -> int | None:
        """Dimension of the stored vectors (from the first row)."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT {self._vector_column} FROM {self._table} LIMIT 1")
            row = cur.fetchone()
            if not row or row[0] is None:
                return None
            return len(_parse_vector(row[0]).ravel())

    def _read_batch(
        self, offset: int, limit: int, include_payload: bool
    ) -> list[tuple[Any, np.ndarray, dict[str, Any] | None]]:
        """Read a batch of (id, vector[, payload]) rows, oldest-first."""
        select = f"{self._id_column}, {self._vector_column}"
        if include_payload:
            select += f", {PAYLOAD_COLUMN}"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {select} FROM {self._table} "
                f"ORDER BY {self._id_column} ASC LIMIT {int(limit)} "
                f"OFFSET {int(offset)}"
            )
            rows = cur.fetchall()
        out: list[tuple[Any, np.ndarray, dict[str, Any] | None]] = []
        for row in rows:
            vec = _parse_vector(row[1]).reshape(1, -1)
            payload = row[2] if include_payload else None
            out.append((row[0], vec, payload))
        return out

    def query(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """Serve-mode query: map queries, cosine-search, return results.

        Cosine distance operator ``<=>``; score = ``1 - distance``. In
        ``"migrated"`` mode the stored vectors are already in new-model space
        and queries are used as-is.
        """
        mapped = query_vectors
        if self._mode != "migrated":
            mapped = self._map_queries(query_vectors)

        has_payload = self._has_payload_column()
        payload_expr = (
            f"COALESCE({PAYLOAD_COLUMN}, '{{}}'::jsonb) AS metadata"
            if has_payload
            else "'{}'::jsonb AS metadata"
        )
        results: list[list[dict[str, Any]]] = []
        with self._connect() as conn, conn.cursor() as cur:
            for vec in mapped:
                cur.execute(
                    f"SELECT {self._id_column}, "
                    f"1 - ({self._vector_column} <=> %s) AS score, "
                    f"{payload_expr} "
                    f"FROM {self._table} "
                    f"ORDER BY {self._vector_column} <=> %s ASC "
                    f"LIMIT {int(k)}",
                    (vec, vec),
                )
                results.append(
                    [
                        {"id": str(r[0]), "score": float(r[1]), "metadata": r[2] or {}}
                        for r in cur.fetchall()
                    ]
                )
        return results

    # ------------------------------------------------------------------
    # Contract: migrate / rollback / restore
    # ------------------------------------------------------------------

    def migrate(
        self,
        batch_size: int = 1000,
        dry_run: bool = False,
        new_collection: str | None = None,
    ) -> MigrationReport:
        """Transform stored vectors in place via a shadow column + atomic swap.

        ``new_collection`` is accepted for contract compatibility; pgvector's
        safety model is an in-place column swap on the configured table, so
        non-None values other than the table name are rejected with a clear
        explanation (use a new adapter instance pointed at a different table
        instead of copying within one migration).
        """
        report = MigrationReport(
            source_collection=self._table,
            target_collection=new_collection or self._table,
        )
        if new_collection and new_collection != self._table:
            raise ValueError(
                "PgvectorAdapter migrates in place via shadow-column swap; "
                "a separate target table is not supported in one migration. "
                "Point a second PgvectorAdapter at the target table instead."
            )

        if dry_run:
            report.rows_processed = self.count()
            report.sampled_recall_at_10 = None
            return report

        dim = self._mapping.d_tgt
        self._ensure_shadow_column(dim)
        include_payload = self._has_payload_column()

        offset = 0
        total = 0
        while True:
            rows = self._read_batch(offset, batch_size, include_payload)
            if not rows:
                break
            with self._connect() as conn, conn.cursor() as cur:
                for row_id, vec, _payload in rows:
                    mapped = self._mapping.transform(vec).ravel().tolist()
                    cur.execute(
                        f"UPDATE {self._table} SET {self._shadow} = %s "
                        f"WHERE {self._id_column} = %s",
                        (mapped, row_id),
                    )
            total += len(rows)
            report.rows_processed = total
            offset += len(rows)

        # Atomic swap: old -> legacy, shadow -> live, drop legacy. One txn.
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"ALTER TABLE {self._table} "
                f"RENAME COLUMN {self._vector_column} TO {self._legacy}"
            )
            cur.execute(
                f"ALTER TABLE {self._table} "
                f"RENAME COLUMN {self._shadow} TO {self._vector_column}"
            )
            cur.execute(
                f"ALTER TABLE {self._table} DROP COLUMN IF EXISTS {self._legacy}"
            )
        return report

    def rollback(self) -> MigrationReport:
        """Drop the shadow column. Source column is untouched until swap.

        After a completed swap the shadow column no longer exists, so this is
        a safe no-op (use ``restore`` semantics with a separate adapter on the
        pre-swap snapshot if you kept one).
        """
        report = MigrationReport(
            source_collection=self._table,
            target_collection=self._table,
            rows_processed=self.count(),
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"ALTER TABLE {self._table} DROP COLUMN IF EXISTS {self._shadow}"
            )
        return report

    def restore(self) -> MigrationReport:
        """Restore the pre-swap column if it still exists (before the swap).

        If the swap already happened the legacy column is gone and this is a
        safe no-op.
        """
        report = MigrationReport(
            source_collection=self._table,
            target_collection=self._table,
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"ALTER TABLE {self._table} DROP COLUMN IF EXISTS {self._shadow}"
            )
        report.rows_processed = self.count()
        return report
