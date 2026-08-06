"""pgvector adapter tests — fake in-memory Postgres, no server needed.

The fake interprets the exact SQL statement shapes the adapter emits so the
shadow-column safety model (idempotent batches, atomic swap, drop-column
rollback) is exercised against a real-ish table, deterministically.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from isotrieve.mapping.linear import RidgeMapping

D_SRC = 8
D_TGT = 12


class _UndefinedColumn(Exception):
    """Simulates psycopg.errors.UndefinedColumn."""


class _FakeCursor:
    def __init__(self, db: _FakeTable) -> None:
        self._db = db
        self._rows: list[tuple[Any, ...]] = []
        self._i = 0

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: Any = ()) -> None:
        self._rows = self._db._execute(sql, params)
        self._i = 0

    def fetchone(self) -> tuple[Any, ...] | None:
        if self._i >= len(self._rows):
            return None
        row = self._rows[self._i]
        self._i += 1
        return row

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows = self._rows[self._i :]
        self._i = len(self._rows)
        return rows


class _FakeConn:
    def __init__(self, db: _FakeTable) -> None:
        self._db = db

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *exc: object) -> bool:
        if exc[0] is None:
            self._db._tx_active = False
        else:
            self._db.rollback()
        return False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._db)

    def commit(self) -> None:
        self._db._tx_active = False

    def rollback(self) -> None:
        self._db.rollback()


class _FakePsycopg:
    def __init__(self, db: _FakeTable) -> None:
        self._db = db
        self.errors = _Errors()

    def connect(self, **kwargs: Any) -> _FakeConn:
        return _FakeConn(self._db)


class _Errors:
    UndefinedColumn = _UndefinedColumn


class _FakeTable:
    """In-memory Postgres table with vector columns + a mini SQL interpreter."""

    def __init__(self, columns: dict[str, tuple[str, int | None]]) -> None:
        # column name -> (type, dim); type in {"vector", "bigint", "jsonb"}
        self.columns = dict(columns)
        self.rows: dict[Any, dict[str, Any]] = {}
        self._tx_active = False

    # -- population helpers -------------------------------------------------
    def add(self, row_id: Any, values: dict[str, Any]) -> None:
        row = {c: None for c in self.columns}
        row.update(values)
        self.rows[row_id] = row

    def get_vector(self, row_id: Any, col: str) -> np.ndarray:
        return np.asarray(self.rows[row_id][col], dtype=np.float64)

    # -- transaction semantics ----------------------------------------------
    def begin_tx(self) -> None:
        self._tx_active = True

    def rollback(self) -> None:
        # The adapter wraps batches in `with conn:` — psycopg commits on clean
        # exit; the fake keeps per-batch writes applied. No shadow rollback
        # needed here; used for the kill/resume simulation.
        self._tx_active = False

    # -- mini SQL interpreter ----------------------------------------------
    def _execute(self, sql: str, params: Any = ()) -> list[tuple[Any, ...]]:
        sql = sql.strip()
        if sql.startswith("CREATE EXTENSION"):
            return []
        if sql.startswith("SELECT COUNT(*)") and "LIMIT" not in sql:
            return [(len(self.rows),)]
        if sql.startswith("ALTER TABLE"):
            return self._alter(sql)
        if sql.startswith("UPDATE"):
            return self._update(sql, params)
        if "<=>" in sql:
            return self._similarity(sql, params)
        if sql.startswith("SELECT"):
            return self._select(sql)
        raise AssertionError(f"unhandled SQL in fake: {sql}")

    def _alter(self, sql: str) -> list[tuple[Any, ...]]:
        m = re.search(r"ADD COLUMN IF NOT EXISTS (\w+) vector\((\d+)\)", sql)
        if m:
            col, dim = m.group(1), int(m.group(2))
            self.columns[col] = ("vector", dim)
            for row in self.rows.values():
                row[col] = None
            return []
        m = re.search(r"RENAME COLUMN (\w+) TO (\w+)", sql)
        if m:
            src, dst = m.group(1), m.group(2)
            self.columns[dst] = self.columns.pop(src)
            for row in self.rows.values():
                row[dst] = row.pop(src)
            return []
        m = re.search(r"DROP COLUMN IF EXISTS (\w+)", sql)
        if m:
            col = m.group(1)
            self.columns.pop(col, None)
            for row in self.rows.values():
                row.pop(col, None)
            return []
        raise AssertionError(f"unhandled ALTER: {sql}")

    def _update(self, sql: str, params: Any) -> list[tuple[Any, ...]]:
        m = re.search(r"SET (\w+) = %s WHERE (\w+) = %s", sql)
        assert m, f"unhandled UPDATE: {sql}"
        col, _id_col = m.group(1), m.group(2)
        value, row_id = params[0], params[1]
        assert row_id in self.rows, f"row {row_id} missing in fake"
        self.rows[row_id][col] = value
        return []

    def _select(self, sql: str) -> list[tuple[Any, ...]]:
        m = re.search(r"FROM (\w+) ORDER BY (\w+) ASC LIMIT (\d+) OFFSET (\d+)", sql)
        if m:
            _table, _id_col, limit, offset = (
                m.group(1),
                m.group(2),
                int(m.group(3)),
                int(m.group(4)),
            )
            if "COUNT(*)" in sql:
                return [(len(self.rows),)]
            select_cols = self._select_cols(sql)
            ids = sorted(self.rows.keys())[offset : offset + limit]
            return [
                tuple(self._col_value(row, c, sql) for c in select_cols)
                for row_id in ids
                for row in [self.rows[row_id]]
            ]

        m = re.search(r"FROM (\w+) LIMIT (\d+)", sql)
        if m:
            _table, limit = m.group(1), int(m.group(2))
            cols = self._select_cols(sql)
            ids = sorted(self.rows.keys())[:limit]
            return [
                tuple(self._col_value(self.rows[i], c, sql) for c in cols) for i in ids
            ]

        m = re.search(r"FROM (\w+)", sql)
        if m:
            cols = self._select_cols(sql)
            return [
                tuple(self._col_value(self.rows[i], c, sql) for c in cols)
                for i in sorted(self.rows)
            ]
        raise AssertionError(f"unhandled SELECT: {sql}")

    def _select_cols(self, sql: str) -> list[str]:
        inner = sql[sql.index("SELECT") + 6 : sql.index(" FROM")]
        cols: list[str] = []
        for part in inner.split(","):
            part = part.strip()
            if part in self.columns:
                cols.append(part)
            elif part.startswith("COALESCE("):
                cols.append(part.split("(", 1)[1].split(",", 1)[0])
            else:
                cols.append(part)  # expression -> ignored for values
        return cols

    def _col_value(self, row: dict[str, Any], col: str, sql: str) -> Any:
        if col not in self.columns:
            if col in ("payload",):
                raise _UndefinedColumn(f'column "{col}" does not exist')
            return None
        return row.get(col)

    def _similarity(self, sql: str, params: Any) -> list[tuple[Any, ...]]:
        m = re.search(r"ORDER BY (\w+) <=> %s ASC LIMIT (\d+)", sql)
        assert m, f"unhandled similarity: {sql}"
        vec_col, limit = m.group(1), int(m.group(2))
        query = np.asarray(params[0], dtype=np.float64)
        scored = []
        for row_id in sorted(self.rows):
            vec = self.get_vector(row_id, vec_col)
            dist = 1.0 - float(
                np.dot(vec, query)
                / (np.linalg.norm(vec) * np.linalg.norm(query) + 1e-12)
            )
            score = 1.0 - dist
            metadata = self.rows[row_id].get("payload")
            scored.append((row_id, score, metadata or {}))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [(r[0], r[1], r[2]) for r in scored[:limit]]


def _make_mapping(d_src=D_SRC, d_tgt=D_TGT, k=120):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(k, d_src))
    W = rng.normal(size=(d_src, d_tgt))
    Y = X @ W
    return RidgeMapping(alpha="auto", seed=0).fit(X, Y)


def _make_adapter(mapping, table: _FakeTable, *, mode="serve", **kwargs):
    from isotrieve.adapters import pgvector as pv

    fake = _FakePsycopg(table)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pv, "_require_psycopg", lambda: fake)
        adapter = pv.PgvectorAdapter(
            mapping,
            table=kwargs.get("table", "items"),
            id_column=kwargs.get("id_column", "id"),
            vector_column=kwargs.get("vector_column", "embedding"),
            mode=mode,
        )
    return adapter


def _populate(table: _FakeTable, n: int, dim: int, *, payload: bool = True) -> None:
    rng = np.random.default_rng(99)
    for i in range(n):
        row = {"id": i, "embedding": rng.normal(size=dim).tolist()}
        if payload:
            row["payload"] = {"idx": i}
        table.add(i, row)


def _make_table(n: int = 0, dim: int = D_SRC, *, payload: bool = True) -> _FakeTable:
    columns = {"id": ("bigint", None), "embedding": ("vector", dim)}
    if payload:
        columns["payload"] = ("jsonb", None)
    table = _FakeTable(columns)
    _populate(table, n, dim, payload=payload)
    return table


class TestPgvectorInMemory:
    def test_count_and_dimensions(self):
        table = _make_table(n=25, dim=D_SRC)
        m = _make_mapping()
        adapter = _make_adapter(m, table)

        assert adapter.count() == 25
        assert adapter.dimensions() == D_SRC

    def test_query_serve_mode(self):
        table = _make_table(n=50, dim=D_SRC)
        m = _make_mapping()
        adapter = _make_adapter(m, table)

        query = np.random.default_rng(7).normal(size=(3, D_TGT))
        results = adapter.query(query, k=5)

        assert len(results) == 3
        for hits in results:
            assert len(hits) <= 5
            for hit in hits:
                assert "id" in hit
                assert "score" in hit
                assert "metadata" in hit

    def test_migrate_swaps_columns(self):
        table = _make_table(n=20, dim=D_SRC)
        m = _make_mapping()
        adapter = _make_adapter(m, table)

        report = adapter.migrate(batch_size=7)

        assert report.rows_processed == 20
        # Shadow column gone, live column now target-dim
        assert "embedding_isotrieve_new" not in table.columns
        assert "embedding_isotrieve_old" not in table.columns
        assert table.columns["embedding"] == ("vector", D_TGT)
        # Vectors actually transformed to target dim
        assert table.get_vector(0, "embedding").shape == (D_TGT,)

    def test_migrate_dry_run(self):
        table = _make_table(n=15, dim=D_SRC)
        m = _make_mapping()
        adapter = _make_adapter(m, table)

        report = adapter.migrate(dry_run=True)

        assert report.rows_processed == 15
        assert "embedding_isotrieve_new" not in table.columns
        assert adapter.dimensions() == D_SRC  # source untouched

    def test_migrate_rejects_target_table(self):
        table = _make_table(n=5, dim=D_SRC)
        m = _make_mapping()
        adapter = _make_adapter(m, table)

        with pytest.raises(ValueError, match="in place"):
            adapter.migrate(new_collection="other")

    def test_migrate_default_target_name(self):
        table = _make_table(n=5, dim=D_SRC)
        m = _make_mapping()
        adapter = _make_adapter(m, table)

        report = adapter.migrate()
        assert report.target_collection == "items"

    def test_query_after_migrate_migrated_mode(self):
        table = _make_table(n=30, dim=D_SRC)
        m = _make_mapping()
        adapter = _make_adapter(m, table)
        adapter.migrate(batch_size=10)

        migrated = _make_adapter(m, table, mode="migrated")
        query = np.random.default_rng(7).normal(size=(2, D_TGT))
        results = migrated.query(query, k=3)
        assert len(results) == 2
        for hits in results:
            assert len(hits) <= 3

    def test_payload_preserved(self):
        table = _make_table(n=10, dim=D_SRC, payload=True)
        m = _make_mapping()
        adapter = _make_adapter(m, table)
        adapter.migrate(batch_size=4)

        assert table.rows[0]["payload"] == {"idx": 0}


class TestPgvectorResilience:
    def test_kill_resume_idempotent(self):
        table = _make_table(n=10_000, dim=D_SRC)
        m = _make_mapping()
        adapter = _make_adapter(m, table)

        # Kill mid-run: make the 4th batch UPDATE raise.
        orig_execute = table._execute
        calls = {"n": 0}

        def flaky(sql, params=()):
            if sql.startswith("UPDATE"):
                calls["n"] += 1
                if calls["n"] == 4:
                    raise RuntimeError("simulated kill mid-stream")
            return orig_execute(sql, params)

        table._execute = flaky
        with pytest.raises(RuntimeError, match="simulated kill"):
            adapter.migrate(batch_size=200)

        # Resume: UPDATE-by-id is idempotent, so re-running completes.
        table._execute = orig_execute
        report = adapter.migrate(batch_size=200)
        assert report.rows_processed == 10_000
        assert table.get_vector(0, "embedding").shape == (D_TGT,)
        assert table.get_vector(9999, "embedding").shape == (D_TGT,)

    def test_rollback_before_swap_drops_shadow_column(self):
        table = _make_table(n=400, dim=D_SRC)
        m = _make_mapping()
        adapter = _make_adapter(m, table)

        # Force a failure before the swap (during the last batch UPDATE),
        # leaving a partial shadow column behind.
        orig_execute = table._execute
        calls = {"n": 0}

        def flaky(sql, params=()):
            if sql.startswith("UPDATE"):
                calls["n"] += 1
                if calls["n"] == 4:
                    raise RuntimeError("simulated kill before swap")
            return orig_execute(sql, params)

        table._execute = flaky
        with pytest.raises(RuntimeError, match="simulated kill"):
            adapter.migrate(batch_size=100)

        # Rollback = drop the shadow column; source stays intact.
        report = adapter.rollback()
        assert report.rows_processed == 400
        assert "embedding_isotrieve_new" not in table.columns
        assert table.columns["embedding"] == ("vector", D_SRC)
        assert table.get_vector(0, "embedding").shape == (D_SRC,)

        # Re-migrate cleanly from untouched source.
        table._execute = orig_execute
        report = adapter.migrate(batch_size=100)
        assert report.rows_processed == 400
        assert table.columns["embedding"] == ("vector", D_TGT)

    def test_exact_batch_boundary(self):
        table = _make_table(n=1000, dim=D_SRC)
        m = _make_mapping()
        adapter = _make_adapter(m, table)

        report = adapter.migrate(batch_size=100)
        assert report.rows_processed == 1000
        assert table.get_vector(999, "embedding").shape == (D_TGT,)


class TestPgvectorGuard:
    def test_require_psycopg_hint(self):
        import isotrieve.adapters.pgvector as pv

        try:
            import psycopg  # noqa: F401

            pytest.skip("psycopg installed; guard not exercised")
        except ImportError:
            pass

        with pytest.raises(ImportError, match="isotrieve\\[pgvector\\]"):
            pv._require_psycopg()


@pytest.mark.integration
class TestPgvectorDocker:
    """Real Postgres+pgvector via DSN (dockerized CI job only).

    Requires ``PGVECTOR_TEST_DSN`` (e.g.
    postgresql://postgres:postgres@localhost:5432/postgres) and the ``vector``
    extension available. Skipped otherwise.
    """

    DSN = "PGVECTOR_TEST_DSN"

    @pytest.fixture()
    def dsn(self):
        import os

        value = os.environ.get(self.DSN)
        if not value:
            pytest.skip(f"{self.DSN} not set — skipping real-DB integration test")
        return value

    def _make_psycopg(self, dsn):
        import psycopg

        return psycopg

    def _setup(self, dsn):
        import psycopg

        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS items_ci")
            cur.execute(
                "CREATE TABLE items_ci (id bigint PRIMARY KEY, embedding vector(8))"
            )
        rng = np.random.default_rng(99)
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            for i in range(20):
                cur.execute(
                    "INSERT INTO items_ci (id, embedding) VALUES (%s, %s)",
                    (i, rng.normal(size=D_SRC).tolist()),
                )

    def _adapter(self, mapping, dsn, mode="serve"):
        from isotrieve.adapters import pgvector as pv

        class _Mod:
            def connect(self, **kwargs):
                import psycopg

                return psycopg.connect(dsn)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(pv, "_require_psycopg", lambda: _Mod())
            return pv.PgvectorAdapter(mapping, table="items_ci", mode=mode)

    def test_migrate_and_query_roundtrip(self, dsn):
        import psycopg

        self._setup(dsn)
        m = _make_mapping()
        adapter = self._adapter(m, dsn)

        assert adapter.count() == 20
        assert adapter.dimensions() == D_SRC

        report = adapter.migrate(batch_size=7)
        assert report.rows_processed == 20

        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT embedding FROM items_ci LIMIT 1")
            row = cur.fetchone()
            assert row is not None and len(row[0]) == D_TGT

        migrated = self._adapter(m, dsn, mode="migrated")
        query = np.random.default_rng(7).normal(size=(2, D_TGT))
        results = migrated.query(query, k=3)
        assert len(results) == 2
        for hits in results:
            assert len(hits) <= 3
            for hit in hits:
                assert "id" in hit
