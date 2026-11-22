"""Database connection pool and schema reconciliation."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

log = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def connect(settings, attempts: int = 30) -> None:
    """Open the pool, waiting for Postgres to accept connections."""
    global _pool

    def configure(connection) -> None:
        register_vector(connection)

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            pool = ConnectionPool(
                settings.database_url,
                min_size=1,
                max_size=10,
                kwargs={"row_factory": dict_row},
                configure=configure,
                open=True,
                timeout=10,
            )
            with pool.connection() as connection:
                connection.execute("SELECT 1")
            _pool = pool
            log.info("connected to the database")
            ensure_settings_table()
            reconcile_dimensions(settings.embed_dim)
            return
        except Exception as exc:
            last = exc
            log.info("database not ready (%s/%s): %s", attempt + 1, attempts, exc)
            time.sleep(2)
    raise RuntimeError(f"could not reach the database: {last}")


def close() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection():
    if _pool is None:
        raise RuntimeError("the database pool is not open")
    with _pool.connection() as conn:
        yield conn


def query(sql: str, params: tuple = ()) -> list[dict]:
    with connection() as conn:
        return conn.execute(sql, params).fetchall()


def query_one(sql: str, params: tuple = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple = ()) -> None:
    with connection() as conn:
        conn.execute(sql, params)


def ensure_settings_table() -> None:
    """Values saved from the settings panel. Created here rather than only in
    init.sql, which runs once and so never reaches an existing database."""
    execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def read_app_settings() -> dict[str, str]:
    return {row["key"]: row["value"] for row in query("SELECT key, value FROM app_settings")}


def write_app_settings(values: dict[str, str | None]) -> None:
    """Upsert each value; None removes it, falling back to the environment."""
    with connection() as conn:
        for key, value in values.items():
            if value is None:
                conn.execute("DELETE FROM app_settings WHERE key = %s", (key,))
            else:
                conn.execute(
                    """
                    INSERT INTO app_settings (key, value) VALUES (%s, %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value,
                                                    updated_at = now()
                    """,
                    (key, value),
                )


def current_dimensions() -> int | None:
    """pgvector stores a column's dimension in atttypmod."""
    row = query_one(
        """
        SELECT atttypmod AS dim
        FROM pg_attribute
        WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
        """
    )
    if not row or row["dim"] is None or row["dim"] < 0:
        return None
    return int(row["dim"])


def reconcile_dimensions(wanted: int) -> None:
    """Resize the embedding column when the model changes.

    Vectors from different models are not comparable, so existing passages are
    discarded and their documents queued for reindexing rather than left to
    return confident nonsense.
    """
    existing = current_dimensions()
    if existing == wanted:
        return

    log.warning(
        "embedding size changed (%s -> %s); clearing passages and queueing "
        "documents for reindexing", existing, wanted,
    )
    with connection() as conn:
        conn.execute("DROP INDEX IF EXISTS chunks_vector_idx")
        conn.execute("TRUNCATE chunks")
        conn.execute(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({wanted})")
        conn.execute(
            "CREATE INDEX chunks_vector_idx ON chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )
        conn.execute(
            "UPDATE documents SET status = 'pending', chunk_count = 0, "
            "progress = 0, indexed_at = NULL"
        )
