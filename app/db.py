"""Read-only PostgreSQL access layer.

This module is the single gateway to the database. It enforces three rules that
are central to the thesis security model:

1. Connections request a read-only transaction characteristic where possible.
2. A statement timeout is applied to every connection.
3. :func:`execute_read_query` rejects any SQL that is not a single read
   statement (SELECT / WITH / EXPLAIN / SHOW), blocking INSERT/UPDATE/DELETE/
   DDL even if such a query were ever constructed.

All queries are parameterised; user input is never concatenated into SQL.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import Connection

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Statements that must never be executed through this read-only layer.
_FORBIDDEN_KEYWORDS = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
        "MERGE",
        "REPLACE",
        "COPY",
        "CALL",
        "DO",
        "VACUUM",
        "COMMENT",
        "REINDEX",
        "REFRESH",
        "LOCK",
        "SET",
        "RESET",
    }
)

# Only these leading keywords are accepted as read statements.
_ALLOWED_LEADING = ("SELECT", "WITH", "EXPLAIN", "SHOW", "TABLE", "VALUES")

_COMMENT_RE = re.compile(r"(--[^\n]*)|(/\*.*?\*/)", re.DOTALL)


class NonReadOnlyQueryError(ValueError):
    """Raised when a query is not a permitted single read-only statement."""


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL line and block comments so they cannot hide keywords."""
    return _COMMENT_RE.sub(" ", sql)


def assert_read_only(sql: str) -> None:
    """Validate that ``sql`` is a single read-only statement.

    Raises :class:`NonReadOnlyQueryError` otherwise. This is a defence-in-depth
    check; the primary guarantee should come from a read-only DB role.
    """
    cleaned = _strip_sql_comments(sql).strip()
    if not cleaned:
        raise NonReadOnlyQueryError("Empty query.")

    # Reject stacked statements (e.g. "SELECT 1; DROP TABLE x").
    # A single trailing semicolon is tolerated.
    inner = cleaned.rstrip(";")
    if ";" in inner:
        raise NonReadOnlyQueryError("Multiple statements are not allowed.")

    upper = inner.upper()
    first_word = upper.split(None, 1)[0] if upper.split() else ""
    if first_word not in _ALLOWED_LEADING:
        raise NonReadOnlyQueryError(
            f"Query must start with one of {_ALLOWED_LEADING}; got '{first_word}'."
        )

    # Word-boundary scan for forbidden keywords anywhere in the statement.
    tokens = set(re.findall(r"[A-Z_]+", upper))
    forbidden = tokens & _FORBIDDEN_KEYWORDS
    if forbidden:
        raise NonReadOnlyQueryError(
            f"Forbidden keyword(s) present: {', '.join(sorted(forbidden))}."
        )


def _build_engine(settings: Settings) -> Engine:
    """Create a SQLAlchemy engine with connect + statement timeouts."""
    connect_args: dict[str, Any] = {
        "connect_timeout": settings.db_connect_timeout,
        # psycopg passes options through to the server; set statement_timeout (ms).
        "options": f"-c statement_timeout={settings.db_statement_timeout * 1000}",
    }
    engine = create_engine(
        settings.sqlalchemy_url(),
        pool_pre_ping=True,
        connect_args=connect_args,
        # Do not log SQL parameters (may contain data); keep echo off.
        echo=False,
    )

    # Belt-and-braces: mark every transaction read-only at the session level.
    @event.listens_for(engine, "connect")
    def _set_read_only(dbapi_connection: Any, _record: Any) -> None:  # noqa: ANN401
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
            cursor.close()
        except Exception:  # pragma: no cover - depends on server permissions
            # If the role cannot set this, we still rely on assert_read_only.
            logger.debug("Could not set session read-only characteristics.")

    return engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a process-wide cached read-only engine."""
    settings = get_settings()
    logger.info("Creating DB engine: %s", settings.safe_summary())
    return _build_engine(settings)


def get_connection() -> Connection:
    """Open a new connection from the shared engine.

    Caller is responsible for closing it (use as a context manager).
    """
    return get_engine().connect()


def execute_read_query(query: str, params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute a validated, parameterised read-only query.

    Args:
        query: A single read-only SQL statement. Use ``:name`` bind parameters.
        params: Parameter values bound safely by the driver (never interpolated).

    Returns:
        A list of row dictionaries.

    Raises:
        NonReadOnlyQueryError: If the query is not a permitted read statement.
    """
    assert_read_only(query)
    with get_connection() as conn:
        result = conn.execute(text(query), dict(params or {}))
        return [dict(row._mapping) for row in result]


def test_connection() -> bool:
    """Verify connectivity. Returns True on success; never exposes the password."""
    try:
        rows = execute_read_query("SELECT 1 AS ok")
        return bool(rows) and rows[0].get("ok") == 1
    except Exception as exc:  # pragma: no cover - network dependent
        logger.error("Connection test failed: %s", type(exc).__name__)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = test_connection()
    print("Database connection:", "OK" if ok else "FAILED")
