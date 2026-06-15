"""Tests for read-only SQL enforcement (no database required)."""

from __future__ import annotations

import pytest

from app.db import NonReadOnlyQueryError, assert_read_only


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM t",
        "select 1",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "EXPLAIN SELECT 1",
        "SELECT 1;",  # single trailing semicolon tolerated
        "  SELECT count(*) FROM schema.table  ",
    ],
)
def test_allows_read_statements(sql: str) -> None:
    assert_read_only(sql)  # should not raise


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a = 1",
        "DELETE FROM t",
        "DROP TABLE t",
        "ALTER TABLE t ADD COLUMN a int",
        "CREATE TABLE t (a int)",
        "TRUNCATE t",
        "SELECT 1; DROP TABLE t",  # stacked statement
        "GRANT ALL ON t TO public",
        "",
        "   ",
    ],
)
def test_rejects_non_read_statements(sql: str) -> None:
    with pytest.raises(NonReadOnlyQueryError):
        assert_read_only(sql)


def test_comment_cannot_hide_keyword() -> None:
    # A DROP hidden behind a comment must still be rejected because the leading
    # keyword after stripping comments is DROP, not SELECT.
    with pytest.raises(NonReadOnlyQueryError):
        assert_read_only("/* SELECT */ DROP TABLE t")
