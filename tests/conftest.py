"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def db_available() -> bool:
    """True if the live database is reachable; used to skip DB-backed tests."""
    try:
        from app.core.db import test_connection

        return test_connection()
    except Exception:
        return False


@pytest.fixture
def require_db(db_available: bool) -> None:
    if not db_available:
        pytest.skip("Live database not reachable; skipping DB-backed test.")
