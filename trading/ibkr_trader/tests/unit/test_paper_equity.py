"""Unit tests for libs.persistence.paper_equity — PaperEquityManager."""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Prevent real asyncpg from being imported by the module under test.
# We inject a lightweight stub so that `import asyncpg` inside
# paper_equity.py resolves without the real C-extension library.
# ---------------------------------------------------------------------------
_fake_asyncpg = MagicMock()
_fake_asyncpg.Pool = MagicMock  # type annotation reference
sys.modules.setdefault("asyncpg", _fake_asyncpg)

from libs.persistence.paper_equity import (  # noqa: E402
    PaperEquityManager,
    apply_paper_pnl,
    load_paper_equity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool() -> MagicMock:
    pool = MagicMock()
    pool.fetchrow = AsyncMock()
    pool.execute = AsyncMock()
    return pool


# ---------------------------------------------------------------------------
# Tests — load()
# ---------------------------------------------------------------------------

class TestLoad:
    """Test PaperEquityManager.load() under various DB states."""

    @pytest.mark.asyncio
    async def test_load_returns_existing_value(self) -> None:
        """When the row already exists, load() returns its equity and
        does NOT call execute (no INSERT)."""
        pool = _make_pool()
        pool.fetchrow.return_value = {"equity": 50_000.0}

        mgr = PaperEquityManager(pool, account_scope="test_acct", initial_equity=100_000.0)
        result = await mgr.load()

        assert result == 50_000.0
        pool.execute.assert_not_called()
        pool.fetchrow.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_seeds_and_reselects(self) -> None:
        """When no row exists, load() INSERTs the initial equity then
        re-SELECTs and returns the committed value."""
        pool = _make_pool()
        # First fetchrow (SELECT) -> None => triggers INSERT path.
        # Second fetchrow (re-SELECT after INSERT) -> the seeded value.
        pool.fetchrow.side_effect = [None, {"equity": 100_000.0}]

        mgr = PaperEquityManager(pool, account_scope="new_acct", initial_equity=100_000.0)
        result = await mgr.load()

        assert result == 100_000.0
        # INSERT must have been called exactly once.
        pool.execute.assert_awaited_once()
        call_sql = pool.execute.call_args[0][0]
        assert "INSERT INTO paper_equity" in call_sql
        assert "ON CONFLICT" in call_sql
        # fetchrow must have been called twice (initial SELECT + re-SELECT).
        assert pool.fetchrow.await_count == 2

    @pytest.mark.asyncio
    async def test_load_concurrent_insert_returns_db_value(self) -> None:
        """If a concurrent process inserts first (ON CONFLICT DO NOTHING),
        the re-SELECT returns the OTHER process's value, NOT _initial."""
        pool = _make_pool()
        # First fetchrow -> None (no row yet).
        # After INSERT (which does nothing due to conflict), the re-SELECT
        # returns a different equity set by the concurrent process.
        concurrent_equity = 75_000.0
        pool.fetchrow.side_effect = [None, {"equity": concurrent_equity}]

        mgr = PaperEquityManager(pool, account_scope="race_acct", initial_equity=100_000.0)
        result = await mgr.load()

        # Must return the DB value, not our _initial.
        assert result == concurrent_equity
        assert result != 100_000.0
        pool.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests — scope property
# ---------------------------------------------------------------------------

class TestScope:
    def test_scope_property(self) -> None:
        pool = _make_pool()
        mgr = PaperEquityManager(pool, account_scope="my_scope", initial_equity=0.0)
        assert mgr.scope == "my_scope"


class TestCompatibilityHelpers:
    @pytest.mark.asyncio
    async def test_load_paper_equity_uses_scope_and_initial(self) -> None:
        pool = _make_pool()
        pool.fetchrow.side_effect = [None, {"equity": 25_000.0}]

        result = await load_paper_equity(
            pool,
            account_scope="family_scope",
            initial_equity=25_000.0,
        )

        assert result == 25_000.0
        assert pool.fetchrow.await_count == 2
        assert pool.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_apply_paper_pnl_seeds_then_updates_scope(self) -> None:
        pool = _make_pool()
        pool.fetchrow.side_effect = [
            None,
            {"equity": 10_000.0},
            {"equity": 10_125.0},
        ]

        result = await apply_paper_pnl(
            pool,
            pnl=150.0,
            commission=25.0,
            account_scope="family_scope",
            initial_equity=10_000.0,
        )

        assert result == 10_125.0
        assert pool.execute.await_count == 1
