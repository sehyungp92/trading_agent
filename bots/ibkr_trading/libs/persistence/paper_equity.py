"""Paper-equity persistence generalized for account or family scopes."""
from __future__ import annotations

import logging

import asyncpg

logger = logging.getLogger(__name__)


class PaperEquityManager:
    """Tracks paper equity for a named account or strategy family."""

    def __init__(self, pool: asyncpg.Pool, account_scope: str, initial_equity: float):
        self._pool = pool
        self._scope = account_scope
        self._initial = initial_equity

    @property
    def scope(self) -> str:
        return self._scope

    async def load(self) -> float:
        row = await self._pool.fetchrow(
            """
            SELECT equity
            FROM paper_equity
            WHERE account_scope = $1
            """,
            self._scope,
        )
        if row is not None:
            return float(row["equity"])

        await self._pool.execute(
            """
            INSERT INTO paper_equity (
                account_scope, equity, initial_equity, total_pnl, total_commission, trade_count
            )
            VALUES ($1, $2, $2, 0.0, 0.0, 0)
            ON CONFLICT (account_scope) DO NOTHING
            """,
            self._scope,
            self._initial,
        )
        # Re-SELECT to get the actual committed value (handles concurrent inserts)
        row = await self._pool.fetchrow(
            "SELECT equity FROM paper_equity WHERE account_scope = $1",
            self._scope,
        )
        actual = float(row["equity"])
        logger.info("Seeded paper equity scope %s at %.2f", self._scope, actual)
        return actual

    async def apply_pnl(self, pnl: float, commission: float = 0.0) -> float:
        row = await self._pool.fetchrow(
            """
            UPDATE paper_equity
            SET equity = equity + $2 - $3,
                total_pnl = total_pnl + $2,
                total_commission = total_commission + $3,
                trade_count = trade_count + CASE WHEN $2 != 0 THEN 1 ELSE 0 END,
                last_update_at = now()
            WHERE account_scope = $1
            RETURNING equity
            """,
            self._scope,
            pnl,
            commission,
        )
        if row is None:
            raise RuntimeError(
                f"paper_equity scope {self._scope!r} is missing; call load() before apply_pnl()"
            )
        return float(row["equity"])

    async def get_current_equity(self) -> float:
        """Return current equity (convenience wrapper for callback use)."""
        row = await self._pool.fetchrow(
            "SELECT equity FROM paper_equity WHERE account_scope = $1",
            self._scope,
        )
        if row is None:
            return self._initial
        return float(row["equity"])

    async def reset(self, new_equity: float | None = None) -> None:
        equity = self._initial if new_equity is None else new_equity
        await self._pool.execute(
            """
            UPDATE paper_equity
            SET equity = $2,
                initial_equity = $2,
                total_pnl = 0.0,
                total_commission = 0.0,
                trade_count = 0,
                last_update_at = now()
            WHERE account_scope = $1
            """,
            self._scope,
            equity,
        )
        logger.info("Reset paper equity scope %s to %.2f", self._scope, equity)


async def load_paper_equity(
    pool: asyncpg.Pool,
    account_scope: str = "paper",
    initial_equity: float = 10_000.0,
) -> float:
    """Compatibility helper used by OMS factories."""
    manager = PaperEquityManager(
        pool,
        account_scope=account_scope,
        initial_equity=initial_equity,
    )
    return await manager.load()


async def apply_paper_pnl(
    pool: asyncpg.Pool,
    pnl: float,
    commission: float = 0.0,
    account_scope: str = "paper",
    initial_equity: float = 10_000.0,
) -> float:
    """Compatibility helper used by OMS factories."""
    manager = PaperEquityManager(
        pool,
        account_scope=account_scope,
        initial_equity=initial_equity,
    )
    await manager.load()
    return await manager.apply_pnl(pnl, commission)
