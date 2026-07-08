"""Bootstrap service for database and related services initialization.

Unified bootstrap shared across all trading families (swing, momentum, stock).
Environment detection uses TRADING_MODE with backward-compatible fallback
to legacy env vars (TRADING_ENV, SWING_TRADER_ENV, ALGO_TRADER_ENV, STOCK_TRADER_ENV).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import asyncpg

from ..oms.persistence.db_config import DBConfig, get_environment, is_db_required
from ..oms.persistence.pool import DatabasePool
from ..oms.persistence.postgres import PgStore
from .heartbeat import HeartbeatService
from .trade_recorder import TradeRecorder

logger = logging.getLogger(__name__)


@dataclass
class BootstrapContext:
    """Container for bootstrapped database services.

    All fields are Optional to support graceful degradation when
    database is not configured or available.
    """

    pool: Optional[asyncpg.Pool] = None
    pg_store: Optional[PgStore] = None
    heartbeat: Optional[HeartbeatService] = None
    trade_recorder: Optional[TradeRecorder] = None

    @property
    def has_db(self) -> bool:
        """Check if database services are available."""
        return self.pool is not None


async def bootstrap_database(
    require_db: Optional[bool] = None,
    init_schema: bool = True,
) -> BootstrapContext:
    """Bootstrap database and related services.

    This function handles graceful degradation:
    - In dev/backtest mode: Returns empty context (InMemory fallback)
    - In paper/live mode: Requires database connection

    Environment is detected via get_environment() which checks env vars
    in priority order:
        TRADING_MODE > TRADING_ENV > SWING_TRADER_ENV > ALGO_TRADER_ENV > STOCK_TRADER_ENV

    Args:
        require_db: Override automatic DB requirement detection.
                   If None, uses is_db_required() based on environment.
        init_schema: If True, initialize database schema (CREATE TABLE IF NOT EXISTS).

    Returns:
        BootstrapContext with database services or empty context for fallback.

    Raises:
        ConnectionError: If database is required but connection fails.
    """
    env = get_environment()
    db_required = require_db if require_db is not None else is_db_required()

    logger.info(f"Bootstrapping database: env={env}, db_required={db_required}")

    # Try to get database config
    db_config = DBConfig.from_env()

    if db_config is None:
        if db_required:
            raise ConnectionError(
                f"Database configuration required for env={env}. "
                "Set DATABASE_URL or DB_HOST environment variables."
            )
        logger.info("No database configured, using in-memory fallback")
        return BootstrapContext()

    # Try to connect
    try:
        pool = await DatabasePool.create(db_config)
    except ConnectionError:
        if db_required:
            raise
        logger.warning("Database connection failed, using in-memory fallback")
        return BootstrapContext()

    # Create PgStore and initialize schema
    pg_store = PgStore(pool)

    if init_schema:
        try:
            await pg_store.init_schema()
            logger.info("Database schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            if db_required:
                await DatabasePool.close(pool)
                raise
            # Always close pool before falling back to in-memory
            await DatabasePool.close(pool)
            logger.warning("Schema init failed, closed pool, using in-memory fallback")
            return BootstrapContext()

    # Create services
    heartbeat = HeartbeatService(pg_store)
    trade_recorder = TradeRecorder(pg_store)

    logger.info(
        f"Database bootstrap complete: "
        f"host={db_config.host}, db={db_config.database}"
    )

    return BootstrapContext(
        pool=pool,
        pg_store=pg_store,
        heartbeat=heartbeat,
        trade_recorder=trade_recorder,
    )


async def shutdown_database(ctx: BootstrapContext) -> None:
    """Shutdown database services gracefully.

    Args:
        ctx: The bootstrap context to shutdown
    """
    if ctx.pool is not None:
        await DatabasePool.close(ctx.pool)
        logger.info("Database shutdown complete")
