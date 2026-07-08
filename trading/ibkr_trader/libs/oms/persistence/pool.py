"""Database connection pool manager."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import asyncpg

from .db_config import DBConfig

logger = logging.getLogger(__name__)


class DatabasePool:
    """Manages asyncpg connection pool lifecycle."""

    @classmethod
    async def create(
        cls,
        config: DBConfig,
        max_retries: int = 3,
        retry_delay_s: float = 2.0,
    ) -> asyncpg.Pool:
        """Create a connection pool with retry logic.

        Args:
            config: Database configuration
            max_retries: Maximum number of connection attempts
            retry_delay_s: Delay between retries in seconds

        Returns:
            asyncpg.Pool instance

        Raises:
            ConnectionError: If all connection attempts fail
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                pool = await asyncpg.create_pool(
                    dsn=config.to_dsn(),
                    min_size=config.pool_min,
                    max_size=config.pool_max,
                    command_timeout=30,
                    statement_cache_size=100,
                    server_settings={"timezone": "America/New_York"},
                )
                logger.info(
                    f"Database pool created: {config.host}:{config.port}/{config.database} "
                    f"(min={config.pool_min}, max={config.pool_max})"
                )
                return pool

            except asyncpg.InvalidCatalogNameError as e:
                # Database doesn't exist - don't retry
                logger.error(f"Database '{config.database}' does not exist")
                raise ConnectionError(f"Database '{config.database}' does not exist") from e

            except asyncpg.InvalidPasswordError as e:
                # Bad credentials - don't retry
                logger.error(f"Invalid database credentials for user '{config.user}'")
                raise ConnectionError(f"Invalid database credentials") from e

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Database connection attempt {attempt}/{max_retries} failed: {e}. "
                        f"Retrying in {retry_delay_s}s..."
                    )
                    await asyncio.sleep(retry_delay_s)
                else:
                    logger.error(f"All database connection attempts failed: {e}")

        raise ConnectionError(
            f"Failed to connect to database after {max_retries} attempts"
        ) from last_error

    @classmethod
    async def close(cls, pool: asyncpg.Pool) -> None:
        """Close the connection pool gracefully.

        Args:
            pool: The pool to close
        """
        if pool is None:
            return

        try:
            await pool.close()
            logger.info("Database pool closed")
        except Exception as e:
            logger.warning(f"Error closing database pool: {e}")

    @classmethod
    async def health_check(cls, pool: asyncpg.Pool) -> bool:
        """Check if the pool is healthy.

        Args:
            pool: The pool to check

        Returns:
            True if healthy, False otherwise
        """
        if pool is None:
            return False

        try:
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False
