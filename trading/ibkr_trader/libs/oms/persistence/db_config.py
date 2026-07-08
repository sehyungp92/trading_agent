"""Database configuration for OMS persistence."""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class DBConfig:
    """Database connection configuration."""

    host: str
    port: int
    database: str
    user: str
    password: str
    pool_min: int = 2
    pool_max: int = 10

    @classmethod
    def from_env(cls) -> Optional["DBConfig"]:
        """Load database config from environment variables.

        Supports two modes:
        1. DATABASE_URL: postgresql://user:pass@host:port/database
        2. Individual vars: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

        Returns None if no database configuration is found.
        """
        database_url = os.getenv("DATABASE_URL")

        if database_url:
            return cls._from_url(database_url)

        # Try individual environment variables
        host = os.getenv("DB_HOST")
        if not host:
            return None

        try:
            return cls(
                host=host,
                port=int(os.getenv("DB_PORT", "5432")),
                database=os.getenv("DB_NAME", "trading"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", ""),
                pool_min=int(os.getenv("DB_POOL_MIN", "2")),
                pool_max=int(os.getenv("DB_POOL_MAX", "10")),
            )
        except ValueError as e:
            logger.error(f"Invalid DB config from environment: {e}")
            return None

    @classmethod
    def _from_url(cls, url: str) -> Optional["DBConfig"]:
        """Parse DATABASE_URL format."""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("postgresql", "postgres"):
                logger.warning(f"Unsupported database scheme: {parsed.scheme}")
                return None

            return cls(
                host=parsed.hostname or "localhost",
                port=parsed.port or 5432,
                database=parsed.path.lstrip("/") or "trading",
                user=parsed.username or "postgres",
                password=parsed.password or "",
                pool_min=int(os.getenv("DB_POOL_MIN", "2")),
                pool_max=int(os.getenv("DB_POOL_MAX", "10")),
            )
        except Exception as e:
            logger.error(f"Failed to parse DATABASE_URL: {e}")
            return None

    def to_dsn(self) -> str:
        """Convert to asyncpg DSN format."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


# Env-var names checked in priority order (legacy fallbacks kept)
_ENV_VAR_CANDIDATES = (
    "TRADING_MODE",         # canonical (shared with IB Gateway)
    "TRADING_ENV",          # legacy alias (deprecated)
    "SWING_TRADER_ENV",     # legacy swing_trader
    "ALGO_TRADER_ENV",      # legacy momentum_trader / stock_trader
    "STOCK_TRADER_ENV",     # legacy stock_trader alias
)


def get_environment() -> str:
    """Get current environment.

    Checks env vars in priority order:
        TRADING_MODE > TRADING_ENV > SWING_TRADER_ENV > ALGO_TRADER_ENV > STOCK_TRADER_ENV

    Returns one of: dev, backtest, paper, live
    Default is 'dev'.
    """
    for var in _ENV_VAR_CANDIDATES:
        val = os.getenv(var)
        if val:
            return val.lower()
    return "dev"


def is_db_required() -> bool:
    """Check if database is required for current environment.

    Database is required for 'live' and 'paper' environments.
    """
    env = get_environment()
    return env in ("live", "paper")
