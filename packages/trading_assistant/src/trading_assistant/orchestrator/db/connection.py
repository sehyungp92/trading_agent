"""SQLite connection factory with WAL mode for concurrent reads."""

from pathlib import Path

import aiosqlite

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


async def create_connection(db_path: str) -> aiosqlite.Connection:
    """Create and initialize a SQLite connection with WAL mode."""
    db = await aiosqlite.connect(db_path)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    # P1-9: wait up to 5s on writer-lock contention rather than failing fast.
    await db.execute("PRAGMA busy_timeout=5000")
    db.row_factory = aiosqlite.Row
    return db


async def initialize_schema(db: aiosqlite.Connection) -> None:
    """Run the schema.sql file to create tables."""
    schema = _SCHEMA_PATH.read_text()
    await db.executescript(schema)
    await db.commit()
