"""Data directory resolution with legacy curated-path compatibility."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataDirs:
    raw_data_dir: Path
    curated_dir: Path
    legacy_curated_dir: Path
    used_legacy_curated: bool = False


def resolve_data_dirs(data_dir: str | Path) -> DataDirs:
    """Resolve raw/curated data directories.

    New layout is ``DATA_DIR/raw`` and ``DATA_DIR/curated``.  The older
    ``DATA_DIR/data/curated`` path is still read when it is the only curated
    directory present.
    """
    base = Path(data_dir)
    raw_data_dir = base / "raw"
    normalized_curated = base / "curated"
    legacy_curated = base / "data" / "curated"

    if legacy_curated.exists() and not normalized_curated.exists():
        logger.warning(
            "Using legacy curated data dir %s. Move it to %s to use the "
            "normalized DATA_DIR layout.",
            legacy_curated,
            normalized_curated,
        )
        return DataDirs(
            raw_data_dir=raw_data_dir,
            curated_dir=legacy_curated,
            legacy_curated_dir=legacy_curated,
            used_legacy_curated=True,
        )

    if legacy_curated.exists() and normalized_curated.exists():
        logger.warning(
            "Both normalized curated data dir %s and legacy dir %s exist; "
            "using normalized path.",
            normalized_curated,
            legacy_curated,
        )

    return DataDirs(
        raw_data_dir=raw_data_dir,
        curated_dir=normalized_curated,
        legacy_curated_dir=legacy_curated,
    )
