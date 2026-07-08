"""Load/save CrisisContext to JSON for live runtime consumption.

Mirrors regime/persistence.py pattern with atomic writes and staleness checks.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from regime.crisis import config as C
from regime.crisis.context import CrisisContext

logger = logging.getLogger(__name__)

CRISIS_CONTEXT_PATH = Path("data/crisis/latest_context.json")
HYSTERESIS_STATE_PATH = Path("data/crisis/hysteresis_state.json")

RECOVERY_DEFAULT = CrisisContext(
    alert_level=C.ALERT_NORMAL,
    alert_level_int=0,
    risk_multiplier=1.0,
    dd_tier_multiplier=1.0,
    computed_at="",
)


def load_crisis_context(path: Path = CRISIS_CONTEXT_PATH) -> CrisisContext:
    """Load from JSON, return RECOVERY_DEFAULT on any failure."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ctx = CrisisContext(**data)

        # Staleness check: revert to NORMAL if context too old
        computed_at = ctx.computed_at
        if computed_at:
            try:
                ts = datetime.fromisoformat(computed_at)
                age = datetime.now(timezone.utc) - ts
                if age.days > C.STALENESS_THRESHOLD_DAYS:
                    logger.warning(
                        "Crisis context is %d days old (computed_at=%s), "
                        "reverting to NORMAL (stale)",
                        age.days, computed_at,
                    )
                    return RECOVERY_DEFAULT
            except (ValueError, TypeError):
                logger.warning("Cannot parse computed_at=%r for staleness check", computed_at)

        data_as_of = getattr(ctx, "data_as_of", "")
        if data_as_of:
            try:
                as_of = datetime.fromisoformat(data_as_of).date()
                data_age = (datetime.now(timezone.utc).date() - as_of).days
                if data_age > C.STALENESS_THRESHOLD_DAYS:
                    logger.warning(
                        "Crisis data is %d days old (data_as_of=%s), "
                        "reverting to NORMAL (stale data)",
                        data_age, data_as_of,
                    )
                    return RECOVERY_DEFAULT
            except (ValueError, TypeError):
                logger.warning("Cannot parse data_as_of=%r for staleness check", data_as_of)

        return ctx

    except FileNotFoundError:
        logger.info("Crisis context file not found (%s), using default NORMAL", path)
        return RECOVERY_DEFAULT
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Crisis context file corrupt (%s): %s, using default NORMAL", path, exc)
        return RECOVERY_DEFAULT


def save_crisis_context(ctx: CrisisContext, path: Path = CRISIS_CONTEXT_PATH) -> None:
    """Persist to JSON atomically. Creates data/crisis/ directory if needed."""
    _atomic_json_write(dataclasses.asdict(ctx), path)


def load_hysteresis_state(path: Path = HYSTERESIS_STATE_PATH) -> dict | None:
    """Load hysteresis tracker state from JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def save_hysteresis_state(state: dict, path: Path = HYSTERESIS_STATE_PATH) -> None:
    """Persist hysteresis tracker state atomically."""
    _atomic_json_write(state, path)


def _atomic_json_write(data: dict, path: Path) -> None:
    """Write JSON atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
