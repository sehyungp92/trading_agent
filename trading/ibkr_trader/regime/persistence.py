"""Load/save RegimeContext to JSON for live runtime consumption."""
from __future__ import annotations

import dataclasses
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from regime.context import RegimeContext

logger = logging.getLogger(__name__)

REGIME_CONTEXT_PATH = Path("data/regime/latest_context.json")

RECOVERY_DEFAULT = RegimeContext(
    regime="S",
    regime_confidence=0.5,
    stress_level=0.0,
    stress_onset=False,
    shift_velocity=0.0,
    suggested_leverage_mult=0.75,
    regime_allocations={
        "SPY": 0.15, "TLT": 0.10, "GLD": 0.10,
        "EFA": 0.05, "CASH": 0.60,
    },
    computed_at="",
)

_STALENESS_DOWNGRADE_DAYS = 7  # downgrade to S if context older than this
_DATA_AS_OF_STALENESS_DAYS = 10


def load_regime_context(path: Path = REGIME_CONTEXT_PATH) -> RegimeContext:
    """Load from JSON, return RECOVERY_DEFAULT on any failure."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ctx = RegimeContext(**data)

        # Staleness check: downgrade to S if context too old
        computed_at = ctx.computed_at
        if computed_at:
            try:
                ts = datetime.fromisoformat(computed_at)
                age = datetime.now(timezone.utc) - ts
                if age.days > _STALENESS_DOWNGRADE_DAYS:
                    logger.warning(
                        "Regime context is %d days old (computed_at=%s), "
                        "downgrading to defensive regime S",
                        age.days, computed_at,
                    )
                    ctx = dataclasses.replace(ctx, regime="S")
            except (ValueError, TypeError):
                logger.warning("Cannot parse computed_at=%r for staleness check", computed_at)

        data_as_of = getattr(ctx, "data_as_of", "")
        if data_as_of:
            try:
                as_of = datetime.fromisoformat(data_as_of).date()
                data_age = (datetime.now(timezone.utc).date() - as_of).days
                if data_age > _DATA_AS_OF_STALENESS_DAYS:
                    logger.warning(
                        "Regime data is %d days old (data_as_of=%s), "
                        "downgrading to defensive regime S",
                        data_age, data_as_of,
                    )
                    ctx = dataclasses.replace(ctx, regime="S", data_status="stale_data_as_of")
            except (ValueError, TypeError):
                logger.warning("Cannot parse data_as_of=%r for staleness check", data_as_of)

        return ctx

    except FileNotFoundError:
        logger.warning("Regime context file not found (%s), using Recovery default", path)
        return RECOVERY_DEFAULT
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Regime context file corrupt (%s): %s, using Recovery default", path, exc)
        return RECOVERY_DEFAULT


def save_regime_context(ctx: RegimeContext, path: Path = REGIME_CONTEXT_PATH) -> None:
    """Persist to JSON atomically. Creates data/regime/ directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dataclasses.asdict(ctx)
    # Atomic write: temp file + rename prevents partial reads
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
