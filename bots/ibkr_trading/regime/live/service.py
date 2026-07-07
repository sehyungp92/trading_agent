"""RegimeService: cross-family regime signal service. Computes weekly, exposes RegimeContext."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from regime.config import MetaConfig, REGIMES
from regime.context import RegimeContext

from .provider import LiveDataProvider

logger = logging.getLogger(__name__)

try:
    import zoneinfo
    _ET = zoneinfo.ZoneInfo("America/New_York")
except Exception:
    _ET = timezone(timedelta(hours=-5))

_DEFAULT_DATA_DIR = Path(os.environ.get("REGIME_DATA_DIR", "data/regime/raw"))


def _next_friday_1630(now_et: datetime, market_cal: Any | None) -> datetime:
    """Find next Friday at 16:30 ET, skipping holidays."""
    # Start from today
    target_date = now_et.date()

    # Find next Friday (or today if Friday and before 16:30)
    days_until_friday = (4 - target_date.weekday()) % 7
    if days_until_friday == 0:
        # It's Friday -- check if we're past 16:30
        cutoff = now_et.replace(hour=16, minute=30, second=0, microsecond=0)
        if now_et >= cutoff:
            days_until_friday = 7  # next Friday
    target_date = target_date + timedelta(days=days_until_friday)

    # Skip holidays
    if market_cal is not None:
        max_skips = 10
        while max_skips > 0 and not market_cal.is_trading_day(target_date):
            target_date += timedelta(days=7)  # skip to next Friday
            max_skips -= 1

    # Build target datetime at 16:30 ET
    target = datetime(
        target_date.year, target_date.month, target_date.day,
        16, 30, 0, tzinfo=_ET,
    )
    return target


_REQUIRED_SIGNAL_COLS = {"P_G", "P_R", "P_S", "P_D", "Conf", "L"}


def _row_to_context(
    row: pd.Series,
    computed_at: str = "",
    *,
    data_as_of: str = "",
    data_status: str = "",
) -> RegimeContext:
    """Convert a signals DataFrame row to RegimeContext."""
    posteriors = [row["P_G"], row["P_R"], row["P_S"], row["P_D"]]
    dominant_idx = int(np.argmax(posteriors))
    dominant = REGIMES[dominant_idx] if dominant_idx < len(REGIMES) else "G"

    sleeves = ["SPY", "EFA", "TLT", "GLD", "CASH"]
    allocations = {s: float(row.get(f"w_{s}", 0.0)) for s in sleeves}

    # stress_velocity = rate of change in stress_level (stress HMM, enabled by default)
    # shift_velocity = scanner shift probability velocity (scanner, disabled by default)
    velocity = float(row.get("stress_velocity", row.get("shift_velocity", 0.0)))

    return RegimeContext(
        regime=dominant,
        regime_confidence=float(row["Conf"]),
        stress_level=float(row.get("stress_level", 0.0)),
        stress_onset=bool(row.get("stress_onset", False)),
        shift_velocity=velocity,
        suggested_leverage_mult=float(row["L"]),
        regime_allocations=allocations,
        computed_at=computed_at,
        data_as_of=data_as_of,
        data_status=data_status,
    )


class RegimeService:
    """Cross-family regime signal service. Computes weekly, exposes RegimeContext."""

    def __init__(
        self,
        ib_session: Any,
        cfg: MetaConfig | None = None,
        data_dir: Path | None = None,
        market_calendar: Any | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._now_provider = now_provider
        self._provider = LiveDataProvider(
            ib_session,
            data_dir or _DEFAULT_DATA_DIR,
            now_provider=now_provider,
        )
        self._cfg = cfg or MetaConfig()
        self._market_cal = market_calendar
        self._context: RegimeContext | None = None
        self._signals_df: pd.DataFrame | None = None
        self._last_compute: datetime | None = None
        self._compute_lock = asyncio.Lock()
        self._scheduler_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Qualify contracts, compute initial signal, launch weekly scheduler."""
        await self._provider.qualify_contracts()
        await self.compute_now()
        self._running = True
        self._scheduler_task = asyncio.create_task(self._weekly_scheduler())

    async def stop(self) -> None:
        """Cancel scheduler."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

    def get_context(self) -> RegimeContext | None:
        """Thread-safe getter. Returns None only before first computation."""
        return self._context

    async def compute_now(self) -> RegimeContext:
        """Force an immediate regime computation and return the fresh context."""
        async with self._compute_lock:
            await self._compute_signal()
        if self._context is None:
            raise RuntimeError("RegimeService compute_now did not produce a context")
        return self._context

    @property
    def signals_df(self) -> pd.DataFrame | None:
        """Full weekly signals DataFrame for diagnostics."""
        return self._signals_df

    @property
    def last_compute(self) -> datetime | None:
        """Timestamp of last successful computation."""
        return self._last_compute

    # ------------------------------------------------------------------
    # Weekly scheduler
    # ------------------------------------------------------------------

    async def _weekly_scheduler(self) -> None:
        """Sleep until Friday 16:30 ET, then recompute."""
        while self._running:
            now_et = self._now().astimezone(_ET)
            target = _next_friday_1630(now_et, self._market_cal)
            wait = (target - now_et).total_seconds()
            logger.info(
                "Regime: next computation scheduled for %s ET (%.1f hours)",
                target.strftime("%Y-%m-%d %H:%M"), wait / 3600,
            )
            await asyncio.sleep(max(1, wait))
            if not self._running:
                break
            try:
                await self.compute_now()
            except Exception:
                logger.exception("RegimeService: weekly computation failed, keeping previous context")

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    async def _compute_signal(self) -> None:
        """Fetch data, run engine, publish RegimeContext."""
        from regime.engine import run_signal_engine

        logger.info("Regime: computing signal ...")

        # 1. Fetch data (IBKR async + FRED in executor)
        macro_df, market_df, strat_ret_df = await self._provider.build_live_data()

        # 2. Run engine in executor (CPU-heavy: HMM fitting, sklearn LedoitWolf)
        loop = asyncio.get_running_loop()
        signals_df = await loop.run_in_executor(
            None,
            lambda: run_signal_engine(
                macro_df, strat_ret_df, market_df,
                growth_feature="GROWTH",
                inflation_feature="INFLATION",
                cfg=self._cfg,
            ),
        )

        # 3. Validate and extract last row -> RegimeContext
        if signals_df is None or signals_df.empty:
            raise ValueError("run_signal_engine returned empty DataFrame")
        missing = _REQUIRED_SIGNAL_COLS - set(signals_df.columns)
        if missing:
            raise ValueError(f"signals_df missing required columns: {missing}")

        self._signals_df = signals_df
        now = self._now()
        _prev_ctx = self._context
        self._context = _row_to_context(
            signals_df.iloc[-1],
            computed_at=now.isoformat(),
            data_as_of=self._provider.last_data_as_of,
            data_status=self._provider.last_data_status,
        )
        self._last_compute = now
        logger.info(
            "Regime signal computed: %s "
            "(conf=%.2f, stress=%.2f, leverage=%.2f, data_as_of=%s)",
            self._context.regime,
            self._context.regime_confidence,
            self._context.stress_level,
            self._context.suggested_leverage_mult,
            self._context.data_as_of or "unknown",
        )

        # 4. Persist to disk for diagnostics and restart resilience
        try:
            from regime.persistence import save_regime_context
            save_regime_context(self._context)
        except Exception:
            logger.warning("Regime: failed to persist context to disk", exc_info=True)

        # 5. Detect regime transition
        if _prev_ctx is not None and _prev_ctx.regime != self._context.regime:
            self._emit_transition(_prev_ctx, self._context)

    def _emit_transition(self, prev: RegimeContext, curr: RegimeContext) -> None:
        """Write RegimeTransitionEvent to JSONL when macro regime changes."""
        try:
            import json
            from dataclasses import asdict
            from regime.events import RegimeTransitionEvent
            event = RegimeTransitionEvent(
                from_regime=prev.regime,
                to_regime=curr.regime,
                regime_confidence=curr.regime_confidence,
                stress_level=curr.stress_level,
                stress_onset=curr.stress_onset,
                shift_velocity=curr.shift_velocity,
                timestamp=curr.computed_at,
            )
            out_dir = Path("data/regime")
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "transitions.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event), default=str) + "\n")
            logger.info("Regime transition: %s -> %s", prev.regime, curr.regime)
        except Exception:
            logger.warning("Failed to emit RegimeTransitionEvent", exc_info=True)

    def _now(self) -> datetime:
        now = self._now_provider() if self._now_provider is not None else datetime.now(timezone.utc)
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now

    def __repr__(self) -> str:
        ctx = self._context
        if ctx is None:
            return "RegimeService(not yet computed)"
        return (
            f"RegimeService(regime={ctx.regime}, conf={ctx.regime_confidence:.2f}, "
            f"stress={ctx.stress_level:.2f}, leverage={ctx.suggested_leverage_mult:.2f})"
        )
