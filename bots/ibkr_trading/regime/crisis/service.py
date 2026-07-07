"""CrisisService: daily async crisis detection service.

Mirrors RegimeService pattern but runs daily instead of weekly.
Fetches 3 FRED series + 2 IBKR ETF bars, computes indicators,
applies conjunction logic + hysteresis, persists context.
"""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import pandas as pd

from regime.crisis import config as C
from regime.crisis.actions import stress_formation_risk_multiplier
from regime.crisis.context import CrisisContext, _risk_mult_for_level, _dd_mult_for_level
from regime.crisis.detector import (
    compute_alert_level,
    compute_advisory_level,
    is_hard_credit_impulse_warning_candidate,
)
from regime.crisis.hysteresis import HysteresisTracker
from regime.crisis.indicators import compute_indicators
from regime.crisis.persistence import (
    load_crisis_context,
    load_hysteresis_state,
    save_crisis_context,
    save_hysteresis_state,
)

logger = logging.getLogger(__name__)

try:
    import zoneinfo
    _ET = zoneinfo.ZoneInfo("America/New_York")
except Exception:
    _ET = timezone(timedelta(hours=-5))

_DEFAULT_DATA_DIR = Path(os.environ.get("CRISIS_DATA_DIR", "data/crisis"))
_DEFAULT_REGIME_DATA_DIR = Path(os.environ.get("REGIME_DATA_DIR", "data/regime/raw"))
_CRISIS_COMPUTE_HOUR_ET = 17
_CRISIS_COMPUTE_MINUTE_ET = 5

# FRED series needed (subset of regime's full set)
_CRISIS_FRED_SERIES = {
    "VIX": "VIXCLS",
    "SPREAD": "BAMLH0A0HYM2",
    "SLOPE_10Y2Y": "T10Y2Y",
    "VIX3M": "VIX3M",
}

# IBKR ETFs needed for correlation computation
_CRISIS_ETF_SYMBOLS = ["SPY", "TLT"]
_CRISIS_MIN_RETURN_OBS = max(C.CORR_WINDOW, 20)
_CRISIS_MIN_MARKET_OBS = max(C.SLOPE_INVERSION_LOOKBACK, 21, C.GRIND_VIX_PERSIST_DAYS)
_CRISIS_MAX_RETURN_LAG_SESSIONS = C.STALENESS_THRESHOLD_DAYS


def _is_trading_day(day: date, market_cal: Any | None) -> bool:
    if market_cal is not None:
        try:
            return bool(market_cal.is_trading_day(day))
        except Exception:
            logger.warning("Crisis: market calendar failed for %s", day, exc_info=True)
    return day.weekday() < 5


def _next_crisis_compute_after_close(
    now_et: datetime,
    market_cal: Any | None = None,
) -> datetime:
    """Return the next daily crisis compute time after US equity close."""
    target_date = now_et.date()
    target = now_et.replace(
        hour=_CRISIS_COMPUTE_HOUR_ET,
        minute=_CRISIS_COMPUTE_MINUTE_ET,
        second=0,
        microsecond=0,
    )
    if not _is_trading_day(target_date, market_cal) or now_et >= target:
        target_date += timedelta(days=1)
        while not _is_trading_day(target_date, market_cal):
            target_date += timedelta(days=1)
        target = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            _CRISIS_COMPUTE_HOUR_ET,
            _CRISIS_COMPUTE_MINUTE_ET,
            tzinfo=_ET,
        )
    return target


class CrisisService:
    """Daily crisis detection service. Computes daily, exposes CrisisContext."""

    def __init__(
        self,
        ib_session: Any,
        data_dir: Path | None = None,
        regime_data_dir: Path | None = None,
        market_calendar: Any | None = None,
        compute_on_start: bool = True,
        auto_schedule: bool = True,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._session = ib_session
        self._data_dir = data_dir or _DEFAULT_DATA_DIR
        self._regime_data_dir = regime_data_dir or _DEFAULT_REGIME_DATA_DIR
        self._market_cal = market_calendar
        self._compute_on_start = compute_on_start
        self._auto_schedule = auto_schedule
        self._now_provider = now_provider
        self._contracts: dict[str, Any] = {}
        self._context: CrisisContext | None = None
        self._tracker: HysteresisTracker | None = None
        self._last_compute: datetime | None = None
        self._compute_lock = asyncio.Lock()
        self._listeners: list[Callable[[CrisisContext], Awaitable[None] | None]] = []
        self._scheduler_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Qualify contracts, load persisted state, compute initial signal, launch scheduler."""
        await self._qualify_contracts()

        # Load persisted hysteresis state
        hyst_state = load_hysteresis_state(self._data_dir / "hysteresis_state.json")
        if hyst_state is not None:
            self._tracker = HysteresisTracker.from_dict(hyst_state)
            logger.info("Hysteresis state loaded: level=%d, days_below=%d",
                        self._tracker.current_level, self._tracker.days_below)
        else:
            self._tracker = HysteresisTracker()

        # Load persisted context as initial (don't compute on startup to avoid
        # leaving downstream callers without a context if startup occurs before
        # the next completed daily bar boundary.
        self._context = load_crisis_context(self._data_dir / "latest_context.json")
        logger.info("Crisis persisted context loaded: %s", self._context.alert_level)

        if self._compute_on_start:
            await self.compute_now(notify=False, require_live_backlog=True)

        self._running = True
        if self._auto_schedule:
            self._scheduler_task = asyncio.create_task(self._daily_scheduler())
        logger.info("Crisis service started: %s", self._context.alert_level)

    async def stop(self) -> None:
        """Cancel scheduler."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

    def get_context(self) -> CrisisContext | None:
        """Thread-safe getter. Returns None only before first load."""
        return self._context

    @property
    def last_compute(self) -> datetime | None:
        """Timestamp of last successful live computation."""
        return self._last_compute

    def add_listener(
        self,
        callback: Callable[[CrisisContext], Awaitable[None] | None],
    ) -> None:
        """Register a callback invoked after every successful live computation."""
        self._listeners.append(callback)

    async def compute_now(
        self,
        *,
        notify: bool = True,
        require_live_backlog: bool = False,
    ) -> CrisisContext:
        """Force a crisis detection computation. Called by scheduler or manually."""
        async with self._compute_lock:
            return await self._compute_now_locked(
                notify=notify,
                require_live_backlog=require_live_backlog,
            )

    async def _compute_now_locked(
        self,
        *,
        notify: bool = True,
        require_live_backlog: bool = False,
    ) -> CrisisContext:
        market_df, strat_ret_df, vix3m_series, data_status = await self._fetch_data()

        if market_df.empty or strat_ret_df.empty:
            return self._unavailable_context(
                "Crisis: insufficient data for computation after cache + IBKR/FRED backfill",
                require_live_backlog,
            )

        compute_date = _latest_return_as_of(strat_ret_df)
        if compute_date is None:
            return self._unavailable_context(
                "Crisis: no completed SPY/TLT return date available after backfill",
                require_live_backlog,
            )

        data_status = _append_data_lag_status(data_status, market_df, compute_date)
        data_status, backlog_ok = _append_backlog_status(
            data_status,
            market_df,
            strat_ret_df,
            compute_date,
        )
        data_status, return_lag_sessions = _append_return_lag_status(
            data_status,
            compute_date,
            self._now(),
            self._market_cal,
        )
        if not backlog_ok:
            return self._unavailable_context(
                "Crisis: insufficient live backlog for computation "
                f"(data_as_of={compute_date.date().isoformat()}, status={data_status})",
                require_live_backlog,
            )
        if return_lag_sessions > _CRISIS_MAX_RETURN_LAG_SESSIONS:
            return self._unavailable_context(
                "Crisis: completed SPY/TLT returns are stale "
                f"(data_as_of={compute_date.date().isoformat()}, "
                f"lag_sessions={return_lag_sessions}, status={data_status})",
                require_live_backlog,
            )

        indicators = compute_indicators(
            market_df,
            strat_ret_df,
            date=compute_date,
            vix3m_series=vix3m_series,
        )
        _, raw_level_int = compute_alert_level(indicators)
        bridge_candidate = is_hard_credit_impulse_warning_candidate(indicators)

        # Apply hysteresis
        if self._tracker is None:
            self._tracker = HysteresisTracker()
        raw_level_int = self._tracker.apply_hard_credit_impulse_bridge(
            raw_level_int,
            bridge_candidate,
        )
        final_level_int = self._tracker.update(raw_level_int)
        final_level_str = C.ALERT_LEVELS[final_level_int]
        advisory_level_str, advisory_level_int, advisory_reason = compute_advisory_level(
            indicators,
            final_level_int,
        )
        formation_risk_mult = stress_formation_risk_multiplier(
            indicators.stress_formation_mode,
            indicators.stress_formation_score,
        )
        portfolio_action_level_int = (
            final_level_int if final_level_int >= 2
            else 1 if formation_risk_mult < 1.0
            else 0
        )
        portfolio_action_level_str = C.ALERT_LEVELS[portfolio_action_level_int]

        now = self._now()
        prev_ctx = self._context
        ctx = CrisisContext(
            alert_level=final_level_str,
            alert_level_int=final_level_int,
            advisory_level=advisory_level_str,
            advisory_level_int=advisory_level_int,
            advisory_reason=advisory_reason,
            portfolio_action_level=portfolio_action_level_str,
            portfolio_action_level_int=portfolio_action_level_int,
            risk_multiplier=(
                _risk_mult_for_level(final_level_int)
                if final_level_int >= 2 else formation_risk_mult
            ),
            dd_tier_multiplier=_dd_mult_for_level(final_level_int),
            vix_level=indicators.vix.value,
            credit_spread_bps=indicators.credit_spread.value,
            yield_curve_slope=indicators.yield_curve.value,
            yield_curve_20d_change=indicators.yield_curve_20d_change,
            spy_3d_return=indicators.spy_3d_return,
            spy_5d_return=indicators.spy_5d_return,
            spy_tlt_correlation=indicators.spy_tlt_corr.value,
            spy_10d_return=indicators.spy_10d_return,
            spy_20d_return=indicators.spy_20d_return,
            vix_3d_change=indicators.vix_3d_change,
            credit_spread_20d_change_bps=indicators.credit_spread_20d_change_bps,
            stress_formation_score=indicators.stress_formation_score,
            stress_formation_mode=indicators.stress_formation_mode,
            stress_formation_reason=indicators.stress_formation_reason,
            vix_term_structure_ratio=(
                indicators.vix_term_structure.value
                if indicators.vix_term_structure else 0.0
            ),
            spy_10d_drawdown=(
                indicators.spy_drawdown.value
                if indicators.spy_drawdown else 0.0
            ),
            primary_watch_count=indicators.watch_count,
            primary_warning_count=indicators.warning_count,
            primary_crisis_count=indicators.crisis_count,
            recovery_ramp_mult=self._tracker.recovery_ramp_mult,
            computed_at=now.isoformat(),
            data_as_of=compute_date.date().isoformat(),
            data_status=data_status,
            days_at_current_level=self._tracker.days_at_level,
            dominant_channel=indicators.dominant_channel,
        )

        self._context = ctx
        self._last_compute = now

        # Persist
        save_crisis_context(ctx, self._data_dir / "latest_context.json")
        save_hysteresis_state(self._tracker.to_dict(), self._data_dir / "hysteresis_state.json")
        self._emit_transition(prev_ctx, ctx)

        logger.info(
            "Crisis detection computed: internal=%s advisory=%s action=%s "
            "(data_as_of=%s, VIX=%.1f, spread=%.0fbps, watch=%d, warn=%d, crisis=%d)",
            final_level_str, advisory_level_str, portfolio_action_level_str,
            ctx.data_as_of or "unknown",
            indicators.vix.value,
            indicators.credit_spread.value,
            indicators.watch_count, indicators.warning_count,
            indicators.crisis_count,
        )

        if notify:
            await self._notify_listeners(ctx)
        return ctx

    async def _daily_scheduler(self) -> None:
        """Run crisis detection once per trading day after completed daily bars."""
        while self._running:
            now_et = self._now().astimezone(_ET)
            target = _next_crisis_compute_after_close(now_et, self._market_cal)
            wait = (target - now_et).total_seconds()
            logger.info(
                "Crisis: next computation scheduled for %s ET (%.1f hours)",
                target.strftime("%Y-%m-%d %H:%M"),
                wait / 3600,
            )
            try:
                await asyncio.sleep(max(1, wait))
            except asyncio.CancelledError:
                return
            if not self._running:
                return
            try:
                await self.compute_now()
            except Exception as exc:
                logger.error("Crisis daily computation failed: %s", exc, exc_info=True)

    async def _qualify_contracts(self) -> None:
        """Qualify ETF contracts for SPY and TLT."""
        from ib_async import Stock

        for sym in _CRISIS_ETF_SYMBOLS:
            try:
                contract = Stock(sym, "SMART", "USD")
                qualified = await self._session.ib.qualifyContractsAsync(contract)
                if qualified:
                    self._contracts[sym] = qualified[0]
            except Exception as e:
                logger.warning("Crisis: could not qualify %s: %s", sym, e)

        logger.info(
            "Crisis: qualified %d/%d ETF contracts",
            len(self._contracts), len(_CRISIS_ETF_SYMBOLS),
        )

    async def _fetch_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, str]:
        """Fetch cached baseline plus fresh FRED/IBKR data for live crisis signals.

        This mirrors the HMM live provider's cache-plus-overlay pattern while
        staying lightweight: the crisis layer only needs market FRED series and
        completed daily SPY/TLT returns.
        """
        market_df, strat_ret_df, seed_status = self._load_cached_inputs()
        cached_return_as_of = _latest_return_as_of(strat_ret_df)
        ibkr_duration = _overlay_duration_str(
            cached_return_as_of,
            self._now(),
            max_days=180,
        )
        fred_start = _overlay_start_date(
            cached_return_as_of,
            self._now(),
            default_days=180,
        )
        status_parts: list[str] = [
            seed_status,
            f"ibkr_window={ibkr_duration.replace(' ', '')}",
            f"fred_start={fred_start}",
        ]

        loop = asyncio.get_running_loop()
        try:
            fred_df = await loop.run_in_executor(
                None,
                lambda: _call_sync_with_optional_kwargs(
                    self._fetch_fred,
                    start=fred_start,
                ),
            )
            if fred_df is not None and not fred_df.empty:
                market_cols = ["VIX", "SPREAD", "SLOPE_10Y2Y"]
                fresh_market_cols = [
                    col for col in market_cols
                    if col in fred_df.columns and not fred_df[col].dropna().empty
                ]
                if fresh_market_cols:
                    market_df = _overlay_columns(market_df, fred_df, market_cols)
                    status_parts.append("fred=fresh")
                else:
                    status_parts.append("fred=cached")
            else:
                status_parts.append("fred=cached")
        except Exception:
            logger.warning("Crisis: FRED fetch failed, using cached market data", exc_info=True)
            status_parts.append("fred=cached")
            fred_df = None

        try:
            fresh_returns = await _call_async_with_optional_kwargs(
                self._fetch_etf_returns,
                duration_str=ibkr_duration,
            )
            if fresh_returns is not None and not fresh_returns.empty:
                strat_ret_df = _overlay_columns(strat_ret_df, fresh_returns, _CRISIS_ETF_SYMBOLS)
                status_parts.append("ibkr=fresh_completed")
            else:
                status_parts.append("ibkr=cached")
        except Exception:
            logger.warning("Crisis: IBKR fetch failed, using cached ETF returns", exc_info=True)
            status_parts.append("ibkr=cached")

        market_df = market_df.sort_index().ffill()
        strat_ret_df = strat_ret_df.sort_index()
        vix3m_series = None
        if fred_df is not None and "VIX3M" in fred_df.columns:
            vix3m_series = fred_df["VIX3M"].dropna()

        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            market_df.to_parquet(self._data_dir / "market_df.parquet")
            strat_ret_df.to_parquet(self._data_dir / "strat_ret_df.parquet")
        except Exception:
            logger.warning("Crisis: failed to update parquet cache", exc_info=True)

        return market_df, strat_ret_df, vix3m_series, ";".join(status_parts)

    def _fetch_fred(self, start: str | None = None) -> pd.DataFrame | None:
        """Fetch recent FRED data for VIX, SPREAD, SLOPE_10Y2Y."""
        key = os.environ.get("FRED_API_KEY", "")
        if not key:
            logger.warning("Crisis: FRED_API_KEY not set, skipping FRED fetch")
            return None

        try:
            from fredapi import Fred
            fred = Fred(api_key=key)
        except ImportError:
            logger.warning("Crisis: fredapi not installed")
            return None

        if start is None:
            start = (self._now() - timedelta(days=180)).strftime("%Y-%m-%d")

        frames: dict[str, pd.Series] = {}
        for col, series_id in _CRISIS_FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=start)
                s.name = col
                frames[col] = s
            except Exception as exc:
                logger.warning("Crisis: FRED fetch failed for %s: %s", col, exc)

        if not frames:
            return None

        df = pd.DataFrame(frames)
        df.index = pd.DatetimeIndex(df.index)
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
        df = df.ffill()
        return df

    async def _fetch_etf_returns(self, duration_str: str = "180 D") -> pd.DataFrame:
        """Fetch daily bars for SPY and TLT, compute returns."""
        frames: dict[str, pd.Series] = {}

        for sym in _CRISIS_ETF_SYMBOLS:
            contract = self._contracts.get(sym)
            if contract is None:
                continue
            try:
                bars = await self._session.req_historical_data(
                    contract,
                    endDateTime="",
                    durationStr=duration_str,
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    request_kind="quick",
                    completed_only=True,
                    as_of=self._now(),
                )
                if bars:
                    dates = pd.to_datetime([b.date for b in bars])
                    closes = pd.Series([b.close for b in bars], index=dates, name=sym, dtype=float)
                    if closes.index.tz is not None:
                        closes.index = closes.index.tz_localize(None)
                    frames[sym] = closes.pct_change().dropna()
            except Exception as exc:
                logger.warning("Crisis: IBKR bars fetch failed for %s: %s", sym, exc)

        if not frames:
            return pd.DataFrame()

        return pd.DataFrame(frames).dropna()

    def _load_cached_inputs(self) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        from regime.seed_manifest import bootstrap_seed_data_dir, validate_seed_data_dir

        seed_dir_raw = os.environ.get("REGIME_SEED_DIR", "").strip()
        seed_status = bootstrap_seed_data_dir(
            self._regime_data_dir,
            Path(seed_dir_raw) if seed_dir_raw else None,
        )
        require_manifest = os.environ.get("REGIME_REQUIRE_SEED_MANIFEST", "0").strip().lower()
        manifest_required = require_manifest in {"1", "true", "yes", "on"}
        should_validate_seed = (
            bool(seed_dir_raw)
            or manifest_required
            or any((self._regime_data_dir / name).exists() for name in (
                "macro_df.parquet",
                "market_df.parquet",
                "strat_ret_df.parquet",
            ))
        )
        if should_validate_seed:
            seed_ok, manifest_status, _ = validate_seed_data_dir(
                self._regime_data_dir,
                require_manifest=manifest_required,
                validate_hashes=True,
            )
            if not seed_ok:
                raise RuntimeError(
                    "Crisis seed manifest validation failed before live overlay: "
                    f"{manifest_status}"
                )
            seed_status = f"{seed_status};{manifest_status}"
        else:
            seed_status = f"{seed_status};seed_manifest=not_checked"
        market_candidates = [
            self._data_dir / "market_df.parquet",
            self._regime_data_dir / "market_df.parquet",
        ]
        strat_candidates = [
            self._data_dir / "strat_ret_df.parquet",
            self._regime_data_dir / "strat_ret_df.parquet",
        ]
        market_df = _read_freshest_existing(market_candidates, required_cols=["VIX", "SPREAD", "SLOPE_10Y2Y"])
        strat_ret_df = _read_freshest_existing(strat_candidates, required_cols=_CRISIS_ETF_SYMBOLS)
        return market_df, strat_ret_df, seed_status

    def _now(self) -> datetime:
        now = self._now_provider() if self._now_provider is not None else datetime.now(timezone.utc)
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now

    def _unavailable_context(
        self,
        message: str,
        require_live_backlog: bool,
    ) -> CrisisContext:
        if require_live_backlog:
            raise RuntimeError(message)
        logger.warning(message)
        return self._context or load_crisis_context(self._data_dir / "latest_context.json")

    async def _notify_listeners(self, ctx: CrisisContext) -> None:
        for listener in list(self._listeners):
            try:
                result = listener(ctx)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Crisis listener failed")

    def _emit_transition(self, prev: CrisisContext | None, curr: CrisisContext) -> None:
        if prev is None:
            return
        if (
            prev.alert_level_int == curr.alert_level_int
            and prev.portfolio_action_level_int == curr.portfolio_action_level_int
        ):
            return
        try:
            from regime.crisis.events import CrisisTransitionEvent
            event = CrisisTransitionEvent(
                from_level=prev.alert_level,
                to_level=curr.alert_level,
                from_level_int=prev.alert_level_int,
                to_level_int=curr.alert_level_int,
                risk_multiplier=curr.risk_multiplier,
                dd_tier_multiplier=curr.dd_tier_multiplier,
                primary_warning_count=curr.primary_warning_count,
                primary_crisis_count=curr.primary_crisis_count,
                dominant_channel=curr.dominant_channel,
                timestamp=curr.computed_at,
            )
            out_dir = self._data_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "transitions.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(dataclasses.asdict(event), default=str) + "\n")
            logger.info("Crisis transition: %s -> %s", prev.alert_level, curr.alert_level)
        except Exception:
            logger.warning("Failed to emit CrisisTransitionEvent", exc_info=True)

    def __repr__(self) -> str:
        ctx = self._context
        if ctx is None:
            return "CrisisService(not yet computed)"
        return (
            f"CrisisService(level={ctx.alert_level}, "
            f"risk_mult={ctx.risk_multiplier:.2f}, "
            f"vix={ctx.vix_level:.1f})"
        )


def _read_first_existing(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        if not path.exists():
            continue
        return _read_parquet_normalized(path)
    return pd.DataFrame()


def _read_freshest_existing(
    paths: list[Path],
    *,
    required_cols: list[str],
) -> pd.DataFrame:
    candidates: list[tuple[pd.Timestamp, int, pd.DataFrame]] = []
    for ordinal, path in enumerate(paths):
        if not path.exists():
            continue
        df = _read_parquet_normalized(path)
        as_of = _latest_required_as_of(df, required_cols)
        if as_of is not None:
            candidates.append((as_of, -ordinal, df))
    if not candidates:
        return _read_first_existing(paths)
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _read_parquet_normalized(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def _latest_required_as_of(
    df: pd.DataFrame,
    required_cols: list[str],
) -> pd.Timestamp | None:
    if df.empty:
        return None
    if any(col not in df.columns for col in required_cols):
        return None
    valid = df[required_cols].dropna(how="any")
    if valid.empty:
        return None
    return pd.Timestamp(valid.index.max())


def _overlay_columns(
    base_df: pd.DataFrame,
    fresh_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    if base_df.empty:
        base_df = pd.DataFrame(index=fresh_df.index)
    out = base_df.copy()
    for col in columns:
        if col not in fresh_df.columns:
            continue
        fresh = fresh_df[col].dropna()
        if fresh.empty:
            continue
        if col not in out.columns:
            out[col] = float("nan")
        out = out.reindex(out.index.union(fresh.index)).sort_index()
        out.loc[fresh.index, col] = fresh
    return out.sort_index()


def _latest_return_as_of(strat_ret_df: pd.DataFrame) -> pd.Timestamp | None:
    if strat_ret_df.empty:
        return None
    cols = [col for col in _CRISIS_ETF_SYMBOLS if col in strat_ret_df.columns]
    if len(cols) == len(_CRISIS_ETF_SYMBOLS):
        valid = strat_ret_df[cols].dropna(how="any")
    else:
        valid = strat_ret_df.dropna(how="all")
    if valid.empty:
        return None
    return pd.Timestamp(valid.index.max())


def _append_data_lag_status(
    status: str,
    market_df: pd.DataFrame,
    compute_date: pd.Timestamp,
) -> str:
    parts = [part for part in status.split(";") if part]
    for col in ("VIX", "SPREAD", "SLOPE_10Y2Y"):
        latest = _latest_non_na_index(market_df, col, compute_date)
        if latest is None:
            parts.append(f"{col.lower()}=missing")
            continue
        lag_days = max(0, (compute_date.normalize() - latest.normalize()).days)
        if lag_days > 0:
            parts.append(f"{col.lower()}_lag_days={lag_days}")
    return ";".join(parts)


def _append_backlog_status(
    status: str,
    market_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    compute_date: pd.Timestamp,
) -> tuple[str, bool]:
    parts = [part for part in status.split(";") if part]
    problems: list[str] = []

    return_cols = [col for col in _CRISIS_ETF_SYMBOLS if col in strat_ret_df.columns]
    if len(return_cols) == len(_CRISIS_ETF_SYMBOLS):
        return_obs = len(strat_ret_df.loc[:compute_date, return_cols].dropna(how="any"))
    else:
        return_obs = 0
    if return_obs < _CRISIS_MIN_RETURN_OBS:
        problems.append(f"spy_tlt={return_obs}/{_CRISIS_MIN_RETURN_OBS}")

    for col in ("VIX", "SPREAD", "SLOPE_10Y2Y"):
        if col not in market_df.columns:
            problems.append(f"{col.lower()}=0/{_CRISIS_MIN_MARKET_OBS}")
            continue
        obs = len(market_df.loc[:compute_date, col].dropna())
        if obs < _CRISIS_MIN_MARKET_OBS:
            problems.append(f"{col.lower()}={obs}/{_CRISIS_MIN_MARKET_OBS}")

    if problems:
        parts.append("backlog=insufficient:" + ",".join(problems))
        return ";".join(parts), False
    parts.append("backlog=ok")
    return ";".join(parts), True


def _append_return_lag_status(
    status: str,
    compute_date: pd.Timestamp,
    now: datetime,
    market_cal: Any | None,
) -> tuple[str, int]:
    parts = [part for part in status.split(";") if part]
    expected = _expected_completed_session_date(now, market_cal)
    lag_sessions = _trading_session_lag(compute_date.date(), expected, market_cal)
    if lag_sessions > 0:
        parts.append(f"return_lag_sessions={lag_sessions}")
    return ";".join(parts), lag_sessions


def _expected_completed_session_date(now: datetime, market_cal: Any | None) -> date:
    now_et = now.astimezone(_ET)
    candidate = now_et.date()
    cutoff = now_et.replace(
        hour=_CRISIS_COMPUTE_HOUR_ET,
        minute=_CRISIS_COMPUTE_MINUTE_ET,
        second=0,
        microsecond=0,
    )
    if now_et < cutoff or not _is_trading_day(candidate, market_cal):
        candidate -= timedelta(days=1)
    while not _is_trading_day(candidate, market_cal):
        candidate -= timedelta(days=1)
    return candidate


def _trading_session_lag(
    data_date: date,
    expected_date: date,
    market_cal: Any | None,
) -> int:
    if data_date >= expected_date:
        return 0
    lag = 0
    cursor = data_date + timedelta(days=1)
    while cursor <= expected_date:
        if _is_trading_day(cursor, market_cal):
            lag += 1
        cursor += timedelta(days=1)
    return lag


def _latest_non_na_index(
    df: pd.DataFrame,
    col: str,
    compute_date: pd.Timestamp,
) -> pd.Timestamp | None:
    if df.empty or col not in df.columns:
        return None
    series = df.loc[:compute_date, col].dropna()
    if series.empty:
        return None
    return pd.Timestamp(series.index.max())


def _overlay_duration_str(
    seed_return_date: pd.Timestamp | None,
    now: datetime,
    *,
    min_days: int = 14,
    max_days: int = 180,
) -> str:
    if seed_return_date is None:
        return f"{max_days} D"
    lag_days = max(1, (now.date() - seed_return_date.date()).days)
    days = min(max(lag_days + 7, min_days), max_days)
    return f"{days} D"


def _overlay_start_date(
    seed_return_date: pd.Timestamp | None,
    now: datetime,
    *,
    min_days: int = 30,
    default_days: int = 180,
) -> str:
    now_ts = pd.Timestamp(now).tz_localize(None) if pd.Timestamp(now).tzinfo else pd.Timestamp(now)
    if seed_return_date is None:
        return (now_ts - pd.Timedelta(days=default_days)).date().isoformat()
    seed_ts = pd.Timestamp(seed_return_date).tz_localize(None) if pd.Timestamp(seed_return_date).tzinfo else pd.Timestamp(seed_return_date)
    start = min(
        seed_ts - pd.Timedelta(days=min_days),
        now_ts - pd.Timedelta(days=min_days),
    )
    return start.date().isoformat()


async def _call_async_with_optional_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    if not _accepts_any_kwargs(func):
        return await func()
    return await func(**kwargs)


def _call_sync_with_optional_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    if not _accepts_any_kwargs(func):
        return func()
    return func(**kwargs)


def _accepts_any_kwargs(func: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    return any(
        param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD, param.VAR_KEYWORD)
        for param in signature.parameters.values()
    )
