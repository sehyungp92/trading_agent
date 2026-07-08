"""Unified swing portfolio backtesting engine.

Runs ATRSS, AKC-Helix, and the idle-capital overlay under one shared replay
with portfolio-level heat caps,
priority-aware reservation, per-strategy ceilings, and cross-strategy
coordination.
"""
from __future__ import annotations

import logging
import json
import asyncio
import copy
from dataclasses import dataclass, field, replace as _dc_replace
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

from strategies.swing.atrss.config import SYMBOL_CONFIGS as ATRSS_SYMBOL_CONFIGS
from strategies.swing.atrss.models import Direction as AtrssDirection

from strategies.swing.akc_helix.config import SYMBOL_CONFIGS as HELIX_SYMBOL_CONFIGS
from strategies.swing.akc_helix.models import Direction as HelixDirection

from strategies.swing.overlay.shared import allocate_weighted_targets, compute_ema

from libs.oms.risk.swing_portfolio_adapter import SwingPortfolioHeatAdapter as PortfolioHeatTracker
from libs.oms.risk.portfolio_rules import PortfolioRuleChecker, PortfolioRulesConfig
from libs.oms.coordination.coordinator import StrategyCoordinator

from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.data.preprocessing import (
    NumpyBars,
    align_4h_to_hourly,
    align_daily_to_hourly,
    build_numpy_arrays,
    filter_rth,
    normalize_timezone,
    resample_1h_to_4h,
)
from backtests.swing.data.multitimeframe import (
    align_15m_to_30m,
    align_15m_to_1h,
    align_15m_to_4h,
    align_daily_to_15m,
    resample_15m_to_30m,
)
from backtests.swing.engine.backtest_engine import BacktestEngine, _AblationPatch, ORDER_EXPIRY_HOURS
from backtests.swing.engine.helix_engine import HelixEngine, _AblationPatch as _HelixAblationPatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class UnifiedPortfolioData:
    """Pre-loaded data for all symbols across all strategies.

    ATRSS requires RTH-filtered hourly data (matching its individual runner).
    Helix uses full hourly data (matching its individual runner).
    For shared symbols (e.g. QQQ), both versions are stored separately.
    """

    daily: dict[str, NumpyBars] = field(default_factory=dict)
    # RTH-filtered hourly data for ATRSS (UTC, RTH only)
    atrss_hourly: dict[str, NumpyBars] = field(default_factory=dict)
    atrss_daily_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    # Full hourly data for Helix (no RTH filter)
    hourly: dict[str, NumpyBars] = field(default_factory=dict)
    four_hour: dict[str, NumpyBars] = field(default_factory=dict)
    daily_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    four_hour_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    # Full 15m ETF data for TPC.
    etf_15m: dict[str, NumpyBars] = field(default_factory=dict)
    etf_30m: dict[str, NumpyBars] = field(default_factory=dict)
    etf_1h: dict[str, NumpyBars] = field(default_factory=dict)
    etf_4h: dict[str, NumpyBars] = field(default_factory=dict)
    etf_daily: dict[str, NumpyBars] = field(default_factory=dict)
    etf_1h_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    etf_30m_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    etf_4h_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    etf_daily_idx_maps: dict[str, np.ndarray] = field(default_factory=dict)
    tpc_replay: dict[str, dict] = field(default_factory=dict)


def _coerce_index_bound(
    value: str | pd.Timestamp | None,
    index: pd.DatetimeIndex,
    *,
    end_of_day: bool,
) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if end_of_day and ts == ts.normalize():
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    index_tz = index.tz
    if index_tz is None:
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
    elif ts.tzinfo is None:
        ts = ts.tz_localize(index_tz)
    else:
        ts = ts.tz_convert(index_tz)
    return ts


def _slice_config_window(df: pd.DataFrame, config: UnifiedBacktestConfig) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    start_ts = _coerce_index_bound(config.start_date, df.index, end_of_day=False)
    end_ts = _coerce_index_bound(config.end_date, df.index, end_of_day=True)
    if start_ts is not None:
        df = df[df.index >= start_ts]
    if end_ts is not None:
        df = df[df.index <= end_ts]
    return df


def load_unified_data(config: UnifiedBacktestConfig) -> UnifiedPortfolioData:
    """Load parquet data for all symbols used by any strategy.

    Each strategy's individual runner loads data differently:
    - ATRSS: normalize_timezone(UTC) + filter_rth
    - Helix: raw parquet (no filter)
    We replicate each strategy's exact data pipeline to ensure trade alignment.
    """
    from pathlib import Path
    data_dir = Path(config.data_dir)
    portfolio = UnifiedPortfolioData()

    atrss_set = set(config.atrss_symbols)
    overlay_syms = config.overlay_symbols if config.overlay_enabled else []
    etf_syms = sorted(set(config.tpc_symbols))
    all_symbols = sorted(set(
        config.atrss_symbols + config.helix_symbols + overlay_syms + etf_syms
    ))

    for sym in all_symbols:
        hourly_path = data_dir / f"{sym}_1h.parquet"
        daily_path = data_dir / f"{sym}_1d.parquet"

        if not hourly_path.exists() or not daily_path.exists():
            logger.warning("Missing data for %s, skipping", sym)
            continue

        hourly_df = _slice_config_window(pd.read_parquet(hourly_path), config)
        daily_df = _slice_config_window(pd.read_parquet(daily_path), config)


        # Date range filtering (reduces bar count → faster engine run)

        # Standardize column names
        hourly_df.columns = hourly_df.columns.str.lower()
        daily_df.columns = daily_df.columns.str.lower()

        # --- Helix: raw unfiltered hourly (always built) ---
        portfolio.daily[sym] = build_numpy_arrays(daily_df)
        portfolio.hourly[sym] = build_numpy_arrays(hourly_df)
        four_hour_df = resample_1h_to_4h(hourly_df)
        portfolio.four_hour[sym] = build_numpy_arrays(four_hour_df)
        portfolio.daily_idx_maps[sym] = align_daily_to_hourly(hourly_df, daily_df)
        portfolio.four_hour_idx_maps[sym] = align_4h_to_hourly(hourly_df, four_hour_df)

        # --- ATRSS: normalize_timezone(UTC) + filter_rth ---
        if sym in atrss_set:
            atrss_h = normalize_timezone(hourly_df.copy())
            atrss_d = normalize_timezone(daily_df.copy())
            atrss_h = filter_rth(atrss_h)
            portfolio.atrss_hourly[sym] = build_numpy_arrays(atrss_h)
            portfolio.atrss_daily_idx_maps[sym] = align_daily_to_hourly(atrss_h, atrss_d)

        # --- TPC: full 15m primary data with completed HTF maps ---
        if sym in etf_syms:
            path_15m = data_dir / f"{sym}_15m.parquet"
            if path_15m.exists():
                df15 = _slice_config_window(pd.read_parquet(path_15m), config)
                df15.columns = df15.columns.str.lower()
                df15 = normalize_timezone(df15)
                hourly_norm = normalize_timezone(hourly_df.copy())
                daily_norm = normalize_timezone(daily_df.copy())
                four_hour_norm = resample_1h_to_4h(hourly_norm)
                df30 = resample_15m_to_30m(df15)
                portfolio.etf_15m[sym] = build_numpy_arrays(df15)
                portfolio.etf_30m[sym] = build_numpy_arrays(df30)
                portfolio.etf_1h[sym] = build_numpy_arrays(hourly_norm)
                portfolio.etf_4h[sym] = build_numpy_arrays(four_hour_norm)
                portfolio.etf_daily[sym] = build_numpy_arrays(daily_norm)
                portfolio.etf_30m_idx_maps[sym] = align_15m_to_30m(df15, df30)
                portfolio.etf_1h_idx_maps[sym] = align_15m_to_1h(df15, hourly_norm)
                portfolio.etf_4h_idx_maps[sym] = align_15m_to_4h(df15, four_hour_norm)
                portfolio.etf_daily_idx_maps[sym] = align_daily_to_15m(df15, daily_norm)
            else:
                logger.warning("Missing 15m ETF data for %s, TPC will skip it", sym)

    if etf_syms:
        from backtests.swing.data.replay_cache import load_tpc_replay_bundle

        tpc_bundle = load_tpc_replay_bundle(
            data_dir,
            symbols=tuple(etf_syms),
            start_date=config.start_date,
            end_date=config.end_date,
        )
        portfolio.tpc_replay = dict(tpc_bundle.data)
        for sym, payload in portfolio.tpc_replay.items():
            portfolio.etf_15m[sym] = payload["bars_15m"]
            portfolio.etf_30m[sym] = payload.get("bars_30m")
            portfolio.etf_1h[sym] = payload["bars_1h"]
            portfolio.etf_4h[sym] = payload["bars_4h"]
            portfolio.etf_daily[sym] = payload["bars_daily"]
            portfolio.etf_30m_idx_maps[sym] = payload.get("idx_30m")
            portfolio.etf_1h_idx_maps[sym] = payload["idx_1h"]
            portfolio.etf_4h_idx_maps[sym] = payload["idx_4h"]
            portfolio.etf_daily_idx_maps[sym] = payload["idx_daily"]

    logger.info("Loaded data for %d symbols: %s", len(portfolio.hourly), list(portfolio.hourly))
    return portfolio


# ---------------------------------------------------------------------------
# Cross-strategy coordination replay adapter
# ---------------------------------------------------------------------------


class _ReplayCoordinationBus:
    """Tiny bus used to run the live StrategyCoordinator in deterministic replay."""

    def __init__(self) -> None:
        self.events: list[dict[str, str]] = []

    def emit_coordination_event(
        self,
        *,
        target_strategy: str,
        event_type: str,
        symbol: str,
    ) -> None:
        self.events.append(
            {
                "target_strategy": target_strategy,
                "event_type": event_type,
                "symbol": symbol,
            }
        )


class BacktestCoordinator:
    """Replay adapter around the live StrategyCoordinator.

    Rule 1: ATRSS entry on symbol X → tighten Helix stop to BE on X.
    Rule 2: ATRSS has active position on symbol X → Helix 1.25x size boost.
    """

    def __init__(self, enable_tighten: bool = True, enable_size_boost: bool = True):
        self._enable_tighten = enable_tighten
        self._enable_size_boost = enable_size_boost
        # symbol → (direction, entry_price)
        self._bus = _ReplayCoordinationBus()
        self._coordinator = StrategyCoordinator(self._bus)
        self.tighten_count = 0
        self.boost_count = 0

    def on_atrss_position_change(
        self, symbol: str, direction: int, entry_price: float,
    ) -> None:
        """Track ATRSS position changes through the live coordinator."""
        existing = self._coordinator.get_position("ATRSS", symbol)
        was_flat = existing is None or existing.qty <= 0
        if direction != 0:
            direction_label = "LONG" if int(direction) > 0 else "SHORT"
            self._coordinator.on_position_update(
                "ATRSS",
                symbol,
                qty=1,
                direction=direction_label,
                entry_price=entry_price,
            )
            if was_flat and self._enable_tighten:
                self._bus.emit_coordination_event(
                    target_strategy="AKC_HELIX",
                    event_type="TIGHTEN_STOP_BE",
                    symbol=symbol,
                )
        else:
            self._coordinator.on_position_update(
                "ATRSS",
                symbol,
                qty=0,
                direction="",
                entry_price=0.0,
            )

    def consume_tighten_events(self) -> list[str]:
        """Return and clear pending tighten events."""
        events = [
            event["symbol"]
            for event in self._bus.events
            if event.get("target_strategy") == "AKC_HELIX"
            and event.get("event_type") == "TIGHTEN_STOP_BE"
        ]
        self._bus.events.clear()
        return events

    def has_atrss_position(self, symbol: str, direction: int) -> bool:
        """Check if ATRSS has an active position on symbol in the given direction."""
        if not self._enable_size_boost:
            return False
        direction_label = "LONG" if int(direction) > 0 else "SHORT"
        return self._coordinator.has_atrss_position(symbol, direction_label)

    def log_action(self, **kwargs) -> None:
        self._coordinator.log_action(**kwargs)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class UnifiedHeatStats:
    """Portfolio heat utilization statistics."""

    avg_heat_pct: float = 0.0
    max_heat_pct: float = 0.0
    pct_time_at_cap: float = 0.0


@dataclass
class StrategyResult:
    """Aggregated results for one strategy."""

    strategy_id: str = ""
    entry_signals_fired: int = 0
    entry_requests: int = 0
    entries_accepted_by_portfolio: int = 0
    total_trades: int = 0
    total_pnl: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    total_r: float = 0.0
    max_drawdown_dollars: float = 0.0
    entries_blocked_by_heat: int = 0
    suppressed_entry_retries: int = 0


@dataclass
class UnifiedPortfolioResult:
    """Combined results from the unified backtest."""

    combined_equity: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_equity_mtm: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_equity_realized: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    heat_stats: UnifiedHeatStats = field(default_factory=UnifiedHeatStats)
    strategy_results: dict[str, StrategyResult] = field(default_factory=dict)
    coordination_tighten_count: int = 0
    coordination_boost_count: int = 0
    portfolio_daily_stop_activations: int = 0
    overlay_pnl: float = 0.0
    overlay_commission: float = 0.0
    overlay_per_symbol_pnl: dict[str, float] = field(default_factory=dict)
    # Raw per-engine results for detailed analysis
    atrss_trades: list = field(default_factory=list)
    helix_trades: list = field(default_factory=list)
    tpc_trades: list = field(default_factory=list)
    # Diagnostic event logs for portfolio diagnostics
    entry_events: list = field(default_factory=list)
    heat_rejections: list = field(default_factory=list)
    coordination_events: list = field(default_factory=list)
    portfolio_rule_events: list = field(default_factory=list)


@dataclass(frozen=True)
class _PortfolioRuleExposure:
    strategy_id: str
    symbol: str
    direction: str
    risk_dollars: float
    risk_R: float
    qty: int


@dataclass(frozen=True)
class SwingFamilyReplayCandidate:
    """Candidate input for the unified swing family replay decision surface."""

    candidate_key: str
    strategy_id: str
    symbol: str
    direction: str
    risk_dollars: float
    qty: int
    entry_time: datetime | None = None


@dataclass(frozen=True)
class SwingFamilyReplayExposure:
    """Existing swing-family exposure consumed by the replay decision surface."""

    strategy_id: str
    symbol: str
    direction: str
    risk_dollars: float
    risk_R: float
    qty: int


@dataclass(frozen=True)
class SwingFamilyReplayDecision:
    """Authoritative replay decision emitted by unified swing portfolio rules."""

    candidate_key: str
    strategy_id: str
    symbol: str
    original_qty: int
    approved_qty: int
    status: str
    reason: str = ""
    size_multiplier: float = 1.0


def _direction_label(direction: object) -> str:
    value = getattr(direction, "value", direction)
    try:
        numeric = int(value)
    except Exception:
        text = str(value).upper()
        return "SHORT" if "SHORT" in text or text == "-1" else "LONG"
    return "LONG" if numeric > 0 else "SHORT"


def _strategy_slot(config: UnifiedBacktestConfig, strategy_id: str):
    if strategy_id == "ATRSS":
        return config.atrss
    if strategy_id == "AKC_HELIX":
        return config.helix
    if strategy_id == "TPC":
        return config.tpc
    return None


def _strategy_unit_risk(config: UnifiedBacktestConfig, strategy_id: str, initial_equity: float) -> float:
    slot = _strategy_slot(config, strategy_id)
    if slot is None:
        return 0.0
    return float(initial_equity) * float(getattr(slot, "unit_risk_pct", 0.0) or 0.0)


class _SwingPortfolioRuleReplayAdapter:
    """Synchronous replay wrapper around the live PortfolioRuleChecker."""

    def __init__(self, config: UnifiedBacktestConfig, initial_equity: float):
        strategy_ids = ("ATRSS", "AKC_HELIX", "TPC")
        dd_tiers = config.drawdown_risk_tiers if not config.dynamic_risk_enabled else ((1.0, 1.0),)
        self._config = config
        self._initial_equity = float(initial_equity)
        self._reference_strategy_id = strategy_ids[0]
        self._reference_unit_risk = _strategy_unit_risk(config, self._reference_strategy_id, initial_equity)
        self._current_equity = float(initial_equity)
        self._exposures: list[_PortfolioRuleExposure] = []
        self._skip_current: tuple[str, str] | None = None
        self._loop = asyncio.new_event_loop()
        rules = PortfolioRulesConfig(
            directional_cap_R=float(config.heat_cap_R),
            directional_cap_long_R=4.0,
            directional_cap_short_R=4.0,
            initial_equity=float(initial_equity),
            family_strategy_ids=strategy_ids,
            symbol_collision_action="half_size",
            strategy_priorities=tuple(
                (sid, int(getattr(_strategy_slot(config, sid), "priority", 99)))
                for sid in strategy_ids
            ),
            priority_headroom_R=0.75,
            priority_reserve_threshold=1,
            reference_unit_risk_dollars=self._reference_unit_risk,
            dd_tiers=tuple(dd_tiers),
            nqdtc_direction_filter_enabled=False,
        )
        self._rules = rules
        self._checker = PortfolioRuleChecker(
            config=rules,
            get_strategy_signal=self._get_strategy_signal,
            get_directional_risk_R=self._get_directional_risk_R,
            get_current_equity=lambda: self._current_equity,
            get_directional_risk_R_for_strategies=self._get_directional_risk_R_for_strategies,
            get_directional_risk_dollars_for_strategies=self._get_directional_risk_dollars_for_strategies,
            get_sibling_positions_for_symbol=self._get_sibling_positions_for_symbol,
        )

    def refresh(
        self,
        *,
        equity: float,
        exposures: list[_PortfolioRuleExposure],
        unit_equity: float | None = None,
    ) -> None:
        self._current_equity = float(equity)
        self._exposures = list(exposures)
        if unit_equity is None:
            return
        reference_unit_risk = _strategy_unit_risk(
            self._config,
            self._reference_strategy_id,
            float(unit_equity),
        )
        if reference_unit_risk <= 0 or abs(reference_unit_risk - self._reference_unit_risk) < 1e-9:
            return
        self._reference_unit_risk = reference_unit_risk
        self._rules = _dc_replace(
            self._rules,
            reference_unit_risk_dollars=reference_unit_risk,
        )
        self._checker.update_config(self._rules)

    def check_entry(
        self,
        *,
        strategy_id: str,
        direction: str,
        risk_dollars: float,
        symbol: str,
        qty: int,
        skip_current_exposure: bool = False,
    ):
        unit = _strategy_unit_risk(self._config, strategy_id, self._initial_equity)
        new_risk_R = float(risk_dollars) / unit if unit > 0 else 0.0
        self._skip_current = (strategy_id, symbol) if skip_current_exposure else None
        try:
            return self._loop.run_until_complete(
                self._checker.check_entry(
                    strategy_id=strategy_id,
                    direction=direction,
                    new_risk_R=new_risk_R,
                    symbol=symbol,
                    new_qty=int(qty),
                    new_risk_dollars=float(risk_dollars),
                )
            )
        finally:
            self._skip_current = None

    def close(self) -> None:
        if not self._loop.is_closed():
            self._loop.close()

    async def _get_strategy_signal(self, _strategy_id: str):
        return None

    async def _get_directional_risk_R(self, direction: str) -> float:
        return sum(exp.risk_R for exp in self._iter_exposures() if exp.direction == direction)

    async def _get_directional_risk_R_for_strategies(self, direction: str, strategy_ids: list[str]) -> float:
        allowed = set(strategy_ids)
        return sum(
            exp.risk_R
            for exp in self._iter_exposures()
            if exp.strategy_id in allowed and exp.direction == direction
        )

    async def _get_directional_risk_dollars_for_strategies(self, direction: str, strategy_ids: list[str]) -> float:
        allowed = set(strategy_ids)
        return sum(
            exp.risk_dollars
            for exp in self._iter_exposures()
            if exp.strategy_id in allowed and exp.direction == direction
        )

    async def _get_sibling_positions_for_symbol(self, strategy_ids: list[str], symbol: str) -> bool:
        allowed = set(strategy_ids)
        return any(
            exp.strategy_id in allowed and exp.symbol == symbol and exp.qty > 0
            for exp in self._iter_exposures()
        )

    def _iter_exposures(self):
        for exp in self._exposures:
            if self._skip_current is not None and (exp.strategy_id, exp.symbol) == self._skip_current:
                continue
            yield exp


def replay_swing_family_candidates(
    config: UnifiedBacktestConfig,
    candidates: list[SwingFamilyReplayCandidate],
    *,
    initial_exposures: list[SwingFamilyReplayExposure] | None = None,
    initial_equity: float | None = None,
    current_equity: float | None = None,
) -> list[SwingFamilyReplayDecision]:
    """Replay swing family portfolio admission for generated child candidates.

    This is the public parity adapter over the same portfolio-rule replay
    checker used by ``run_unified``. It returns matcher-level decisions so the
    parity OMS timeline is driven by family replay outcomes instead of raw
    child actions.
    """

    equity = float(initial_equity if initial_equity is not None else config.initial_equity)
    active_equity = float(current_equity if current_equity is not None else equity)
    replay = _SwingPortfolioRuleReplayAdapter(config, equity)
    exposures = [
        _PortfolioRuleExposure(
            strategy_id=item.strategy_id,
            symbol=item.symbol,
            direction=_direction_label(item.direction),
            risk_dollars=float(item.risk_dollars),
            risk_R=float(item.risk_R),
            qty=int(item.qty),
        )
        for item in (initial_exposures or [])
        if float(item.risk_dollars) > 0.0 and int(item.qty) > 0
    ]
    decisions: list[SwingFamilyReplayDecision] = []
    try:
        for candidate in candidates:
            qty = max(0, int(candidate.qty))
            risk_dollars = max(0.0, float(candidate.risk_dollars))
            replay.refresh(
                equity=active_equity,
                unit_equity=active_equity,
                exposures=exposures,
            )
            result = replay.check_entry(
                strategy_id=candidate.strategy_id,
                direction=_direction_label(candidate.direction),
                risk_dollars=risk_dollars,
                symbol=candidate.symbol,
                qty=qty,
            )
            if not result.approved:
                decisions.append(
                    SwingFamilyReplayDecision(
                        candidate_key=candidate.candidate_key,
                        strategy_id=candidate.strategy_id,
                        symbol=candidate.symbol,
                        original_qty=qty,
                        approved_qty=0,
                        status="rejected",
                        reason=str(result.denial_reason or "portfolio_rule"),
                        size_multiplier=0.0,
                    )
                )
                continue

            size_multiplier = max(float(result.size_multiplier or 1.0), 0.0)
            approved_qty = max(1, int(qty * size_multiplier)) if qty > 0 else 0
            status = "reduced" if 0 < approved_qty < qty else "accepted"
            decisions.append(
                SwingFamilyReplayDecision(
                    candidate_key=candidate.candidate_key,
                    strategy_id=candidate.strategy_id,
                    symbol=candidate.symbol,
                    original_qty=qty,
                    approved_qty=approved_qty,
                    status=status,
                    reason="",
                    size_multiplier=size_multiplier,
                )
            )
            if approved_qty <= 0:
                continue
            risk_ratio = (approved_qty / qty) if qty > 0 else 0.0
            adjusted_risk = risk_dollars * risk_ratio
            unit = _strategy_unit_risk(config, candidate.strategy_id, active_equity)
            exposures.append(
                _PortfolioRuleExposure(
                    strategy_id=candidate.strategy_id,
                    symbol=candidate.symbol,
                    direction=_direction_label(candidate.direction),
                    risk_dollars=adjusted_risk,
                    risk_R=adjusted_risk / unit if unit > 0 else 0.0,
                    qty=approved_qty,
                )
            )
    finally:
        replay.close()
    return decisions


# ---------------------------------------------------------------------------
# Helper: get point value from strategy configs
# ---------------------------------------------------------------------------

def _get_point_value(symbol: str) -> float:
    """Get point value from whichever strategy config has it."""
    for cfgs in (ATRSS_SYMBOL_CONFIGS, HELIX_SYMBOL_CONFIGS):
        cfg = cfgs.get(symbol)
        if cfg is not None:
            return cfg.multiplier
    return 1.0


# ---------------------------------------------------------------------------
# Compute open risk across all engines
# ---------------------------------------------------------------------------

def _compute_open_risk(
    atrss_engines: dict[str, BacktestEngine],
    helix_engines: dict[str, HelixEngine],
    tpc_open_risks: dict[int, float] | None = None,
) -> dict[str, float]:
    """Returns {strategy_id: total_risk_dollars}."""
    risk: dict[str, float] = {
        "ATRSS": 0.0, "AKC_HELIX": 0.0, "TPC": 0.0,
    }

    for sym, eng in atrss_engines.items():
        pos = eng.position
        if pos.direction != AtrssDirection.FLAT and pos.base_leg is not None:
            r = abs(pos.base_leg.entry_price - pos.current_stop) * eng.point_value * pos.total_qty
            risk["ATRSS"] += r

    for sym, eng in helix_engines.items():
        pos = eng.active_position
        if pos is not None and pos.qty_open > 0:
            r = abs(pos.fill_price - pos.current_stop) * eng.point_value * pos.qty_open
            risk["AKC_HELIX"] += r

    if tpc_open_risks:
        risk["TPC"] = float(sum(tpc_open_risks.values()))

    return risk


def _portfolio_rule_exposures(
    config: UnifiedBacktestConfig,
    initial_equity: float,
    atrss_engines: dict[str, BacktestEngine],
    helix_engines: dict[str, HelixEngine],
    active_tpc_trades: dict[int, object],
    active_tpc_risks: dict[int, float],
) -> list[_PortfolioRuleExposure]:
    exposures: list[_PortfolioRuleExposure] = []

    def _append(strategy_id: str, symbol: str, direction: object, risk_dollars: float, qty: float) -> None:
        risk_dollars_f = float(risk_dollars or 0.0)
        if risk_dollars_f <= 0:
            return
        unit = _strategy_unit_risk(config, strategy_id, initial_equity)
        exposures.append(
            _PortfolioRuleExposure(
                strategy_id=strategy_id,
                symbol=str(symbol or ""),
                direction=_direction_label(direction),
                risk_dollars=risk_dollars_f,
                risk_R=risk_dollars_f / unit if unit > 0 else 0.0,
                qty=max(0, int(abs(qty or 0))),
            )
        )

    for sym, eng in atrss_engines.items():
        pos = eng.position
        if pos.direction != AtrssDirection.FLAT and pos.base_leg is not None:
            risk_dollars = abs(pos.base_leg.entry_price - pos.current_stop) * eng.point_value * pos.total_qty
            _append("ATRSS", sym, pos.direction, risk_dollars, pos.total_qty)

    for sym, eng in helix_engines.items():
        pos = eng.active_position
        if pos is not None and pos.qty_open > 0:
            risk_dollars = abs(pos.fill_price - pos.current_stop) * eng.point_value * pos.qty_open
            _append("AKC_HELIX", sym, pos.setup.direction, risk_dollars, pos.qty_open)

    for trade_id, trade in active_tpc_trades.items():
        sym = str(getattr(trade, "symbol", "") or "")
        qty = float(getattr(trade, "qty", 0.0) or 0.0)
        risk_dollars = float(active_tpc_risks.get(trade_id, _trade_risk_dollars(trade)) or 0.0)
        _append("TPC", sym, getattr(trade, "direction", 0), risk_dollars, qty)

    return exposures


def _apply_portfolio_size_multiplier_to_atrss_candidate(cand, size_multiplier: float) -> tuple[float, int]:
    original_qty = int(getattr(cand, "qty", 0) or 0)
    if original_qty <= 0 or size_multiplier == 1.0:
        return 1.0, original_qty
    new_qty = max(1, int(original_qty * float(size_multiplier)))
    cand.qty = new_qty
    return (new_qty / original_qty if original_qty else 1.0), new_qty


def _apply_portfolio_size_multiplier_to_helix(engine: HelixEngine, size_multiplier: float) -> tuple[float, int]:
    pos = engine.active_position
    if pos is None or pos.qty_open <= 0 or size_multiplier == 1.0:
        return 1.0, int(pos.qty_open if pos is not None else 0)
    original_qty = int(pos.qty_open)
    new_qty = max(1, int(original_qty * float(size_multiplier)))
    ratio = new_qty / original_qty if original_qty else 1.0
    if new_qty == original_qty:
        return ratio, new_qty

    pos.qty_open = new_qty
    pos.setup.qty_planned = new_qty
    pos.setup.fill_qty = new_qty
    if hasattr(pos.setup, "qty_open"):
        pos.setup.qty_open = new_qty
    for order in getattr(engine.broker, "pending_orders", []):
        if order.symbol == engine.symbol and order.tag == "protective_stop":
            order.qty = new_qty
    core_setup = getattr(engine, "_core_state", None)
    if core_setup is not None:
        setup = core_setup.active_setups.get(pos.setup.setup_id)
        if setup is not None:
            setup.qty_planned = new_qty
            setup.fill_qty = new_qty
            setup.qty_open = new_qty
    return ratio, new_qty


def _scaled_tpc_trade(trade, size_multiplier: float) -> tuple[object, float, int]:
    original_qty = int(abs(float(getattr(trade, "qty", 0.0) or 0.0)))
    if original_qty <= 0 or size_multiplier == 1.0:
        return trade, 1.0, original_qty
    new_qty = max(1, int(original_qty * float(size_multiplier)))
    ratio = new_qty / original_qty if original_qty else 1.0
    if ratio == 1.0:
        return trade, ratio, new_qty
    adjusted = copy.copy(trade)
    setattr(adjusted, "qty", new_qty)
    for attr in ("pnl_dollars", "commission"):
        if hasattr(adjusted, attr):
            setattr(adjusted, attr, float(getattr(adjusted, attr, 0.0) or 0.0) * ratio)
    setattr(adjusted, "portfolio_size_mult", ratio)
    return adjusted, ratio, new_qty


_TPC_REPLAY_CACHE: dict[str, list] = {}


def _timeline_key(ts) -> int:
    """Normalize numpy/pandas/Python timestamps to UTC nanoseconds."""
    if isinstance(ts, (int, np.integer)):
        return int(ts)
    if hasattr(ts, "item"):
        item = ts.item()
        if isinstance(item, (int, np.integer)):
            return int(item)
    return int(pd.Timestamp(ts).value)


def _tpc_cache_key(data: UnifiedPortfolioData, config: UnifiedBacktestConfig) -> str:
    spans = []
    for sym in config.tpc_symbols:
        replay_payload = data.tpc_replay.get(sym, {})
        bars = replay_payload.get("bars_15m") or data.etf_15m.get(sym)
        if bars is None or len(bars) == 0:
            spans.append((sym, 0, 0, 0, ()))
            continue
        context_keys = tuple(sorted((replay_payload.get("context_indicators") or {}).keys()))
        spans.append((sym, len(bars), _timeline_key(bars.times[0]), _timeline_key(bars.times[-1]), context_keys))
    payload = {
        "initial_equity": config.initial_equity,
        "symbols": list(config.tpc_symbols),
        "warmup_15m": config.warmup_15m,
        "overrides": config.tpc_param_overrides,
        "spans": spans,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _build_tpc_replay_data(
    data: UnifiedPortfolioData,
    config: UnifiedBacktestConfig,
) -> dict[str, dict]:
    if data.tpc_replay:
        return {
            sym: data.tpc_replay[sym]
            for sym in config.tpc_symbols
            if sym in data.tpc_replay
        }
    replay: dict[str, dict] = {}
    for sym in config.tpc_symbols:
        if (
            sym not in data.etf_15m
            or sym not in data.etf_1h
            or sym not in data.etf_4h
            or sym not in data.etf_daily
        ):
            continue
        replay[sym] = {
            "bars_15m": data.etf_15m[sym],
            "bars_30m": data.etf_30m.get(sym),
            "bars_1h": data.etf_1h[sym],
            "bars_4h": data.etf_4h[sym],
            "bars_daily": data.etf_daily[sym],
            "idx_30m": data.etf_30m_idx_maps.get(sym),
            "idx_1h": data.etf_1h_idx_maps.get(sym),
            "idx_4h": data.etf_4h_idx_maps.get(sym),
            "idx_daily": data.etf_daily_idx_maps.get(sym),
        }
    return replay


def _run_tpc_source_replay(data: UnifiedPortfolioData, config: UnifiedBacktestConfig) -> list:
    """Run/cache TPC's own shared-core replay and return source trades."""
    if not config.tpc_symbols:
        return []
    cache_key = _tpc_cache_key(data, config)
    cached = _TPC_REPLAY_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    replay = _build_tpc_replay_data(data, config)
    if not replay:
        _TPC_REPLAY_CACHE[cache_key] = []
        return []

    from backtests.swing.engine.tpc_engine import run_tpc_independent

    result = run_tpc_independent(replay, config.build_tpc_config())
    trades = sorted(
        list(getattr(result, "trades", []) or []),
        key=lambda t: (_timeline_key(getattr(t, "entry_time", 0) or 0), _timeline_key(getattr(t, "exit_time", 0) or 0)),
    )
    _TPC_REPLAY_CACHE[cache_key] = trades
    return list(trades)


def _build_tpc_trade_events(trades: list) -> tuple[dict[int, list[tuple[int, object]]], dict[int, list[tuple[int, object]]]]:
    entries: dict[int, list[tuple[int, object]]] = {}
    exits: dict[int, list[tuple[int, object]]] = {}
    for idx, trade in enumerate(trades):
        entry_time = getattr(trade, "entry_time", None)
        exit_time = getattr(trade, "exit_time", None)
        if entry_time is None or exit_time is None:
            continue
        entries.setdefault(_timeline_key(entry_time), []).append((idx, trade))
        exits.setdefault(_timeline_key(exit_time), []).append((idx, trade))
    return entries, exits


def _trade_risk_dollars(trade) -> float:
    entry = float(getattr(trade, "entry_price", 0.0) or 0.0)
    stop = float(getattr(trade, "initial_stop", 0.0) or 0.0)
    qty = abs(float(getattr(trade, "qty", 0.0) or 0.0))
    risk = abs(entry - stop) * qty
    return risk if np.isfinite(risk) and risk > 0 else 1.0


def _strategy_initial_unit_risk_dollars(
    config: UnifiedBacktestConfig,
    strategy_id: str,
    initial_equity: float,
) -> float:
    slot_attr = {
        "ATRSS": "atrss",
        "AKC_HELIX": "helix",
        "TPC": "tpc",
    }.get(strategy_id)
    if slot_attr is None:
        return 0.0
    slot = getattr(config, slot_attr, None)
    unit_risk_pct = float(getattr(slot, "unit_risk_pct", 0.0) or 0.0)
    return float(initial_equity) * unit_risk_pct


def _trade_portfolio_size_mult(trade) -> float:
    value = float(getattr(trade, "portfolio_size_mult", 1.0) or 1.0)
    return value if np.isfinite(value) and value >= 0.0 else 1.0


def _trade_net_pnl(trade) -> float:
    if hasattr(trade, "net_pnl_dollars"):
        return float(getattr(trade, "net_pnl_dollars", 0.0) or 0.0)
    return float(getattr(trade, "pnl_dollars", 0.0) or 0.0) - float(getattr(trade, "commission", 0.0) or 0.0)


def _strategy_trade_net_pnl(strategy_id: str, trade) -> float:
    if hasattr(trade, "net_pnl_dollars"):
        return float(getattr(trade, "net_pnl_dollars", 0.0) or 0.0)
    pnl = float(getattr(trade, "pnl_dollars", 0.0) or 0.0)
    if strategy_id == "TPC":
        return pnl
    return pnl - float(getattr(trade, "commission", 0.0) or 0.0)


def _trade_net_r(trade) -> float:
    if hasattr(trade, "net_r_multiple"):
        return float(getattr(trade, "net_r_multiple", 0.0) or 0.0)
    return float(getattr(trade, "r_multiple", 0.0) or 0.0)


def _normalised_trade_r(trade) -> float:
    total_r = _trade_net_r(trade)
    if not np.isfinite(total_r):
        return 0.0
    return total_r * _trade_portfolio_size_mult(trade)


def _normalised_trade_pnl(
    strategy_id: str,
    trade,
    config: UnifiedBacktestConfig,
    initial_equity: float,
) -> float:
    unit_risk = _strategy_initial_unit_risk_dollars(config, strategy_id, initial_equity)
    total_r = _normalised_trade_r(trade)
    if unit_risk > 0.0 and np.isfinite(total_r):
        return total_r * unit_risk
    return _strategy_trade_net_pnl(strategy_id, trade)


def _normalised_trade_copy(
    strategy_id: str,
    trade,
    config: UnifiedBacktestConfig,
    initial_equity: float,
):
    normalised = copy.copy(trade)
    source_pnl = _strategy_trade_net_pnl(strategy_id, trade)
    source_r = _trade_net_r(trade)
    pnl = _normalised_trade_pnl(strategy_id, trade, config, initial_equity)
    total_r = _normalised_trade_r(trade)
    setattr(normalised, "source_pnl_dollars", source_pnl)
    setattr(normalised, "source_r_multiple", source_r)
    setattr(normalised, "pnl_dollars", pnl)
    setattr(normalised, "r_multiple", total_r)
    setattr(normalised, "net_pnl_dollars", pnl)
    setattr(normalised, "net_r_multiple", total_r)
    qty = abs(float(getattr(normalised, "qty", 0.0) or 0.0))
    if qty > 0:
        setattr(normalised, "pnl_points", pnl / qty)
    setattr(normalised, "portfolio_normalised", True)
    return normalised


def _tpc_open_mtm_pnl(
    data: UnifiedPortfolioData,
    active_trades: dict[int, object],
    active_risks: dict[int, float],
    config: UnifiedBacktestConfig,
    initial_equity: float,
    ts_key: int,
) -> float:
    """Approximate MTM for accepted open TPC source trades."""
    pnl = 0.0
    unit_risk = _strategy_initial_unit_risk_dollars(config, "TPC", initial_equity)
    for trade_id, trade in active_trades.items():
        sym = getattr(trade, "symbol", "")
        bars = data.etf_15m.get(sym)
        if bars is None or len(bars) == 0:
            continue
        times = bars.times.astype("datetime64[ns]").astype(np.int64)
        idx = int(np.searchsorted(times, ts_key, side="right") - 1)
        if idx < 0:
            continue
        close = float(bars.closes[min(idx, len(bars.closes) - 1)])
        direction = 1 if int(getattr(trade, "direction", 0) or 0) > 0 else -1
        qty = abs(float(getattr(trade, "qty", 0.0) or 0.0))
        entry = float(getattr(trade, "entry_price", 0.0) or 0.0)
        raw_open_pnl = (close - entry) * direction * qty
        risk_dollars = float(active_risks.get(trade_id, _trade_risk_dollars(trade)) or 0.0)
        if unit_risk > 0.0 and risk_dollars > 0.0:
            pnl += raw_open_pnl / risk_dollars * unit_risk * _trade_portfolio_size_mult(trade)
        else:
            pnl += raw_open_pnl
    return float(pnl)


def _latest_engine_mark(engine, default_equity: float) -> float:
    """Return the latest per-engine MTM equity mark when available."""
    eq_arr = getattr(engine, "_eq_arr", None)
    bar_idx = int(getattr(engine, "_bar_idx", 0) or 0)
    if eq_arr is not None and bar_idx > 0:
        try:
            value = float(eq_arr[bar_idx - 1])
            if np.isfinite(value):
                return value
        except (IndexError, TypeError, ValueError):
            pass

    curve = getattr(engine, "equity_curve", None)
    if curve is not None and len(curve) > 0:
        try:
            value = float(curve[-1])
            if np.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass

    value = float(getattr(engine, "equity", default_equity) or default_equity)
    return value if np.isfinite(value) else float(default_equity)


def _active_strategy_ids(config: UnifiedBacktestConfig) -> list[str]:
    active: list[str] = []
    if config.atrss_symbols:
        active.append("ATRSS")
    if config.helix_symbols:
        active.append("AKC_HELIX")
    if config.tpc_symbols:
        active.append("TPC")
    return active


def _native_strategy_result(strategy_id: str, trades: list, entry_requests: int | None = None) -> StrategyResult:
    total_pnl = sum(_strategy_trade_net_pnl(strategy_id, t) for t in trades)
    wins = sum(1 for t in trades if _strategy_trade_net_pnl(strategy_id, t) > 0.0)
    losses = sum(1 for t in trades if _strategy_trade_net_pnl(strategy_id, t) <= 0.0)
    total_r = sum(_trade_net_r(t) for t in trades)
    requests = len(trades) if entry_requests is None else int(entry_requests)
    return StrategyResult(
        strategy_id=strategy_id,
        entry_signals_fired=requests,
        entry_requests=requests,
        entries_accepted_by_portfolio=len(trades),
        total_trades=len(trades),
        total_pnl=total_pnl,
        winning_trades=wins,
        losing_trades=losses,
        total_r=total_r,
    )


def _empty_strategy_results(active_id: str, trades: list, entry_requests: int | None = None) -> dict[str, StrategyResult]:
    return {
        sid: _native_strategy_result(sid, trades if sid == active_id else [], entry_requests if sid == active_id else 0)
        for sid in ("ATRSS", "AKC_HELIX", "TPC")
    }


def _native_heat_stats(native_result) -> UnifiedHeatStats:
    heat = getattr(native_result, "heat_stats", None)
    if heat is None:
        return UnifiedHeatStats()
    return UnifiedHeatStats(
        avg_heat_pct=float(getattr(heat, "avg_heat_pct", 0.0) or 0.0),
        max_heat_pct=float(getattr(heat, "max_heat_pct", 0.0) or 0.0),
        pct_time_at_cap=float(
            getattr(heat, "pct_time_at_cap", getattr(heat, "pct_time_at_limit", 0.0)) or 0.0
        ),
    )


def _collect_native_trades(symbol_results: dict) -> list:
    trades: list = []
    for sym, result in symbol_results.items():
        for trade in getattr(result, "trades", []) or []:
            if not getattr(trade, "symbol", ""):
                setattr(trade, "symbol", sym)
            trades.append(trade)
    trades.sort(
        key=lambda t: (
            _timeline_key(getattr(t, "entry_time", 0) or 0),
            _timeline_key(getattr(t, "exit_time", 0) or 0),
        )
    )
    return trades


def _run_native_standalone_if_requested(
    data: UnifiedPortfolioData,
    config: UnifiedBacktestConfig,
) -> UnifiedPortfolioResult | None:
    """Use the individual synchronized runner for exact standalone parity.

    The portfolio driver intentionally changes order admission semantics. For
    a no-overlay, no-constraints single-strategy run, the correct reference is
    the same synchronized engine used by the latest individual diagnostics.
    """
    if config.portfolio_constraints_enabled or config.overlay_enabled:
        return None
    active = _active_strategy_ids(config)
    if len(active) != 1:
        return None

    strategy_id = active[0]
    if strategy_id == "ATRSS":
        from backtests.swing.engine.portfolio_engine import PortfolioData, run_synchronized

        portfolio_data = PortfolioData(
            daily={sym: data.daily[sym] for sym in config.atrss_symbols if sym in data.daily},
            hourly={sym: data.atrss_hourly[sym] for sym in config.atrss_symbols if sym in data.atrss_hourly},
            daily_idx_maps={
                sym: data.atrss_daily_idx_maps[sym]
                for sym in config.atrss_symbols
                if sym in data.atrss_daily_idx_maps
            },
        )
        native = run_synchronized(portfolio_data, config.build_atrss_config())
        trades = _collect_native_trades(native.symbol_results)
        equity = np.array(native.combined_equity)
        return UnifiedPortfolioResult(
            combined_equity=equity,
            combined_equity_mtm=equity,
            combined_equity_realized=equity,
            combined_timestamps=np.array(native.combined_timestamps),
            heat_stats=_native_heat_stats(native),
            strategy_results=_empty_strategy_results("ATRSS", trades),
            atrss_trades=trades,
        )

    if strategy_id == "AKC_HELIX":
        from backtests.swing.engine.helix_portfolio_engine import HelixPortfolioData, run_helix_synchronized

        portfolio_data = HelixPortfolioData(
            daily={sym: data.daily[sym] for sym in config.helix_symbols if sym in data.daily},
            hourly={sym: data.hourly[sym] for sym in config.helix_symbols if sym in data.hourly},
            four_hour={sym: data.four_hour[sym] for sym in config.helix_symbols if sym in data.four_hour},
            daily_idx_maps={
                sym: data.daily_idx_maps[sym]
                for sym in config.helix_symbols
                if sym in data.daily_idx_maps
            },
            four_hour_idx_maps={
                sym: data.four_hour_idx_maps[sym]
                for sym in config.helix_symbols
                if sym in data.four_hour_idx_maps
            },
        )
        native = run_helix_synchronized(portfolio_data, config.build_helix_config())
        trades = _collect_native_trades(native.symbol_results)
        equity = np.array(native.combined_equity)
        return UnifiedPortfolioResult(
            combined_equity=equity,
            combined_equity_mtm=equity,
            combined_equity_realized=equity,
            combined_timestamps=np.array(native.combined_timestamps),
            heat_stats=_native_heat_stats(native),
            strategy_results=_empty_strategy_results("AKC_HELIX", trades),
            helix_trades=trades,
        )

    if strategy_id == "TPC":
        trades = _run_tpc_source_replay(data, config)
        total_pnl = sum(_strategy_trade_net_pnl("TPC", t) for t in trades)
        equity = np.array([float(config.initial_equity), float(config.initial_equity) + total_pnl])
        return UnifiedPortfolioResult(
            combined_equity=equity,
            combined_equity_mtm=equity,
            combined_equity_realized=equity,
            strategy_results=_empty_strategy_results("TPC", trades),
            tpc_trades=trades,
        )

    return None


def _combined_child_open_mtm_pnl(
    engine_groups: list[tuple[str, dict[str, object]]],
    config: UnifiedBacktestConfig | None = None,
    initial_equity: float | None = None,
) -> float:
    """Return only unrealized MTM from child engines, excluding realized equity."""
    open_mtm = 0.0
    strategy_by_label = {"atrss": "ATRSS", "helix": "AKC_HELIX"}
    for label, engines in engine_groups:
        unit_risk = (
            _strategy_initial_unit_risk_dollars(config, strategy_by_label.get(label, ""), float(initial_equity))
            if config is not None and initial_equity is not None
            else 0.0
        )
        for engine in engines.values():
            realized_equity = float(getattr(engine, "equity", 0.0) or 0.0)
            mark = _latest_engine_mark(engine, realized_equity)
            if np.isfinite(mark) and np.isfinite(realized_equity):
                raw_open_mtm = mark - realized_equity
                actual_open_risk = _engine_open_risk_dollars(engine)
                if unit_risk > 0.0 and actual_open_risk > 0.0:
                    open_mtm += raw_open_mtm / actual_open_risk * unit_risk
                else:
                    open_mtm += raw_open_mtm
    return float(open_mtm)


def _engine_open_risk_dollars(engine) -> float:
    point_value = float(getattr(engine, "point_value", 1.0) or 1.0)
    position = getattr(engine, "position", None)
    direction = getattr(position, "direction", None)
    if position is not None and str(direction).split(".")[-1] != "FLAT":
        legs = getattr(position, "legs", []) or []
        risk = 0.0
        for leg in legs:
            entry = float(getattr(leg, "entry_price", 0.0) or 0.0)
            stop = float(getattr(leg, "initial_stop", 0.0) or 0.0)
            qty = abs(float(getattr(leg, "qty", 0.0) or 0.0))
            risk += abs(entry - stop) * point_value * qty
        if risk > 0.0 and np.isfinite(risk):
            return float(risk)

    active = getattr(engine, "active_position", None)
    if active is not None:
        entry = float(getattr(active, "fill_price", 0.0) or 0.0)
        stop = float(getattr(active, "initial_stop", 0.0) or 0.0)
        qty = abs(float(getattr(active, "qty_open", 0.0) or 0.0))
        risk = abs(entry - stop) * point_value * qty
        if risk > 0.0 and np.isfinite(risk):
            return float(risk)
    return 0.0


def _overlay_transaction_cost(shares_delta: float, raw_price: float, slippage) -> float:
    shares = abs(float(shares_delta))
    price = float(raw_price)
    if shares <= 0 or price <= 0:
        return 0.0
    per_share_commission = shares * float(getattr(slippage, "commission_per_share_etf", 0.0) or 0.0)
    min_commission = float(getattr(slippage, "commission_min_etf_order", 0.0) or 0.0)
    commission = max(per_share_commission, min_commission) if min_commission > 0 else per_share_commission
    slip = price * float(getattr(slippage, "overlay_slip_bps", 0.0) or 0.0) / 10_000
    return commission + slip * shares


# ---------------------------------------------------------------------------
# Idle-capital overlay helpers
# ---------------------------------------------------------------------------

def _overlay_ema(series: np.ndarray, period: int) -> np.ndarray:
    """EMA with SMA seed (matches idle_capital_study.py)."""
    return compute_ema(series, period)


def _overlay_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder-smoothed RSI for overlay (self-contained, no cross-strategy imports)."""
    n = len(closes)
    out = np.full(n, 50.0)
    if n < 2:
        return out

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.empty(len(deltas), dtype=float)
    avg_loss = np.empty(len(deltas), dtype=float)

    seed = min(period, len(deltas))
    avg_gain[0] = float(np.mean(gains[:seed]))
    avg_loss[0] = float(np.mean(losses[:seed]))

    alpha = 1.0 / period
    for i in range(1, len(deltas)):
        avg_gain[i] = avg_gain[i - 1] * (1 - alpha) + gains[i] * alpha
        avg_loss[i] = avg_loss[i - 1] * (1 - alpha) + losses[i] * alpha

    for i in range(len(deltas)):
        ag = avg_gain[i]
        al = avg_loss[i]
        if al == 0:
            out[i + 1] = 100.0 if ag > 0 else 50.0
        else:
            rs = ag / al
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return out


def _overlay_macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD for overlay -> (line, signal_line, histogram). Self-contained.

    The MACD line is valid from index (slow-1) onward.  The signal line EMA
    is computed only over the valid portion of the MACD line, then placed back
    into a full-length array so indices align with the input closes.
    """
    n = len(closes)
    ema_fast = _overlay_ema(closes, fast)
    ema_slow = _overlay_ema(closes, slow)
    line = ema_fast - ema_slow  # valid from index slow-1

    # Compute signal EMA over the valid tail of the MACD line
    valid_start = slow - 1  # first index where line is not NaN
    valid_line = line[valid_start:]
    valid_sig = _overlay_ema(valid_line, signal)

    # Place back into full-length arrays
    sig = np.full(n, np.nan, dtype=float)
    sig[valid_start:] = valid_sig
    hist = line - sig
    return line, sig, hist


def _precompute_overlay_rsi(
    daily: dict[str, NumpyBars],
    config: UnifiedBacktestConfig,
) -> dict[str, np.ndarray]:
    """Precompute RSI arrays per overlay symbol."""
    rsi_arrays: dict[str, np.ndarray] = {}
    for sym in config.overlay_symbols:
        bars = daily.get(sym)
        if bars is None:
            continue
        overrides = config.overlay_rsi_overrides.get(sym, {})
        period = overrides.get("period", config.overlay_rsi_period)
        rsi_arrays[sym] = _overlay_rsi(bars.closes, period)
    return rsi_arrays


def _precompute_overlay_macd(
    daily: dict[str, NumpyBars],
    config: UnifiedBacktestConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Precompute MACD arrays per overlay symbol."""
    macd_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sym in config.overlay_symbols:
        bars = daily.get(sym)
        if bars is None:
            continue
        overrides = config.overlay_macd_overrides.get(sym, {})
        fast = overrides.get("fast", config.overlay_macd_fast)
        slow = overrides.get("slow", config.overlay_macd_slow)
        sig = overrides.get("signal", config.overlay_macd_signal)
        macd_arrays[sym] = _overlay_macd(bars.closes, fast, slow, sig)
    return macd_arrays


def _compute_overlay_score(
    sym: str,
    d_idx: int,
    overlay_emas: dict[str, tuple[np.ndarray, np.ndarray]],
    overlay_rsi: dict[str, np.ndarray],
    overlay_macd: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: UnifiedBacktestConfig,
) -> float:
    """Compute composite overlay score [0, 1] for a symbol at a daily bar index.

    Components:
    - EMA score (weight 0.40): 0-1 based on spread magnitude
    - RSI score (weight 0.30): 1.0 in sweet spot [50-65], 0.0 if overbought or < bull_min
    - MACD score (weight 0.30): 1.0 if histogram positive+rising, 0.0 if negative+falling
    """
    W_EMA, W_RSI, W_MACD = config.overlay_score_weights

    # --- EMA score ---
    ema_score = 0.0
    ema_pair = overlay_emas.get(sym)
    if ema_pair is not None:
        ema_f, ema_s = ema_pair
        if not (np.isnan(ema_f[d_idx]) or np.isnan(ema_s[d_idx])):
            spread = (ema_f[d_idx] - ema_s[d_idx]) / ema_s[d_idx] if ema_s[d_idx] != 0 else 0.0
            if spread > 0:
                ema_score = min(spread / config.overlay_ema_spread_norm, 1.0)

    # --- RSI score ---
    rsi_score = 0.0
    rsi_arr = overlay_rsi.get(sym)
    if rsi_arr is not None:
        rsi_overrides = config.overlay_rsi_overrides.get(sym, {})
        overbought = rsi_overrides.get("overbought", config.overlay_rsi_overbought)
        bull_min = rsi_overrides.get("bull_min", config.overlay_rsi_bull_min)
        rsi_val = rsi_arr[d_idx]
        if rsi_val >= overbought:
            rsi_score = 0.0
        elif rsi_val < bull_min:
            rsi_score = 0.0
        elif 50 <= rsi_val <= 65:
            rsi_score = 1.0
        elif rsi_val < 50:
            # Ramp from bull_min to 50
            rsi_score = (rsi_val - bull_min) / (50 - bull_min) if (50 - bull_min) > 0 else 0.0
        else:
            # Ramp down from 65 to overbought
            rsi_score = (overbought - rsi_val) / (overbought - 65) if (overbought - 65) > 0 else 0.0

    # --- MACD score ---
    macd_score = 0.0
    macd_data = overlay_macd.get(sym)
    if macd_data is not None:
        _line, _sig, hist = macd_data
        if not np.isnan(hist[d_idx]):
            hist_val = hist[d_idx]
            hist_prev = hist[d_idx - 1] if d_idx > 0 and not np.isnan(hist[d_idx - 1]) else hist_val
            rising = hist_val > hist_prev
            s_pr, s_pf, s_nf, s_nr = config.overlay_macd_scores
            if hist_val > 0 and rising:
                macd_score = s_pr
            elif hist_val > 0:
                macd_score = s_pf  # positive but falling
            elif hist_val < 0 and not rising:
                macd_score = s_nf  # negative and falling
            else:
                macd_score = s_nr  # negative but rising (recovery)

    return W_EMA * ema_score + W_RSI * rsi_score + W_MACD * macd_score


def _precompute_overlay_emas(
    daily: dict[str, NumpyBars],
    config: UnifiedBacktestConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Precompute fast/slow EMAs on daily closes for overlay symbols."""
    emas: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym in config.overlay_symbols:
        bars = daily.get(sym)
        if bars is None:
            continue
        fast, slow = config.overlay_ema_overrides.get(
            sym, (config.overlay_ema_fast, config.overlay_ema_slow),
        )
        ema_fast = _overlay_ema(bars.closes, fast)
        ema_slow = _overlay_ema(bars.closes, slow)
        emas[sym] = (ema_fast, ema_slow)
    return emas


def _build_daily_date_index(
    daily: dict[str, NumpyBars],
    symbols: list[str],
) -> dict[str, dict[date, int]]:
    """Build {symbol: {date: bar_index}} for daily data lookup."""
    idx: dict[str, dict[date, int]] = {}
    for sym in symbols:
        bars = daily.get(sym)
        if bars is None:
            continue
        mapping: dict[date, int] = {}
        for i in range(len(bars.times)):
            t = bars.times[i]
            if hasattr(t, 'item'):
                t = t.item()
            dt = pd.Timestamp(t).date() if not isinstance(t, date) else t
            if isinstance(dt, datetime):
                dt = dt.date()
            mapping[dt] = i
        idx[sym] = mapping
    return idx


def _rebalance_overlay(
    config: UnifiedBacktestConfig,
    daily: dict[str, NumpyBars],
    overlay_emas: dict[str, tuple[np.ndarray, np.ndarray]],
    daily_date_idx: dict[str, dict[date, int]],
    current_date: date,
    portfolio_equity: float,
    overlay_shares: dict[str, float],
    overlay_rsi: dict[str, np.ndarray] | None = None,
    overlay_macd: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    overlay_in_position: dict[str, bool] | None = None,
    bear_regime_active: bool = False,
) -> None:
    """Rebalance overlay positions at day boundary.

    Legacy "ema" mode: equal/weighted across bullish symbols (EMA fast > slow).
    "multi" mode: EMA+RSI+MACD composite scoring with hysteresis and adaptive sizing.
    """
    if config.overlay_mode == "multi" and overlay_rsi is not None and overlay_macd is not None:
        _rebalance_overlay_multi(
            config, daily, overlay_emas, daily_date_idx, current_date,
            portfolio_equity, overlay_shares, overlay_rsi, overlay_macd,
            overlay_in_position or {}, bear_regime_active=bear_regime_active,
        )
        return

    # --- Legacy "ema" mode (shared decision/allocation semantics) ---
    signals: dict[str, bool] = {}

    for sym in config.overlay_symbols:
        sym_idx = daily_date_idx.get(sym)
        if sym_idx is None:
            signals[sym] = False
            continue
        d_idx = sym_idx.get(current_date)
        fast, slow = config.overlay_ema_overrides.get(
            sym, (config.overlay_ema_fast, config.overlay_ema_slow),
        )
        if d_idx is None or d_idx < slow:
            signals[sym] = False
            continue
        ema_f, ema_s = overlay_emas[sym]
        signals[sym] = bool(ema_f[d_idx] > ema_s[d_idx])

    execution_prices: dict[str, float] = {}
    for sym in config.overlay_symbols:
        sym_idx = daily_date_idx.get(sym)
        if sym_idx is None:
            continue
        d_idx = sym_idx.get(current_date)
        if d_idx is None:
            continue
        if d_idx + 1 < len(daily[sym].opens):
            execution_prices[sym] = float(daily[sym].opens[d_idx + 1])
        else:
            execution_prices[sym] = float(daily[sym].closes[d_idx])

    target_shares = allocate_weighted_targets(
        config.overlay_symbols,
        signals=signals,
        prices=execution_prices,
        portfolio_equity=portfolio_equity,
        max_equity_pct=config.overlay_max_pct,
        weights=config.overlay_weights,
    )
    if bear_regime_active:
        for sym in config.overlay_symbols:
            if overlay_shares.get(sym, 0.0) == 0.0 and target_shares.get(sym, 0) > 0:
                target_shares[sym] = 0
    for sym in config.overlay_symbols:
        overlay_shares[sym] = float(target_shares.get(sym, 0))


def _rebalance_overlay_multi(
    config: UnifiedBacktestConfig,
    daily: dict[str, NumpyBars],
    overlay_emas: dict[str, tuple[np.ndarray, np.ndarray]],
    daily_date_idx: dict[str, dict[date, int]],
    current_date: date,
    portfolio_equity: float,
    overlay_shares: dict[str, float],
    overlay_rsi: dict[str, np.ndarray],
    overlay_macd: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    overlay_in_position: dict[str, bool],
    bear_regime_active: bool = False,
) -> None:
    """Multi-indicator overlay rebalance with scoring, hysteresis, and adaptive sizing."""
    available = max(portfolio_equity * config.overlay_max_pct, 0.0)
    scores: dict[str, float] = {}
    active_syms: dict[str, float] = {}  # sym -> score for syms that pass hysteresis

    for sym in config.overlay_symbols:
        sym_idx = daily_date_idx.get(sym)
        if sym_idx is None:
            overlay_shares[sym] = 0.0
            continue
        d_idx = sym_idx.get(current_date)
        fast, slow = config.overlay_ema_overrides.get(
            sym, (config.overlay_ema_fast, config.overlay_ema_slow),
        )
        if d_idx is None or d_idx < slow:
            overlay_shares[sym] = 0.0
            overlay_in_position[sym] = False
            continue

        score = _compute_overlay_score(
            sym, d_idx, overlay_emas, overlay_rsi, overlay_macd, config,
        )
        scores[sym] = score

        currently_in = overlay_in_position.get(sym, False)
        if currently_in:
            # Hysteresis: stay in unless score drops below exit threshold
            if score < config.overlay_exit_score_max:
                overlay_in_position[sym] = False
            else:
                active_syms[sym] = score
        else:
            # Entry: require score >= entry threshold
            if score >= config.overlay_entry_score_min:
                if bear_regime_active and overlay_shares.get(sym, 0.0) == 0.0:
                    overlay_in_position[sym] = False
                    continue
                overlay_in_position[sym] = True
                active_syms[sym] = score

    # Allocate capital across active symbols
    if not active_syms:
        for sym in config.overlay_symbols:
            overlay_shares[sym] = 0.0
        return

    # Weighted allocation: base weight * adaptive scaling
    if config.overlay_weights is None:
        base_weights = {s: 1.0 for s in active_syms}
    else:
        base_weights = {s: config.overlay_weights.get(s, 1.0) for s in active_syms}
    total_base_w = sum(base_weights.values())

    for sym in config.overlay_symbols:
        if sym not in active_syms:
            overlay_shares[sym] = 0.0
            continue

        sym_idx = daily_date_idx.get(sym, {})
        d_idx = sym_idx.get(current_date)
        if d_idx is None:
            overlay_shares[sym] = 0.0
            continue
        # Execute at next day's open (signal at close, fill at next open)
        if d_idx + 1 < len(daily[sym].opens):
            price = daily[sym].opens[d_idx + 1]
        else:
            price = daily[sym].closes[d_idx]  # last day fallback
        if price <= 0 or total_base_w <= 0:
            overlay_shares[sym] = 0.0
            continue

        # Base allocation from weights
        alloc_pct = base_weights[sym] / total_base_w

        # Adaptive sizing: scale allocation by signal strength
        if config.overlay_adaptive_sizing:
            score = active_syms[sym]
            # Map score to [min_alloc_pct, max_alloc_pct]
            alloc_range = config.overlay_max_alloc_pct - config.overlay_min_alloc_pct
            size_factor = config.overlay_min_alloc_pct + alloc_range * score
        else:
            size_factor = 1.0

        alloc = available * alloc_pct * size_factor
        overlay_shares[sym] = int(alloc / price)


def _compute_overlay_value(
    overlay_shares: dict[str, float],
    daily: dict[str, NumpyBars],
    daily_date_idx: dict[str, dict[date, int]],
    current_date: date,
) -> float:
    """MTM value of overlay positions using daily close prices."""
    value = 0.0
    for sym, shares in overlay_shares.items():
        if shares == 0.0:
            continue
        sym_idx = daily_date_idx.get(sym)
        if sym_idx is None:
            continue
        d_idx = sym_idx.get(current_date)
        if d_idx is None:
            continue
        value += shares * daily[sym].closes[d_idx]
    return value


# ---------------------------------------------------------------------------
# Main entry: run_unified
# ---------------------------------------------------------------------------

def run_unified(
    data: UnifiedPortfolioData,
    config: UnifiedBacktestConfig,
) -> UnifiedPortfolioResult:
    """Run all strategies stepping through time with shared portfolio constraints."""

    standalone_result = _run_native_standalone_if_requested(data, config)
    if standalone_result is not None:
        return standalone_result

    # When fixed_qty is used, heat enforcement is disabled because position
    # sizes don't scale with equity — the R-based thresholds become meaningless.
    skip_heat = (not config.portfolio_constraints_enabled) or config.fixed_qty is not None

    atrss_bt = config.build_atrss_config()
    helix_bt = config.build_helix_config()
    tpc_source_trades = _run_tpc_source_replay(data, config)
    tpc_entries_by_ts, tpc_exits_by_ts = _build_tpc_trade_events(tpc_source_trades)

    # --- Instantiate engines ---
    atrss_engines: dict[str, BacktestEngine] = {}
    for sym in config.atrss_symbols:
        cfg = ATRSS_SYMBOL_CONFIGS.get(sym)
        if cfg is None or sym not in data.atrss_hourly:
            continue
        mult = config.symbol_risk_multipliers.get(f"ATRSS:{sym}", 1.0)
        if mult != 1.0:
            cfg = _dc_replace(cfg, base_risk_pct=cfg.base_risk_pct * mult)
        atrss_engines[sym] = BacktestEngine(
            symbol=sym, cfg=cfg, bt_config=atrss_bt,
            point_value=cfg.multiplier,
        )

    helix_engines: dict[str, HelixEngine] = {}
    for sym in config.helix_symbols:
        cfg = HELIX_SYMBOL_CONFIGS.get(sym)
        if cfg is None or sym not in data.hourly:
            continue
        mult = config.symbol_risk_multipliers.get(f"AKC_HELIX:{sym}", 1.0)
        if mult != 1.0:
            cfg = _dc_replace(cfg, base_risk_pct=cfg.base_risk_pct * mult)
        eng = HelixEngine(
            symbol=sym, cfg=cfg, bt_config=helix_bt,
            point_value=cfg.multiplier,
        )
        # Precompute indicator arrays once (avoids per-bar recomputation)
        eng._precompute_indicators(data.hourly[sym], data.four_hour[sym])
        helix_engines[sym] = eng

    if not atrss_engines and not helix_engines and not tpc_source_trades:
        logger.warning("No engines created — check symbols and data")
        return UnifiedPortfolioResult()

    # --- Build unified timestamp index ---
    # ATRSS uses RTH-filtered hourly; Helix uses full hourly.
    time_sets: dict[str, dict] = {}
    all_times_set: set = set()
    all_engines_flat: dict[str, object] = {}

    for sym, eng in atrss_engines.items():
        key = f"atrss_{sym}"
        all_engines_flat[key] = eng
        times = data.atrss_hourly[sym].times
        mapping = {}
        for i in range(len(times)):
            t = times[i].item() if hasattr(times[i], 'item') else times[i]
            mapping[t] = i
        time_sets[key] = mapping
        all_times_set.update(mapping.keys())

    for sym, eng in helix_engines.items():
        key = f"helix_{sym}"
        all_engines_flat[key] = eng
        times = data.hourly[sym].times
        mapping = {}
        for i in range(len(times)):
            t = times[i].item() if hasattr(times[i], 'item') else times[i]
            mapping[t] = i
        time_sets[key] = mapping
        all_times_set.update(mapping.keys())

    all_times_set.update(tpc_entries_by_ts.keys())
    all_times_set.update(tpc_exits_by_ts.keys())

    unified_ts = sorted(all_times_set)
    logger.info("Unified timeline: %d bars", len(unified_ts))

    # --- Pre-cache _to_datetime conversions for engine hot-paths ---
    # Helix converts numpy datetime64 to Python
    # datetime thousands of times per bar (lookback-window scans).
    # Pre-converting all unique timestamps into a cache and monkey-patching
    # the engines' _to_datetime eliminates redundant pd.Timestamp() calls.
    from datetime import timezone as _tz
    _dt_cache: dict = {}
    def _fast_to_datetime(ts, _c=_dt_cache) -> datetime:
        r = _c.get(ts)
        if r is not None:
            return r
        if isinstance(ts, datetime):
            r = ts if ts.tzinfo else ts.replace(tzinfo=_tz.utc)
        else:
            r = pd.Timestamp(ts).to_pydatetime()
        _c[ts] = r
        return r
    # Pre-populate from all timestamp arrays
    for _bars in (*data.hourly.values(), *data.atrss_hourly.values(),
                  *data.four_hour.values(), *data.daily.values()):
        for _t in _bars.times:
            _fast_to_datetime(_t)
    # Patch engine static methods to use the cache
    HelixEngine._to_datetime = staticmethod(_fast_to_datetime)

    # --- Initialize trackers ---
    init_eq = config.initial_equity
    portfolio_equity = init_eq
    peak_risk_equity = init_eq
    risk_sizing_equity = init_eq
    risk_scale = 1.0

    heat_tracker = PortfolioHeatTracker(
        heat_cap_R=config.heat_cap_R,
        portfolio_daily_stop_R=config.portfolio_daily_stop_R,
        strategy_slots=[config.atrss, config.helix, config.tpc],
        equity=init_eq,
        simulate_live_r_normalization=config.simulate_live_r_normalization,
        reserve_idle_higher_priority=config.reserve_idle_higher_priority,
    )
    portfolio_rule_replay = None if skip_heat else _SwingPortfolioRuleReplayAdapter(config, init_eq)
    coordinator = BacktestCoordinator(
        enable_tighten=config.enable_atrss_helix_tighten,
        enable_size_boost=config.enable_atrss_helix_size_boost,
    )

    # --- Overlay initialization ---
    overlay_shares: dict[str, float] = {}
    overlay_prev_value: float = 0.0
    overlay_cumulative_pnl: float = 0.0
    overlay_total_commission: float = 0.0
    overlay_per_sym_pnl: dict[str, float] = {}
    overlay_prev_sym_value: dict[str, float] = {}
    overlay_emas: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    overlay_rsi: dict[str, np.ndarray] = {}
    overlay_macd: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    overlay_in_position: dict[str, bool] = {}
    daily_date_idx_syms = list(set(config.overlay_symbols if config.overlay_enabled else []))
    daily_date_idx: dict[str, dict[date, int]] = _build_daily_date_index(data.daily, daily_date_idx_syms)
    if config.overlay_enabled:
        overlay_emas = _precompute_overlay_emas(data.daily, config)
        if config.overlay_mode == "multi":
            overlay_rsi = _precompute_overlay_rsi(data.daily, config)
            overlay_macd = _precompute_overlay_macd(data.daily, config)
        for sym in config.overlay_symbols:
            overlay_shares[sym] = 0.0
            overlay_per_sym_pnl[sym] = 0.0
            overlay_prev_sym_value[sym] = 0.0
            overlay_in_position[sym] = False

    # Track previous trade counts to detect actual trade closes
    prev_trade_counts: dict[str, int] = {}
    for label, engines in [("atrss", atrss_engines), ("helix", helix_engines)]:
        for sym in engines:
            prev_trade_counts[f"{label}_{sym}"] = 0

    equity_curve_mtm: list[float] = []
    equity_curve_realized: list[float] = []
    timestamps: list = []
    heat_samples: list[float] = []
    prev_date: date | None = None
    entry_attempts: dict[str, int] = {"ATRSS": 0, "AKC_HELIX": 0, "TPC": 0}
    accepted_entries: dict[str, int] = {"ATRSS": 0, "AKC_HELIX": 0, "TPC": 0}
    blocked_entries: dict[str, int] = {"ATRSS": 0, "AKC_HELIX": 0, "TPC": 0}
    suppressed_entry_retries: dict[str, int] = {"ATRSS": 0, "AKC_HELIX": 0, "TPC": 0}
    entry_event_log: list[dict] = []
    heat_rejection_log: list[dict] = []
    coordination_event_log: list[dict] = []
    portfolio_rule_event_log: list[dict] = []
    daily_stop_activations = 0
    active_tpc_trades: dict[int, object] = {}
    active_tpc_risks: dict[int, float] = {}
    accepted_tpc_trades: list = []
    atrss_blocked_until: dict[tuple[str, str, int], datetime] = {}
    atrss_block_ttl = timedelta(hours=float(ORDER_EXPIRY_HOURS))

    warmup_d = config.warmup_daily
    warmup_h = config.warmup_hourly

    # Track ATRSS position state for coordination
    atrss_prev_positions: dict[str, bool] = {sym: False for sym in atrss_engines}

    # --- Main bar loop ---
    # Apply ablation patches for ATRSS module-level constants (matches run_independent)
    _patch = _AblationPatch(atrss_bt.flags, atrss_bt.param_overrides)
    _patch.__enter__()
    _helix_patch = _HelixAblationPatch(helix_bt.flags, helix_bt.param_overrides)
    _helix_patch.__enter__()

    for ts in unified_ts:

        # Detect daily boundary
        ts_dt = pd.Timestamp(ts).to_pydatetime() if not isinstance(ts, datetime) else ts
        current_date = ts_dt.date() if hasattr(ts_dt, 'date') else pd.Timestamp(ts).date()
        if prev_date is not None and current_date != prev_date:
            heat_tracker.on_new_day()

            # --- Overlay: MTM delta from previous day, then rebalance ---
            # Overlay PnL is tracked separately so active engines don't see it
            # in their sizing equity.  Use overlay_last_daily (most recent
            # trading day with a daily bar) for MTM/rebalance.  Skip weekends
            # and holidays where the hourly data has bars but daily ETF data
            # does not.
            if config.overlay_enabled and prev_date in daily_date_idx.get(config.overlay_symbols[0], {}):
                new_value = _compute_overlay_value(
                    overlay_shares, data.daily, daily_date_idx, prev_date,
                )
                delta = new_value - overlay_prev_value
                overlay_cumulative_pnl += delta

                # Per-symbol overlay PnL tracking
                for _osym in config.overlay_symbols:
                    _sh = overlay_shares.get(_osym, 0.0)
                    _sv = 0.0
                    if _sh != 0.0:
                        _sidx = daily_date_idx.get(_osym, {}).get(prev_date)
                        if _sidx is not None:
                            _sv = _sh * data.daily[_osym].closes[_sidx]
                    _sdelta = _sv - overlay_prev_sym_value.get(_osym, 0.0)
                    overlay_per_sym_pnl[_osym] = overlay_per_sym_pnl.get(_osym, 0.0) + _sdelta

                total_equity = portfolio_equity + overlay_cumulative_pnl
                old_overlay_shares = dict(overlay_shares)
                _rebalance_overlay(
                    config, data.daily, overlay_emas, daily_date_idx,
                    prev_date, total_equity, overlay_shares,
                    overlay_rsi=overlay_rsi or None,
                    overlay_macd=overlay_macd or None,
                    overlay_in_position=overlay_in_position,
                )
                # Overlay transaction costs: commission + slippage on share deltas
                for _osym in config.overlay_symbols:
                    old_sh = old_overlay_shares.get(_osym, 0.0)
                    new_sh = overlay_shares.get(_osym, 0.0)
                    delta_sh = abs(new_sh - old_sh)
                    _sidx = daily_date_idx.get(_osym, {}).get(prev_date)
                    if delta_sh > 0 and _sidx is not None and _sidx + 1 < len(data.daily[_osym].opens):
                        cost = _overlay_transaction_cost(
                            delta_sh,
                            data.daily[_osym].opens[_sidx + 1],
                            config.slippage,
                        )
                        overlay_total_commission += cost
                        overlay_cumulative_pnl -= cost

                # Baseline for next MTM: next-open for changed positions
                # (execution price), close for held (preserves overnight gap).
                overlay_prev_value = 0.0
                for _osym in config.overlay_symbols:
                    old_sh = old_overlay_shares.get(_osym, 0.0)
                    new_sh = overlay_shares.get(_osym, 0.0)
                    _sv = 0.0
                    if new_sh != 0.0:
                        _sidx = daily_date_idx.get(_osym, {}).get(prev_date)
                        if _sidx is not None:
                            if old_sh != new_sh and _sidx + 1 < len(data.daily[_osym].opens):
                                _sv = new_sh * data.daily[_osym].opens[_sidx + 1]
                            else:
                                _sv = new_sh * data.daily[_osym].closes[_sidx]
                    overlay_prev_sym_value[_osym] = _sv
                    overlay_prev_value += _sv

        if config.dynamic_risk_enabled:
            peak_risk_equity = max(peak_risk_equity, portfolio_equity)
            drawdown_pct = (peak_risk_equity - portfolio_equity) / peak_risk_equity if peak_risk_equity > 0 else 0.0
            risk_scale = _drawdown_risk_multiplier(drawdown_pct, config.drawdown_risk_tiers)
            risk_sizing_equity = max(portfolio_equity * risk_scale, 0.0)
        else:
            risk_sizing_equity = portfolio_equity
            risk_scale = 1.0
        heat_tracker.update_unit_risk(
            risk_sizing_equity,
            [config.atrss, config.helix, config.tpc],
        )

        prev_date = current_date

        # === Step 1: ATRSS engines (priority 0) ===
        for sym, engine in atrss_engines.items():
            key = f"atrss_{sym}"
            bar_idx = time_sets[key].get(ts)
            if bar_idx is None:
                continue

            engine.sizing_equity = risk_sizing_equity
            try:
                candidates = engine.step_bar(
                    data.daily[sym], data.atrss_hourly[sym],
                    data.atrss_daily_idx_maps[sym], bar_idx,
                    warmup_d, warmup_h,
                )
            except (OverflowError, ValueError):
                # Edge case: trigger ≈ stop → infinite qty; skip this bar
                candidates = []

            # Heat-gated candidate submission
            if candidates:
                bar_time = _to_datetime(ts)
                for cand in candidates:
                    signal_key = (sym, _enum_label(cand.type), int(cand.direction))
                    blocked_until = atrss_blocked_until.get(signal_key)
                    if blocked_until is not None:
                        if bar_time < blocked_until:
                            suppressed_entry_retries["ATRSS"] += 1
                            continue
                        atrss_blocked_until.pop(signal_key, None)
                    entry_attempts["ATRSS"] += 1
                    risk_dollars = abs(cand.trigger_price - cand.initial_stop) * engine.point_value * cand.qty
                    if skip_heat:
                        setattr(cand, "portfolio_size_mult", 1.0)
                        engine.submit_candidate(cand, bar_time)
                        accepted_entries["ATRSS"] += 1
                        atrss_blocked_until.pop(signal_key, None)
                        _log_entry_event(
                            entry_event_log, heat_tracker, bar_time, "ATRSS", sym, "entry",
                            "accepted", "", risk_dollars,
                            direction=int(cand.direction),
                            entry_price=cand.trigger_price,
                            stop_price=cand.initial_stop,
                            quantity=cand.qty,
                            signal_type=_enum_label(cand.type),
                            quality_score=getattr(cand, "rank_score", None),
                            entry_already_filled=False,
                            metadata={"risk_scale": risk_scale, "atrh": getattr(cand, "atrh", 0.0)},
                        )
                    else:
                        ok, reason = heat_tracker.can_enter("ATRSS", risk_dollars)
                        adjusted_risk_dollars = risk_dollars
                        portfolio_size_mult = 1.0
                        if ok and portfolio_rule_replay is not None:
                            portfolio_rule_replay.refresh(
                                equity=portfolio_equity,
                                unit_equity=risk_sizing_equity,
                                exposures=_portfolio_rule_exposures(
                                    config,
                                    init_eq,
                                    atrss_engines,
                                    helix_engines,
                                    active_tpc_trades,
                                    active_tpc_risks,
                                ),
                            )
                            rule_result = portfolio_rule_replay.check_entry(
                                strategy_id="ATRSS",
                                direction=_direction_label(cand.direction),
                                risk_dollars=risk_dollars,
                                symbol=sym,
                                qty=cand.qty,
                            )
                            if not rule_result.approved:
                                ok = False
                                reason = f"Portfolio rule: {rule_result.denial_reason}"
                                portfolio_rule_event_log.append(
                                    {
                                        "time": bar_time,
                                        "strategy": "ATRSS",
                                        "symbol": sym,
                                        "result": "blocked",
                                        "reason": reason,
                                    }
                                )
                            else:
                                portfolio_size_mult = float(rule_result.size_multiplier or 1.0)
                                ratio, _new_qty = _apply_portfolio_size_multiplier_to_atrss_candidate(
                                    cand,
                                    portfolio_size_mult,
                                )
                                adjusted_risk_dollars = risk_dollars * ratio
                                if portfolio_size_mult != 1.0:
                                    portfolio_rule_event_log.append(
                                        {
                                            "time": bar_time,
                                            "strategy": "ATRSS",
                                            "symbol": sym,
                                            "result": "sized",
                                            "size_multiplier": portfolio_size_mult,
                                            "quantity": _new_qty,
                                        }
                                    )
                        if ok:
                            setattr(cand, "portfolio_size_mult", portfolio_size_mult)
                            engine.submit_candidate(cand, bar_time)
                            heat_tracker.reserve_entry("ATRSS", adjusted_risk_dollars)
                            accepted_entries["ATRSS"] += 1
                            atrss_blocked_until.pop(signal_key, None)
                            _log_entry_event(
                                entry_event_log, heat_tracker, bar_time, "ATRSS", sym, "entry",
                                "accepted", "", adjusted_risk_dollars,
                                direction=int(cand.direction),
                                entry_price=cand.trigger_price,
                                stop_price=cand.initial_stop,
                                quantity=cand.qty,
                                signal_type=_enum_label(cand.type),
                                quality_score=getattr(cand, "rank_score", None),
                                entry_already_filled=False,
                                metadata={
                                    "risk_scale": risk_scale,
                                    "atrh": getattr(cand, "atrh", 0.0),
                                    "portfolio_size_mult": portfolio_size_mult,
                                    "requested_risk_dollars": risk_dollars,
                                },
                            )
                        else:
                            blocked_entries["ATRSS"] += 1
                            atrss_blocked_until[signal_key] = bar_time + atrss_block_ttl
                            event = _log_entry_event(
                                entry_event_log, heat_tracker, bar_time, "ATRSS", sym, "entry",
                                "blocked", reason, risk_dollars,
                                direction=int(cand.direction),
                                entry_price=cand.trigger_price,
                                stop_price=cand.initial_stop,
                                quantity=cand.qty,
                                signal_type=_enum_label(cand.type),
                                quality_score=getattr(cand, "rank_score", None),
                                entry_already_filled=False,
                                metadata={"risk_scale": risk_scale, "atrh": getattr(cand, "atrh", 0.0)},
                            )
                            heat_rejection_log.append(event)
                            if "daily stop" in reason.lower():
                                daily_stop_activations += 1

            # Track ATRSS position changes for coordination
            has_pos = engine.position.direction != AtrssDirection.FLAT
            had_pos = atrss_prev_positions.get(sym, False)
            if has_pos and not had_pos:
                entry_price = engine.position.base_leg.entry_price if engine.position.base_leg else 0.0
                coordinator.on_atrss_position_change(sym, engine.position.direction, entry_price)
            elif not has_pos and had_pos:
                coordinator.on_atrss_position_change(sym, 0, 0.0)
            atrss_prev_positions[sym] = has_pos

        # === Step 2: Coordination Rule 1 — tighten Helix stops ===
        for tighten_sym in coordinator.consume_tighten_events():
            if tighten_sym in helix_engines:
                h_eng = helix_engines[tighten_sym]
                pos = h_eng.active_position
                if pos is not None and pos.qty_open > 0:
                    # Tighten to breakeven (fill_price)
                    be_price = pos.fill_price
                    if pos.setup.direction == HelixDirection.LONG:
                        if be_price > pos.current_stop:
                            pos.current_stop = be_price
                            coordinator.tighten_count += 1
                            coordination_event_log.append({"time": _to_datetime(ts), "type": "tighten", "trigger_strategy": "ATRSS", "target_strategy": "AKC_HELIX", "symbol": tighten_sym})
                    elif pos.setup.direction == HelixDirection.SHORT:
                        if be_price < pos.current_stop:
                            pos.current_stop = be_price
                            coordinator.tighten_count += 1
                            coordination_event_log.append({"time": _to_datetime(ts), "type": "tighten", "trigger_strategy": "ATRSS", "target_strategy": "AKC_HELIX", "symbol": tighten_sym})

        # === Step 3: Helix engines (priority 1) ===
        for sym, engine in helix_engines.items():
            key = f"helix_{sym}"
            bar_idx = time_sets[key].get(ts)
            if bar_idx is None:
                continue

            engine.sizing_equity = risk_sizing_equity

            # Snapshot position state before step
            had_position = engine.active_position is not None and engine.active_position.qty_open > 0

            try:
                engine._step_bar(
                    data.daily[sym], data.hourly[sym], data.four_hour[sym],
                    data.daily_idx_maps[sym], data.four_hour_idx_maps[sym],
                    bar_idx, warmup_d, warmup_h,
                )
            except (OverflowError, ValueError):
                continue

            # Detect new entry
            has_position = engine.active_position is not None and engine.active_position.qty_open > 0
            if has_position and not had_position:
                pos = engine.active_position
                risk_dollars = abs(pos.fill_price - pos.current_stop) * engine.point_value * pos.qty_open
                entry_attempts["AKC_HELIX"] += 1
                if skip_heat:
                    ok, reason = True, ""
                else:
                    ok, reason = heat_tracker.can_enter("AKC_HELIX", risk_dollars)
                adjusted_risk_dollars = risk_dollars
                portfolio_size_mult = 1.0
                if ok and not skip_heat and portfolio_rule_replay is not None:
                    portfolio_rule_replay.refresh(
                        equity=portfolio_equity,
                        unit_equity=risk_sizing_equity,
                        exposures=_portfolio_rule_exposures(
                            config,
                            init_eq,
                            atrss_engines,
                            helix_engines,
                            active_tpc_trades,
                            active_tpc_risks,
                        ),
                    )
                    rule_result = portfolio_rule_replay.check_entry(
                        strategy_id="AKC_HELIX",
                        direction=_direction_label(pos.setup.direction),
                        risk_dollars=risk_dollars,
                        symbol=sym,
                        qty=pos.qty_open,
                        skip_current_exposure=True,
                    )
                    if not rule_result.approved:
                        ok = False
                        reason = f"Portfolio rule: {rule_result.denial_reason}"
                        portfolio_rule_event_log.append(
                            {
                                "time": _to_datetime(ts),
                                "strategy": "AKC_HELIX",
                                "symbol": sym,
                                "result": "blocked",
                                "reason": reason,
                            }
                        )
                    else:
                        portfolio_size_mult = float(rule_result.size_multiplier or 1.0)
                        ratio, new_qty = _apply_portfolio_size_multiplier_to_helix(
                            engine,
                            portfolio_size_mult,
                        )
                        adjusted_risk_dollars = risk_dollars * ratio
                        if portfolio_size_mult != 1.0:
                            portfolio_rule_event_log.append(
                                {
                                    "time": _to_datetime(ts),
                                    "strategy": "AKC_HELIX",
                                    "symbol": sym,
                                    "result": "sized",
                                    "size_multiplier": portfolio_size_mult,
                                    "quantity": new_qty,
                                }
                            )
                if not ok:
                    # Force flatten: reverse the entry
                    blocked_entries["AKC_HELIX"] += 1
                    event = _log_entry_event(
                        entry_event_log, heat_tracker, _to_datetime(ts), "AKC_HELIX", sym, "entry",
                        "blocked", reason, risk_dollars,
                        direction=int(pos.setup.direction),
                        entry_price=pos.fill_price,
                        stop_price=pos.current_stop,
                        quantity=pos.qty_open,
                        signal_type=_enum_label(pos.setup.setup_class),
                        quality_score=float(getattr(pos.setup, "adx_at_entry", 0.0) or 0.0),
                        entry_already_filled=True,
                        metadata={
                            "risk_scale": risk_scale,
                            "origin_tf": str(getattr(pos.setup, "origin_tf", "") or ""),
                            "regime": str(getattr(pos, "regime_at_entry", "") or ""),
                            "regime_4h": str(getattr(pos.setup, "regime_4h_at_entry", "") or ""),
                            "div_mag_norm": float(getattr(pos.setup, "div_mag_norm", 0.0) or 0.0),
                        },
                    )
                    heat_rejection_log.append(event)
                    _force_flatten_helix(engine, pos)
                    if "daily stop" in reason.lower():
                        daily_stop_activations += 1
                else:
                    accepted_entries["AKC_HELIX"] += 1
                    heat_tracker.reserve_entry("AKC_HELIX", adjusted_risk_dollars)
                    _log_entry_event(
                        entry_event_log, heat_tracker, _to_datetime(ts), "AKC_HELIX", sym, "entry",
                        "accepted", "", adjusted_risk_dollars,
                        direction=int(pos.setup.direction),
                        entry_price=pos.fill_price,
                        stop_price=pos.current_stop,
                        quantity=engine.active_position.qty_open if engine.active_position is not None else pos.qty_open,
                        signal_type=_enum_label(pos.setup.setup_class),
                        quality_score=float(getattr(pos.setup, "adx_at_entry", 0.0) or 0.0),
                        entry_already_filled=True,
                        metadata={
                            "risk_scale": risk_scale,
                            "portfolio_size_mult": portfolio_size_mult,
                            "requested_risk_dollars": risk_dollars,
                            "origin_tf": str(getattr(pos.setup, "origin_tf", "") or ""),
                            "regime": str(getattr(pos, "regime_at_entry", "") or ""),
                            "regime_4h": str(getattr(pos.setup, "regime_4h_at_entry", "") or ""),
                            "div_mag_norm": float(getattr(pos.setup, "div_mag_norm", 0.0) or 0.0),
                        },
                    )
                    # Check coordination Rule 2: size boost
                    if pos.setup and hasattr(pos.setup, 'direction'):
                        direction = pos.setup.direction
                        if coordinator.has_atrss_position(sym, direction):
                            coordinator.boost_count += 1
                            coordination_event_log.append({"time": _to_datetime(ts), "type": "boost", "trigger_strategy": "ATRSS", "target_strategy": "AKC_HELIX", "symbol": sym})

        # === Step 3b: TPC completed-trade source replay (priority 2) ===
        # TPC uses its shared-core ETF replay to generate source trades; this
        # layer then admits those trades through family heat and daily stops.
        for trade_id, trade in tpc_exits_by_ts.get(ts, []):
            if trade_id not in active_tpc_trades:
                continue
            admitted_trade = active_tpc_trades[trade_id]
            trade_pnl = _normalised_trade_pnl("TPC", admitted_trade, config, init_eq)
            portfolio_equity += trade_pnl
            heat_tracker.record_trade_close("TPC", trade_pnl)
            accepted_tpc_trades.append(admitted_trade)
            active_tpc_trades.pop(trade_id, None)
            active_tpc_risks.pop(trade_id, None)

        for trade_id, trade in tpc_entries_by_ts.get(ts, []):
            sym = str(getattr(trade, "symbol", ""))
            risk_dollars = _trade_risk_dollars(trade)
            entry_attempts["TPC"] += 1
            if skip_heat:
                ok, reason = True, ""
            else:
                ok, reason = heat_tracker.can_enter("TPC", risk_dollars)
            adjusted_trade = trade
            adjusted_risk_dollars = risk_dollars
            portfolio_size_mult = 1.0
            adjusted_qty = int(abs(float(getattr(trade, "qty", 0.0) or 0.0)))
            if ok and not skip_heat and portfolio_rule_replay is not None:
                portfolio_rule_replay.refresh(
                    equity=portfolio_equity,
                    unit_equity=risk_sizing_equity,
                    exposures=_portfolio_rule_exposures(
                        config,
                        init_eq,
                        atrss_engines,
                        helix_engines,
                        active_tpc_trades,
                        active_tpc_risks,
                    ),
                )
                rule_result = portfolio_rule_replay.check_entry(
                    strategy_id="TPC",
                    direction=_direction_label(getattr(trade, "direction", 0)),
                    risk_dollars=risk_dollars,
                    symbol=sym,
                    qty=adjusted_qty,
                )
                if not rule_result.approved:
                    ok = False
                    reason = f"Portfolio rule: {rule_result.denial_reason}"
                    portfolio_rule_event_log.append(
                        {
                            "time": _to_datetime(ts),
                            "strategy": "TPC",
                            "symbol": sym,
                            "result": "blocked",
                            "reason": reason,
                        }
                    )
                else:
                    portfolio_size_mult = float(rule_result.size_multiplier or 1.0)
                    adjusted_trade, ratio, adjusted_qty = _scaled_tpc_trade(trade, portfolio_size_mult)
                    adjusted_risk_dollars = risk_dollars * ratio
                    if portfolio_size_mult != 1.0:
                        portfolio_rule_event_log.append(
                            {
                                "time": _to_datetime(ts),
                                "strategy": "TPC",
                                "symbol": sym,
                                "result": "sized",
                                "size_multiplier": portfolio_size_mult,
                                "quantity": adjusted_qty,
                            }
                        )
            if ok:
                accepted_entries["TPC"] += 1
                active_tpc_trades[trade_id] = adjusted_trade
                active_tpc_risks[trade_id] = adjusted_risk_dollars
                heat_tracker.reserve_entry("TPC", adjusted_risk_dollars)
                _log_entry_event(
                    entry_event_log, heat_tracker, _to_datetime(ts), "TPC", sym, "entry",
                    "accepted", "", adjusted_risk_dollars,
                    direction=int(getattr(adjusted_trade, "direction", 0) or 0),
                    entry_price=float(getattr(adjusted_trade, "entry_price", 0.0) or 0.0),
                    stop_price=float(getattr(adjusted_trade, "initial_stop", 0.0) or 0.0),
                    quantity=float(getattr(adjusted_trade, "qty", 0.0) or 0.0),
                    signal_type=str(getattr(adjusted_trade, "entry_type", "") or ""),
                    quality_score=float(getattr(adjusted_trade, "quality_score", 0.0) or 0.0),
                    entry_already_filled=True,
                    metadata={
                        "risk_scale": risk_scale,
                        "source": "tpc_completed_trade_replay",
                        "portfolio_size_mult": portfolio_size_mult,
                        "requested_risk_dollars": risk_dollars,
                    },
                )
            else:
                blocked_entries["TPC"] += 1
                event = _log_entry_event(
                    entry_event_log, heat_tracker, _to_datetime(ts), "TPC", sym, "entry",
                    "blocked", reason, risk_dollars,
                    direction=int(getattr(trade, "direction", 0) or 0),
                    entry_price=float(getattr(trade, "entry_price", 0.0) or 0.0),
                    stop_price=float(getattr(trade, "initial_stop", 0.0) or 0.0),
                    quantity=float(getattr(trade, "qty", 0.0) or 0.0),
                    signal_type=str(getattr(trade, "entry_type", "") or ""),
                    quality_score=float(getattr(trade, "quality_score", 0.0) or 0.0),
                    entry_already_filled=True,
                    metadata={"risk_scale": risk_scale, "source": "tpc_completed_trade_replay"},
                )
                heat_rejection_log.append(event)
                if "daily stop" in reason.lower():
                    daily_stop_activations += 1

        # === Step 4: Update portfolio equity from normalized closed-trade PnL ===
        for label, engines, strat_id in [
            ("atrss", atrss_engines, "ATRSS"),
            ("helix", helix_engines, "AKC_HELIX"),
        ]:
            for sym, eng in engines.items():
                key = f"{label}_{sym}"
                # Record realized PnL only when a trade actually closes
                n_trades = len(eng.trades)
                if n_trades > prev_trade_counts[key]:
                    # New trade(s) closed — record the PnL from the most recent trade
                    for t_idx in range(prev_trade_counts[key], n_trades):
                        trade_pnl = _normalised_trade_pnl(strat_id, eng.trades[t_idx], config, init_eq)
                        portfolio_equity += trade_pnl
                        heat_tracker.record_trade_close(strat_id, trade_pnl)
                    prev_trade_counts[key] = n_trades

        # === Step 5: Update heat tracker ===
        risk_by_strat = _compute_open_risk(atrss_engines, helix_engines, active_tpc_risks)
        heat_tracker.update_open_risk(risk_by_strat)

        engine_groups = [
            ("atrss", atrss_engines),
            ("helix", helix_engines),
        ]
        tpc_open_mtm = _tpc_open_mtm_pnl(
            data,
            active_tpc_trades,
            active_tpc_risks,
            config,
            init_eq,
            int(ts),
        )
        equity_curve_realized.append(portfolio_equity + overlay_cumulative_pnl)
        equity_curve_mtm.append(
            portfolio_equity
            + overlay_cumulative_pnl
            + _combined_child_open_mtm_pnl(engine_groups, config, init_eq)
            + tpc_open_mtm
        )
        timestamps.append(ts)

        # Track heat utilization (normalized to ATRSS R-units for consistency)
        total_risk = sum(risk_by_strat.values())
        # Use the same portfolio unit risk as the gate, including dynamic NAV throttles.
        norm_base = heat_tracker.portfolio_unit_risk_dollars
        heat_R = total_risk / norm_base if norm_base > 0 else 0.0
        heat_samples.append(heat_R)

    finalized_helix = False
    for sym, engine in helix_engines.items():
        if engine.active_position is None:
            continue
        hourly = data.hourly.get(sym)
        if hourly is None or len(hourly) == 0:
            continue
        engine._flatten_at_end_of_data(
            hourly.closes[-1],
            engine._to_datetime(hourly.times[-1]),
        )
        key = f"helix_{sym}"
        n_trades = len(engine.trades)
        if n_trades > prev_trade_counts.get(key, 0):
            for t_idx in range(prev_trade_counts.get(key, 0), n_trades):
                trade_pnl = _normalised_trade_pnl("AKC_HELIX", engine.trades[t_idx], config, init_eq)
                portfolio_equity += trade_pnl
                heat_tracker.record_trade_close("AKC_HELIX", trade_pnl)
            prev_trade_counts[key] = n_trades
            finalized_helix = True

    if finalized_helix and equity_curve_realized:
        final_ts = int(timestamps[-1]) if timestamps else 0
        final_engine_groups = [
            ("atrss", atrss_engines),
            ("helix", helix_engines),
        ]
        equity_curve_realized[-1] = portfolio_equity + overlay_cumulative_pnl
        equity_curve_mtm[-1] = (
            portfolio_equity
            + overlay_cumulative_pnl
            + _combined_child_open_mtm_pnl(final_engine_groups, config, init_eq)
            + _tpc_open_mtm_pnl(
                data,
                active_tpc_trades,
                active_tpc_risks,
                config,
                init_eq,
                final_ts,
            )
        )

    _helix_patch.__exit__(None, None, None)
    _patch.__exit__(None, None, None)

    # --- Final overlay settlement: MTM the last day's positions ---
    if config.overlay_enabled and prev_date is not None:
        # Find last valid daily date for settlement
        ref_sym = config.overlay_symbols[0]
        if ref_sym in daily_date_idx and prev_date in daily_date_idx[ref_sym]:
            final_value = _compute_overlay_value(
                overlay_shares, data.daily, daily_date_idx, prev_date,
            )
            delta = final_value - overlay_prev_value
            overlay_cumulative_pnl += delta
            # Final per-symbol settlement
            for _osym in config.overlay_symbols:
                _sh = overlay_shares.get(_osym, 0.0)
                _sv = 0.0
                if _sh != 0.0:
                    _sidx = daily_date_idx.get(_osym, {}).get(prev_date)
                    if _sidx is not None:
                        _sv = _sh * data.daily[_osym].closes[_sidx]
                _sdelta = _sv - overlay_prev_sym_value.get(_osym, 0.0)
                overlay_per_sym_pnl[_osym] = overlay_per_sym_pnl.get(_osym, 0.0) + _sdelta
        # Update last equity curve point to include final overlay PnL
        if equity_curve_realized:
            equity_curve_realized[-1] = portfolio_equity + overlay_cumulative_pnl
        if equity_curve_mtm:
            final_ts = int(timestamps[-1]) if timestamps else 0
            final_engine_groups = [
                ("atrss", atrss_engines),
                ("helix", helix_engines),
            ]
            equity_curve_mtm[-1] = (
                portfolio_equity
                + overlay_cumulative_pnl
                + _combined_child_open_mtm_pnl(final_engine_groups, config, init_eq)
                + _tpc_open_mtm_pnl(
                    data,
                    active_tpc_trades,
                    active_tpc_risks,
                    config,
                    init_eq,
                    final_ts,
                )
            )

    # --- Build results ---
    heat_arr = np.array(heat_samples) if heat_samples else np.array([0.0])
    heat_stats = UnifiedHeatStats(
        avg_heat_pct=float(np.mean(heat_arr)),
        max_heat_pct=float(np.max(heat_arr)),
        pct_time_at_cap=float(np.mean(heat_arr >= config.heat_cap_R)) * 100,
    )

    # Collect portfolio-normalised trades per strategy. Raw child-engine
    # trade dollars are source diagnostics; portfolio reports use the same
    # static initial-risk basis that drove the shared equity ledger.
    raw_atrss_trades = []
    for eng in atrss_engines.values():
        raw_atrss_trades.extend(eng.trades)

    raw_helix_trades = []
    for eng in helix_engines.values():
        raw_helix_trades.extend(eng.trades)

    all_atrss_trades = [
        _normalised_trade_copy("ATRSS", trade, config, init_eq)
        for trade in raw_atrss_trades
    ]
    all_helix_trades = [
        _normalised_trade_copy("AKC_HELIX", trade, config, init_eq)
        for trade in raw_helix_trades
    ]
    all_tpc_trades = [
        _normalised_trade_copy("TPC", trade, config, init_eq)
        for trade in accepted_tpc_trades
    ]

    strategy_results = {}
    for sid, trades, blocked in [
        ("ATRSS", all_atrss_trades, blocked_entries["ATRSS"]),
        ("AKC_HELIX", all_helix_trades, blocked_entries["AKC_HELIX"]),
        ("TPC", all_tpc_trades, blocked_entries["TPC"]),
    ]:
        total_pnl = sum(_trade_net_pnl(t) for t in trades)
        wins = sum(1 for t in trades if _trade_net_pnl(t) > 0)
        losses = sum(1 for t in trades if _trade_net_pnl(t) <= 0)
        total_r = sum(_trade_net_r(t) for t in trades)
        strategy_results[sid] = StrategyResult(
            strategy_id=sid,
            entry_signals_fired=entry_attempts[sid],
            entry_requests=entry_attempts[sid],
            entries_accepted_by_portfolio=accepted_entries[sid],
            total_trades=len(trades),
            total_pnl=total_pnl,
            winning_trades=wins,
            losing_trades=losses,
            total_r=total_r,
            entries_blocked_by_heat=blocked,
            suppressed_entry_retries=suppressed_entry_retries[sid],
        )

    if portfolio_rule_replay is not None:
        portfolio_rule_replay.close()

    return UnifiedPortfolioResult(
        combined_equity=np.array(equity_curve_mtm),
        combined_equity_mtm=np.array(equity_curve_mtm),
        combined_equity_realized=np.array(equity_curve_realized),
        combined_timestamps=np.array(timestamps),
        heat_stats=heat_stats,
        strategy_results=strategy_results,
        coordination_tighten_count=coordinator.tighten_count,
        coordination_boost_count=coordinator.boost_count,
        portfolio_daily_stop_activations=daily_stop_activations,
        overlay_pnl=overlay_cumulative_pnl,
        overlay_commission=overlay_total_commission,
        overlay_per_symbol_pnl=dict(overlay_per_sym_pnl),
        atrss_trades=all_atrss_trades,
        helix_trades=all_helix_trades,
        tpc_trades=all_tpc_trades,
        entry_events=entry_event_log,
        heat_rejections=heat_rejection_log,
        coordination_events=coordination_event_log,
        portfolio_rule_events=portfolio_rule_event_log,
    )


# ---------------------------------------------------------------------------
# Force-flatten helpers (when heat check fails post-entry)
# ---------------------------------------------------------------------------

def _drawdown_risk_multiplier(drawdown_pct: float, tiers: tuple[tuple[float, float], ...] | list[tuple[float, float]]) -> float:
    multiplier = 1.0
    for threshold, tier_multiplier in sorted((float(t), float(m)) for t, m in tiers):
        if drawdown_pct >= threshold:
            multiplier = tier_multiplier
    return max(0.0, multiplier)


def _enum_label(value) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", getattr(value, "name", value)))


def _log_entry_event(
    events: list[dict],
    heat_tracker: PortfolioHeatTracker,
    event_time: datetime,
    strategy_id: str,
    symbol: str,
    stage: str,
    status: str,
    reason: str,
    risk_dollars: float,
    *,
    direction: int | None = None,
    entry_price: float | None = None,
    stop_price: float | None = None,
    quantity: float | None = None,
    signal_type: str | None = None,
    quality_score: float | None = None,
    entry_already_filled: bool | None = None,
    metadata: dict | None = None,
) -> dict:
    event = {
        "time": event_time,
        "strategy": strategy_id,
        "symbol": symbol,
        "stage": stage,
        "status": status,
        "reason": reason,
        "risk_dollars": float(risk_dollars or 0.0),
        "risk_context": _entry_risk_context(heat_tracker, strategy_id, risk_dollars),
    }
    if direction is not None:
        event["direction"] = int(direction)
    if entry_price is not None:
        event["entry_price"] = float(entry_price)
    if stop_price is not None:
        event["stop_price"] = float(stop_price)
    if quantity is not None:
        event["quantity"] = float(quantity)
    if signal_type:
        event["signal_type"] = str(signal_type)
    if quality_score is not None:
        event["quality_score"] = float(quality_score)
    if entry_already_filled is not None:
        event["entry_already_filled"] = bool(entry_already_filled)
    if metadata:
        event["metadata"] = dict(metadata)
    events.append(event)
    return event


def _entry_risk_context(
    heat_tracker: PortfolioHeatTracker,
    strategy_id: str,
    risk_dollars: float,
) -> dict[str, float]:
    if hasattr(heat_tracker, "entry_risk_context"):
        return heat_tracker.entry_risk_context(strategy_id, risk_dollars)
    strat = heat_tracker._strats.get(strategy_id)
    portfolio_unit = heat_tracker._portfolio_unit_risk
    total_open_dollars = heat_tracker.total_open_risk_dollars
    portfolio_open_r = total_open_dollars / portfolio_unit if portfolio_unit > 0 else 0.0
    portfolio_request_r = risk_dollars / portfolio_unit if portfolio_unit > 0 else 0.0
    context = {
        "portfolio_open_risk_R": float(portfolio_open_r),
        "portfolio_request_risk_R": float(portfolio_request_r),
        "portfolio_after_request_R": float(portfolio_open_r + portfolio_request_r),
        "portfolio_heat_cap_R": float(heat_tracker._heat_cap_R),
    }
    if strat is not None:
        strategy_request_r = risk_dollars / strat.unit_risk_dollars if strat.unit_risk_dollars > 0 else 0.0
        context.update(
            {
                "strategy_open_risk_R": float(strat.open_risk_R),
                "strategy_request_risk_R": float(strategy_request_r),
                "strategy_after_request_R": float(strat.open_risk_R + strategy_request_r),
                "strategy_heat_cap_R": float(strat.max_heat_R),
                "strategy_daily_realized_R": float(strat.daily_realized_R),
                "strategy_daily_stop_R": float(strat.daily_stop_R),
            }
        )
    return context


def _force_flatten_helix(engine: HelixEngine, pos) -> None:
    """Reverse a Helix entry that breached the heat cap."""
    engine.broker.cancel_all(engine.symbol)
    # Reverse entry commission leaked by the phantom fill
    entry_comm = getattr(pos, "commission", 0.0)
    if entry_comm > 0:
        engine.equity += entry_comm
        engine.total_commission -= entry_comm
    engine.setups_filled = max(0, engine.setups_filled - 1)
    engine.active_position = None


def _to_datetime(ts) -> datetime:
    """Convert numpy datetime64 or pandas Timestamp to UTC-aware datetime.

    Must match BacktestEngine._to_datetime which returns UTC-aware datetimes,
    otherwise SimBroker comparisons fail with naive/aware mismatch.
    """
    from datetime import timezone
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    if hasattr(ts, 'astype'):
        # numpy datetime64
        unix_epoch = np.datetime64(0, 'ns')
        one_second = np.timedelta64(1, 's')
        seconds = (ts - unix_epoch) / one_second
        return datetime.fromtimestamp(float(seconds), tz=timezone.utc)
    if hasattr(ts, 'to_pydatetime'):
        dt = ts.to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    dt = pd.Timestamp(ts).to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_unified_report(result: UnifiedPortfolioResult, config: UnifiedBacktestConfig) -> None:
    """Print a summary of the unified backtest results."""
    eq = result.combined_equity
    if len(eq) == 0:
        print("No results to report.")
        return

    init_eq = config.initial_equity
    final_eq = eq[-1]
    total_return = (final_eq - init_eq) / init_eq * 100

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = float(np.min(dd))
    max_dd_dollars = float(np.min(eq - peak))

    # Sharpe (annualized from hourly returns)
    returns = np.diff(eq) / eq[:-1]
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 7)  # ~7 trading hours/day
    else:
        sharpe = 0.0

    total_trades = sum(sr.total_trades for sr in result.strategy_results.values())
    active_pnl = sum(sr.total_pnl for sr in result.strategy_results.values())
    total_pnl = active_pnl + result.overlay_pnl

    print("\n" + "=" * 70)
    print("UNIFIED SWING PORTFOLIO BACKTEST RESULTS")
    print("=" * 70)
    print(f"Initial Equity:  ${init_eq:,.2f}")
    print(f"Final Equity:    ${final_eq:,.2f}")
    print(f"Total Return:    {total_return:+.2f}%")
    print(f"Total PnL:       ${total_pnl:+,.2f}")
    print(f"Max Drawdown:    {max_dd:.2f}% (${max_dd_dollars:,.2f})")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Total Trades:    {total_trades}")
    print(f"Timeline:        {len(eq):,} bars")

    print(f"\n{'Heat Statistics':}")
    print(f"  Avg Heat (R):        {result.heat_stats.avg_heat_pct:.3f}")
    print(f"  Max Heat (R):        {result.heat_stats.max_heat_pct:.3f}")
    print(f"  % Time at Cap:       {result.heat_stats.pct_time_at_cap:.1f}%")

    print(f"\n{'Cross-Strategy Coordination':}")
    print(f"  Helix Stop Tightens: {result.coordination_tighten_count}")
    print(f"  Helix Size Boosts:   {result.coordination_boost_count}")
    print(f"  Portfolio Daily Stops: {result.portfolio_daily_stop_activations}")

    if config.overlay_enabled:
        ovl_pct = result.overlay_pnl / init_eq * 100 if init_eq > 0 else 0
        act_pct = active_pnl / init_eq * 100 if init_eq > 0 else 0
        if config.overlay_mode == "multi":
            mode_label = f"Multi (EMA+RSI+MACD) entry>={config.overlay_entry_score_min} exit<={config.overlay_exit_score_max}"
            print(f"\n{'Idle-Capital Overlay (' + mode_label + ')':}")
            print(f"  Symbols:             {', '.join(config.overlay_symbols)}")
            print(f"  Adaptive Sizing:     {'ON' if config.overlay_adaptive_sizing else 'OFF'} [{config.overlay_min_alloc_pct:.0%}-{config.overlay_max_alloc_pct:.0%}]")
        else:
            print(f"\n{'Idle-Capital Overlay (EMA {0}/{1})'.format(config.overlay_ema_fast, config.overlay_ema_slow):}")
        print(f"  Overlay PnL:         ${result.overlay_pnl:+,.2f} ({ovl_pct:+.1f}%)")
        print(f"  Overlay Costs:       ${result.overlay_commission:,.2f}")
        print(f"  Active Strategy PnL: ${active_pnl:+,.2f} ({act_pct:+.1f}%)")
        print(f"  Overlay Max Pct:     {config.overlay_max_pct:.0%}")
        if result.overlay_per_symbol_pnl:
            for osym, opnl in sorted(result.overlay_per_symbol_pnl.items()):
                opnl_pct = opnl / init_eq * 100 if init_eq > 0 else 0
                print(f"    {osym:<6} overlay PnL: ${opnl:+,.2f} ({opnl_pct:+.1f}%)")

    print(f"\n{'Per-Strategy Breakdown':}")
    print(f"{'Strategy':<22} {'Trades':>7} {'Win%':>6} {'PnL':>10} {'Total R':>8} {'Blocked':>8}")
    print("-" * 65)
    for sid in ["ATRSS", "AKC_HELIX", "TPC"]:
        sr = result.strategy_results.get(sid)
        if sr is None:
            continue
        win_pct = sr.winning_trades / sr.total_trades * 100 if sr.total_trades > 0 else 0
        print(f"{sid:<22} {sr.total_trades:>7} {win_pct:>5.1f}% ${sr.total_pnl:>9,.2f} {sr.total_r:>7.1f}R {sr.entries_blocked_by_heat:>8}")

    print("=" * 70)
