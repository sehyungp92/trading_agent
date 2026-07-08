"""Tier 3 IARIC pullback-buy mean-reversion backtest engine.

Scans the full S&P universe (~415 stocks) for short-term oversold pullbacks
in uptrends. Entry triggers: RSI(2) < threshold, consecutive down days,
or price in MA zone. Exits: RSI mean-reversion, time stop, profit target,
ATR trailing stop. Optional overnight carry.

Completely independent of T1 sponsorship/conviction logic. Reuses the same
ResearchReplayEngine data layer, IARICBacktestConfig, StrategySettings
(pb_* params), TradeRecord, and scoring infrastructure.
"""
from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from math import floor
from typing import Any

import numpy as np

from backtests.shared.parity.legacy_result_outputs import trade_outcomes_from_records
from backtests.stock.config_iaric import IARICBacktestConfig
from backtests.stock.engine.iaric_pullback_indicators import (
    adx_suite,
    atr,
    bollinger_pctb,
    consecutive_down_days,
    ema,
    pullback_depth,
    rate_of_change,
    relative_strength_ratio,
    rolling_sma,
    rsi,
    sma_slope_positive,
    volume_climax_ratio,
)
from backtests.stock.engine.research_replay import (
    ResearchReplayEngine,
    _iloc_upto,
)
from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from backtests.stock.models import Direction as BTDirection, TradeRecord
from strategies.stock.iaric.core import logic as iaric_core_logic
from strategies.stock.iaric.core.state import (
    IARICCoreState,
    IARICEntryRequest,
    IARICFill,
    IARICFlattenRequest,
)

from strategies.stock.iaric.config import StrategySettings
from strategies.stock.iaric.models import PBSymbolState, WatchlistArtifact, WatchlistItem

logger = logging.getLogger(__name__)

_PULLBACK_ENGINE_SHARED_CACHE: "weakref.WeakKeyDictionary[ResearchReplayEngine, dict[str, Any]]" = weakref.WeakKeyDictionary()


# ---------------------------------------------------------------------------
# Internal position tracking
# ---------------------------------------------------------------------------


@dataclass
class _PBPosition:
    """Backtest-internal pullback-buy position."""

    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: int
    risk_per_share: float
    sector: str
    regime_tier: str
    stop: float
    entry_rsi: float
    trigger_type: str             # "RSI" | "CDD" | "MA_ZONE"
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    hold_days: int = 0
    carry_eligible: bool = False
    highest_carry_close: float = 0.0
    commission_entry: float = 0.0
    slippage_entry: float = 0.0
    entry_atr: float = 0.0
    entry_gap_pct: float = 0.0
    entry_sma_dist_pct: float = 0.0
    entry_cdd: int = 0
    entry_rank: int = 0
    entry_rank_pct: float = 0.0
    n_candidates: int = 0
    daily_signal_score: float = 0.0
    daily_signal_rank_pct: float = 100.0
    signal_family: str = ""
    route_family: str = "OPEN_SCORED_ENTRY"
    selection_reason: str = ""
    close_r: float = 0.0
    close_pct: float = 0.0
    exit_rsi: float = 0.0
    ledger_ref: dict[str, Any] | None = None
    # V2 fields
    trigger_types: list[str] | None = None
    trigger_tier: str = ""
    trend_tier: str = "STRONG"
    mfe_stage: int = 0
    partial_taken: bool = False
    partial_qty_exited: int = 0
    realized_partial_pnl: float = 0.0
    realized_partial_commission: float = 0.0

    @property
    def total_risk(self) -> float:
        return self.risk_per_share * self.quantity

    def unrealized_r(self, price: float) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        return (price - self.entry_price) / self.risk_per_share

    def build_metadata(self) -> dict:
        return {
            "trigger_type": self.trigger_type,
            "entry_rsi": round(self.entry_rsi, 2),
            "hold_days": self.hold_days,
            "setup_type": "PULLBACK_BUY",
            "setup_tag": "PULLBACK_BUY",
            "mfe_r": round(
                (self.max_favorable - self.entry_price) / max(self.risk_per_share, 0.01), 4
            ) if self.max_favorable > 0 else 0.0,
            "mae_r": round(
                (self.entry_price - self.max_adverse) / max(self.risk_per_share, 0.01), 4
            ) if self.max_adverse > 0 and self.max_adverse < self.entry_price else 0.0,
            "entry_atr": round(self.entry_atr, 4),
            "stop_distance_pct": round(
                (self.entry_price - self.stop) / self.entry_price * 100, 3
            ) if self.entry_price > 0 else 0.0,
            "entry_gap_pct": round(self.entry_gap_pct, 3),
            "entry_sma_dist_pct": round(self.entry_sma_dist_pct, 3),
            "entry_cdd": self.entry_cdd,
            "entry_rank": self.entry_rank,
            "entry_rank_pct": round(self.entry_rank_pct, 2),
            "n_candidates": self.n_candidates,
            "daily_signal_score": round(self.daily_signal_score, 2),
            "daily_signal_rank_pct": round(self.daily_signal_rank_pct, 2),
            "signal_family": self.signal_family,
            "entry_trigger": self.route_family,
            "entry_route_family": self.route_family,
            "selection_reason": self.selection_reason,
            "close_r": round(self.close_r, 4),
            "close_pct": round(self.close_pct, 4),
            "exit_rsi": round(self.exit_rsi, 2),
            "trigger_types": self.trigger_types or [self.trigger_type],
            "trigger_tier": self.trigger_tier,
            "trend_tier": self.trend_tier,
            "mfe_stage": self.mfe_stage,
            "partial_taken": self.partial_taken,
        }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class IARICPullbackResult:
    """Result from running the IARIC pullback backtest."""

    trades: list[TradeRecord]
    equity_curve: np.ndarray
    timestamps: np.ndarray
    daily_selections: dict[date, WatchlistArtifact]
    candidate_ledger: dict[date, list[dict[str, Any]]] | None = None
    funnel_counters: dict[str, int] | None = None
    rejection_log: list[dict[str, Any]] | None = None
    shadow_outcomes: list[dict[str, Any]] | None = None
    selection_attribution: dict[date, dict[str, Any]] | None = None
    fsm_log: list[dict[str, Any]] | None = None
    decision_stream: list[dict[str, Any]] = field(default_factory=list)
    trade_outcomes: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _iloc_for_date(didx: tuple[list[date], list[int]], trade_date: date) -> int:
    """Find the iloc for the last bar on or before trade_date. Returns -1 if before data start."""
    return _iloc_upto(didx[0], didx[1], trade_date)


def _close_in_range_pct(high: float, low: float, close: float) -> float:
    if high <= low:
        return 1.0
    return float(min(max((close - low) / (high - low), 0.0), 1.0))


def _rank_percent(rank: int, total: int) -> float:
    if rank <= 0 or total <= 0:
        return 100.0
    if total == 1:
        return 50.0
    return float(min(max(rank / total * 100.0, 0.0), 100.0))


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _passes_rank_gate(rank: int, total: int, settings: StrategySettings) -> bool:
    if str(getattr(settings, "pb_signal_rank_gate_mode", "score_rank") or "score_rank").lower() != "percentile_only":
        if rank < settings.pb_entry_rank_min or rank > settings.pb_entry_rank_max:
            return False
    rank_pct = _rank_percent(rank, total)
    return settings.pb_entry_rank_pct_min <= rank_pct <= settings.pb_entry_rank_pct_max


def _rank_gate_reason(rank: int, total: int, settings: StrategySettings) -> str | None:
    if str(getattr(settings, "pb_signal_rank_gate_mode", "score_rank") or "score_rank").lower() != "percentile_only":
        if rank < settings.pb_entry_rank_min or rank > settings.pb_entry_rank_max:
            return "rank_abs_reject"
    rank_pct = _rank_percent(rank, total)
    if rank_pct < settings.pb_entry_rank_pct_min or rank_pct > settings.pb_entry_rank_pct_max:
        return "rank_pct_reject"
    return None


def _regime_signal_score(regime_tier: str) -> float:
    return {
        "A": 1.0,
        "B": 0.70,
        "C": 0.20,
    }.get(str(regime_tier or "B").upper(), 0.50)


def _gap_signal_score(gap_pct: float, *, family_key: str = "") -> float:
    if family_key == "meanrev_sweetspot_v1":
        if gap_pct <= -1.0:
            return 0.80
        if gap_pct <= 0.0:
            return 0.70
        if gap_pct <= 1.0:
            return 0.20
        if gap_pct <= 2.5:
            return 1.00
        return 0.45
    if gap_pct <= -1.0:
        return 1.0
    if gap_pct <= 0.0:
        return 0.85
    if gap_pct <= 1.0:
        return 0.60
    if gap_pct <= 2.5:
        return 0.45
    return 0.20


def _cdd_signal_score(cdd: int, *, family_key: str = "") -> float:
    if family_key == "meanrev_sweetspot_v1":
        if cdd >= 6:
            return 0.85
        if cdd >= 4:
            return 1.0
        if cdd >= 2:
            return 0.55
        return 0.10
    if cdd >= 4:
        return 1.0
    if cdd == 3:
        return 0.70
    if cdd == 2:
        return 0.50
    if cdd == 1:
        return 0.35
    return 0.25


def _sma_dist_signal_score(sma_dist_pct: float, *, family_key: str = "") -> float:
    if family_key == "meanrev_sweetspot_v1":
        if sma_dist_pct < 0:
            return 0.0
        if sma_dist_pct < 2.0:
            return 0.80
        if sma_dist_pct < 5.0:
            return 1.00
        if sma_dist_pct < 10.0:
            return 0.70
        return 0.10
    if sma_dist_pct < 0:
        return 0.0
    if sma_dist_pct < 1.0:
        return 0.90
    if sma_dist_pct < 2.5:
        return 0.65
    if sma_dist_pct < 5.5:
        return 0.60
    if sma_dist_pct < 10.0:
        return 0.95
    return 0.35


def _rsi_signal_score(entry_rsi: float, threshold: float, *, family_key: str = "") -> float:
    if family_key == "meanrev_sweetspot_v1":
        if entry_rsi < 3.0:
            return 0.75
        if entry_rsi < 6.0:
            return 1.0
        if entry_rsi < threshold:
            return 0.20
        return 0.0
    if threshold <= 0:
        return 0.0
    score = (threshold - entry_rsi) / max(threshold, 1.0)
    return _clip01(score)


def _sector_crowding_penalty(sector_count: int) -> float:
    if sector_count <= 1:
        return 0.0
    return _clip01((sector_count - 1) / 4.0)


def _breadth_penalty(total_candidates: int, min_candidates_day: int, *, hard_gate: bool) -> float:
    if hard_gate or min_candidates_day <= 0 or total_candidates >= min_candidates_day:
        return 0.0
    return _clip01((min_candidates_day - total_candidates) / max(min_candidates_day, 1))


def _bounded_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clip01((float(value) - low) / (high - low))


def _watch_item_signal_scores(item: WatchlistItem | None) -> dict[str, float]:
    if item is None:
        return {
            "sponsorship": 0.25,
            "persistence": 0.25,
            "rs": 0.35,
            "trend_strength": 0.30,
            "acceptance": 0.25,
            "conviction": 0.25,
            "missing_item": 1.0,
        }
    return {
        "sponsorship": _clip01(float(getattr(item, "sponsorship_score", 0.0))),
        "persistence": _clip01(float(getattr(item, "persistence", 0.0))),
        "rs": _clip01(float(getattr(item, "rs_percentile", 0.0)) / 100.0),
        "trend_strength": _bounded_score(float(getattr(item, "trend_strength", 0.0)), 0.01, 0.12),
        "acceptance": 1.0 if bool(getattr(item, "acceptance_pass", False)) else 0.0,
        "conviction": _bounded_score(float(getattr(item, "conviction_multiplier", 0.0)), 0.70, 1.50),
        "missing_item": 0.0,
    }


def _effective_min_candidates_day(
    settings: StrategySettings,
    *,
    trade_universe_size: int,
    full_universe_size: int,
) -> int:
    base = int(getattr(settings, "pb_min_candidates_day", 0) or 0)
    if base <= 0:
        return 0
    if (
        not bool(getattr(settings, "pb_backtest_intraday_universe_only", False))
        or trade_universe_size <= 0
        or full_universe_size <= 0
        or trade_universe_size >= full_universe_size
    ):
        return base
    scaled = int(np.ceil(base * trade_universe_size / max(full_universe_size, 1)))
    return max(1, scaled)


def _signal_family_weights(family: str) -> dict[str, float]:
    family_key = str(family or "balanced_v1").lower()
    zeroed = {
        "daily_rank": 0.0,
        "gap": 0.0,
        "cdd": 0.0,
        "sma": 0.0,
        "rsi": 0.0,
        "flow": 0.0,
        "regime": 0.0,
        "sector_penalty": 0.0,
        "breadth_penalty": 0.0,
        "sponsorship": 0.0,
        "persistence": 0.0,
        "rs": 0.0,
        "trend_strength": 0.0,
        "acceptance": 0.0,
        "conviction": 0.0,
        "missing_item_penalty": 0.0,
    }
    weights = {
        "balanced_v1": {
            "daily_rank": 22.0,
            "gap": 12.0,
            "cdd": 12.0,
            "sma": 12.0,
            "rsi": 18.0,
            "flow": 10.0,
            "regime": 6.0,
            "sector_penalty": 8.0,
            "breadth_penalty": 6.0,
        },
        "trend_guard": {
            "daily_rank": 28.0,
            "gap": 10.0,
            "cdd": 8.0,
            "sma": 16.0,
            "rsi": 12.0,
            "flow": 12.0,
            "regime": 8.0,
            "sector_penalty": 8.0,
            "breadth_penalty": 6.0,
        },
        "meanrev_v1": {
            "daily_rank": 16.0,
            "gap": 16.0,
            "cdd": 14.0,
            "sma": 10.0,
            "rsi": 22.0,
            "flow": 8.0,
            "regime": 6.0,
            "sector_penalty": 8.0,
            "breadth_penalty": 6.0,
        },
        "hybrid_alpha_v1": {
            "daily_rank": 14.0,
            "gap": 14.0,
            "cdd": 12.0,
            "sma": 10.0,
            "rsi": 14.0,
            "flow": 6.0,
            "regime": 4.0,
            "sector_penalty": 6.0,
            "breadth_penalty": 4.0,
            "sponsorship": 8.0,
            "persistence": 6.0,
            "rs": 8.0,
            "trend_strength": 5.0,
            "acceptance": 4.0,
            "conviction": 5.0,
            "missing_item_penalty": 6.0,
        },
        "quality_hybrid_v1": {
            "daily_rank": 10.0,
            "gap": 12.0,
            "cdd": 10.0,
            "sma": 8.0,
            "rsi": 12.0,
            "flow": 6.0,
            "regime": 4.0,
            "sector_penalty": 6.0,
            "breadth_penalty": 4.0,
            "sponsorship": 8.0,
            "persistence": 6.0,
            "rs": 8.0,
            "trend_strength": 4.0,
            "acceptance": 3.0,
            "conviction": 3.0,
            "missing_item_penalty": 6.0,
        },
        "sponsor_rs_hybrid_v1": {
            "daily_rank": 10.0,
            "gap": 12.0,
            "cdd": 10.0,
            "sma": 8.0,
            "rsi": 12.0,
            "flow": 6.0,
            "regime": 4.0,
            "sector_penalty": 6.0,
            "breadth_penalty": 4.0,
            "sponsorship": 10.0,
            "persistence": 4.0,
            "rs": 12.0,
            "trend_strength": 6.0,
            "acceptance": 2.0,
            "conviction": 2.0,
            "missing_item_penalty": 6.0,
        },
        "meanrev_plus_v1": {
            "daily_rank": 12.0,
            "gap": 14.0,
            "cdd": 14.0,
            "sma": 12.0,
            "rsi": 18.0,
            "flow": 6.0,
            "regime": 4.0,
            "sector_penalty": 6.0,
            "breadth_penalty": 4.0,
            "sponsorship": 4.0,
            "persistence": 2.0,
            "rs": 4.0,
            "trend_strength": 2.0,
            "acceptance": 0.0,
            "conviction": 2.0,
            "missing_item_penalty": 6.0,
        },
        "meanrev_sweetspot_v1": {
            "daily_rank": 10.0,
            "gap": 16.0,
            "cdd": 16.0,
            "sma": 14.0,
            "rsi": 24.0,
            "flow": 6.0,
            "regime": 4.0,
            "sector_penalty": 6.0,
            "breadth_penalty": 4.0,
        },
    }
    selected = dict(zeroed)
    selected.update(weights.get(family_key, weights["balanced_v1"]))
    return selected


def _daily_signal_bundle(
    *,
    settings: StrategySettings,
    regime_tier: str,
    item: WatchlistItem | None,
    entry_rsi: float,
    gap_pct: float,
    sma_dist_pct: float,
    cdd: int,
    flow_negative: bool,
    sector_count: int,
    total_candidates: int,
    effective_min_candidates_day: int,
) -> dict[str, float]:
    family_key = str(getattr(settings, "pb_daily_signal_family", "balanced_v1") or "balanced_v1").lower()
    weights = _signal_family_weights(family_key)
    daily_rank = _clip01(float(getattr(item, "daily_rank", 0.5) if item is not None else 0.5))
    watch_item_scores = _watch_item_signal_scores(item)
    components = {
        "daily_rank": daily_rank * weights["daily_rank"],
        "gap": _gap_signal_score(gap_pct, family_key=family_key) * weights["gap"],
        "cdd": _cdd_signal_score(cdd, family_key=family_key) * weights["cdd"],
        "sma": _sma_dist_signal_score(sma_dist_pct, family_key=family_key) * weights["sma"],
        "rsi": _rsi_signal_score(entry_rsi, settings.pb_rsi_entry, family_key=family_key) * weights["rsi"],
        "flow_state": (1.0 if not flow_negative else -1.0) * weights["flow"],
        "regime": _regime_signal_score(regime_tier) * weights["regime"],
        "sponsorship": watch_item_scores["sponsorship"] * weights["sponsorship"],
        "persistence": watch_item_scores["persistence"] * weights["persistence"],
        "rs": watch_item_scores["rs"] * weights["rs"],
        "trend_strength": watch_item_scores["trend_strength"] * weights["trend_strength"],
        "acceptance": watch_item_scores["acceptance"] * weights["acceptance"],
        "conviction": watch_item_scores["conviction"] * weights["conviction"],
        "sector_crowding_penalty": -_sector_crowding_penalty(sector_count) * weights["sector_penalty"],
        "breadth_penalty": -_breadth_penalty(
            total_candidates,
            effective_min_candidates_day,
            hard_gate=bool(getattr(settings, "pb_min_candidates_day_hard_gate", False)),
        ) * weights["breadth_penalty"],
        "missing_item_penalty": -watch_item_scores["missing_item"] * weights["missing_item_penalty"],
    }
    total = sum(components.values())
    components["score"] = float(max(total, 0.0))
    return components


def _passes_daily_signal_floor(
    settings: StrategySettings,
    score: float,
    *,
    rescue_candidate: bool = False,
) -> bool:
    floor = float(getattr(settings, "pb_daily_rescue_min_score", 0.0)) if rescue_candidate else float(getattr(settings, "pb_daily_signal_min_score", 0.0))
    return float(score) >= floor


def _risk_budget_mult(trade_date: date, settings: StrategySettings) -> float:
    weekday = trade_date.weekday()
    if weekday == 1:
        return float(settings.pb_tuesday_mult)
    if weekday == 2:
        return float(settings.pb_wednesday_mult)
    if weekday == 3:
        return float(settings.pb_thursday_mult)
    if weekday == 4:
        return float(settings.pb_friday_mult)
    return 1.0


# ---------------------------------------------------------------------------
# V2 helpers (shared by daily + hybrid engines)
# ---------------------------------------------------------------------------


def _evaluate_v2_triggers(
    ind: dict[str, np.ndarray],
    iloc: int,
    closes: np.ndarray,
    prev_close: float,
    gap_pct: float,
    trend_tier: str,
    rs_val: float,
    settings: StrategySettings,
) -> list[tuple[str, str]]:
    """Return list of (trigger_name, tier) pairs. Empty = no trigger."""
    results: list[tuple[str, str]] = []

    rsi2_val = ind["rsi"][iloc] if not np.isnan(ind["rsi"][iloc]) else 100.0
    rsi5_val = ind["rsi5"][iloc] if not np.isnan(ind["rsi5"][iloc]) else 100.0
    cdd_val = int(ind["cdd"][iloc])
    depth_val = ind["depth"][iloc] if not np.isnan(ind["depth"][iloc]) else 0.0
    bb_val = ind["bb_pctb"][iloc] if not np.isnan(ind["bb_pctb"][iloc]) else 0.5
    vcr_val = ind["vcr"][iloc] if not np.isnan(ind["vcr"][iloc]) else 1.0
    roc_val = ind["roc5"][iloc] if not np.isnan(ind["roc5"][iloc]) else 0.0
    is_down_day = closes[iloc] < closes[iloc - 1] if iloc > 0 else False

    # A: DEEP_RSI
    if rsi2_val < settings.pb_v2_rsi2_thresh:
        results.append(("DEEP_RSI", "HIGH"))

    # B: MOD_RSI
    if rsi5_val < settings.pb_v2_rsi5_thresh and cdd_val >= settings.pb_v2_cdd_min_for_rsi5:
        results.append(("MOD_RSI", "MEDIUM"))

    # C: ATR_DEPTH
    if depth_val >= settings.pb_v2_depth_thresh and trend_tier:
        results.append(("ATR_DEPTH", "MEDIUM"))

    # D: BB_EXTREME
    if bb_val < settings.pb_v2_bb_pctb_thresh and trend_tier:
        results.append(("BB_EXTREME", "HIGH"))

    # E: VOL_CAPITULATION
    if vcr_val > settings.pb_v2_vol_climax_thresh and is_down_day and rsi2_val < 30:
        results.append(("VOL_CAPITULATION", "HIGH"))

    # F: RS_DIP
    if rs_val > settings.pb_v2_rs_ratio_thresh and roc_val < settings.pb_v2_roc_thresh and trend_tier:
        results.append(("RS_DIP", "MEDIUM"))

    # G: GAP_FILL (uses observed T open via gap_pct)
    if gap_pct < settings.pb_v2_gap_fill_thresh and trend_tier == "STRONG":
        results.append(("GAP_FILL", "HIGH"))

    return results


def _daily_signal_bundle_v2(
    *,
    trend_tier: str,
    trigger_types: list[str],
    trigger_tier: str,
    adx: float,
    plus_di: float,
    minus_di: float,
    sma_slope_pos: bool,
    sma_dist_pct: float,
    pullback_depth_atr: float,
    rsi2: float,
    rsi5: float,
    vcr: float,
    is_down_day: bool,
    rs_ratio: float,
    gap_pct: float,
    regime_tier: str,
    sector_count: int,
    item: WatchlistItem | None,
    n_triggers: int,
    candidate_yesterday: bool,
) -> dict[str, float]:
    """V2 scoring model -- 100 base + up to 15 bonus."""

    # --- Core Technical (70) ---

    # trend_quality (max 22)
    tq = 0.0
    if sma_slope_pos:
        tq += 7.0
    if not np.isnan(adx) and adx > 25 and not np.isnan(plus_di) and not np.isnan(minus_di) and plus_di > minus_di:
        tq += 8.0
    if trend_tier == "STRONG":
        tq += 7.0
    elif trend_tier == "SECULAR":
        tq += 4.0

    # pullback_depth (max 18)
    pd_score = 0.0
    if not np.isnan(pullback_depth_atr):
        if pullback_depth_atr > 3.5:
            pd_score = 4.0
        elif pullback_depth_atr > 2.5:
            pd_score = 10.0
        elif pullback_depth_atr > 1.5:
            pd_score = 18.0
        elif pullback_depth_atr > 1.0:
            pd_score = 12.0
        elif pullback_depth_atr > 0.5:
            pd_score = 7.0
        else:
            pd_score = 2.0

    # trigger_quality (max 15)
    trig_q = 15.0 if trigger_tier == "HIGH" else 10.0
    if n_triggers >= 2:
        trig_q += 3.0

    # oversold_depth (max 15)
    rsi2_score = 0.0
    if rsi2 < 5:
        rsi2_score = 15.0
    elif rsi2 < 10:
        rsi2_score = 12.0
    elif rsi2 < 15:
        rsi2_score = 9.0
    elif rsi2 < 20:
        rsi2_score = 6.0

    rsi5_score = 0.0
    if rsi5 < 20:
        rsi5_score = 12.0
    elif rsi5 < 25:
        rsi5_score = 9.0
    elif rsi5 < 30:
        rsi5_score = 6.0

    oversold = max(rsi2_score, rsi5_score)

    # --- Context (30) ---

    # volume_context (max 10)
    vol_ctx = 3.0
    if is_down_day:
        if vcr > 2.5:
            vol_ctx = 10.0
        elif vcr > 2.0:
            vol_ctx = 8.0
        elif vcr > 1.5:
            vol_ctx = 6.0
    elif vcr > 2.0:
        vol_ctx = 4.0

    # relative_strength (max 8)
    rs_score = 1.0
    if not np.isnan(rs_ratio):
        if rs_ratio > 1.10:
            rs_score = 8.0
        elif rs_ratio > 1.05:
            rs_score = 7.0
        elif rs_ratio > 1.00:
            rs_score = 5.0
        elif rs_ratio > 0.95:
            rs_score = 3.0

    # gap_context (max 6)
    gap_ctx = _gap_signal_score(gap_pct) * 0.06  # scale existing 0-100 to 0-6

    # regime (max 6)
    regime_ctx = 6.0 if regime_tier == "A" else (4.0 if regime_tier == "B" else 1.0)

    # --- Penalties (up to -10) ---
    sector_pen = min(_sector_crowding_penalty(sector_count), 6.0)
    trend_pen = 4.0 if trend_tier == "SECULAR" else 0.0

    # --- Optional bonuses (up to +15) ---
    sponsor_bonus = 0.0
    persist_bonus = 0.0
    if item is not None:
        sp = getattr(item, "sponsorship_state", "NEUTRAL")
        if sp == "STRONG":
            sponsor_bonus = 5.0
        elif sp == "ACCUMULATE":
            sponsor_bonus = 3.0
        elif sp == "NEUTRAL":
            sponsor_bonus = 1.0
        p = float(getattr(item, "persistence", 0.0) or 0.0)
        if p > 0.7:
            persist_bonus = 3.0
        elif p > 0.5:
            persist_bonus = 2.0

    yesterday_bonus = 4.0 if candidate_yesterday else 0.0
    multi_bonus = 3.0 if n_triggers >= 3 else 0.0

    total = (
        tq + pd_score + min(trig_q, 18.0) + oversold  # core
        + vol_ctx + rs_score + gap_ctx + regime_ctx  # context
        - sector_pen - trend_pen  # penalties
        + sponsor_bonus + persist_bonus + yesterday_bonus + multi_bonus  # bonuses
    )

    components = {
        "trend_quality": round(tq, 2),
        "pullback_depth": round(pd_score, 2),
        "trigger_quality": round(min(trig_q, 18.0), 2),
        "oversold_depth": round(oversold, 2),
        "volume_context": round(vol_ctx, 2),
        "relative_strength": round(rs_score, 2),
        "gap_context": round(gap_ctx, 2),
        "regime": round(regime_ctx, 2),
        "sector_penalty": round(-sector_pen, 2),
        "trend_tier_discount": round(-trend_pen, 2),
        "sponsorship_bonus": round(sponsor_bonus, 2),
        "persistence_bonus": round(persist_bonus, 2),
        "candidate_yesterday": round(yesterday_bonus, 2),
        "multi_trigger": round(multi_bonus, 2),
        "score": round(max(total, 0.0), 2),
    }
    return components


def _v2_score_sizing_mult(score: float, trend_tier: str, route_family: str, settings: StrategySettings) -> float:
    """Map V2 score to sizing multiplier, stacking trend and route adjustments."""
    if score >= 75:
        base = settings.pb_v2_sizing_premium
    elif score >= 60:
        base = settings.pb_v2_sizing_standard
    elif score >= 45:
        base = settings.pb_v2_sizing_reduced
    elif score >= 30:
        base = settings.pb_v2_sizing_minimum
    else:
        return 0.0  # REJECT

    mult = base
    if trend_tier == "SECULAR":
        mult *= settings.pb_v2_secular_sizing_mult
    if route_family == "AFTERNOON_RETEST":
        mult *= settings.pb_v2_afternoon_retest_sizing_mult
    return mult


def _should_flatten_v2(
    pos: _PBPosition,
    close: float,
    close_pct: float,
    regime_tier: str,
    flow_last_n: list[float] | None,
    settings: StrategySettings,
) -> tuple[bool, str]:
    """Inverted carry logic: default is CARRY. Returns (should_flatten, reason)."""
    ur = pos.unrealized_r(close)

    # Deep loser
    if ur < settings.pb_v2_flatten_loss_r:
        return True, "DEEP_LOSS"

    # Regime C + not enough profit
    if regime_tier == "C" and ur < settings.pb_v2_flatten_regime_c_min_r:
        return True, "REGIME_C_UNDERWATER"

    # 2 consecutive negative flow days (skip during grace period -- pullback
    # entries inherently have negative flow on entry day and preceding days)
    grace = getattr(settings, "pb_v2_flow_grace_days", 2)
    if pos.hold_days > grace and flow_last_n is not None and len(flow_last_n) >= 2 and all(v < 0 for v in flow_last_n[-2:]):
        return True, "FLOW_REVERSAL"

    # Time stop
    if pos.hold_days >= settings.pb_max_hold_days:
        return True, "TIME_STOP"

    return False, ""


def _v2_rsi_exit_threshold(route_family: str, settings: StrategySettings) -> float:
    """Route-specific RSI exit threshold for V2."""
    if route_family == "OPEN_SCORED_ENTRY":
        return settings.pb_v2_rsi_exit_open_scored
    if route_family == "DELAYED_CONFIRM":
        return settings.pb_v2_rsi_exit_delayed
    if route_family == "VWAP_BOUNCE":
        return settings.pb_v2_rsi_exit_vwap_bounce
    if route_family == "AFTERNOON_RETEST":
        return settings.pb_v2_rsi_exit_afternoon
    return settings.pb_v2_rsi_exit_open_scored


def _pullback_indicator_cache_key(settings: StrategySettings) -> tuple:
    return (
        int(settings.pb_rsi_period),
        int(settings.pb_atr_period),
        int(settings.pb_trend_sma),
        int(settings.pb_trend_slope_lookback),
        bool(getattr(settings, "pb_v2_enabled", False)),
    )


def _shared_pullback_state(
    replay: ResearchReplayEngine,
    settings: StrategySettings,
    *,
    trade_universe: list[tuple[str, str, str]] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[date, int]], dict[str, np.ndarray]]:
    cache = _PULLBACK_ENGINE_SHARED_CACHE.setdefault(
        replay,
        {
            "indicators": {},
            "date_iloc": None,
            "flow_negative": None,
        },
    )

    date_iloc = cache.get("date_iloc")
    if date_iloc is None:
        date_iloc = {
            sym: dict(zip(dates_list, ilocs_list))
            for sym, (dates_list, ilocs_list) in replay._daily_didx.items()
        }
        cache["date_iloc"] = date_iloc

    flow_negative = cache.get("flow_negative")
    if flow_negative is None:
        flow_negative = {
            sym: flow < 0
            for sym, flow in replay._daily_flow.items()
        }
        cache["flow_negative"] = flow_negative

    active_universe = trade_universe or replay._universe
    v2 = bool(getattr(settings, "pb_v2_enabled", False))
    indicator_key = (
        _pullback_indicator_cache_key(settings),
        tuple(sym for sym, _, _ in active_universe),
    )
    indicators_by_key = cache["indicators"]
    indicators = indicators_by_key.get(indicator_key)
    if indicators is None:
        indicators = {}
        # Pre-extract SPY closes for RS ratio when V2 is enabled
        spy_closes: np.ndarray | None = None
        if v2:
            spy_arrs = replay._daily_arrs.get("SPY")
            if spy_arrs is not None:
                spy_closes = spy_arrs["close"]
        for sym, _, _ in active_universe:
            arrs = replay._daily_arrs.get(sym)
            if arrs is None:
                continue
            closes = arrs["close"]
            highs = arrs["high"]
            lows = arrs["low"]
            volumes = arrs["volume"]
            sma_trend = rolling_sma(closes, settings.pb_trend_sma)
            atr_arr = atr(highs, lows, closes, settings.pb_atr_period)
            ind_dict: dict[str, Any] = {
                "rsi": rsi(closes, settings.pb_rsi_period),
                "atr": atr_arr,
                "sma_trend": sma_trend,
                "sma20": rolling_sma(closes, 20),
                "sma_slope": sma_slope_positive(sma_trend, settings.pb_trend_slope_lookback),
                "cdd": consecutive_down_days(closes),
            }
            if v2:
                adx_arr, pdi_arr, mdi_arr = adx_suite(highs, lows, closes, 14)
                ind_dict.update({
                    "rsi5": rsi(closes, 5),
                    "depth": pullback_depth(highs, closes, atr_arr, 10),
                    "roc5": rate_of_change(closes, 5),
                    "bb_pctb": bollinger_pctb(closes, 20, 2.0),
                    "vcr": volume_climax_ratio(volumes, 20),
                    "adx": adx_arr,
                    "plus_di": pdi_arr,
                    "minus_di": mdi_arr,
                    "sma200": rolling_sma(closes, 200),
                    "ema10": ema(closes, 10),
                    "rs_ratio": relative_strength_ratio(closes, spy_closes, 20) if spy_closes is not None else np.full(len(closes), np.nan),
                })
            indicators[sym] = ind_dict
        indicators_by_key[indicator_key] = indicators

    return indicators, date_iloc, flow_negative


def _resolve_trade_universe(
    replay: ResearchReplayEngine,
    settings: StrategySettings,
) -> list[tuple[str, str, str]]:
    universe = list(replay._universe)
    # Add SPY as benchmark for V2 RS computation (if not already in universe)
    if bool(getattr(settings, "pb_v2_enabled", False)):
        syms = {sym for sym, _, _ in universe}
        if "SPY" not in syms and replay._daily_arrs.get("SPY") is not None:
            universe.append(("SPY", "benchmark", "benchmark"))
    if not bool(getattr(settings, "pb_backtest_intraday_universe_only", False)):
        return universe
    intraday_symbols = set(getattr(replay, "_5m_didx", {}).keys()) | set(getattr(replay, "_intraday_5m_cache", {}).keys()) | set(getattr(replay, "_5m_paths", {}).keys())
    if not intraday_symbols:
        return universe
    filtered = [row for row in universe if row[0] in intraday_symbols]
    return filtered or universe


def _ensure_candidate_ledger(day_records: dict[date, list[dict[str, Any]]], trade_date: date) -> list[dict[str, Any]]:
    return day_records.setdefault(trade_date, [])


def _build_selection_attribution(
    candidate_ledger: dict[date, list[dict[str, Any]]],
) -> dict[date, dict[str, Any]]:
    selection_attribution: dict[date, dict[str, Any]] = {}
    for trade_date, records in sorted(candidate_ledger.items()):
        if not records:
            continue
        entered = [record for record in records if record.get("disposition") == "entered"]
        skipped = [record for record in records if record.get("disposition") != "entered"]
        gate_groups: dict[str, list[dict[str, Any]]] = {}
        for record in skipped:
            gate_groups.setdefault(record.get("disposition", "unknown"), []).append(record)

        def _avg_actual(items: list[dict[str, Any]]) -> float:
            vals = [float(item["actual_r"]) for item in items if item.get("actual_r") is not None]
            return float(np.mean(vals)) if vals else 0.0

        def _avg_shadow(items: list[dict[str, Any]]) -> float:
            vals = [float(item["shadow_r"]) for item in items if item.get("shadow_r") is not None]
            return float(np.mean(vals)) if vals else 0.0

        best_skipped = max(
            (record for record in skipped if record.get("shadow_r") is not None),
            key=lambda item: float(item.get("shadow_r", -999.0)),
            default=None,
        )
        worst_entered = min(
            (record for record in entered if record.get("actual_r") is not None),
            key=lambda item: float(item.get("actual_r", 999.0)),
            default=None,
        )
        skipped_beating_entered = 0
        if entered:
            entered_rs = [float(item["actual_r"]) for item in entered if item.get("actual_r") is not None]
            if entered_rs:
                worst_taken = min(entered_rs)
                skipped_beating_entered = sum(
                    1
                    for item in skipped
                    if item.get("shadow_r") is not None and float(item["shadow_r"]) > worst_taken
                )

        selection_attribution[trade_date] = {
            "candidate_count": len(records),
            "entered_count": len(entered),
            "entered_symbols": [record["symbol"] for record in entered],
            "entered_avg_r": _avg_actual(entered),
            "skipped_avg_shadow_r": _avg_shadow(skipped),
            "best_skipped_symbol": best_skipped.get("symbol") if best_skipped else None,
            "best_skipped_shadow_r": float(best_skipped["shadow_r"]) if best_skipped and best_skipped.get("shadow_r") is not None else None,
            "worst_entered_symbol": worst_entered.get("symbol") if worst_entered else None,
            "worst_entered_actual_r": float(worst_entered["actual_r"]) if worst_entered and worst_entered.get("actual_r") is not None else None,
            "skipped_beating_worst_entered": skipped_beating_entered,
            "gates": {
                gate: {
                    "count": len(items),
                    "symbols": [item["symbol"] for item in items[:10]],
                    "avg_shadow_r": _avg_shadow(items),
                }
                for gate, items in sorted(gate_groups.items(), key=lambda item: (-len(item[1]), item[0]))
            },
        }
    return selection_attribution


def _mfe_r(position: _PBPosition) -> float:
    if position.risk_per_share <= 0 or position.max_favorable <= 0:
        return 0.0
    return float((position.max_favorable - position.entry_price) / position.risk_per_share)


def _can_carry(position: _PBPosition, close: float, close_pct: float, settings: StrategySettings) -> bool:
    if not settings.pb_carry_enabled:
        return False
    if position.unrealized_r(close) <= settings.pb_carry_min_r:
        return False
    if position.hold_days >= settings.pb_max_hold_days:
        return False
    if close_pct < settings.pb_carry_close_pct_min:
        return False
    if _mfe_r(position) < settings.pb_carry_mfe_gate_r:
        return False
    return True


class IARICPullbackDailyEngine:
    """Tier 3 IARIC pullback-buy backtest engine.

    For each trading day:
    1. Manage carry positions (stop, RSI exit, time stop, profit target)
    2. Check regime gate
    3. Scan universe for pullback candidates (using prev_date indicators)
    4. Rank by RSI ascending (most oversold first)
    5. Enter at today's open with ATR-based stop
    6. Intraday exit management or carry overnight
    """

    def __init__(
        self,
        config: IARICBacktestConfig,
        replay: ResearchReplayEngine,
        settings: StrategySettings | None = None,
        *,
        collect_diagnostics: bool = False,
    ):
        self._config = config
        self._replay = replay
        self._settings = settings or StrategySettings()
        self._collect_diagnostics = collect_diagnostics

        if config.param_overrides:
            self._settings = replace(self._settings, **config.param_overrides)

        self._slippage = config.slippage
        self._trade_universe = _resolve_trade_universe(replay, self._settings)
        self._trade_symbols = {sym for sym, _, _ in self._trade_universe}
        self._sector_map = {sym: sector for sym, sector, _ in self._trade_universe}
        self._effective_min_candidates_day = _effective_min_candidates_day(
            self._settings,
            trade_universe_size=len(self._trade_universe),
            full_universe_size=len(replay._universe),
        )

        # Share expensive pullback pre-computes across engine instances when the
        # upstream indicator settings are unchanged. This is the hot path during
        # phased auto candidate scoring, where many runs only vary downstream params.
        self._indicators, self._date_iloc, self._flow_negative = _shared_pullback_state(
            replay,
            self._settings,
            trade_universe=self._trade_universe,
        )
        # V2: multi-day candidate persistence
        self._prev_day_candidates: set[str] = set()

        # ---- parity: core-logic replay state ----
        self._core_state = IARICCoreState(
            trade_date=date.today(),
            saved_at=datetime.now(timezone.utc),
            symbols=[],
        )
        self._decision_events: list = []
        self._order_counter: int = 0

    # ---- parity helpers ------------------------------------------------
    def _replay_core_step(self, *, bar_input=None, order_updates=None, fills=None):
        result = run_replay(
            self._core_state,
            steps=[ReplayStep(bar_input=bar_input, order_updates=order_updates or [], fills=fills or [])],
            on_bar=lambda state, payload: iaric_core_logic.on_bar(state, **payload),
            on_order_update=iaric_core_logic.on_order_update,
            on_fill=iaric_core_logic.on_fill,
        )
        self._core_state = result.state
        self._decision_events.extend(result.events)
        return result

    def _ensure_core_symbol(self, symbol: str, stop_level: float, route_family: str) -> None:
        for s in self._core_state.symbols:
            if s.symbol == symbol:
                s.stop_level = stop_level
                s.route_family = route_family
                return
        self._core_state.symbols.append(
            PBSymbolState(symbol=symbol, stop_level=stop_level, route_family=route_family)
        )

    def _record_shadow_outcome(
        self,
        record: dict[str, Any],
        shadow_outcomes: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        outcome = self._simulate_shadow_trade(record)
        if outcome is None:
            return None
        record.update(outcome)
        if shadow_outcomes is not None:
            shadow_outcomes.append({
                "trade_date": record.get("trade_date"),
                "symbol": record.get("symbol"),
                "gate": record.get("disposition"),
                **outcome,
            })
        return outcome

    def _record_rejection(
        self,
        record: dict[str, Any],
        gate: str,
        rejection_log: list[dict[str, Any]] | None,
        shadow_outcomes: list[dict[str, Any]] | None,
        funnel_counters: dict[str, int] | None,
    ) -> None:
        record["disposition"] = gate
        record["skip_reason"] = gate
        if funnel_counters is not None:
            funnel_counters[gate] = funnel_counters.get(gate, 0) + 1
        outcome = self._record_shadow_outcome(record, shadow_outcomes)
        if rejection_log is not None:
            rejection_log.append({
                "trade_date": record.get("trade_date"),
                "symbol": record.get("symbol"),
                "gate": gate,
                "trigger_type": record.get("trigger_type"),
                "entry_rsi": record.get("entry_rsi"),
                "entry_gap_pct": record.get("entry_gap_pct"),
                "entry_sma_dist_pct": record.get("entry_sma_dist_pct"),
                "entry_cdd": record.get("entry_cdd"),
                "entry_rank": record.get("entry_rank"),
                "entry_rank_pct": record.get("entry_rank_pct"),
                "n_candidates": record.get("n_candidates"),
                "sector": record.get("sector"),
                "risk_per_share": record.get("risk_per_share"),
                "shadow_r": outcome.get("shadow_r") if outcome else None,
                "shadow_exit_reason": outcome.get("shadow_exit_reason") if outcome else None,
            })

    def _attach_trade_outcome(self, position: _PBPosition, trade: TradeRecord) -> None:
        record = position.ledger_ref
        if record is None:
            return
        record.update({
            "actual_r": float(trade.r_multiple),
            "actual_exit_reason": trade.exit_reason or "UNKNOWN",
            "actual_hold_days": int(trade.hold_bars),
            "actual_mfe_r": float(trade.metadata.get("mfe_r", 0.0)),
            "actual_mae_r": float(trade.metadata.get("mae_r", 0.0)),
            "close_r": float(trade.metadata.get("close_r", 0.0)),
            "close_pct": float(trade.metadata.get("close_pct", 0.0)),
            "exit_rsi": float(trade.metadata.get("exit_rsi", 0.0)),
            "selected_route": str(trade.metadata.get("entry_route_family", "") or record.get("selected_route") or ""),
        })

    def _simulate_shadow_trade(self, record: dict[str, Any]) -> dict[str, Any] | None:
        sym = str(record.get("symbol") or "")
        trade_date = record.get("trade_date")
        entry_price = record.get("entry_price")
        risk_per_share = record.get("risk_per_share")
        stop_price = record.get("stop_price")
        trigger_type = str(record.get("trigger_type") or "UNKNOWN")
        entry_rsi = float(record.get("entry_rsi") or 50.0)
        sector = str(record.get("sector") or "UNKNOWN")
        regime_tier = str(record.get("regime_tier") or "?")

        if (
            not sym
            or not isinstance(trade_date, date)
            or entry_price is None
            or risk_per_share is None
            or stop_price is None
            or float(risk_per_share) <= 0
        ):
            return None

        settings = self._settings
        entry_price = float(entry_price)
        risk_per_share = float(risk_per_share)
        stop_price = float(stop_price)
        ts = datetime(trade_date.year, trade_date.month, trade_date.day, tzinfo=timezone.utc)
        ohlc = self._replay.get_daily_ohlc(sym, trade_date)
        if ohlc is None:
            return None

        O, H, L, C = ohlc
        pos = _PBPosition(
            symbol=sym,
            entry_price=entry_price,
            entry_time=ts,
            quantity=1,
            risk_per_share=risk_per_share,
            sector=sector,
            regime_tier=regime_tier,
            stop=stop_price,
            entry_rsi=entry_rsi,
            trigger_type=trigger_type,
            max_favorable=H,
            max_adverse=L,
            entry_atr=float(record.get("entry_atr") or 0.0),
            entry_gap_pct=float(record.get("entry_gap_pct") or 0.0),
            entry_sma_dist_pct=float(record.get("entry_sma_dist_pct") or 0.0),
            entry_cdd=int(record.get("entry_cdd") or 0),
            entry_rank=int(record.get("entry_rank") or 0),
            entry_rank_pct=float(record.get("entry_rank_pct") or 0.0),
            n_candidates=int(record.get("n_candidates") or 0),
            daily_signal_score=float(record.get("daily_signal_score") or 0.0),
            daily_signal_rank_pct=float(record.get("daily_signal_rank_pct") or 100.0),
            signal_family=str(record.get("signal_family") or getattr(self._settings, "pb_daily_signal_family", "balanced_v1")),
            route_family=str(record.get("route_family") or "OPEN_SCORED_ENTRY"),
            selection_reason=str(record.get("selection_reason") or "daily_signal_score"),
        )
        pos.close_pct = _close_in_range_pct(H, L, C)
        pos.close_r = pos.unrealized_r(C)

        ind = self._indicators.get(sym)
        didx = self._replay._daily_didx.get(sym)
        if ind is not None and didx is not None:
            iloc_today = _iloc_for_date(didx, trade_date)
            if iloc_today >= 0 and not np.isnan(ind["rsi"][iloc_today]):
                pos.exit_rsi = float(ind["rsi"][iloc_today])

        exit_price = None
        exit_reason = ""

        if L <= pos.stop:
            exit_price = C if settings.pb_use_close_stop else pos.stop
            exit_reason = "STOP_HIT"
        elif settings.pb_profit_target_r > 0 and pos.unrealized_r(C) > settings.pb_profit_target_r:
            exit_price = C
            exit_reason = "PROFIT_TARGET"
        elif settings.pb_carry_enabled and _can_carry(pos, C, pos.close_pct, settings):
            pos.carry_eligible = True
            pos.hold_days = 1
            pos.highest_carry_close = C
            cur_date = trade_date
            while True:
                next_date = self._replay.get_next_trading_date(cur_date)
                if next_date is None:
                    exit_price = C
                    exit_reason = "END_OF_DATA"
                    break
                cur_date = next_date
                ohlc_next = self._replay.get_daily_ohlc(sym, cur_date)
                if ohlc_next is None:
                    exit_price = C
                    exit_reason = "END_OF_DATA"
                    break
                O2, H2, L2, C2 = ohlc_next
                pos.hold_days += 1
                pos.max_favorable = max(pos.max_favorable, H2)
                pos.max_adverse = min(pos.max_adverse, L2) if pos.max_adverse > 0 else L2
                pos.highest_carry_close = max(pos.highest_carry_close, C2)
                pos.close_pct = _close_in_range_pct(H2, L2, C2)

                prev_date = self._replay.get_prev_trading_date(cur_date)
                last_n = None
                if prev_date is not None:
                    last_n = self._replay.get_flow_proxy_last_n(
                        sym,
                        prev_date,
                        max(1, settings.pb_flow_reversal_lookback),
                    )
                if last_n is not None and all(v < 0 for v in last_n):
                    exit_price = O2
                    exit_reason = "FLOW_REVERSAL"
                elif L2 <= pos.stop:
                    exit_price = C2 if settings.pb_use_close_stop else pos.stop
                    exit_reason = "STOP_HIT"
                elif ind is not None and didx is not None:
                    iloc_cur = _iloc_for_date(didx, cur_date)
                    if iloc_cur >= 0 and not np.isnan(ind["rsi"][iloc_cur]) and ind["rsi"][iloc_cur] > settings.pb_rsi_exit:
                        pos.exit_rsi = float(ind["rsi"][iloc_cur])
                        exit_price = C2
                        exit_reason = "RSI_EXIT"
                if exit_price is None and pos.hold_days >= settings.pb_max_hold_days:
                    exit_price = C2
                    exit_reason = "TIME_STOP"
                if exit_price is None and settings.pb_profit_target_r > 0 and pos.unrealized_r(C2) > settings.pb_profit_target_r:
                    exit_price = C2
                    exit_reason = "PROFIT_TARGET"
                if exit_price is None:
                    if _can_carry(pos, C2, pos.close_pct, settings):
                        C = C2
                        continue
                    exit_price = C2
                    exit_reason = "CARRY_EXIT"
                C = C2
                break
        else:
            exit_price = C
            exit_reason = "EOD_FLATTEN"

        if exit_price is None:
            exit_price = C
            exit_reason = "EOD_FLATTEN"

        slip = float(exit_price) * self._slippage.slip_bps_normal / 10_000
        fill = round(float(exit_price) - slip, 2)
        return {
            "shadow_r": pos.unrealized_r(fill),
            "shadow_exit_reason": exit_reason,
            "shadow_hold_days": int(max(pos.hold_days, 1)),
            "shadow_mfe_r": float((pos.max_favorable - pos.entry_price) / max(pos.risk_per_share, 0.01)) if pos.max_favorable > 0 else 0.0,
            "shadow_mae_r": float((pos.entry_price - pos.max_adverse) / max(pos.risk_per_share, 0.01)) if pos.max_adverse > 0 and pos.max_adverse < pos.entry_price else 0.0,
            "shadow_close_r": float(pos.close_r),
            "shadow_close_pct": float(pos.close_pct),
        }

    def run(self) -> IARICPullbackResult:
        """Execute the full pullback-buy backtest."""
        cfg = self._config
        settings = self._settings
        collect_diagnostics = self._collect_diagnostics
        start = date.fromisoformat(cfg.start_date)
        end = date.fromisoformat(cfg.end_date)

        trading_dates = self._replay.tradable_dates(start, end)
        if not trading_dates:
            logger.warning("No trading dates in range %s to %s", start, end)
            return IARICPullbackResult(
                trades=[], equity_curve=np.array([cfg.initial_equity]),
                timestamps=np.array([]), daily_selections={},
            )

        # State
        equity = cfg.initial_equity
        carry_positions: dict[str, _PBPosition] = {}
        trades: list[TradeRecord] = []
        equity_history: list[float] = [equity]
        ts_history: list[datetime] = []
        daily_selections: dict[date, WatchlistArtifact] = {}
        candidate_ledger: dict[date, list[dict[str, Any]]] | None = {} if collect_diagnostics else None
        funnel_counters: dict[str, int] | None = ({
            "universe_seen": 0,
            "triggered": 0,
            "flow_reject": 0,
            "flow_rescue_pool": 0,
            "candidate_pool": 0,
            "min_candidates_day_reject": 0,
            "rank_abs_reject": 0,
            "rank_pct_reject": 0,
            "sector_cap_reject": 0,
            "position_cap_reject": 0,
            "sizing_reject": 0,
            "buying_power_reject": 0,
            "entered": 0,
        } if collect_diagnostics else None)
        rejection_log: list[dict[str, Any]] | None = [] if collect_diagnostics else None
        shadow_outcomes: list[dict[str, Any]] | None = [] if collect_diagnostics else None

        for trade_date in trading_dates:
            ts = datetime(trade_date.year, trade_date.month, trade_date.day, tzinfo=timezone.utc)
            ts_history.append(ts)

            prev_date = self._replay.get_prev_trading_date(trade_date)
            if prev_date is None:
                equity_history.append(equity)
                continue

            # ===== 1. CARRY MANAGEMENT =====
            closed_carry: list[str] = []
            for sym, pos in list(carry_positions.items()):
                ohlc = self._replay.get_daily_ohlc(sym, trade_date)
                if ohlc is None:
                    closed_carry.append(sym)
                    continue

                O, H, L, C = ohlc
                pos.hold_days += 1
                pos.max_favorable = max(pos.max_favorable, H)
                pos.max_adverse = min(pos.max_adverse, L) if pos.max_adverse > 0 else L
                pos.highest_carry_close = max(pos.highest_carry_close, C)
                pos.close_pct = _close_in_range_pct(H, L, C)

                exit_price = None
                exit_reason = ""
                v2 = bool(settings.pb_v2_enabled)

                if v2:
                    # V2 carry exits: stop, EMA reversion, RSI, flatten check
                    if L <= pos.stop:
                        exit_price = pos.stop
                        exit_reason = "STOP_HIT"

                    # EMA reversion exit (Day 2+)
                    if exit_price is None and settings.pb_v2_ema_reversion_exit:
                        ind = self._indicators.get(sym)
                        if ind is not None and "ema10" in ind:
                            didx = self._replay._daily_didx.get(sym)
                            if didx is not None:
                                prev_iloc = _iloc_for_date(didx, prev_date)
                                if prev_iloc >= 0 and not np.isnan(ind["ema10"][prev_iloc]):
                                    if C >= ind["ema10"][prev_iloc] and pos.unrealized_r(C) > settings.pb_v2_ema_reversion_min_r:
                                        exit_price = C
                                        exit_reason = "EMA_REVERSION"

                    # Route-specific RSI exit
                    if exit_price is None:
                        ind = self._indicators.get(sym)
                        if ind is not None:
                            didx = self._replay._daily_didx.get(sym)
                            if didx is not None:
                                iloc = _iloc_for_date(didx, trade_date)
                                if iloc >= 0 and not np.isnan(ind["rsi"][iloc]):
                                    rsi_thresh = _v2_rsi_exit_threshold(pos.route_family, settings)
                                    if ind["rsi"][iloc] > rsi_thresh:
                                        exit_price = C
                                        exit_reason = "RSI_EXIT"

                    # V2 flatten check (inverted carry)
                    if exit_price is None:
                        last_n = self._replay.get_flow_proxy_last_n(sym, prev_date, 2)
                        flatten, flatten_reason = _should_flatten_v2(
                            pos, C, pos.close_pct, regime_tier, list(last_n) if last_n is not None else None, settings,
                        )
                        if flatten:
                            exit_price = C
                            exit_reason = flatten_reason
                        else:
                            continue  # Keep carrying (V2 default)
                else:
                    # Legacy carry management
                    # Flow reversal exit (2 neg flow days -> exit at open)
                    last_n = self._replay.get_flow_proxy_last_n(
                        sym,
                        prev_date,
                        max(1, settings.pb_flow_reversal_lookback),
                    )
                    if last_n is not None and all(v < 0 for v in last_n):
                        exit_price = O
                        exit_reason = "FLOW_REVERSAL"

                    # Stop hit
                    if exit_price is None and L <= pos.stop:
                        exit_price = C if settings.pb_use_close_stop else pos.stop
                        exit_reason = "STOP_HIT"

                    # RSI exit: mean-reversion target reached
                    if exit_price is None:
                        ind = self._indicators.get(sym)
                        if ind is not None:
                            didx = self._replay._daily_didx.get(sym)
                            if didx is not None:
                                iloc = _iloc_for_date(didx, trade_date)
                                if iloc >= 0 and not np.isnan(ind["rsi"][iloc]):
                                    if ind["rsi"][iloc] > settings.pb_rsi_exit:
                                        exit_price = C
                                        exit_reason = "RSI_EXIT"

                    # Time stop
                    if exit_price is None and pos.hold_days >= settings.pb_max_hold_days:
                        exit_price = C
                        exit_reason = "TIME_STOP"

                    # Profit target
                    if exit_price is None and settings.pb_profit_target_r > 0:
                        if pos.unrealized_r(C) > settings.pb_profit_target_r:
                            exit_price = C
                            exit_reason = "PROFIT_TARGET"

                    # Still carry-eligible? Check if should continue carrying
                    if exit_price is None:
                        if _can_carry(pos, C, pos.close_pct, settings):
                            continue  # Keep carrying
                        exit_price = C
                        exit_reason = "CARRY_EXIT"

                # Populate close_r and exit_rsi for carry exit diagnostics
                pos.close_r = pos.unrealized_r(C)
                ind = self._indicators.get(sym)
                if ind is not None:
                    didx = self._replay._daily_didx.get(sym)
                    if didx is not None:
                        iloc = _iloc_for_date(didx, trade_date)
                        if iloc >= 0 and not np.isnan(ind["rsi"][iloc]):
                            pos.exit_rsi = ind["rsi"][iloc]

                # Close carry position
                slip = exit_price * self._slippage.slip_bps_normal / 10_000
                fill = round(exit_price - slip, 2)
                commission = self._slippage.commission_per_share * pos.quantity
                pnl = (fill - pos.entry_price) * pos.quantity
                r_mult = (pnl - pos.commission_entry - commission) / pos.total_risk if pos.total_risk > 0 else 0.0
                equity += pnl - commission - pos.commission_entry

                trade = TradeRecord(
                    strategy="IARIC_PB",
                    symbol=sym,
                    direction=BTDirection.LONG,
                    entry_time=pos.entry_time,
                    exit_time=ts,
                    entry_price=pos.entry_price,
                    exit_price=fill,
                    quantity=pos.quantity,
                    pnl=pnl,
                    r_multiple=r_mult,
                    risk_per_share=pos.risk_per_share,
                    commission=pos.commission_entry + commission,
                    slippage=pos.slippage_entry + slip * pos.quantity,
                    entry_type=pos.trigger_type,
                    exit_reason=exit_reason,
                    sector=pos.sector,
                    regime_tier=pos.regime_tier,
                    hold_bars=pos.hold_days,
                    max_favorable=pos.max_favorable,
                    max_adverse=pos.max_adverse,
                    metadata=pos.build_metadata(),
                )
                trades.append(trade)
                self._attach_trade_outcome(pos, trade)

                # ---- parity: notify core of carry exit ----
                _exit_oid = f"iaric-x-{sym}-{self._order_counter}"
                self._order_counter += 1
                self._replay_core_step(
                    bar_input={"bar_ts": ts, "flatten_request": IARICFlattenRequest(symbol=sym, reason=exit_reason)},
                    fills=[
                        IARICFill(
                            oms_order_id=_exit_oid,
                            fill_price=fill,
                            fill_qty=pos.quantity,
                            fill_time=ts,
                            commission=commission,
                            symbol=sym,
                            order_role="EXIT",
                            exit_type=exit_reason,
                        )
                    ],
                )

                if cfg.verbose:
                    logger.info(
                        "[%s] CARRY EXIT %s @ %.2f reason=%s PnL=%.2f R=%.2f",
                        trade_date, sym, fill, exit_reason, pnl, r_mult,
                    )

                closed_carry.append(sym)

            for sym in closed_carry:
                carry_positions.pop(sym, None)

            # ===== 2. REGIME CHECK =====
            artifact = self._replay.iaric_selection_for_date(prev_date, self._settings)
            daily_selections[trade_date] = artifact
            item_lookup = getattr(artifact, "by_symbol", {})

            regime_tier = artifact.regime.tier
            if settings.pb_regime_gate == "C_only_skip":
                if regime_tier == "C":
                    equity_history.append(equity)
                    continue
            elif settings.pb_regime_gate == "B_and_above":
                if regime_tier not in ("A", "B"):
                    equity_history.append(equity)
                    continue
            # "any" -> no regime filter

            # ===== 3. SCAN UNIVERSE FOR PULLBACK CANDIDATES =====
            candidates: list[dict[str, Any]] = []
            sector_raw_counts: dict[str, int] = {}
            v2 = bool(settings.pb_v2_enabled)

            flow_policy = str(getattr(settings, "pb_flow_policy", "soft_penalty_rescue") or "soft_penalty_rescue").lower()
            use_cdd = settings.pb_cdd_min > 0
            use_mazone = settings.pb_ma_zone_entry
            rsi_thresh = settings.pb_rsi_entry
            cdd_thresh = settings.pb_cdd_min
            today_candidate_syms: set[str] = set()

            for sym, sector_raw, _ in self._trade_universe:
                if funnel_counters is not None:
                    funnel_counters["universe_seen"] += 1
                # Skip SPY benchmark in entry evaluation
                if v2 and sector_raw == "benchmark":
                    continue
                if sym in carry_positions:
                    continue

                ind = self._indicators.get(sym)
                if ind is None:
                    continue

                di = self._date_iloc.get(sym)
                if di is None:
                    continue
                iloc = di.get(prev_date, -1)
                if iloc < 0:
                    continue

                closes = self._replay._daily_arrs[sym]["close"]
                sma_trend_val = ind["sma_trend"][iloc]
                if np.isnan(sma_trend_val):
                    continue

                # --- Trend filter ---
                trend_tier = ""
                if v2:
                    above_sma50 = closes[iloc] > sma_trend_val
                    slope_ok = bool(ind["sma_slope"][iloc])
                    sma200_val = ind["sma200"][iloc]
                    above_sma200 = not np.isnan(sma200_val) and closes[iloc] > sma200_val
                    sma50_above_200 = not np.isnan(sma200_val) and sma_trend_val > sma200_val
                    if above_sma50 and slope_ok:
                        trend_tier = "STRONG"
                    elif settings.pb_v2_allow_secular and above_sma200 and sma50_above_200:
                        trend_tier = "SECULAR"
                    else:
                        continue
                else:
                    if closes[iloc] <= sma_trend_val:
                        continue
                    if not ind["sma_slope"][iloc]:
                        continue
                    trend_tier = "STRONG"

                prev_close_val = closes[iloc]
                if prev_close_val <= 0:
                    continue
                ohlc = self._replay.get_daily_ohlc(sym, trade_date)
                if ohlc is None:
                    continue
                O, _H, _L, _C = ohlc
                gap_pct = (O - prev_close_val) / prev_close_val * 100
                sma_dist_pct = (prev_close_val - sma_trend_val) / sma_trend_val * 100 if sma_trend_val > 0 else 0.0
                cdd_val = int(ind["cdd"][iloc])

                # --- Range filters ---
                if v2:
                    if gap_pct < settings.pb_v2_gap_min_pct or gap_pct > settings.pb_v2_gap_max_pct:
                        continue
                    if sma_dist_pct < settings.pb_v2_sma_dist_min_pct or sma_dist_pct > settings.pb_v2_sma_dist_max_pct:
                        continue
                else:
                    if gap_pct < settings.pb_gap_min_pct or gap_pct > settings.pb_gap_max_pct:
                        continue
                    if sma_dist_pct < settings.pb_sma_dist_min_pct or sma_dist_pct > settings.pb_sma_dist_max_pct:
                        continue
                if cdd_val > settings.pb_cdd_max:
                    continue

                # --- Entry triggers ---
                trigger_types_list: list[str] = []
                trigger_tier = ""
                trigger_type = ""
                rsi_val = ind["rsi"][iloc]

                if v2:
                    rs_val = ind["rs_ratio"][iloc] if not np.isnan(ind["rs_ratio"][iloc]) else 1.0
                    v2_triggers = _evaluate_v2_triggers(ind, iloc, closes, prev_close_val, gap_pct, trend_tier, rs_val, settings)
                    if not v2_triggers:
                        continue
                    trigger_types_list = [t[0] for t in v2_triggers]
                    trigger_tier = "HIGH" if any(t[1] == "HIGH" for t in v2_triggers) else "MEDIUM"
                    trigger_type = trigger_types_list[0]
                else:
                    if not np.isnan(rsi_val) and rsi_val < rsi_thresh:
                        trigger_type = "RSI"
                    elif use_cdd and ind["cdd"][iloc] >= cdd_thresh:
                        trigger_type = "CDD"
                    elif use_mazone:
                        sma20_val = ind["sma20"][iloc]
                        if not np.isnan(sma20_val) and closes[iloc] < sma20_val and closes[iloc] > sma_trend_val:
                            trigger_type = "MA_ZONE"
                    if not trigger_type:
                        continue
                    trigger_types_list = [trigger_type]
                    trigger_tier = "MEDIUM"

                today_candidate_syms.add(sym)
                day_records = _ensure_candidate_ledger(candidate_ledger, trade_date) if candidate_ledger is not None else None
                sector = self._sector_map.get(sym, "Unknown")
                sector_raw_counts[sector] = sector_raw_counts.get(sector, 0) + 1
                slip = O * self._slippage.slip_bps_normal / 10_000
                fill_price = round(O + slip, 2)
                atr_val = ind["atr"][iloc]
                stop_price = None
                risk_per_share = None
                if not np.isnan(atr_val) and atr_val > 0:
                    stop_price = fill_price - settings.pb_atr_stop_mult * atr_val
                    risk_per_share = fill_price - stop_price

                record: dict[str, Any] | None = None
                if day_records is not None:
                    watch_item = item_lookup.get(sym)
                    record = {
                        "trade_date": trade_date,
                        "symbol": sym,
                        "trigger_type": trigger_type,
                        "trigger_types": trigger_types_list,
                        "trigger_tier": trigger_tier,
                        "trend_tier": trend_tier,
                        "entry_rsi": float(rsi_val) if not np.isnan(rsi_val) else 50.0,
                        "entry_gap_pct": float(gap_pct),
                        "entry_sma_dist_pct": float(sma_dist_pct),
                        "entry_cdd": int(cdd_val),
                        "entry_rank": 0,
                        "entry_rank_pct": 100.0,
                        "n_candidates": 0,
                        "sector": sector,
                        "regime_tier": regime_tier,
                        "entry_price": float(fill_price),
                        "entry_open": float(O),
                        "entry_atr": float(atr_val) if not np.isnan(atr_val) else 0.0,
                        "stop_price": float(stop_price) if stop_price is not None else None,
                        "risk_per_share": float(risk_per_share) if risk_per_share is not None else None,
                        "risk_budget_mult": _risk_budget_mult(trade_date, settings),
                        "candidate_count_raw": 0,
                        "daily_signal_score": 0.0,
                        "daily_signal_rank_pct": 100.0,
                        "signal_family": "v2" if v2 else str(getattr(settings, "pb_daily_signal_family", "balanced_v1")),
                        "daily_signal_min_score_threshold": float(settings.pb_v2_signal_floor if v2 else getattr(settings, "pb_daily_signal_min_score", 0.0)),
                        "selection_reason": "",
                        "skip_reason": "",
                        "capacity_reason": "",
                        "selected_route": "",
                        "route_family": "",
                        "route_score": 0.0,
                        "route_feasible": False,
                        "route_feasible_bar_index": None,
                        "flow_negative": False,
                        "flow_policy": flow_policy,
                        "flow_proxy_gate_pass": bool(getattr(watch_item, "flow_proxy_gate_pass", True)) if watch_item is not None else True,
                        "disposition": "triggered",
                        "actual_r": None,
                        "shadow_r": None,
                    }
                    day_records.append(record)
                if funnel_counters is not None:
                    funnel_counters["triggered"] += 1

                effective_rsi = rsi_val if not np.isnan(rsi_val) else 50.0
                flow_negative = False
                if settings.pb_flow_gate:
                    fn = self._flow_negative.get(sym)
                    flow_negative = bool(fn is not None and fn[iloc])
                if record is not None:
                    record["flow_negative"] = flow_negative
                candidates.append({
                    "symbol": sym,
                    "effective_rsi": float(effective_rsi),
                    "trigger_type": trigger_type,
                    "trigger_types": trigger_types_list,
                    "trigger_tier": trigger_tier,
                    "trend_tier": trend_tier,
                    "sector": sector,
                    "item": item_lookup.get(sym),
                    "flow_negative": flow_negative,
                    "entry_rsi": float(rsi_val) if not np.isnan(rsi_val) else 50.0,
                    "entry_gap_pct": float(gap_pct),
                    "entry_sma_dist_pct": float(sma_dist_pct),
                    "entry_cdd": int(cdd_val),
                    "entry_price": float(fill_price),
                    "entry_open": float(O),
                    "entry_atr": float(atr_val) if not np.isnan(atr_val) else 0.0,
                    "stop_price": float(stop_price) if stop_price is not None else None,
                    "risk_per_share": float(risk_per_share) if risk_per_share is not None else None,
                    "record": record,
                    "prev_iloc": iloc,
                })

            scored_candidates: list[dict[str, Any]] = []
            rescue_floor = float(getattr(settings, "pb_daily_rescue_min_score", settings.pb_rescue_min_score))
            for candidate in candidates:
                record = candidate.get("record")
                sym = str(candidate["symbol"])
                iloc = int(candidate["prev_iloc"])

                if v2:
                    ind = self._indicators.get(sym, {})
                    _c_arrs = self._replay._daily_arrs.get(sym)
                    is_down_day = bool(_c_arrs["close"][iloc] < _c_arrs["close"][iloc - 1]) if _c_arrs is not None and iloc > 0 else False

                    def _safe_v2(key: str, default: float = float("nan")) -> float:
                        arr = ind.get(key)
                        if arr is not None and iloc < len(arr) and not np.isnan(arr[iloc]):
                            return float(arr[iloc])
                        return default

                    bundle = _daily_signal_bundle_v2(
                        trend_tier=str(candidate.get("trend_tier", "STRONG")),
                        trigger_types=list(candidate.get("trigger_types", [candidate.get("trigger_type", "")])),
                        trigger_tier=str(candidate.get("trigger_tier", "MEDIUM")),
                        adx=_safe_v2("adx"),
                        plus_di=_safe_v2("plus_di"),
                        minus_di=_safe_v2("minus_di"),
                        sma_slope_pos=bool(ind["sma_slope"][iloc]) if "sma_slope" in ind else False,
                        sma_dist_pct=float(candidate.get("entry_sma_dist_pct", 0.0)),
                        pullback_depth_atr=_safe_v2("depth"),
                        rsi2=float(candidate.get("entry_rsi", 50.0)),
                        rsi5=_safe_v2("rsi5", 50.0),
                        vcr=_safe_v2("vcr", 1.0),
                        is_down_day=is_down_day,
                        rs_ratio=_safe_v2("rs_ratio"),
                        gap_pct=float(candidate.get("entry_gap_pct", 0.0)),
                        regime_tier=regime_tier,
                        sector_count=sector_raw_counts.get(str(candidate.get("sector")), 1),
                        item=candidate.get("item"),
                        n_triggers=len(candidate.get("trigger_types", [])),
                        candidate_yesterday=sym in self._prev_day_candidates,
                    )
                    signal_floor = settings.pb_v2_signal_floor
                    if regime_tier == "B" and settings.pb_v2_signal_floor_tier_b > 0:
                        signal_floor = settings.pb_v2_signal_floor_tier_b
                else:
                    bundle = _daily_signal_bundle(
                        settings=settings,
                        regime_tier=regime_tier,
                        item=candidate.get("item"),
                        entry_rsi=float(candidate.get("entry_rsi", 50.0)),
                        gap_pct=float(candidate.get("entry_gap_pct", 0.0)),
                        sma_dist_pct=float(candidate.get("entry_sma_dist_pct", 0.0)),
                        cdd=int(candidate.get("entry_cdd", 0)),
                        flow_negative=bool(candidate.get("flow_negative")),
                        sector_count=sector_raw_counts.get(str(candidate.get("sector")), 1),
                        total_candidates=len(candidates),
                        effective_min_candidates_day=self._effective_min_candidates_day,
                    )
                    signal_floor = 0.0  # legacy uses _passes_daily_signal_floor

                candidate["daily_signal_score"] = float(bundle["score"])
                candidate["daily_signal_components"] = dict(bundle)
                rescue_candidate = bool(candidate.get("flow_negative")) and flow_policy == "soft_penalty_rescue" and candidate["daily_signal_score"] >= rescue_floor
                candidate["rescue_flow_candidate"] = rescue_candidate
                if record is not None:
                    record["candidate_count_raw"] = len(candidates)
                    record["daily_signal_score"] = float(bundle["score"])
                    record["signal_family"] = "v2" if v2 else str(getattr(settings, "pb_daily_signal_family", "balanced_v1"))
                    record["selection_reason"] = "daily_signal_score"
                    record["rescue_flow_candidate"] = rescue_candidate
                    for name, value in bundle.items():
                        record[f"daily_signal_component_{name}"] = round(float(value), 4)

                # Floor check
                if v2:
                    if candidate["daily_signal_score"] < signal_floor:
                        if record is not None:
                            self._record_rejection(record, "daily_signal_floor_reject", rejection_log, shadow_outcomes, funnel_counters)
                        continue
                else:
                    if not _passes_daily_signal_floor(settings, candidate["daily_signal_score"], rescue_candidate=rescue_candidate):
                        if record is not None:
                            self._record_rejection(record, "daily_signal_floor_reject", rejection_log, shadow_outcomes, funnel_counters)
                        continue
                if bool(candidate.get("flow_negative")) and flow_policy == "hard_reject":
                    if record is not None:
                        self._record_rejection(record, "flow_reject", rejection_log, shadow_outcomes, funnel_counters)
                    continue
                if bool(candidate.get("flow_negative")) and flow_policy == "soft_penalty_rescue" and not rescue_candidate:
                    if record is not None:
                        self._record_rejection(record, "flow_reject", rejection_log, shadow_outcomes, funnel_counters)
                    continue
                if rescue_candidate and funnel_counters is not None:
                    funnel_counters["flow_rescue_pool"] = funnel_counters.get("flow_rescue_pool", 0) + 1
                if record is not None:
                    record["disposition"] = "candidate_pool"
                if funnel_counters is not None:
                    funnel_counters["candidate_pool"] += 1
                scored_candidates.append(candidate)

            candidates = sorted(
                scored_candidates,
                key=lambda row: (
                    -float(row.get("daily_signal_score", 0.0)),
                    -float(getattr(row.get("item"), "daily_rank", 0.0) if row.get("item") is not None else 0.0),
                    float(row.get("effective_rsi", 50.0)),
                ),
            )
            if bool(getattr(settings, "pb_min_candidates_day_hard_gate", False)) and len(candidates) < self._effective_min_candidates_day:
                if candidate_ledger is not None:
                    for candidate in candidates:
                        record = candidate.get("record")
                        if record is not None:
                            record["n_candidates"] = len(candidates)
                            self._record_rejection(
                                record,
                                "min_candidates_day_reject",
                                rejection_log,
                                shadow_outcomes,
                                funnel_counters,
                            )
                equity_history.append(equity)
                continue

            available_slots = settings.pb_max_positions - len(carry_positions)
            sector_counts: dict[str, int] = {}
            for pos in carry_positions.values():
                sector_counts[pos.sector] = sector_counts.get(pos.sector, 0) + 1

            # ===== 5. ENTRY AT TODAY'S OPEN =====
            intraday_positions: list[_PBPosition] = []
            n_candidates_total = len(candidates)
            rank_counter = 0

            for candidate in candidates:
                sym = str(candidate["symbol"])
                rsi_val = float(candidate["effective_rsi"])
                trigger_type = str(candidate["trigger_type"])
                record = candidate.get("record")
                daily_signal_score = float(candidate.get("daily_signal_score", 0.0))
                rank_counter += 1
                rank_pct = _rank_percent(rank_counter, n_candidates_total)
                if record is not None:
                    record["entry_rank"] = rank_counter
                    record["entry_rank_pct"] = rank_pct
                    record["n_candidates"] = n_candidates_total
                    record["daily_signal_rank_pct"] = rank_pct
                gate_reason = _rank_gate_reason(rank_counter, n_candidates_total, settings)
                if gate_reason is not None:
                    if record is not None:
                        self._record_rejection(record, gate_reason, rejection_log, shadow_outcomes, funnel_counters)
                    continue

                sector = str(candidate["sector"])
                sec_count = sector_counts.get(sector, 0)
                if sec_count >= settings.max_positions_per_sector:
                    if record is not None:
                        self._record_rejection(record, "sector_cap_reject", rejection_log, shadow_outcomes, funnel_counters)
                    continue
                if len(intraday_positions) >= available_slots:
                    if record is not None:
                        self._record_rejection(record, "position_cap_reject", rejection_log, shadow_outcomes, funnel_counters)
                    if not collect_diagnostics:
                        break
                    continue

                ohlc = self._replay.get_daily_ohlc(sym, trade_date)
                if ohlc is None:
                    continue
                O, H, L, C = ohlc

                fill_price = float(record["entry_price"]) if record is not None and record.get("entry_price") is not None else round(O + O * self._slippage.slip_bps_normal / 10_000, 2)
                ind = self._indicators[sym]
                iloc = int(candidate["prev_iloc"])
                if iloc < 0:
                    continue
                atr_val = float(record["entry_atr"]) if record is not None and record.get("entry_atr") is not None else float(ind["atr"][iloc])
                if np.isnan(atr_val) or atr_val <= 0:
                    if record is not None:
                        self._record_rejection(record, "sizing_reject", rejection_log, shadow_outcomes, funnel_counters)
                    continue

                stop_price = float(record["stop_price"]) if record is not None and record.get("stop_price") is not None else fill_price - settings.pb_atr_stop_mult * atr_val
                risk_per_share = float(record["risk_per_share"]) if record is not None and record.get("risk_per_share") is not None else fill_price - stop_price
                if risk_per_share <= 0:
                    if record is not None:
                        self._record_rejection(record, "sizing_reject", rejection_log, shadow_outcomes, funnel_counters)
                    continue

                # Position sizing
                risk_dollars = equity * settings.base_risk_fraction * _risk_budget_mult(trade_date, settings)
                if v2:
                    sizing_mult = _v2_score_sizing_mult(daily_signal_score, str(candidate.get("trend_tier", "STRONG")), "OPEN_SCORED_ENTRY", settings)
                    if sizing_mult <= 0:
                        if record is not None:
                            self._record_rejection(record, "sizing_reject", rejection_log, shadow_outcomes, funnel_counters)
                        continue
                    risk_dollars *= sizing_mult
                elif bool(candidate.get("rescue_flow_candidate")):
                    risk_dollars *= float(getattr(settings, "pb_rescue_size_mult", 0.65))
                if risk_dollars <= 0:
                    if record is not None:
                        self._record_rejection(record, "sizing_reject", rejection_log, shadow_outcomes, funnel_counters)
                    continue
                qty = int(floor(risk_dollars / risk_per_share))
                if qty < 1:
                    if record is not None:
                        self._record_rejection(record, "sizing_reject", rejection_log, shadow_outcomes, funnel_counters)
                    continue

                # Buying power check
                if settings.intraday_leverage > 0:
                    carry_notional = sum(
                        p.entry_price * p.quantity for p in carry_positions.values()
                    )
                    intraday_notional = sum(
                        p.entry_price * p.quantity for p in intraday_positions
                    )
                    available_bp = equity * settings.intraday_leverage - carry_notional - intraday_notional
                    max_qty_bp = int(available_bp / fill_price) if fill_price > 0 else 0
                    qty = min(qty, max_qty_bp)
                    if qty < 1:
                        if record is not None:
                            self._record_rejection(record, "buying_power_reject", rejection_log, shadow_outcomes, funnel_counters)
                        continue

                commission = self._slippage.commission_per_share * qty

                # Compute entry diagnostics metadata
                prev_close_val = self._replay._daily_arrs[sym]["close"][iloc]
                sma_val = ind["sma_trend"][iloc]
                cdd_val = int(ind["cdd"][iloc])

                pos = _PBPosition(
                    symbol=sym,
                    entry_price=fill_price,
                    entry_time=ts,
                    quantity=qty,
                    risk_per_share=risk_per_share,
                    sector=sector,
                    regime_tier=regime_tier,
                    stop=stop_price,
                    entry_rsi=rsi_val,
                    trigger_type=trigger_type,
                    max_favorable=H,
                    max_adverse=L,
                    commission_entry=commission,
                    slippage_entry=slip * qty,
                    entry_atr=atr_val,
                    entry_gap_pct=(O - prev_close_val) / prev_close_val * 100 if prev_close_val > 0 else 0.0,
                    entry_sma_dist_pct=(prev_close_val - sma_val) / sma_val * 100 if sma_val > 0 else 0.0,
                    entry_cdd=cdd_val,
                    entry_rank=rank_counter,
                    entry_rank_pct=rank_pct,
                    n_candidates=n_candidates_total,
                    daily_signal_score=daily_signal_score,
                    daily_signal_rank_pct=rank_pct,
                    signal_family="v2" if v2 else str(getattr(settings, "pb_daily_signal_family", "balanced_v1")),
                    route_family="OPEN_SCORED_ENTRY",
                    selection_reason="daily_signal_score",
                    ledger_ref=record,
                    trigger_types=list(candidate.get("trigger_types", [trigger_type])),
                    trigger_tier=str(candidate.get("trigger_tier", "")),
                    trend_tier=str(candidate.get("trend_tier", "STRONG")),
                )

                intraday_positions.append(pos)
                sector_counts[sector] = sec_count + 1
                if record is not None:
                    record["disposition"] = "entered"
                    record["quantity"] = qty
                    record["selected_route"] = "OPEN_SCORED_ENTRY"
                    record["route_family"] = "OPEN_SCORED_ENTRY"
                    record["route_score"] = daily_signal_score
                    record["route_feasible"] = True
                    record["route_feasible_bar_index"] = 0
                    record["selection_reason"] = "daily_signal_score"
                if funnel_counters is not None:
                    funnel_counters["entered"] += 1

                # ---- parity: notify core of entry ----
                _entry_oid = f"iaric-e-{sym}-{self._order_counter}"
                self._order_counter += 1
                self._ensure_core_symbol(sym, stop_price, "OPEN_SCORED_ENTRY")
                self._replay_core_step(
                    bar_input={
                        "bar_ts": ts,
                        "entry_request": IARICEntryRequest(
                            client_order_id=_entry_oid,
                            symbol=sym,
                            route="OPEN_SCORED_ENTRY",
                            qty=qty,
                            limit_price=fill_price,
                            stop_price=stop_price,
                        ),
                    },
                )
                self._replay_core_step(
                    fills=[
                        IARICFill(
                            oms_order_id=_entry_oid,
                            fill_price=fill_price,
                            fill_qty=qty,
                            fill_time=ts,
                            commission=commission,
                            symbol=sym,
                            order_role="ENTRY",
                        )
                    ],
                )

                if cfg.verbose:
                    logger.info(
                        "[%s] ENTRY %s @ %.2f qty=%d trigger=%s RSI=%.1f regime=%s",
                        trade_date, sym, fill_price, qty, trigger_type, rsi_val, regime_tier,
                    )

            # ===== 6. INTRADAY EXIT (same-day management) =====
            for pos in intraday_positions:
                ohlc = self._replay.get_daily_ohlc(pos.symbol, trade_date)
                if ohlc is None:
                    continue
                O, H, L, C = ohlc

                # Update MFE/MAE
                pos.max_favorable = max(pos.max_favorable, H)
                pos.max_adverse = min(pos.max_adverse, L) if pos.max_adverse > 0 else L
                pos.close_pct = _close_in_range_pct(H, L, C)

                exit_price = None
                exit_reason = ""

                # Stop hit
                if L <= pos.stop:
                    exit_price = C if settings.pb_use_close_stop else pos.stop
                    exit_reason = "STOP_HIT"

                if v2 and exit_price is None:
                    # V2 EMA reversion exit (same-day: close >= prior day EMA10)
                    if settings.pb_v2_ema_reversion_exit:
                        ind = self._indicators.get(pos.symbol)
                        if ind is not None and "ema10" in ind:
                            didx = self._replay._daily_didx.get(pos.symbol)
                            if didx is not None:
                                prev_iloc = _iloc_for_date(didx, prev_date)
                                if prev_iloc >= 0 and not np.isnan(ind["ema10"][prev_iloc]):
                                    if C >= ind["ema10"][prev_iloc] and pos.unrealized_r(C) > settings.pb_v2_ema_reversion_min_r:
                                        exit_price = C
                                        exit_reason = "EMA_REVERSION"

                    # V2 route-specific RSI exit
                    if exit_price is None:
                        ind = self._indicators.get(pos.symbol)
                        if ind is not None:
                            didx = self._replay._daily_didx.get(pos.symbol)
                            if didx is not None:
                                iloc_today = _iloc_for_date(didx, trade_date)
                                if iloc_today >= 0 and not np.isnan(ind["rsi"][iloc_today]):
                                    rsi_thresh = _v2_rsi_exit_threshold(pos.route_family, settings)
                                    if ind["rsi"][iloc_today] > rsi_thresh:
                                        exit_price = C
                                        exit_reason = "RSI_EXIT"

                    # V2 carry decision (inverted: default carry, unconditional when V2)
                    if exit_price is None:
                        last_n = self._replay.get_flow_proxy_last_n(pos.symbol, prev_date, 2)
                        flatten, flatten_reason = _should_flatten_v2(
                            pos, C, pos.close_pct, regime_tier, list(last_n) if last_n is not None else None, settings,
                        )
                        if not flatten:
                            pos.carry_eligible = True
                            pos.hold_days = 1
                            pos.highest_carry_close = C
                            carry_positions[pos.symbol] = pos
                            continue  # Default: carry overnight
                        exit_price = C
                        exit_reason = flatten_reason

                    # V2 fallback: EOD flatten (shouldn't reach here often)
                    if exit_price is None:
                        exit_price = C
                        exit_reason = "EOD_FLATTEN"
                else:
                    # Legacy exit logic
                    # RSI exit: use today's RSI
                    if exit_price is None:
                        ind = self._indicators.get(pos.symbol)
                        if ind is not None:
                            didx = self._replay._daily_didx.get(pos.symbol)
                            if didx is not None:
                                iloc_today = _iloc_for_date(didx, trade_date)
                                if iloc_today >= 0 and not np.isnan(ind["rsi"][iloc_today]):
                                    if ind["rsi"][iloc_today] > settings.pb_rsi_exit:
                                        exit_price = C
                                        exit_reason = "RSI_EXIT"

                    # Profit target
                    if exit_price is None and settings.pb_profit_target_r > 0:
                        if pos.unrealized_r(C) > settings.pb_profit_target_r:
                            exit_price = C
                            exit_reason = "PROFIT_TARGET"

                    # Carry check
                    if exit_price is None and settings.pb_carry_enabled:
                        if _can_carry(pos, C, pos.close_pct, settings):
                            pos.carry_eligible = True
                            pos.hold_days = 1
                            pos.highest_carry_close = C
                            carry_positions[pos.symbol] = pos
                            continue  # Don't close -- carry overnight

                    # Default: EOD flatten
                    if exit_price is None:
                        exit_price = C
                        exit_reason = "EOD_FLATTEN"

                # Populate close_r and exit_rsi for diagnostics
                pos.close_r = pos.unrealized_r(C)
                ind = self._indicators.get(pos.symbol)
                if ind is not None:
                    didx = self._replay._daily_didx.get(pos.symbol)
                    if didx is not None:
                        iloc_today = _iloc_for_date(didx, trade_date)
                        if iloc_today >= 0 and not np.isnan(ind["rsi"][iloc_today]):
                            pos.exit_rsi = ind["rsi"][iloc_today]

                # Close position
                slip = exit_price * self._slippage.slip_bps_normal / 10_000
                fill = round(exit_price - slip, 2)
                commission = self._slippage.commission_per_share * pos.quantity
                pnl = (fill - pos.entry_price) * pos.quantity
                r_mult = (pnl - pos.commission_entry - commission) / pos.total_risk if pos.total_risk > 0 else 0.0
                equity += pnl - commission - pos.commission_entry

                trade = TradeRecord(
                    strategy="IARIC_PB",
                    symbol=pos.symbol,
                    direction=BTDirection.LONG,
                    entry_time=pos.entry_time,
                    exit_time=ts,
                    entry_price=pos.entry_price,
                    exit_price=fill,
                    quantity=pos.quantity,
                    pnl=pnl,
                    r_multiple=r_mult,
                    risk_per_share=pos.risk_per_share,
                    commission=pos.commission_entry + commission,
                    slippage=pos.slippage_entry + slip * pos.quantity,
                    entry_type=pos.trigger_type,
                    exit_reason=exit_reason,
                    sector=pos.sector,
                    regime_tier=pos.regime_tier,
                    hold_bars=1,
                    max_favorable=pos.max_favorable,
                    max_adverse=pos.max_adverse,
                    metadata=pos.build_metadata(),
                )
                trades.append(trade)
                self._attach_trade_outcome(pos, trade)

                # ---- parity: notify core of intraday exit ----
                _exit_oid = f"iaric-x-{pos.symbol}-{self._order_counter}"
                self._order_counter += 1
                self._replay_core_step(
                    bar_input={"bar_ts": ts, "flatten_request": IARICFlattenRequest(symbol=pos.symbol, reason=exit_reason)},
                    fills=[
                        IARICFill(
                            oms_order_id=_exit_oid,
                            fill_price=fill,
                            fill_qty=pos.quantity,
                            fill_time=ts,
                            commission=commission,
                            symbol=pos.symbol,
                            order_role="EXIT",
                            exit_type=exit_reason,
                        )
                    ],
                )

                if cfg.verbose:
                    logger.info(
                        "[%s] EXIT %s @ %.2f reason=%s PnL=%.2f R=%.2f",
                        trade_date, pos.symbol, fill, exit_reason, pnl, r_mult,
                    )

            # Update V2 candidate persistence tracking
            if v2:
                self._prev_day_candidates = today_candidate_syms

            equity_history.append(equity)

        # Close any remaining carry positions at last available price
        if carry_positions and trading_dates:
            last_date = trading_dates[-1]
            for sym, pos in list(carry_positions.items()):
                close_price = self._replay.get_daily_close(sym, last_date)
                if close_price is None:
                    continue
                ts = datetime(last_date.year, last_date.month, last_date.day, tzinfo=timezone.utc)
                slip = close_price * self._slippage.slip_bps_normal / 10_000
                fill = round(close_price - slip, 2)
                commission = self._slippage.commission_per_share * pos.quantity
                pnl = (fill - pos.entry_price) * pos.quantity
                r_mult = (pnl - pos.commission_entry - commission) / pos.total_risk if pos.total_risk > 0 else 0.0
                equity += pnl - commission - pos.commission_entry

                trade = TradeRecord(
                    strategy="IARIC_PB",
                    symbol=sym,
                    direction=BTDirection.LONG,
                    entry_time=pos.entry_time,
                    exit_time=ts,
                    entry_price=pos.entry_price,
                    exit_price=fill,
                    quantity=pos.quantity,
                    pnl=pnl,
                    r_multiple=r_mult,
                    risk_per_share=pos.risk_per_share,
                    commission=pos.commission_entry + commission,
                    slippage=pos.slippage_entry + slip * pos.quantity,
                    entry_type=pos.trigger_type,
                    exit_reason="END_OF_BACKTEST",
                    sector=pos.sector,
                    regime_tier=pos.regime_tier,
                    hold_bars=pos.hold_days,
                    max_favorable=pos.max_favorable,
                    max_adverse=pos.max_adverse,
                    metadata=pos.build_metadata(),
                )
                trades.append(trade)
                self._attach_trade_outcome(pos, trade)

                # ---- parity: notify core of end-of-backtest exit ----
                _exit_oid = f"iaric-x-{sym}-{self._order_counter}"
                self._order_counter += 1
                self._replay_core_step(
                    bar_input={"bar_ts": ts, "flatten_request": IARICFlattenRequest(symbol=sym, reason="END_OF_BACKTEST")},
                    fills=[
                        IARICFill(
                            oms_order_id=_exit_oid,
                            fill_price=fill,
                            fill_qty=pos.quantity,
                            fill_time=ts,
                            commission=commission,
                            symbol=sym,
                            order_role="EXIT",
                            exit_type="END_OF_BACKTEST",
                        )
                    ],
                )

            # Update equity curve to reflect EOB closures
            equity_history[-1] = equity

        logger.info(
            "IARIC Tier 3 (Pullback) complete: %d trades, final equity: $%.2f (%.1f%%)",
            len(trades), equity, (equity / cfg.initial_equity - 1) * 100,
        )

        selection_attribution = _build_selection_attribution(candidate_ledger or {}) if candidate_ledger else None

        return IARICPullbackResult(
            trades=trades,
            equity_curve=np.array(equity_history),
            timestamps=np.array([
                np.datetime64(ts.replace(tzinfo=None)) for ts in ts_history
            ]),
            daily_selections=daily_selections,
            candidate_ledger=candidate_ledger,
            funnel_counters=funnel_counters,
            rejection_log=rejection_log,
            shadow_outcomes=shadow_outcomes,
            selection_attribution=selection_attribution,
            fsm_log=None,
            decision_stream=normalize_decision_stream(self._decision_events),
            trade_outcomes=trade_outcomes_from_records(trades),
        )


class IARICPullbackEngine:
    """Stable public entrypoint for pullback backtests.

    Defaults to the legacy daily engine and dispatches to the hybrid 5m engine
    only when explicitly requested via ``pb_execution_mode``.
    """

    def __init__(
        self,
        config: IARICBacktestConfig,
        replay: ResearchReplayEngine,
        settings: StrategySettings | None = None,
        *,
        collect_diagnostics: bool = False,
    ):
        self._config = config
        self._replay = replay
        self._settings = settings or StrategySettings()
        self._collect_diagnostics = collect_diagnostics

        if config.param_overrides:
            self._settings = replace(self._settings, **config.param_overrides)

    def run(self) -> IARICPullbackResult:
        mode = str(getattr(self._settings, "pb_execution_mode", "daily") or "daily").lower()
        if mode == "daily":
            engine = IARICPullbackDailyEngine(
                self._config,
                self._replay,
                self._settings,
                collect_diagnostics=self._collect_diagnostics,
            )
            return engine.run()
        if mode == "intraday_hybrid":
            from backtests.stock.engine.iaric_pullback_intraday_hybrid_engine import (
                IARICPullbackIntradayHybridEngine,
            )

            engine = IARICPullbackIntradayHybridEngine(
                self._config,
                self._replay,
                self._settings,
                collect_diagnostics=self._collect_diagnostics,
            )
            return engine.run()
        raise ValueError(f"Unsupported pb_execution_mode: {mode}")
