"""Hybrid 5m execution layer for IARIC pullback candidates."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from math import floor
from typing import Any

import numpy as np

from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.legacy_result_outputs import trade_outcomes_from_records
from strategies.stock.iaric.core import logic as iaric_core_logic
from strategies.stock.iaric.core.state import (
    IARICEntryRequest,
    IARICFill,
    IARICFlattenRequest,
    IARICPartialExitRequest,
)
from backtests.stock.engine.iaric_pullback_engine import (
    IARICPullbackDailyEngine,
    IARICPullbackResult,
    _build_selection_attribution,
    _close_in_range_pct,
    _daily_signal_bundle,
    _daily_signal_bundle_v2,
    _ensure_candidate_ledger,
    _evaluate_v2_triggers,
    _iloc_for_date,
    _passes_daily_signal_floor,
    _rank_gate_reason,
    _rank_percent,
    _risk_budget_mult,
    _should_flatten_v2,
    _v2_rsi_exit_threshold,
    _v2_score_sizing_mult,
)
from backtests.stock.models import Direction as BTDirection, TradeRecord

from strategies.stock.iaric.config import ET
from strategies.stock.iaric.models import Bar, MarketSnapshot, WatchlistArtifact, WatchlistItem

logger = logging.getLogger(__name__)

_MKT_OPEN = time(9, 30)
_MKT_CLOSE = time(16, 0)


@dataclass
class _PBHybridState:
    symbol: str
    item: WatchlistItem
    record: dict[str, Any] | None
    trigger_type: str
    entry_rsi: float
    entry_gap_pct: float
    entry_sma_dist_pct: float
    entry_cdd: int
    entry_rank: int
    entry_rank_pct: float
    n_candidates: int
    prev_iloc: int
    sector: str
    daily_atr: float
    daily_signal_score: float = 0.0
    daily_signal_rank_pct: float = 100.0
    daily_signal_components: dict[str, float] = field(default_factory=dict)
    rescue_flow_candidate: bool = False
    stage: str = "WATCHING"
    intraday_setup_type: str = ""
    route_family: str = ""
    setup_low: float = 0.0
    reclaim_level: float = 0.0
    stop_level: float = 0.0
    flush_bar_idx: int = 0
    ready_bar_idx: int = 0
    acceptance_count: int = 0
    required_acceptance: int = 0
    intraday_score: float = 0.0
    target_entry_price: float = 0.0
    improvement_expires: int = 0
    invalid_reason: str = ""
    invalid_reset_bar: int = 0
    stopped_out_today: bool = False
    reentry_count: int = 0
    priority_skip_count: int = 0
    score_components: dict[str, float] = field(default_factory=dict)
    ready_cpr: float = 0.0
    ready_volume_ratio: float = 0.0
    ready_timestamp: datetime | None = None
    accepted_bar_idx: int = -1
    accepted_timestamp: datetime | None = None
    accepted_entry_price: float = 0.0
    accepted_entry_trigger: str = ""
    accepted_route_family: str = ""
    accepted_score: float = 0.0
    accepted_session_atr: float = 0.0
    accepted_score_components: dict[str, float] = field(default_factory=dict)
    # V2 fields
    trigger_types: list[str] = field(default_factory=list)
    trigger_tier: str = ""
    trend_tier: str = "STRONG"

    def reset_for_watch(self) -> None:
        self.stage = "WATCHING"
        self.intraday_setup_type = ""
        self.route_family = ""
        self.setup_low = 0.0
        self.reclaim_level = 0.0
        self.stop_level = 0.0
        self.flush_bar_idx = 0
        self.ready_bar_idx = 0
        self.acceptance_count = 0
        self.required_acceptance = 0
        self.intraday_score = 0.0
        self.target_entry_price = 0.0
        self.improvement_expires = 0
        self.invalid_reason = ""
        self.invalid_reset_bar = 0
        self.score_components = {}
        self.ready_cpr = 0.0
        self.ready_volume_ratio = 0.0
        self.ready_timestamp = None
        self.accepted_bar_idx = -1
        self.accepted_timestamp = None
        self.accepted_entry_price = 0.0
        self.accepted_entry_trigger = ""
        self.accepted_route_family = ""
        self.accepted_score = 0.0
        self.accepted_session_atr = 0.0
        self.accepted_score_components = {}


@dataclass
class _PBHybridPosition:
    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: int
    risk_per_share: float
    sector: str
    regime_tier: str
    stop: float
    current_stop: float
    trigger_type: str
    entry_rsi: float
    entry_gap_pct: float
    entry_sma_dist_pct: float
    entry_cdd: int
    entry_rank: int
    entry_rank_pct: float
    n_candidates: int
    daily_signal_score: float
    daily_signal_rank_pct: float
    signal_family: str
    intraday_setup_type: str
    entry_trigger: str
    route_family: str
    carry_profile: str
    selection_reason: str
    intraday_score: float
    reclaim_bars: int
    rescue_flow_candidate: bool
    reentry_count: int
    entry_atr: float
    item: WatchlistItem
    ready_timestamp: datetime | None = None
    accepted_timestamp: datetime | None = None
    accepted_bar_idx: int = -1
    accepted_entry_price: float = 0.0
    accepted_intraday_score: float = 0.0
    accepted_entry_trigger: str = ""
    acceptance_count: int = 0
    required_acceptance: int = 0
    micropressure_signal: str = "NEUTRAL"
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    highest_close: float = 0.0
    hold_bars: int = 0
    hold_days: int = 1
    carry_days: int = 0
    close_r: float = 0.0
    close_pct: float = 0.0
    exit_rsi: float = 0.0
    carry_score: float = 0.0
    partial_taken: bool = False
    trail_active: bool = False
    partial_qty_exited: int = 0
    realized_partial_pnl: float = 0.0
    realized_partial_commission: float = 0.0
    realized_partial_slippage: float = 0.0
    commission_entry: float = 0.0
    slippage_entry: float = 0.0
    entry_bar_idx: int = 0
    ledger_ref: dict[str, Any] | None = None
    score_components: dict[str, float] = field(default_factory=dict)
    breakeven_activated: bool = False
    carry_binary_ok: bool = False
    carry_score_ok: bool = False
    carry_decision_path: str = ""
    bars_to_mfe: int = 0
    # V2 fields
    trigger_types: list[str] = field(default_factory=list)
    trigger_tier: str = ""
    trend_tier: str = "STRONG"
    mfe_stage: int = 0
    v2_partial_taken: bool = False

    def unrealized_r(self, price: float) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        return (price - self.entry_price) / self.risk_per_share

    def mfe_r(self) -> float:
        if self.risk_per_share <= 0 or self.max_favorable <= 0:
            return 0.0
        return float((self.max_favorable - self.entry_price) / self.risk_per_share)

    def build_metadata(self) -> dict[str, Any]:
        metadata = {
            "trigger_type": self.trigger_type,
            "entry_rsi": round(self.entry_rsi, 2),
            "hold_days": self.hold_days,
            "setup_type": "PULLBACK_BUY",
            "setup_tag": "PULLBACK_BUY",
            "mfe_r": round(self.mfe_r(), 4),
            "mae_r": round(
                (self.entry_price - self.max_adverse) / max(self.risk_per_share, 0.01),
                4,
            ) if self.max_adverse > 0 and self.max_adverse < self.entry_price else 0.0,
            "entry_atr": round(self.entry_atr, 4),
            "stop_distance_pct": round((self.entry_price - self.stop) / self.entry_price * 100, 3) if self.entry_price > 0 else 0.0,
            "entry_gap_pct": round(self.entry_gap_pct, 3),
            "entry_sma_dist_pct": round(self.entry_sma_dist_pct, 3),
            "entry_cdd": self.entry_cdd,
            "entry_rank": self.entry_rank,
            "entry_rank_pct": round(self.entry_rank_pct, 2),
            "n_candidates": self.n_candidates,
            "daily_signal_score": round(self.daily_signal_score, 2),
            "daily_signal_rank_pct": round(self.daily_signal_rank_pct, 2),
            "signal_family": self.signal_family,
            "close_r": round(self.close_r, 4),
            "close_pct": round(self.close_pct, 4),
            "exit_rsi": round(self.exit_rsi, 2),
            "intraday_setup_type": self.intraday_setup_type,
            "entry_trigger": self.entry_trigger,
            "route_family": self.route_family,
            "carry_profile": self.carry_profile,
            "selection_reason": self.selection_reason,
            "intraday_score": round(self.intraday_score, 2),
            "ready_timestamp": self.ready_timestamp.isoformat() if self.ready_timestamp is not None else "",
            "accepted_timestamp": self.accepted_timestamp.isoformat() if self.accepted_timestamp is not None else "",
            "accepted_bar_index": self.accepted_bar_idx,
            "accepted_entry_price": round(self.accepted_entry_price, 2),
            "accepted_intraday_score": round(self.accepted_intraday_score, 2),
            "accepted_entry_trigger": self.accepted_entry_trigger,
            "reclaim_bars": self.reclaim_bars,
            "acceptance_count": self.acceptance_count,
            "required_acceptance_count": self.required_acceptance,
            "micropressure_signal": self.micropressure_signal,
            "sponsorship_state": self.item.sponsorship_state,
            "rescue_flow_candidate": self.rescue_flow_candidate,
            "partial_taken": self.partial_taken,
            "trail_active": self.trail_active,
            "carry_score": round(self.carry_score, 2),
            "reentry_count": self.reentry_count,
            "entry_bar_index": self.entry_bar_idx,
            "entry_route_family": self.route_family,
            "breakeven_activated": self.breakeven_activated,
            "carry_binary_ok": self.carry_binary_ok,
            "carry_score_ok": self.carry_score_ok,
            "carry_decision_path": self.carry_decision_path,
            "trigger_types": self.trigger_types,
            "trigger_tier": self.trigger_tier,
            "trend_tier": self.trend_tier,
            "mfe_stage": self.mfe_stage,
        }
        for name, value in self.score_components.items():
            metadata[f"entry_score_component_{name}"] = round(float(value), 4)
        return metadata


class IARICPullbackIntradayHybridEngine(IARICPullbackDailyEngine):
    """Hybrid intraday execution for pullback candidates."""

    def _log_fsm(
        self,
        fsm_log: list[dict[str, Any]],
        symbol: str,
        trade_date: date,
        timestamp: datetime,
        from_state: str,
        to_state: str,
        reason: str,
        *,
        score: float | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "symbol": symbol,
            "date": trade_date,
            "timestamp": timestamp,
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason,
        }
        if score is not None:
            row["score"] = round(float(score), 2)
        fsm_log.append(row)

    def _mark_stage(self, record: dict[str, Any] | None, stage: str) -> None:
        if record is None:
            return
        record["intraday_last_stage"] = stage
        record[f"stage_{stage.lower()}"] = True
        path = record.setdefault("intraday_stage_path", [])
        if not path or path[-1] != stage:
            path.append(stage)

    def _attach_hybrid_trade_outcome(self, position: _PBHybridPosition, trade: TradeRecord) -> None:
        record = position.ledger_ref
        if record is None:
            return
        existing_r = record.get("actual_r")
        record["actual_r"] = float(trade.r_multiple) if existing_r is None else float(existing_r) + float(trade.r_multiple)
        record["actual_exit_reason"] = trade.exit_reason or "UNKNOWN"
        record["actual_hold_days"] = int(record.get("actual_hold_days", 0) or 0) + int(max(position.hold_days, 1))
        record["actual_mfe_r"] = max(float(record.get("actual_mfe_r", 0.0) or 0.0), float(trade.metadata.get("mfe_r", 0.0)))
        record["actual_mae_r"] = max(float(record.get("actual_mae_r", 0.0) or 0.0), float(trade.metadata.get("mae_r", 0.0)))
        record["close_r"] = float(trade.metadata.get("close_r", 0.0))
        record["close_pct"] = float(trade.metadata.get("close_pct", 0.0))
        record["exit_rsi"] = float(trade.metadata.get("exit_rsi", 0.0))
        record["partial_taken"] = bool(record.get("partial_taken")) or bool(trade.metadata.get("partial_taken"))
        record["trail_active"] = bool(record.get("trail_active")) or bool(trade.metadata.get("trail_active"))
        record["breakeven_activated"] = bool(record.get("breakeven_activated")) or bool(trade.metadata.get("breakeven_activated"))
        record["carry_score"] = max(float(record.get("carry_score", 0.0) or 0.0), float(trade.metadata.get("carry_score", 0.0)))
        record["reentry_count"] = max(int(record.get("reentry_count", 0) or 0), int(trade.metadata.get("reentry_count", 0) or 0))
        record["carry_decision_path"] = str(trade.metadata.get("carry_decision_path") or record.get("carry_decision_path") or "")
        record["entry_route_family"] = str(trade.metadata.get("entry_route_family") or record.get("entry_route_family") or "")
        record["actual_trade_count"] = int(record.get("actual_trade_count", 0) or 0) + 1

    def _volume_ratio(self, bar: Bar, item: WatchlistItem) -> float:
        return iaric_core_logic.compute_volume_ratio(bar, item)

    def _session_atr(self, item: WatchlistItem, bars: list[Bar]) -> float:
        ref_price = bars[0].open if bars else max(item.avwap_ref, 1.0)
        if item.intraday_atr_seed > 0:
            return max(item.intraday_atr_seed * ref_price, ref_price * 0.0025)
        if item.daily_atr_estimate > 0:
            return max(item.daily_atr_estimate * 0.25, ref_price * 0.0025)
        return ref_price * 0.01

    def _micropressure_label(
        self,
        bars: list[Bar],
        bar_idx: int,
        reclaim_level: float,
        item: WatchlistItem,
        *,
        lookback_bars: int = 3,
    ) -> str:
        return iaric_core_logic.micropressure_label(
            bars,
            bar_idx,
            reclaim_level,
            item,
            lookback_bars=lookback_bars,
        )

    def _thirty_min_context_bonus(self, market: MarketSnapshot, *, weight: float) -> float:
        return iaric_core_logic.thirty_min_context_bonus(market, weight=weight)

    def _apply_score_components(
        self,
        payload: dict[str, Any] | None,
        components: dict[str, float],
        *,
        prefix: str,
    ) -> None:
        if payload is None:
            return
        for name, value in components.items():
            payload[f"{prefix}{name}"] = round(float(value), 4)

    def _route_prefix(self, route_family: str) -> str:
        return iaric_core_logic.route_prefix(route_family)

    def _route_enabled(self, route_family: str) -> bool:
        return iaric_core_logic.route_enabled(self._settings, route_family)

    def _route_setting(self, route_family: str, suffix: str, fallback_suffix: str | None = None):
        return iaric_core_logic.route_setting(self._settings, route_family, suffix, fallback_suffix)

    def _route_carry_profile(self, route_family: str) -> str:
        return iaric_core_logic.route_carry_profile(route_family)

    def _route_min_daily_signal_score(self, route_family: str) -> float:
        return iaric_core_logic.route_min_daily_signal_score(self._settings, route_family)

    def _open_scored_eligible(self, payload: dict[str, Any] | None) -> bool:
        return iaric_core_logic.open_scored_eligible(self._settings, payload)

    def _entry_score_bundle(
        self,
        state: _PBHybridState,
        bar: Bar,
        market: MarketSnapshot,
        bars: list[Bar],
        bar_idx: int,
    ) -> dict[str, float]:
        return iaric_core_logic.compute_route_entry_score_bundle(
            self._settings,
            state,
            state.item,
            bar,
            market,
            bar_idx,
            bars=bars,
        )

    def _compute_entry_score(
        self,
        state: _PBHybridState,
        bar: Bar,
        market: MarketSnapshot,
        bars: list[Bar],
        bar_idx: int,
    ) -> float:
        bundle = self._entry_score_bundle(state, bar, market, bars, bar_idx)
        return float(bundle["score"])

    def _compute_carry_score(
        self,
        position: _PBHybridPosition,
        bar: Bar,
        market: MarketSnapshot,
        bars: list[Bar],
        bar_idx: int,
    ) -> float:
        route_family = position.route_family
        score = 0.0
        cur_r = position.unrealized_r(bar.close)
        score += min(max(cur_r / 2.0, 0.0), 1.0) * 25.0
        score += min(max(position.daily_signal_score / 100.0, 0.0), 1.0) * 20.0

        close_pct = _close_in_range_pct(
            market.session_high if market.session_high is not None else bar.high,
            market.session_low if market.session_low is not None else bar.low,
            bar.close,
        )
        score += close_pct * 18.0
        score += min(max(position.mfe_r() / 2.0, 0.0), 1.0) * 15.0

        vwap = market.session_vwap or bar.close
        if position.entry_atr > 0:
            vwap_score = min(max(((bar.close - vwap) / position.entry_atr + 0.25) / 0.75, 0.0), 1.0)
        else:
            vwap_score = 0.0
        score += vwap_score * 10.0

        recent_label = self._micropressure_label(
            bars,
            bar_idx,
            position.entry_price,
            position.item,
            lookback_bars=6,
        )
        if recent_label == "ACCUMULATE":
            score += 6.0
        elif recent_label == "DISTRIBUTE":
            score -= 10.0

        flow_score = 0.0
        if position.item.sponsorship_state == "STRONG":
            flow_score = 1.0
        elif position.item.sponsorship_state in {"ACCUMULATE", "NEUTRAL"}:
            flow_score = 0.65
        elif position.item.sponsorship_state == "STALE":
            flow_score = 0.4
        score += flow_score * 8.0
        score += min(max(self._thirty_min_context_bonus(market, weight=4.0), -4.0), 2.0)
        if position.rescue_flow_candidate:
            score -= 10.0
        if route_family == "OPENING_RECLAIM":
            score -= 5.0
        elif route_family == "OPEN_SCORED_ENTRY":
            score += 3.0
        return float(max(score, 0.0))

    def _binary_carry_ok(
        self,
        position: _PBHybridPosition,
        bar: Bar,
        market: MarketSnapshot,
        trade_date: date,
    ) -> bool:
        settings = self._settings
        route_family = position.route_family
        if not settings.pb_carry_enabled:
            return False
        if position.rescue_flow_candidate:
            return False
        if position.item.earnings_risk_flag or position.item.blacklist_flag:
            return False
        if position.item.sponsorship_state not in {"STRONG", "ACCUMULATE"}:
            return False
        cur_r = position.unrealized_r(bar.close)
        close_pct = _close_in_range_pct(
            market.session_high if market.session_high is not None else bar.high,
            market.session_low if market.session_low is not None else bar.low,
            bar.close,
        )
        if cur_r <= float(self._route_setting(route_family, "carry_min_r", "pb_carry_min_r")):
            return False
        if close_pct < float(self._route_setting(route_family, "carry_close_pct_min", "pb_carry_close_pct_min")):
            return False
        if position.mfe_r() < float(self._route_setting(route_family, "carry_mfe_gate_r", "pb_carry_mfe_gate_r")):
            return False
        if position.daily_signal_score < float(self._route_setting(route_family, "carry_min_daily_signal_score", "pb_carry_min_daily_signal_score")):
            return False
        last_n = self._replay.get_flow_proxy_last_n(
            position.symbol,
            trade_date,
            max(1, int(self._route_setting(route_family, "flow_reversal_lookback", "pb_flow_reversal_lookback"))),
        )
        if last_n is not None and all(v < 0 for v in last_n):
            return False
        return True

    def _should_exit_for_vwap_fail(
        self,
        position: _PBHybridPosition,
        bars: list[Bar],
        bar_idx: int,
        market: MarketSnapshot,
    ) -> bool:
        lookback_setting = int(self._route_setting(position.route_family, "vwap_fail_lookback_bars", "pb_vwap_fail_lookback_bars"))
        cpr_max = float(self._route_setting(position.route_family, "vwap_fail_cpr_max", "pb_vwap_fail_cpr_max"))
        if lookback_setting <= 1 or cpr_max < 0:
            return False
        lookback = max(2, lookback_setting)
        if bar_idx + 1 < lookback:
            return False
        recent = bars[bar_idx + 1 - lookback:bar_idx + 1]
        if len(recent) < lookback:
            return False
        vwap = market.session_vwap
        if vwap is None or recent[-1].close >= vwap:
            return False
        if recent[-1].cpr > cpr_max:
            return False
        highs = [bar.high for bar in recent]
        return all(highs[idx] <= highs[idx - 1] + 1e-9 for idx in range(1, len(highs)))

    def _close_position(
        self,
        position: _PBHybridPosition,
        exit_price: float,
        ts: datetime,
        reason: str,
    ) -> tuple[TradeRecord, float]:
        slip = exit_price * self._slippage.slip_bps_normal / 10_000
        fill = round(exit_price - slip, 2)
        commission = self._slippage.commission_per_share * position.quantity
        runner_pnl = (fill - position.entry_price) * position.quantity
        total_pnl = runner_pnl + position.realized_partial_pnl
        total_commission = position.commission_entry + commission + position.realized_partial_commission
        total_qty = position.quantity + position.partial_qty_exited
        total_risk = position.risk_per_share * max(total_qty, 1)
        r_mult = (total_pnl - total_commission) / total_risk if total_risk > 0 else 0.0
        metadata = position.build_metadata()
        bars_to_exit = max(position.hold_bars, 1)
        if bars_to_exit <= 1:
            bars_to_exit = max(int(round((ts - position.entry_time).total_seconds() / 300.0)), 1)
        metadata["bars_to_exit"] = int(bars_to_exit)
        metadata["bars_to_mfe"] = int(position.bars_to_mfe or bars_to_exit)
        metadata["mfe_before_exit_r"] = round(position.mfe_r(), 4)
        metadata["mfe_before_negative_exit_r"] = round(position.mfe_r(), 4) if r_mult < 0 else 0.0
        trade = TradeRecord(
            strategy="IARIC_PB",
            symbol=position.symbol,
            direction=BTDirection.LONG,
            entry_time=position.entry_time,
            exit_time=ts,
            entry_price=position.entry_price,
            exit_price=fill,
            quantity=total_qty,
            pnl=total_pnl,
            r_multiple=r_mult,
            risk_per_share=position.risk_per_share,
            commission=total_commission,
            slippage=position.slippage_entry + slip * position.quantity + position.realized_partial_slippage,
            entry_type=position.trigger_type,
            exit_reason=reason,
            sector=position.sector,
            regime_tier=position.regime_tier,
            hold_bars=max(position.hold_days, 1),
            max_favorable=position.max_favorable,
            max_adverse=position.max_adverse,
            metadata=metadata,
        )

        # ---- parity: notify core of exit ----
        _exit_oid = f"iaric-x-{position.symbol}-{self._order_counter}"
        self._order_counter += 1
        self._replay_core_step(
            bar_input={"bar_ts": ts, "flatten_request": IARICFlattenRequest(symbol=position.symbol, reason=reason)},
            fills=[
                IARICFill(
                    oms_order_id=_exit_oid,
                    fill_price=fill,
                    fill_qty=position.quantity,
                    fill_time=ts,
                    commission=commission,
                    symbol=position.symbol,
                    order_role="EXIT",
                    exit_type=reason,
                )
            ],
        )

        return trade, runner_pnl - commission - position.commission_entry

    def _process_overnight_carries(
        self,
        carry_positions: dict[str, _PBHybridPosition],
        trade_date: date,
        equity: float,
        trades: list[TradeRecord],
    ) -> float:
        settings = self._settings
        closed: list[str] = []
        prev_date = self._replay.get_prev_trading_date(trade_date)
        for sym, pos in list(carry_positions.items()):
            route_family = pos.route_family or "OPEN_SCORED_ENTRY"
            ohlc = self._replay.get_daily_ohlc(sym, trade_date)
            if ohlc is None:
                trade, eq_delta = self._close_position(
                    pos,
                    pos.entry_price,
                    self._market_open_timestamp(trade_date),
                    "DATA_GAP",
                )
                trades.append(trade)
                equity += eq_delta
                self._attach_hybrid_trade_outcome(pos, trade)
                closed.append(sym)
                continue

            O, H, L, C = ohlc
            pos.hold_days += 1
            pos.carry_days += 1
            if H > pos.max_favorable + 1e-9:
                pos.max_favorable = H
                pos.bars_to_mfe = max(pos.bars_to_mfe, pos.hold_bars + pos.carry_days * 78)
            pos.max_adverse = min(pos.max_adverse, L) if pos.max_adverse > 0 else L
            pos.highest_close = max(pos.highest_close, C)
            pos.close_pct = _close_in_range_pct(H, L, C)
            pos.close_r = pos.unrealized_r(C)

            exit_price = None
            exit_reason = ""
            if O <= pos.current_stop:
                exit_price = O
                exit_reason = "GAP_STOP"
            elif prev_date is not None:
                last_n = self._replay.get_flow_proxy_last_n(
                    sym,
                    prev_date,
                    max(1, int(self._route_setting(route_family, "flow_reversal_lookback", "pb_flow_reversal_lookback"))),
                )
                if last_n is not None and all(v < 0 for v in last_n):
                    exit_price = O
                    exit_reason = "FLOW_REVERSAL"

            if exit_price is None and L <= pos.current_stop:
                exit_price = C if settings.pb_use_close_stop else pos.current_stop
                exit_reason = "STOP_HIT"

            # V2: EMA reversion exit for carried positions
            if exit_price is None and settings.pb_v2_enabled and settings.pb_v2_ema_reversion_exit:
                ind = self._indicators.get(sym)
                didx = self._date_iloc.get(sym)
                if ind is not None and didx is not None:
                    p_iloc = didx.get(prev_date, -1) if prev_date is not None else -1
                    ema10_arr = ind.get("ema10") if ind is not None else None
                    if ema10_arr is not None and p_iloc >= 0 and not np.isnan(ema10_arr[p_iloc]):
                        ema10_val = float(ema10_arr[p_iloc])
                        if C >= ema10_val and pos.unrealized_r(C) > settings.pb_v2_ema_reversion_min_r:
                            exit_price = C
                            exit_reason = "EMA_REVERSION"

            if exit_price is None:
                ind = self._indicators.get(sym)
                didx = self._replay._daily_didx.get(sym)
                if ind is not None and didx is not None:
                    iloc = _iloc_for_date(didx, trade_date)
                    if iloc >= 0 and not np.isnan(ind["rsi"][iloc]):
                        pos.exit_rsi = float(ind["rsi"][iloc])
                        # V2: route-specific RSI exit threshold
                        rsi_thresh = (
                            _v2_rsi_exit_threshold(route_family, settings)
                            if settings.pb_v2_enabled
                            else float(self._route_setting(route_family, "rsi_exit", "pb_rsi_exit"))
                        )
                        if ind["rsi"][iloc] > rsi_thresh:
                            exit_price = C
                            exit_reason = "RSI_EXIT"

            if exit_price is None and pos.hold_days >= int(self._route_setting(route_family, "max_hold_days", "pb_max_hold_days")):
                exit_price = C
                exit_reason = "TIME_STOP"

            if exit_price is None and settings.pb_profit_target_r > 0 and pos.unrealized_r(C) >= settings.pb_profit_target_r:
                exit_price = C
                exit_reason = "PROFIT_TARGET"

            if exit_price is not None:
                pos.carry_decision_path = exit_reason.lower()
                trade, eq_delta = self._close_position(
                    pos,
                    float(exit_price),
                    self._market_close_timestamp(trade_date),
                    exit_reason,
                )
                trades.append(trade)
                equity += eq_delta
                self._attach_hybrid_trade_outcome(pos, trade)
                closed.append(sym)

        for sym in closed:
            carry_positions.pop(sym, None)
        return equity

    def _build_watchlists(
        self,
        trade_date: date,
        prev_date: date,
        artifact: WatchlistArtifact,
        regime_tier: str,
        carry_positions: dict[str, _PBHybridPosition],
        candidate_ledger: dict[date, list[dict[str, Any]]] | None,
        funnel_counters: dict[str, int] | None,
        rejection_log: list[dict[str, Any]] | None,
        shadow_outcomes: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        settings = self._settings
        candidates: list[dict[str, Any]] = []
        rescue_candidates: list[dict[str, Any]] = []
        item_lookup = getattr(artifact, "by_symbol", {})
        flow_policy = str(getattr(settings, "pb_flow_policy", "soft_penalty_rescue") or "soft_penalty_rescue").lower()
        use_cdd = settings.pb_cdd_min > 0
        use_mazone = settings.pb_ma_zone_entry
        sector_raw_counts: dict[str, int] = {}

        for sym, sym_sector_raw, _ in self._trade_universe:
            if funnel_counters is not None:
                funnel_counters["universe_seen"] = funnel_counters.get("universe_seen", 0) + 1
            if sym in carry_positions:
                continue

            ind = self._indicators.get(sym)
            di = self._date_iloc.get(sym)
            if ind is None or di is None:
                continue
            iloc = di.get(prev_date, -1)
            if iloc < 0:
                continue

            closes = self._replay._daily_arrs[sym]["close"]
            sma_trend_val = ind["sma_trend"][iloc]
            v2 = settings.pb_v2_enabled

            # --- V2 two-tier trend filter vs legacy hard SMA(50) gate ---
            trend_tier = "STRONG"
            if v2:
                if sym_sector_raw == "benchmark":
                    continue
                above_sma50 = not np.isnan(sma_trend_val) and closes[iloc] > sma_trend_val
                slope_ok = bool(ind["sma_slope"][iloc])
                sma200_val = ind.get("sma200")
                sma200_v = sma200_val[iloc] if sma200_val is not None else np.nan
                above_sma200 = not np.isnan(sma200_v) and closes[iloc] > sma200_v
                sma50_above_200 = not np.isnan(sma200_v) and not np.isnan(sma_trend_val) and sma_trend_val > sma200_v
                if above_sma50 and slope_ok:
                    trend_tier = "STRONG"
                elif settings.pb_v2_allow_secular and above_sma200 and sma50_above_200:
                    trend_tier = "SECULAR"
                else:
                    continue
            else:
                if np.isnan(sma_trend_val) or closes[iloc] <= sma_trend_val or not ind["sma_slope"][iloc]:
                    continue

            prev_close_val = closes[iloc]
            if prev_close_val <= 0:
                continue
            ohlc = self._replay.get_daily_ohlc(sym, trade_date)
            if ohlc is None:
                continue
            O, _H, _L, _C = ohlc
            gap_pct = (O - prev_close_val) / prev_close_val * 100
            sma_dist_pct = (prev_close_val - sma_trend_val) / sma_trend_val * 100 if sma_trend_val > 0 and not np.isnan(sma_trend_val) else 0.0
            cdd_val = int(ind["cdd"][iloc])

            # --- V2 widened range filters ---
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

            # --- V2 multi-trigger union vs legacy single trigger ---
            triggered = False
            trigger_type = ""
            v2_trigger_types: list[str] = []
            v2_trigger_tier = ""
            rsi_val = ind["rsi"][iloc]
            if v2:
                rs_val = 1.0
                rs_arr = ind.get("rs_ratio")
                if rs_arr is not None and not np.isnan(rs_arr[iloc]):
                    rs_val = float(rs_arr[iloc])
                v2_triggers = _evaluate_v2_triggers(
                    ind=ind, iloc=iloc, closes=closes, prev_close=prev_close_val,
                    gap_pct=gap_pct, trend_tier=trend_tier, rs_val=rs_val, settings=settings,
                )
                if v2_triggers:
                    triggered = True
                    v2_trigger_types = [t[0] for t in v2_triggers]
                    tiers = [t[1] for t in v2_triggers]
                    v2_trigger_tier = "HIGH" if "HIGH" in tiers else "MEDIUM"
                    trigger_type = v2_trigger_types[0]
            else:
                if not np.isnan(rsi_val) and rsi_val < settings.pb_rsi_entry:
                    triggered = True
                    trigger_type = "RSI"
                if not triggered and use_cdd and int(ind["cdd"][iloc]) >= settings.pb_cdd_min:
                    triggered = True
                    trigger_type = "CDD"
                if not triggered and use_mazone:
                    sma20_val = ind["sma20"][iloc]
                    if not np.isnan(sma20_val) and closes[iloc] < sma20_val and closes[iloc] > sma_trend_val:
                        triggered = True
                        trigger_type = "MA_ZONE"
            if not triggered:
                continue

            if funnel_counters is not None:
                funnel_counters["triggered"] = funnel_counters.get("triggered", 0) + 1

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
                    "signal_family": str(getattr(settings, "pb_daily_signal_family", "balanced_v1")),
                    "daily_signal_min_score_threshold": float(getattr(settings, "pb_daily_signal_min_score", 0.0)),
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
                    "actual_r": None,
                    "shadow_r": None,
                    "intraday_stage_path": ["WATCHING"],
                    "candidate_kind": "core",
                    "live_intraday_candidate": False,
                    "intraday_data_available": False,
                    "rescue_flow_candidate": False,
                    "partial_taken": False,
                    "trail_active": False,
                    "breakeven_activated": False,
                    "carry_score": 0.0,
                    "carry_decision_path": "",
                    "reentry_count": 0,
                    "actual_trade_count": 0,
                    "blocked_by_capacity_reason": "",
                    "entry_window_feasible": False,
                    "entry_window_feasible_bar_index": None,
                    "max_feasible_intraday_score": 0.0,
                    "ready_timestamp": None,
                    "ready_bar_index": None,
                    "ready_cpr": 0.0,
                    "ready_volume_ratio": 0.0,
                    "refinement_route": "",
                    "entry_route_family": "",
                "entry_score_threshold": float(settings.pb_entry_score_min),
                "delayed_confirm_score_threshold": float(settings.pb_delayed_confirm_score_min),
                "ready_min_cpr_threshold": float(settings.pb_ready_min_cpr),
                "ready_min_volume_ratio_threshold": float(settings.pb_ready_min_volume_ratio),
                "delayed_confirm_after_bar_threshold": int(settings.pb_delayed_confirm_after_bar),
                "delayed_confirm_min_daily_signal_threshold": float(getattr(settings, "pb_delayed_confirm_min_daily_signal_score", 0.0)),
                "opening_reclaim_min_daily_signal_threshold": float(getattr(settings, "pb_opening_reclaim_min_daily_signal_score", 0.0)),
                "open_scored_min_score_threshold": float(getattr(settings, "pb_open_scored_min_score", 0.0)),
                "open_scored_rank_pct_max_threshold": float(getattr(settings, "pb_open_scored_rank_pct_max", 100.0)),
                "daily_rescue_min_score_threshold": float(getattr(settings, "pb_daily_rescue_min_score", 0.0)),
                    "disposition": "triggered",
                }
                day_records.append(record)

            candidate = {
                "symbol": sym,
                "effective_rsi": float(rsi_val) if not np.isnan(rsi_val) else 50.0,
                "item": item_lookup.get(sym),
                "entry_rsi": float(rsi_val) if not np.isnan(rsi_val) else 50.0,
                "entry_gap_pct": float(gap_pct),
                "entry_sma_dist_pct": float(sma_dist_pct),
                "entry_cdd": int(cdd_val),
                "entry_rank": 0,
                "entry_rank_pct": 100.0,
                "n_candidates": 0,
                "trigger_type": trigger_type,
                "sector": sector,
                "entry_price": float(fill_price),
                "entry_open": float(O),
                "entry_atr": float(atr_val) if not np.isnan(atr_val) else 0.0,
                "stop_price": float(stop_price) if stop_price is not None else None,
                "risk_per_share": float(risk_per_share) if risk_per_share is not None else None,
                "record": record,
                "prev_iloc": iloc,
                "daily_atr": float(atr_val) if not np.isnan(atr_val) else 0.0,
                "rescue_flow_candidate": False,
                "flow_negative": False,
                "trend_tier": trend_tier,
                "v2_trigger_types": v2_trigger_types,
                "v2_trigger_tier": v2_trigger_tier,
            }
            flow_negative = False
            if settings.pb_flow_gate:
                flow_flags = self._flow_negative.get(sym)
                flow_negative = bool(flow_flags is not None and flow_flags[iloc])
            candidate["flow_negative"] = flow_negative
            if record is not None:
                record["flow_negative"] = flow_negative
            candidates.append(candidate)

        scored_candidates: list[dict[str, Any]] = []
        rescue_floor = float(getattr(settings, "pb_daily_rescue_min_score", settings.pb_rescue_min_score))
        v2 = settings.pb_v2_enabled
        today_candidate_syms: set[str] = set()
        for candidate in candidates:
            record = candidate.get("record")
            if v2:
                c_sym = str(candidate["symbol"])
                c_ind = self._indicators.get(c_sym)
                c_iloc = int(candidate.get("prev_iloc", -1))
                c_closes = self._replay._daily_arrs.get(c_sym, {}).get("close")
                adx_v = plus_di_v = minus_di_v = 0.0
                depth_v = rsi5_v = vcr_v = rs_v = 0.0
                sma_slope_v = False
                is_down_day = False
                if c_ind is not None and c_iloc >= 0:
                    adx_arr = c_ind.get("adx")
                    if adx_arr is not None and not np.isnan(adx_arr[c_iloc]):
                        adx_v = float(adx_arr[c_iloc])
                    pdi_arr = c_ind.get("plus_di")
                    if pdi_arr is not None and not np.isnan(pdi_arr[c_iloc]):
                        plus_di_v = float(pdi_arr[c_iloc])
                    mdi_arr = c_ind.get("minus_di")
                    if mdi_arr is not None and not np.isnan(mdi_arr[c_iloc]):
                        minus_di_v = float(mdi_arr[c_iloc])
                    depth_arr = c_ind.get("depth")
                    if depth_arr is not None and not np.isnan(depth_arr[c_iloc]):
                        depth_v = float(depth_arr[c_iloc])
                    rsi5_arr = c_ind.get("rsi5")
                    if rsi5_arr is not None and not np.isnan(rsi5_arr[c_iloc]):
                        rsi5_v = float(rsi5_arr[c_iloc])
                    vcr_arr = c_ind.get("vcr")
                    if vcr_arr is not None and not np.isnan(vcr_arr[c_iloc]):
                        vcr_v = float(vcr_arr[c_iloc])
                    rs_arr = c_ind.get("rs_ratio")
                    if rs_arr is not None and not np.isnan(rs_arr[c_iloc]):
                        rs_v = float(rs_arr[c_iloc])
                    sma_slope_v = bool(c_ind["sma_slope"][c_iloc])
                    if c_closes is not None and c_iloc >= 1:
                        is_down_day = bool(c_closes[c_iloc] < c_closes[c_iloc - 1])
                bundle = _daily_signal_bundle_v2(
                    trend_tier=str(candidate.get("trend_tier", "STRONG")),
                    trigger_types=list(candidate.get("v2_trigger_types", [])),
                    trigger_tier=str(candidate.get("v2_trigger_tier", "MEDIUM")),
                    adx=adx_v, plus_di=plus_di_v, minus_di=minus_di_v,
                    sma_slope_pos=sma_slope_v,
                    sma_dist_pct=float(candidate.get("entry_sma_dist_pct", 0.0)),
                    pullback_depth_atr=depth_v,
                    rsi2=float(candidate.get("entry_rsi", 50.0)),
                    rsi5=rsi5_v,
                    vcr=vcr_v, is_down_day=is_down_day,
                    rs_ratio=rs_v,
                    gap_pct=float(candidate.get("entry_gap_pct", 0.0)),
                    regime_tier=regime_tier,
                    sector_count=sector_raw_counts.get(str(candidate.get("sector")), 1),
                    item=candidate.get("item"),
                    n_triggers=len(candidate.get("v2_trigger_types", [])),
                    candidate_yesterday=c_sym in getattr(self, "_prev_day_candidates", set()),
                )
                today_candidate_syms.add(c_sym)
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
            # V2 uses its own floor -- no rescue bypass (matches daily engine)
            if v2:
                effective_floor = settings.pb_v2_signal_floor
                if regime_tier == "B" and settings.pb_v2_signal_floor_tier_b > 0:
                    effective_floor = settings.pb_v2_signal_floor_tier_b
                if candidate["daily_signal_score"] < effective_floor:
                    if record is not None:
                        self._record_rejection(record, "daily_signal_floor_reject", rejection_log, shadow_outcomes, funnel_counters)
                    continue
            elif not _passes_daily_signal_floor(settings, candidate["daily_signal_score"], rescue_candidate=rescue_candidate):
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
            if rescue_candidate:
                if record is not None:
                    record["candidate_kind"] = "rescue"
                    record["disposition"] = "flow_rescue_pool"
                if funnel_counters is not None:
                    funnel_counters["flow_rescue_pool"] = funnel_counters.get("flow_rescue_pool", 0) + 1
                rescue_candidates.append(candidate)
            else:
                if record is not None:
                    record["disposition"] = "candidate_pool"
                if funnel_counters is not None:
                    funnel_counters["candidate_pool"] = funnel_counters.get("candidate_pool", 0) + 1
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
            for candidate in candidates:
                record = candidate.get("record")
                if record is not None:
                    record["n_candidates"] = len(candidates)
                    self._record_rejection(record, "min_candidates_day_reject", rejection_log, shadow_outcomes, funnel_counters)
            for candidate in rescue_candidates:
                record = candidate.get("record")
                if record is not None:
                    record["n_candidates"] = len(candidates)
                    self._record_rejection(record, "flow_reject", rejection_log, shadow_outcomes, funnel_counters)
            return {}, {}

        core_watchlist: dict[str, dict[str, Any]] = {}
        for rank_counter, candidate in enumerate(candidates, start=1):
            record = candidate.get("record")
            rank_pct = _rank_percent(rank_counter, len(candidates))
            candidate["entry_rank"] = rank_counter
            candidate["entry_rank_pct"] = rank_pct
            candidate["n_candidates"] = len(candidates)
            candidate["daily_signal_rank_pct"] = rank_pct
            if record is not None:
                record["entry_rank"] = rank_counter
                record["entry_rank_pct"] = rank_pct
                record["n_candidates"] = len(candidates)
                record["daily_signal_rank_pct"] = rank_pct
            gate_reason = _rank_gate_reason(rank_counter, len(candidates), settings)
            if gate_reason is not None:
                if record is not None:
                    self._record_rejection(record, gate_reason, rejection_log, shadow_outcomes, funnel_counters)
                continue
            if record is not None:
                record["disposition"] = "watchlist"
            core_watchlist[str(candidate["symbol"])] = candidate

        rescue_watchlist: dict[str, dict[str, Any]] = {}
        rescue_candidates.sort(key=lambda row: (-float(row.get("daily_signal_score", 0.0)), float(row.get("effective_rsi", 50.0))))
        for rescue_rank, candidate in enumerate(rescue_candidates, start=1):
            record = candidate.get("record")
            candidate["entry_rank"] = len(candidates) + rescue_rank
            candidate["entry_rank_pct"] = 100.0
            candidate["n_candidates"] = len(candidates)
            candidate["rescue_rank"] = rescue_rank
            candidate["daily_signal_rank_pct"] = 100.0
            if record is not None:
                record["entry_rank"] = len(candidates) + rescue_rank
                record["entry_rank_pct"] = 100.0
                record["n_candidates"] = len(candidates)
                record["rescue_rank"] = rescue_rank
                record["daily_signal_rank_pct"] = 100.0
                record["disposition"] = "rescue_watchlist"
            rescue_watchlist[str(candidate["symbol"])] = candidate

        if funnel_counters is not None:
            funnel_counters["watchlist"] = funnel_counters.get("watchlist", 0) + len(core_watchlist) + len(rescue_watchlist)
        # V2: update candidate persistence tracking
        if v2:
            self._prev_day_candidates = today_candidate_syms
        return core_watchlist, rescue_watchlist

    def _fallback_watch_item(
        self,
        symbol: str,
        candidate: dict[str, Any],
        artifact: WatchlistArtifact,
        bars: list[Bar],
        prev_date: date,
    ) -> WatchlistItem:
        prev_close = 0.0
        arrs = self._replay._daily_arrs.get(symbol)
        if arrs is not None:
            prev_iloc = int(candidate.get("prev_iloc", -1))
            if 0 <= prev_iloc < len(arrs["close"]):
                prev_close = float(arrs["close"][prev_iloc])
        ref_price = prev_close if prev_close > 0 else (bars[0].open if bars else 1.0)
        avg_5m_volume = float(np.mean([bar.volume for bar in bars[: min(len(bars), 12)]])) if bars else 0.0
        expected_5m = max(avg_5m_volume, 1.0)
        avg_30m = expected_5m * 6.0
        band = ref_price * self._settings.avwap_band_pct
        return WatchlistItem(
            symbol=symbol,
            exchange="SMART",
            primary_exchange="SMART",
            currency="USD",
            tick_size=0.01,
            point_value=1.0,
            sector=str(candidate.get("sector") or "Unknown"),
            regime_score=float(getattr(artifact.regime, "score", 0.0)),
            regime_tier=str(getattr(artifact.regime, "tier", "B")),
            regime_risk_multiplier=float(getattr(artifact.regime, "risk_multiplier", 1.0)),
            sector_score=0.0,
            sector_rank_weight=0.0,
            sponsorship_score=0.0,
            sponsorship_state="NEUTRAL",
            persistence=0.5,
            intensity_z=0.0,
            accel_z=0.0,
            rs_percentile=50.0,
            leader_pass=False,
            trend_pass=True,
            trend_strength=0.0,
            earnings_risk_flag=False,
            blacklist_flag=False,
            anchor_date=prev_date,
            anchor_type="PULLBACK_FALLBACK",
            acceptance_pass=True,
            avwap_ref=ref_price,
            avwap_band_lower=max(ref_price - band, 0.01),
            avwap_band_upper=ref_price + band,
            daily_atr_estimate=float(candidate.get("daily_atr") or 0.0),
            intraday_atr_seed=0.0,
            daily_rank=0.0,
            tradable_flag=True,
            conviction_bucket="BASE",
            conviction_multiplier=1.0,
            recommended_risk_r=1.0,
            average_30m_volume=avg_30m,
            expected_5m_volume=expected_5m,
            flow_proxy_gate_pass=not bool(candidate.get("rescue_flow_candidate")),
            overflow_rank=None,
        )

    def _aggregate_30m_bar(self, symbol: str, bars: list[Bar]) -> Bar | None:
        if not bars:
            return None
        return Bar(
            symbol=symbol,
            start_time=bars[0].start_time,
            end_time=bars[-1].end_time,
            open=bars[0].open,
            high=max(bar.high for bar in bars),
            low=min(bar.low for bar in bars),
            close=bars[-1].close,
            volume=sum(bar.volume for bar in bars),
        )

    def _entry_threshold(self, state: _PBHybridState) -> float:
        return iaric_core_logic.entry_threshold(self._settings, state)

    def _market_open_timestamp(self, trade_date: date) -> datetime:
        return datetime(trade_date.year, trade_date.month, trade_date.day, 9, 30, tzinfo=ET).astimezone(timezone.utc)

    def _market_close_timestamp(self, trade_date: date) -> datetime:
        return datetime(trade_date.year, trade_date.month, trade_date.day, 16, 0, tzinfo=ET).astimezone(timezone.utc)

    def _open_scored_fill_timing(self) -> str:
        timing = str(getattr(self._settings, "pb_open_scored_fill_timing", "next_5m_open") or "next_5m_open").lower()
        if timing not in {"next_5m_open", "same_open"}:
            raise ValueError(
                "pb_open_scored_fill_timing must be 'next_5m_open' or 'same_open'"
            )
        return timing

    def _open_scored_fill_bar(self, bars: list[Bar]) -> tuple[int, Bar] | None:
        start = self._settings.pb_intraday_entry_start
        end = self._settings.pb_intraday_entry_end
        for idx, bar in enumerate(bars):
            ts = bar.start_time
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            now_et = ts.astimezone(ET).time()
            if start <= now_et <= end:
                return idx, bar
        return None

    def _apply_open_scored_delayed_fill(
        self,
        payload: dict[str, Any],
        fallback: dict[str, Any],
        *,
        fill_bar_idx: int,
        fill_bar: Bar,
    ) -> bool:
        candidate = fallback.get("candidate") or {}
        item = fallback.get("item")
        entry_open = float(fill_bar.open)
        slip = entry_open * self._slippage.slip_bps_normal / 10_000
        fill_price = round(entry_open + slip, 2)
        entry_atr = float(
            payload.get("entry_atr")
            or candidate.get("entry_atr")
            or candidate.get("daily_atr")
            or getattr(item, "daily_atr_estimate", 0.0)
            or 0.0
        )
        if fill_price <= 0 or entry_atr <= 0:
            return False
        stop_price = fill_price - self._settings.pb_atr_stop_mult * entry_atr
        risk_per_share = fill_price - stop_price
        if stop_price <= 0 or risk_per_share <= 0:
            return False
        payload["entry_open"] = entry_open
        payload["entry_price"] = float(fill_price)
        payload["entry_atr"] = entry_atr
        payload["stop_price"] = float(stop_price)
        payload["risk_per_share"] = float(risk_per_share)
        payload["entry_timestamp"] = fill_bar.start_time
        payload["entry_bar_index"] = int(fill_bar_idx)
        payload["open_scored_fill_timing"] = "next_5m_open"
        return True

    def _activate_delayed_confirm(
        self,
        state: _PBHybridState,
        market: MarketSnapshot,
        bars: list[Bar],
        bar_idx: int,
        session_atr: float,
        trade_date: date,
        record: dict[str, Any] | None,
        funnel_counters: dict[str, int] | None,
        fsm_log: list[dict[str, Any]] | None,
    ) -> bool:
        if bar_idx >= len(bars):
            return False
        bar = bars[bar_idx]
        step = iaric_core_logic.activate_delayed_confirm_route(
            self._settings,
            state,
            state.item,
            bar,
            market,
            bar_idx,
            session_atr,
            bars=bars,
        )
        if step is None:
            return False
        self._mark_stage(record, "READY")
        if funnel_counters is not None:
            funnel_counters["ready"] = funnel_counters.get("ready", 0) + 1
        if record is not None:
            record["intraday_setup_type"] = state.intraday_setup_type
            record["intraday_score"] = round(state.intraday_score, 2)
            record["reclaim_bars"] = max(bar_idx - state.flush_bar_idx + 1, 1)
            record["selection_refine_score"] = round(state.intraday_score, 2)
            record["ready_timestamp"] = state.ready_timestamp
            record["ready_bar_index"] = int(state.ready_bar_idx)
            record["ready_cpr"] = round(state.ready_cpr, 4)
            record["ready_volume_ratio"] = round(state.ready_volume_ratio, 4)
            record["refinement_route"] = "DELAYED_CONFIRM"
            self._apply_score_components(record, state.score_components, prefix="score_component_")
        if fsm_log is not None:
            self._log_fsm(
                fsm_log,
                state.symbol,
                trade_date,
                bar.end_time,
                step.prior_stage,
                "READY",
                step.reason or "delayed_confirm",
                score=state.intraday_score,
            )
        return True

    # ------------------------------------------------------------------
    # V2 Route: VWAP_BOUNCE (after bar 12 / 11:00 ET)
    # ------------------------------------------------------------------
    def _activate_vwap_bounce(
        self,
        state: _PBHybridState,
        market: MarketSnapshot,
        bars: list[Bar],
        bar_idx: int,
        session_atr: float,
        trade_date: date,
        record: dict[str, Any] | None,
        funnel_counters: dict[str, int] | None,
        fsm_log: list[dict[str, Any]] | None,
    ) -> bool:
        if bar_idx >= len(bars):
            return False
        bar = bars[bar_idx]
        step = iaric_core_logic.activate_vwap_bounce_route(
            self._settings,
            state,
            state.item,
            bar,
            market,
            bar_idx,
            session_atr,
            bars=bars,
        )
        if step is None:
            return False
        self._mark_stage(record, "READY")
        if funnel_counters is not None:
            funnel_counters["ready"] = funnel_counters.get("ready", 0) + 1
        if record is not None:
            record["intraday_setup_type"] = "VWAP_BOUNCE"
            record["intraday_score"] = round(state.intraday_score, 2)
            record["reclaim_bars"] = 1
            record["selection_refine_score"] = round(state.intraday_score, 2)
            record["ready_timestamp"] = state.ready_timestamp
            record["ready_bar_index"] = int(state.ready_bar_idx)
            record["ready_cpr"] = round(state.ready_cpr, 4)
            record["ready_volume_ratio"] = round(state.ready_volume_ratio, 4)
            record["refinement_route"] = "VWAP_BOUNCE"
            self._apply_score_components(record, state.score_components, prefix="score_component_")
        if fsm_log is not None:
            self._log_fsm(fsm_log, state.symbol, trade_date, bar.end_time, step.prior_stage, "READY", step.reason or "vwap_bounce", score=state.intraday_score)
        return True

    # ------------------------------------------------------------------
    # V2 Route: AFTERNOON_RETEST (after bar 48 / 13:30 ET)
    # ------------------------------------------------------------------
    def _activate_afternoon_retest(
        self,
        state: _PBHybridState,
        market: MarketSnapshot,
        bars: list[Bar],
        bar_idx: int,
        session_atr: float,
        trade_date: date,
        record: dict[str, Any] | None,
        funnel_counters: dict[str, int] | None,
        fsm_log: list[dict[str, Any]] | None,
    ) -> bool:
        if bar_idx >= len(bars):
            return False
        bar = bars[bar_idx]
        step = iaric_core_logic.activate_afternoon_retest_route(
            self._settings,
            state,
            state.item,
            bar,
            market,
            bar_idx,
            session_atr,
            bars=bars,
        )
        if step is None:
            return False
        self._mark_stage(record, "READY")
        if funnel_counters is not None:
            funnel_counters["ready"] = funnel_counters.get("ready", 0) + 1
        if record is not None:
            record["intraday_setup_type"] = "AFTERNOON_RETEST"
            record["intraday_score"] = round(state.intraday_score, 2)
            record["reclaim_bars"] = 1
            record["selection_refine_score"] = round(state.intraday_score, 2)
            record["ready_timestamp"] = state.ready_timestamp
            record["ready_bar_index"] = int(state.ready_bar_idx)
            record["ready_cpr"] = round(state.ready_cpr, 4)
            record["ready_volume_ratio"] = round(state.ready_volume_ratio, 4)
            record["refinement_route"] = "AFTERNOON_RETEST"
            self._apply_score_components(record, state.score_components, prefix="score_component_")
        if fsm_log is not None:
            self._log_fsm(fsm_log, state.symbol, trade_date, bar.end_time, step.prior_stage, "READY", step.reason or "afternoon_retest", score=state.intraday_score)
        return True

    def _invalidate_state(
        self,
        state: _PBHybridState,
        record: dict[str, Any] | None,
        fsm_log: list[dict[str, Any]] | None,
        trade_date: date,
        timestamp: datetime,
        reason: str,
        reset_bar: int,
    ) -> None:
        prior = state.stage
        state.stage = "INVALIDATED"
        state.invalid_reason = reason
        state.invalid_reset_bar = reset_bar
        if record is not None:
            record["intraday_invalid_reason"] = reason
        self._mark_stage(record, "INVALIDATED")
        if fsm_log is not None:
            self._log_fsm(fsm_log, state.symbol, trade_date, timestamp, prior, "INVALIDATED", reason)

    def _initial_stop(self, state: _PBHybridState, session_atr: float) -> float:
        return iaric_core_logic.compute_initial_stop(
            self._settings,
            state.setup_low,
            state.daily_atr,
            session_atr,
        )

    def _build_daily_fallback_position(
        self,
        *,
        symbol: str,
        item: WatchlistItem,
        record: dict[str, Any],
        trade_date: date,
        regime_tier: str,
        quantity: int,
        entry_time: datetime | None = None,
        entry_bar_idx: int = 0,
    ) -> _PBHybridPosition | None:
        entry_price = float(record.get("entry_price") or 0.0)
        stop_price = float(record.get("stop_price") or 0.0)
        risk_per_share = float(record.get("risk_per_share") or 0.0)
        if entry_price <= 0 or stop_price <= 0 or risk_per_share <= 0 or quantity < 1:
            return None
        raw_entry_open = record.get("entry_open")
        if raw_entry_open is not None:
            entry_open = float(raw_entry_open)
            entry_slip_per_share = max(entry_price - entry_open, 0.0)
        else:
            entry_slip_per_share = entry_price * self._slippage.slip_bps_normal / 10_000
            entry_price = round(entry_price + entry_slip_per_share, 2)
            risk_per_share = entry_price - stop_price
        ts = entry_time or self._market_open_timestamp(trade_date)
        return _PBHybridPosition(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=ts,
            quantity=quantity,
            risk_per_share=risk_per_share,
            sector=str(record.get("sector") or item.sector),
            regime_tier=regime_tier,
            stop=stop_price,
            current_stop=stop_price,
            trigger_type=str(record.get("trigger_type") or "UNKNOWN"),
            entry_rsi=float(record.get("entry_rsi") or 50.0),
            entry_gap_pct=float(record.get("entry_gap_pct") or 0.0),
            entry_sma_dist_pct=float(record.get("entry_sma_dist_pct") or 0.0),
            entry_cdd=int(record.get("entry_cdd") or 0),
            entry_rank=int(record.get("entry_rank") or 0),
            entry_rank_pct=float(record.get("entry_rank_pct") or 100.0),
            n_candidates=int(record.get("n_candidates") or 0),
            daily_signal_score=float(record.get("daily_signal_score") or 0.0),
            daily_signal_rank_pct=float(record.get("daily_signal_rank_pct") or 100.0),
            signal_family=str(record.get("signal_family") or getattr(self._settings, "pb_daily_signal_family", "balanced_v1")),
            intraday_setup_type="OPEN_SCORED_ENTRY",
            entry_trigger="OPEN_SCORED_ENTRY",
            route_family="OPEN_SCORED_ENTRY",
            carry_profile=self._route_carry_profile("OPEN_SCORED_ENTRY"),
            selection_reason=str(record.get("selection_reason") or "daily_signal_score"),
            intraday_score=float(
                record.get("intraday_score")
                or record.get("route_score")
                or record.get("daily_signal_score")
                or record.get("selection_refine_score")
                or 0.0
            ),
            reclaim_bars=0,
            rescue_flow_candidate=bool(record.get("rescue_flow_candidate")),
            reentry_count=0,
            entry_atr=float(record.get("entry_atr") or 0.0),
            item=item,
            acceptance_count=0,
            required_acceptance=0,
            micropressure_signal="N/A",
            max_favorable=entry_price,
            max_adverse=entry_price,
            highest_close=entry_price,
            commission_entry=self._slippage.commission_per_share * quantity,
            slippage_entry=entry_slip_per_share * quantity,
            entry_bar_idx=entry_bar_idx,
            ledger_ref=record,
        )

    def _manage_daily_fallback_position(
        self,
        position: _PBHybridPosition,
        trade_date: date,
        carry_positions: dict[str, _PBHybridPosition],
        trades: list[TradeRecord],
        equity: float,
        funnel_counters: dict[str, int] | None,
    ) -> float:
        ohlc = self._replay.get_daily_ohlc(position.symbol, trade_date)
        if ohlc is None:
            trade, eq_delta = self._close_position(
                position,
                position.entry_price,
                self._market_close_timestamp(trade_date),
                "DATA_GAP",
            )
            trades.append(trade)
            self._attach_hybrid_trade_outcome(position, trade)
            return equity + eq_delta

        _O, H, L, C = ohlc
        if H > position.max_favorable + 1e-9:
            position.max_favorable = H
            position.bars_to_mfe = max(position.bars_to_mfe, 78)
        position.max_adverse = min(position.max_adverse, L)
        position.highest_close = max(position.highest_close, C)
        position.close_pct = _close_in_range_pct(H, L, C)
        position.close_r = position.unrealized_r(C)

        ind = self._indicators.get(position.symbol)
        didx = self._replay._daily_didx.get(position.symbol)
        if ind is not None and didx is not None:
            iloc_today = _iloc_for_date(didx, trade_date)
            if iloc_today >= 0 and not np.isnan(ind["rsi"][iloc_today]):
                position.exit_rsi = float(ind["rsi"][iloc_today])

        exit_price: float | None = None
        exit_reason = ""
        route_family = position.route_family or "OPEN_SCORED_ENTRY"
        if L <= position.stop:
            exit_price = C if self._settings.pb_use_close_stop else position.stop
            exit_reason = "STOP_HIT"
        elif self._settings.pb_profit_target_r > 0 and position.unrealized_r(C) >= self._settings.pb_profit_target_r:
            exit_price = C
            exit_reason = "PROFIT_TARGET"

        if exit_price is None and self._settings.pb_carry_enabled:
            position.carry_score = max(
                float(position.carry_score),
                min(max(position.daily_signal_score / 100.0, 0.0), 1.0) * 45.0
                + min(max(position.close_pct, 0.0), 1.0) * 25.0
                + min(max(position.close_r / 2.0, 0.0), 1.0) * 15.0
                + min(max(position.mfe_r() / 2.0, 0.0), 1.0) * 15.0,
            )
            last_n = self._replay.get_flow_proxy_last_n(
                position.symbol,
                trade_date,
                max(1, int(self._route_setting(route_family, "flow_reversal_lookback", "pb_flow_reversal_lookback"))),
            )
            flow_ok = last_n is None or not all(v < 0 for v in last_n)
            binary_ok = (
                position.unrealized_r(C) > float(self._route_setting(route_family, "carry_min_r", "pb_carry_min_r"))
                and position.close_pct >= float(self._route_setting(route_family, "carry_close_pct_min", "pb_carry_close_pct_min"))
                and position.mfe_r() >= float(self._route_setting(route_family, "carry_mfe_gate_r", "pb_carry_mfe_gate_r"))
                and flow_ok
                and position.daily_signal_score >= float(self._route_setting(route_family, "carry_min_daily_signal_score", "pb_carry_min_daily_signal_score"))
            )
            carry_score_fallback_enabled = bool(
                self._route_setting(route_family, "carry_score_fallback_enabled", "pb_carry_score_fallback")
            )
            carry_score_threshold = float(
                self._route_setting(route_family, "carry_score_threshold", "pb_carry_score_threshold")
            )
            score_ok = (
                self._settings.pb_carry_enabled
                and carry_score_fallback_enabled
                and position.carry_score >= carry_score_threshold
            )
            position.carry_binary_ok = bool(binary_ok)
            position.carry_score_ok = bool(score_ok)
            if position.close_r > 0 and (binary_ok or score_ok):
                position.current_stop = position.stop
                position.carry_decision_path = "binary" if binary_ok else "score_fallback"
                carry_positions[position.symbol] = position
                if funnel_counters is not None:
                    funnel_counters["carried"] = funnel_counters.get("carried", 0) + 1
                return equity
        position.carry_decision_path = "flatten" if exit_price is None else exit_reason.lower()

        trade, eq_delta = self._close_position(
            position,
            float(C if exit_price is None else exit_price),
            self._market_close_timestamp(trade_date),
            "EOD_FLATTEN" if exit_price is None else exit_reason,
        )
        trades.append(trade)
        self._attach_hybrid_trade_outcome(position, trade)
        return equity + eq_delta

    def run(self) -> IARICPullbackResult:
        cfg = self._config
        settings = self._settings
        open_scored_fill_timing = self._open_scored_fill_timing()
        collect_diagnostics = self._collect_diagnostics
        start = date.fromisoformat(cfg.start_date)
        end = date.fromisoformat(cfg.end_date)
        if settings.pb_intraday_bar_minutes != 5:
            raise ValueError(
                f"IARIC pullback intraday hybrid currently requires 5-minute bars; got {settings.pb_intraday_bar_minutes}."
            )

        trading_dates = self._replay.tradable_dates(start, end)
        if not trading_dates:
            logger.warning("No trading dates in range %s to %s", start, end)
            return IARICPullbackResult(
                trades=[],
                equity_curve=np.array([cfg.initial_equity]),
                timestamps=np.array([]),
                daily_selections={},
                candidate_ledger={} if collect_diagnostics else None,
                funnel_counters={} if collect_diagnostics else None,
                rejection_log=[] if collect_diagnostics else None,
                shadow_outcomes=[] if collect_diagnostics else None,
                selection_attribution={} if collect_diagnostics else None,
                fsm_log=[] if collect_diagnostics else None,
                decision_stream=[],
                trade_outcomes=[],
            )

        equity = cfg.initial_equity
        carry_positions: dict[str, _PBHybridPosition] = {}
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
            "watchlist": 0,
            "min_candidates_day_reject": 0,
            "rank_abs_reject": 0,
            "rank_pct_reject": 0,
            "sector_cap_reject": 0,
            "position_cap_reject": 0,
            "sizing_reject": 0,
            "buying_power_reject": 0,
            "no_intraday_data": 0,
            "no_intraday_setup": 0,
            "never_ready": 0,
            "intraday_invalidated": 0,
            "priority_reject": 0,
            "intraday_priority_reserve": 0,
            "rescue_cap_reject": 0,
            "entry_window_expired": 0,
            "flush_locked": 0,
            "reclaiming": 0,
            "ready": 0,
            "entered": 0,
            "rescue_entered": 0,
            "pm_reentry": 0,
            "open_scored_entry": 0,
            "partial": 0,
            "trailed": 0,
            "carried": 0,
        } if collect_diagnostics else None)
        rejection_log: list[dict[str, Any]] | None = [] if collect_diagnostics else None
        shadow_outcomes: list[dict[str, Any]] | None = [] if collect_diagnostics else None
        fsm_log: list[dict[str, Any]] | None = [] if collect_diagnostics else None

        for trade_idx, trade_date in enumerate(trading_dates):
            has_next_backtest_day = trade_idx < len(trading_dates) - 1
            ts_history.append(datetime(trade_date.year, trade_date.month, trade_date.day, tzinfo=timezone.utc))
            prev_date = self._replay.get_prev_trading_date(trade_date)
            if prev_date is None:
                equity_history.append(equity)
                continue

            equity = self._process_overnight_carries(carry_positions, trade_date, equity, trades)

            artifact = self._replay.iaric_selection_for_date(prev_date, settings)
            daily_selections[trade_date] = artifact
            regime_tier = artifact.regime.tier
            if settings.pb_regime_gate == "C_only_skip" and regime_tier == "C":
                equity_history.append(equity)
                continue
            if settings.pb_regime_gate == "B_and_above" and regime_tier not in ("A", "B"):
                equity_history.append(equity)
                continue

            core_watchlist, rescue_watchlist = self._build_watchlists(
                trade_date,
                prev_date,
                artifact,
                regime_tier,
                carry_positions,
                candidate_ledger,
                funnel_counters,
                rejection_log,
                shadow_outcomes,
            )
            if not core_watchlist and not rescue_watchlist:
                equity_history.append(equity)
                continue

            item_lookup = getattr(artifact, "by_symbol", {})
            watch_symbols = sorted({*core_watchlist.keys(), *rescue_watchlist.keys()})
            bars_by_symbol: dict[str, list[Bar]] = {}
            market_by_symbol: dict[str, MarketSnapshot] = {}
            state_by_symbol: dict[str, _PBHybridState] = {}
            session_atr_by_symbol: dict[str, float] = {}
            open_scored_candidates: list[dict[str, Any]] = []

            for symbol in watch_symbols:
                bars = list(self._replay.get_5m_bar_objects_for_date(symbol, trade_date))
                candidate = core_watchlist.get(symbol) or rescue_watchlist.get(symbol)
                if candidate is None:
                    continue
                rescue_candidate = symbol in rescue_watchlist
                item = item_lookup.get(symbol) or self._fallback_watch_item(symbol, candidate, artifact, bars, prev_date)
                record = candidate.get("record")
                if not bars:
                    if rescue_candidate:
                        if record is not None:
                            record["live_intraday_candidate"] = False
                            record["intraday_data_available"] = False
                            self._record_rejection(record, "no_intraday_data", rejection_log, shadow_outcomes, funnel_counters)
                        continue
                    if record is not None:
                        record["live_intraday_candidate"] = False
                        record["intraday_data_available"] = False
                    if not bool(getattr(settings, "pb_open_scored_missing_5m_allow", True)):
                        if record is not None:
                            self._record_rejection(record, "no_intraday_data", rejection_log, shadow_outcomes, funnel_counters)
                        continue
                    if not self._open_scored_eligible(record or candidate):
                        if record is not None:
                            self._record_rejection(record, "open_scored_gate_reject", rejection_log, shadow_outcomes, funnel_counters)
                        continue
                    if open_scored_fill_timing == "next_5m_open":
                        if record is not None:
                            self._record_rejection(record, "open_scored_no_post_open_bar", rejection_log, shadow_outcomes, funnel_counters)
                        continue
                    if record is not None:
                        record["refinement_route"] = "OPEN_SCORED_ENTRY"
                        record["route_feasible"] = True
                        record["route_feasible_bar_index"] = 0
                        record["open_scored_fill_timing"] = "same_open"
                    open_scored_candidates.append({
                        "symbol": symbol,
                        "candidate": candidate,
                        "item": item,
                        "record": record,
                        "sector": str(candidate.get("sector") or item.sector),
                        "missing_5m": True,
                    })
                    continue

                bars_by_symbol[symbol] = bars
                market_by_symbol[symbol] = MarketSnapshot(symbol=symbol)
                session_atr_by_symbol[symbol] = self._session_atr(item, bars)
                state_by_symbol[symbol] = _PBHybridState(
                    symbol=symbol,
                    item=item,
                    record=record,
                    trigger_type=str(candidate.get("trigger_type") or "UNKNOWN"),
                    entry_rsi=float(record.get("entry_rsi") if record is not None else candidate.get("entry_rsi", candidate.get("effective_rsi", 50.0))),
                    entry_gap_pct=float(record.get("entry_gap_pct") if record is not None else candidate.get("entry_gap_pct", 0.0)),
                    entry_sma_dist_pct=float(record.get("entry_sma_dist_pct") if record is not None else candidate.get("entry_sma_dist_pct", 0.0)),
                    entry_cdd=int(record.get("entry_cdd") if record is not None else candidate.get("entry_cdd", 0)),
                    entry_rank=int(record.get("entry_rank") if record is not None else candidate.get("entry_rank", 0)),
                    entry_rank_pct=float(record.get("entry_rank_pct") if record is not None else candidate.get("entry_rank_pct", 100.0)),
                    n_candidates=int(record.get("n_candidates") if record is not None else candidate.get("n_candidates", len(core_watchlist))),
                    prev_iloc=int(candidate.get("prev_iloc", -1)),
                    sector=str(candidate.get("sector") or item.sector),
                    daily_atr=float(candidate.get("daily_atr") or 0.0),
                    daily_signal_score=float(record.get("daily_signal_score") if record is not None else candidate.get("daily_signal_score", 0.0)),
                    daily_signal_rank_pct=float(record.get("daily_signal_rank_pct") if record is not None else candidate.get("daily_signal_rank_pct", 100.0)),
                    daily_signal_components=dict(candidate.get("daily_signal_components") or {}),
                    rescue_flow_candidate=rescue_candidate,
                    trigger_types=list(candidate.get("v2_trigger_types", [])),
                    trigger_tier=str(candidate.get("v2_trigger_tier", "")),
                    trend_tier=str(candidate.get("trend_tier", "STRONG")),
                )
                if record is not None:
                    record["live_intraday_candidate"] = True
                    record["intraday_data_available"] = True
                    record["intraday_setup_type"] = ""
                    record["entry_trigger"] = ""
                    record["intraday_score"] = 0.0
                    record["reclaim_bars"] = 0
                    record["refinement_route"] = ""
                    record["blocked_by_capacity_reason"] = ""
                    record["entry_window_feasible"] = False
                    record["entry_window_feasible_bar_index"] = None
                    record["max_feasible_intraday_score"] = 0.0
                    record["selection_refine_score"] = float(record.get("selection_refine_score") or 0.0)
                if self._open_scored_eligible(candidate):
                    fill_bar_idx = 0
                    fill_bar = bars[0]
                    if open_scored_fill_timing == "next_5m_open":
                        fill_ref = self._open_scored_fill_bar(bars)
                        if fill_ref is None:
                            if record is not None:
                                self._record_rejection(
                                    record,
                                    "open_scored_no_post_open_bar",
                                    rejection_log,
                                    shadow_outcomes,
                                    funnel_counters,
                                )
                            continue
                        fill_bar_idx, fill_bar = fill_ref
                    if record is not None:
                        record["route_feasible"] = True
                        record["route_feasible_bar_index"] = int(fill_bar_idx)
                        record["open_scored_fill_timing"] = open_scored_fill_timing
                    open_scored_candidates.append({
                        "symbol": symbol,
                        "candidate": candidate,
                        "item": item,
                        "record": record,
                        "sector": str(candidate.get("sector") or item.sector),
                        "missing_5m": False,
                        "fill_bar_idx": fill_bar_idx,
                        "fill_bar": fill_bar,
                    })

            if not state_by_symbol and not open_scored_candidates:
                equity_history.append(equity)
                continue

            max_pos = settings.pb_max_positions
            if regime_tier == "B":
                tier_b_cap = settings.max_positions_tier_b
                max_pos = min(max_pos, tier_b_cap)
            sector_cap = settings.max_positions_per_sector
            available_slots = max(max_pos - len(carry_positions), 0)
            sector_counts: dict[str, int] = {}
            for position in carry_positions.values():
                sector_counts[position.sector] = sector_counts.get(position.sector, 0) + 1
            core_intraday_candidates = sum(1 for state in state_by_symbol.values() if not state.rescue_flow_candidate)
            if settings.pb_v2_enabled:
                # V2: OPEN_SCORED is the primary route -- no reserve, full capacity
                intraday_priority_reserve = 0
                open_scored_slot_cap = available_slots
            else:
                intraday_priority_reserve = min(
                    available_slots,
                    max(int(settings.pb_intraday_priority_reserve_slots), 0),
                    core_intraday_candidates,
                )
                open_scored_slot_cap = min(
                    max(available_slots - intraday_priority_reserve, 0),
                    max(int(np.ceil(available_slots * float(getattr(settings, "pb_open_scored_max_share", 0.45)))), 0),
                )
            max_total_open_scored = available_slots if not state_by_symbol else open_scored_slot_cap
            open_scored_positions: dict[str, _PBHybridPosition] = {}
            open_scored_candidates.sort(
                key=lambda row: (
                    1 if bool(row.get("missing_5m")) else 0,
                    -float(((row.get("record") or row.get("candidate") or {}).get("daily_signal_score", 0.0)) or 0.0),
                    float(((row.get("record") or row.get("candidate") or {}).get("entry_rank_pct", 100.0)) or 100.0),
                )
            )
            open_scored_cap_reason = "open_route_cap"
            if state_by_symbol and max_total_open_scored == max(available_slots - intraday_priority_reserve, 0) and max_total_open_scored < available_slots:
                open_scored_cap_reason = "intraday_priority_reserve"
            for fallback in open_scored_candidates:
                record = fallback["record"]
                payload = record or fallback.get("candidate") or {}
                sector = str(fallback["sector"])
                missing_5m = bool(fallback.get("missing_5m"))
                symbol = str(fallback["symbol"])
                if len(open_scored_positions) >= max_total_open_scored:
                    if record is not None:
                        record["capacity_reason"] = open_scored_cap_reason
                        record["blocked_by_capacity_reason"] = record["capacity_reason"]
                        if missing_5m:
                            record["disposition"] = "intraday_priority_reserve" if open_scored_cap_reason == "intraday_priority_reserve" else "position_cap_reject"
                    continue
                if sector_counts.get(sector, 0) >= sector_cap:
                    if record is not None and missing_5m:
                        record["blocked_by_capacity_reason"] = "sector_cap"
                        self._record_rejection(record, "sector_cap_reject", rejection_log, shadow_outcomes, funnel_counters)
                    continue
                entry_time = self._market_open_timestamp(trade_date)
                entry_bar_idx = 0
                if open_scored_fill_timing == "next_5m_open" and not missing_5m:
                    fill_bar = fallback.get("fill_bar")
                    fill_bar_idx = int(fallback.get("fill_bar_idx", -1))
                    if not isinstance(fill_bar, Bar) or fill_bar_idx < 0:
                        if record is not None:
                            self._record_rejection(record, "open_scored_no_post_open_bar", rejection_log, shadow_outcomes, funnel_counters)
                        continue
                    if not self._apply_open_scored_delayed_fill(
                        payload,
                        fallback,
                        fill_bar_idx=fill_bar_idx,
                        fill_bar=fill_bar,
                    ):
                        if record is not None:
                            self._record_rejection(record, "sizing_reject", rejection_log, shadow_outcomes, funnel_counters)
                        continue
                    entry_time = fill_bar.start_time
                    entry_bar_idx = fill_bar_idx
                else:
                    payload["open_scored_fill_timing"] = "same_open"
                risk_per_share = float(payload.get("risk_per_share") or 0.0)
                entry_price = float(payload.get("entry_price") or 0.0)
                if risk_per_share <= 0 or entry_price <= 0:
                    if record is not None and missing_5m:
                        self._record_rejection(record, "sizing_reject", rejection_log, shadow_outcomes, funnel_counters)
                    continue
                risk_dollars = equity * settings.base_risk_fraction * _risk_budget_mult(trade_date, settings)
                if settings.pb_v2_enabled:
                    v2_mult = _v2_score_sizing_mult(
                        float(payload.get("daily_signal_score", 0.0)),
                        str(payload.get("trend_tier", "STRONG") or "STRONG"),
                        "OPEN_SCORED_ENTRY",
                        settings,
                    )
                    risk_dollars *= v2_mult
                if regime_tier == "B" and settings.t2_regime_b_sizing_mult != 1.0:
                    risk_dollars *= settings.t2_regime_b_sizing_mult
                if bool(payload.get("rescue_flow_candidate")):
                    risk_dollars *= float(getattr(settings, "pb_rescue_size_mult", 0.65))
                qty = int(floor(risk_dollars / risk_per_share)) if risk_dollars > 0 else 0
                if qty < 1:
                    if record is not None and missing_5m:
                        self._record_rejection(record, "sizing_reject", rejection_log, shadow_outcomes, funnel_counters)
                    continue
                if settings.intraday_leverage > 0:
                    carry_notional = sum(position.entry_price * position.quantity for position in carry_positions.values())
                    fallback_notional = sum(position.entry_price * position.quantity for position in open_scored_positions.values())
                    available_bp = equity * settings.intraday_leverage - carry_notional - fallback_notional
                    max_qty_bp = int(available_bp / entry_price) if entry_price > 0 else 0
                    qty = min(qty, max_qty_bp)
                    if qty < 1:
                        if record is not None and missing_5m:
                            record["blocked_by_capacity_reason"] = "buying_power"
                            self._record_rejection(record, "buying_power_reject", rejection_log, shadow_outcomes, funnel_counters)
                        continue
                position = self._build_daily_fallback_position(
                    symbol=symbol,
                    item=fallback["item"],
                    record=payload,
                    trade_date=trade_date,
                    regime_tier=regime_tier,
                    quantity=qty,
                    entry_time=entry_time,
                    entry_bar_idx=entry_bar_idx,
                )
                if position is None:
                    if record is not None and missing_5m:
                        self._record_rejection(record, "sizing_reject", rejection_log, shadow_outcomes, funnel_counters)
                    continue
                open_scored_positions[position.symbol] = position
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                if record is not None:
                    record["disposition"] = "entered"
                    record["intraday_data_available"] = not missing_5m
                    record["intraday_setup_type"] = "OPEN_SCORED_ENTRY"
                    record["entry_trigger"] = "OPEN_SCORED_ENTRY"
                    record["entry_route_family"] = "OPEN_SCORED_ENTRY"
                    record["selected_route"] = "OPEN_SCORED_ENTRY"
                    record["route_family"] = "OPEN_SCORED_ENTRY"
                    record["refinement_route"] = "OPEN_SCORED_ENTRY"
                    record["intraday_score"] = float(record.get("daily_signal_score") or 0.0)
                    record["route_score"] = float(record.get("daily_signal_score") or 0.0)
                    record["selection_reason"] = "daily_signal_score"
                    record["quantity"] = qty
                    record["entry_timestamp"] = position.entry_time
                    record["entry_bar_index"] = int(position.entry_bar_idx)
                    record["open_scored_fill_timing"] = payload.get("open_scored_fill_timing", open_scored_fill_timing)
                if funnel_counters is not None:
                    funnel_counters["entered"] = funnel_counters.get("entered", 0) + 1
                    funnel_counters["open_scored_entry"] = funnel_counters.get("open_scored_entry", 0) + 1

                # ---- parity: notify core of daily fallback entry ----
                _entry_oid = f"iaric-e-{position.symbol}-{self._order_counter}"
                self._order_counter += 1
                self._ensure_core_symbol(position.symbol, position.stop, "OPEN_SCORED_ENTRY")
                self._replay_core_step(
                    bar_input={
                        "bar_ts": position.entry_time,
                        "entry_request": IARICEntryRequest(
                            client_order_id=_entry_oid,
                            symbol=position.symbol,
                            route="OPEN_SCORED_ENTRY",
                            qty=position.quantity,
                            limit_price=position.entry_price,
                            stop_price=position.stop,
                        ),
                    },
                )
                self._replay_core_step(
                    fills=[
                        IARICFill(
                            oms_order_id=_entry_oid,
                            fill_price=position.entry_price,
                            fill_qty=position.quantity,
                            fill_time=position.entry_time,
                            commission=position.commission_entry,
                            symbol=position.symbol,
                            order_role="ENTRY",
                        )
                    ],
                )

            intraday_positions: dict[str, _PBHybridPosition] = {}
            # Delayed OPEN_SCORED fills use the 5m stream so pre-entry bars cannot affect risk.
            if settings.pb_v2_enabled or open_scored_fill_timing == "next_5m_open":
                for _os_sym in list(open_scored_positions):
                    if _os_sym in bars_by_symbol:
                        intraday_positions[_os_sym] = open_scored_positions.pop(_os_sym)
            rescue_entries_today = 0
            max_bars = max((len(bars) for bars in bars_by_symbol.values()), default=0)
            for bar_idx in range(max_bars):
                for symbol, bars in bars_by_symbol.items():
                    if bar_idx >= len(bars):
                        continue
                    bar = bars[bar_idx]
                    market = market_by_symbol[symbol]
                    market.last_price = bar.close
                    market.last_5m_bar = bar
                    market.bars_5m.append(bar)
                    market.session_high = bar.high if market.session_high is None else max(market.session_high, bar.high)
                    market.session_low = bar.low if market.session_low is None else min(market.session_low, bar.low)
                    # Incremental VWAP: O(1) per bar instead of O(N) re-sum
                    market._cum_pv += bar.typical_price * bar.volume
                    market._cum_vol += bar.volume
                    market.session_vwap = market._cum_pv / max(market._cum_vol, 1.0)
                    if (bar_idx + 1) % max(1, settings.pb_opening_range_bars) == 0:
                        agg = self._aggregate_30m_bar(
                            symbol,
                            list(market.bars_5m)[-settings.pb_opening_range_bars :],
                        )
                        if agg is not None:
                            market.last_30m_bar = agg
                            market.bars_30m.append(agg)

                queued_entry_candidates: list[dict[str, Any]] = []
                for symbol, state in state_by_symbol.items():
                    if state.stage != "ENTRY_QUEUED" or symbol in intraday_positions or symbol in open_scored_positions:
                        continue
                    bars = bars_by_symbol.get(symbol)
                    if bars is None or bar_idx >= len(bars) or bar_idx != state.accepted_bar_idx + 1:
                        continue
                    queued_entry_candidates.append({
                        "symbol": symbol,
                        "state": state,
                        "bar": bars[bar_idx],
                        "score": state.accepted_score,
                        "entry_trigger": state.accepted_entry_trigger,
                        "route_family": state.accepted_route_family,
                        "session_atr": state.accepted_session_atr,
                        "score_components": dict(state.accepted_score_components),
                    })

                queued_entry_candidates.sort(
                    key=lambda row: (
                        -float(row["score"]),
                        float(row["state"].entry_rank_pct),
                        float(row["state"].entry_rsi),
                    )
                )
                for candidate in queued_entry_candidates:
                    state = candidate["state"]
                    symbol = str(candidate["symbol"])
                    if state.stage != "ENTRY_QUEUED" or symbol in intraday_positions or symbol in open_scored_positions:
                        continue

                    bar = candidate["bar"]
                    record = state.record
                    sector = state.sector
                    if len(intraday_positions) + len(open_scored_positions) >= available_slots:
                        state.priority_skip_count += 1
                        if record is not None and not record.get("blocked_by_capacity_reason"):
                            record["blocked_by_capacity_reason"] = "slot_cap"
                        self._invalidate_state(
                            state,
                            record,
                            fsm_log,
                            trade_date,
                            bar.start_time,
                            "slot_cap_reject",
                            max_bars + 1,
                        )
                        continue
                    if sector_counts.get(sector, 0) >= sector_cap:
                        state.priority_skip_count += 1
                        if record is not None and not record.get("blocked_by_capacity_reason"):
                            record["blocked_by_capacity_reason"] = "sector_cap"
                        self._invalidate_state(
                            state,
                            record,
                            fsm_log,
                            trade_date,
                            bar.start_time,
                            "sector_cap_reject",
                            max_bars + 1,
                        )
                        continue
                    if state.rescue_flow_candidate and rescue_entries_today >= settings.pb_rescue_max_per_day:
                        state.priority_skip_count += 1
                        if record is not None and not record.get("blocked_by_capacity_reason"):
                            record["blocked_by_capacity_reason"] = "rescue_cap"
                        self._invalidate_state(
                            state,
                            record,
                            fsm_log,
                            trade_date,
                            bar.start_time,
                            "rescue_cap_reject",
                            max_bars + 1,
                        )
                        continue

                    entry_price = float(bar.open)
                    stop_level = self._initial_stop(state, float(candidate["session_atr"]))
                    slip = entry_price * self._slippage.slip_bps_normal / 10_000
                    fill_price = round(entry_price + slip, 2)
                    risk_per_share = fill_price - stop_level
                    if risk_per_share <= 0:
                        self._invalidate_state(
                            state,
                            record,
                            fsm_log,
                            trade_date,
                            bar.start_time,
                            "sizing_reject",
                            max_bars + 1,
                        )
                        continue

                    risk_dollars = equity * settings.base_risk_fraction * _risk_budget_mult(trade_date, settings)
                    if settings.pb_v2_enabled:
                        v2_mult = _v2_score_sizing_mult(
                            state.daily_signal_score,
                            state.trend_tier,
                            str(candidate["route_family"]),
                            settings,
                        )
                        risk_dollars *= v2_mult
                    elif settings.pb_entry_strength_sizing:
                        risk_dollars *= min(max(float(candidate["score"]) / 80.0, 0.6), 1.4)
                    if regime_tier == "B" and settings.t2_regime_b_sizing_mult != 1.0:
                        risk_dollars *= settings.t2_regime_b_sizing_mult
                    if state.rescue_flow_candidate:
                        risk_dollars *= 0.65
                    qty = int(floor(risk_dollars / risk_per_share))
                    if qty < 1:
                        self._invalidate_state(
                            state,
                            record,
                            fsm_log,
                            trade_date,
                            bar.start_time,
                            "sizing_reject",
                            max_bars + 1,
                        )
                        continue

                    if settings.intraday_leverage > 0:
                        carry_notional = sum(position.entry_price * position.quantity for position in carry_positions.values())
                        fallback_notional = sum(position.entry_price * position.quantity for position in open_scored_positions.values())
                        intraday_notional = sum(position.entry_price * position.quantity for position in intraday_positions.values())
                        available_bp = equity * settings.intraday_leverage - carry_notional - fallback_notional - intraday_notional
                        max_qty_bp = int(available_bp / entry_price) if entry_price > 0 else 0
                        qty = min(qty, max_qty_bp)
                        if qty < 1:
                            if record is not None:
                                record["blocked_by_capacity_reason"] = "buying_power"
                            self._invalidate_state(
                                state,
                                record,
                                fsm_log,
                                trade_date,
                                bar.start_time,
                                "buying_power_reject",
                                max_bars + 1,
                            )
                            continue

                    commission = self._slippage.commission_per_share * qty
                    is_pm_reentry = bool(state.stopped_out_today and settings.pb_pm_reentry)
                    reentry_count = state.reentry_count + 1 if is_pm_reentry else state.reentry_count
                    micro_signal = self._micropressure_label(
                        bars_by_symbol[symbol],
                        min(bar_idx, len(bars_by_symbol[symbol]) - 1),
                        state.reclaim_level,
                        state.item,
                    )
                    position = _PBHybridPosition(
                        symbol=symbol,
                        entry_price=fill_price,
                        entry_time=bar.start_time,
                        quantity=qty,
                        risk_per_share=risk_per_share,
                        sector=sector,
                        regime_tier=regime_tier,
                        stop=stop_level,
                        current_stop=stop_level,
                        trigger_type=state.trigger_type,
                        entry_rsi=state.entry_rsi,
                        entry_gap_pct=state.entry_gap_pct,
                        entry_sma_dist_pct=state.entry_sma_dist_pct,
                        entry_cdd=state.entry_cdd,
                        entry_rank=state.entry_rank,
                        entry_rank_pct=state.entry_rank_pct,
                        n_candidates=state.n_candidates,
                        daily_signal_score=state.daily_signal_score,
                        daily_signal_rank_pct=state.daily_signal_rank_pct,
                        signal_family="v2" if settings.pb_v2_enabled else str(getattr(self._settings, "pb_daily_signal_family", "balanced_v1")),
                        intraday_setup_type=state.intraday_setup_type,
                        entry_trigger=str(candidate["entry_trigger"]),
                        route_family=str(candidate["route_family"]),
                        carry_profile=self._route_carry_profile(str(candidate["route_family"])),
                        selection_reason="pm_reentry" if is_pm_reentry else "route_confirmation",
                        intraday_score=float(candidate["score"]),
                        reclaim_bars=max(state.ready_bar_idx - state.flush_bar_idx + 1, 1),
                        rescue_flow_candidate=state.rescue_flow_candidate,
                        reentry_count=reentry_count,
                        entry_atr=float(candidate["session_atr"]),
                        item=state.item,
                        ready_timestamp=state.ready_timestamp,
                        accepted_timestamp=state.accepted_timestamp,
                        accepted_bar_idx=state.accepted_bar_idx,
                        accepted_entry_price=state.accepted_entry_price,
                        accepted_intraday_score=state.accepted_score,
                        accepted_entry_trigger=state.accepted_entry_trigger,
                        acceptance_count=state.acceptance_count,
                        required_acceptance=state.required_acceptance,
                        micropressure_signal=micro_signal,
                        trigger_types=list(state.trigger_types),
                        trigger_tier=state.trigger_tier,
                        trend_tier=state.trend_tier,
                        max_favorable=max(fill_price, bar.high),
                        max_adverse=min(fill_price, bar.low),
                        highest_close=bar.close,
                        commission_entry=commission,
                        slippage_entry=slip * qty,
                        entry_bar_idx=bar_idx,
                        ledger_ref=record,
                        score_components=dict(candidate["score_components"]),
                    )
                    intraday_positions[symbol] = position
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    prior = state.stage
                    state.stage = "IN_POSITION"
                    state.accepted_bar_idx = -1
                    state.accepted_timestamp = None
                    state.accepted_entry_price = 0.0
                    state.accepted_entry_trigger = ""
                    state.accepted_route_family = ""
                    state.accepted_score = 0.0
                    state.accepted_session_atr = 0.0
                    state.accepted_score_components = {}
                    if is_pm_reentry:
                        state.reentry_count = reentry_count
                    state.stopped_out_today = False
                    if record is not None:
                        record["disposition"] = "entered"
                        record["quantity"] = qty
                        record["entry_timestamp"] = bar.start_time
                        record["entry_bar_index"] = int(bar_idx)
                        record["entry_trigger"] = position.entry_trigger
                        record["entry_route_family"] = position.route_family
                        record["selected_route"] = position.route_family
                        record["route_family"] = position.route_family
                        record["intraday_score"] = round(position.intraday_score, 2)
                        record["route_score"] = round(position.intraday_score, 2)
                        record["reclaim_bars"] = position.reclaim_bars
                        record["rescue_flow_candidate"] = position.rescue_flow_candidate
                        record["refinement_route"] = position.route_family
                        record["selection_reason"] = position.selection_reason
                        self._apply_score_components(record, position.score_components, prefix="score_component_")
                    if funnel_counters is not None:
                        funnel_counters["entered"] = funnel_counters.get("entered", 0) + 1
                        if position.rescue_flow_candidate:
                            funnel_counters["rescue_entered"] = funnel_counters.get("rescue_entered", 0) + 1
                        if position.reentry_count > 0:
                            funnel_counters["pm_reentry"] = funnel_counters.get("pm_reentry", 0) + 1
                    if position.rescue_flow_candidate:
                        rescue_entries_today += 1
                    if fsm_log is not None:
                        self._log_fsm(
                            fsm_log,
                            symbol,
                            trade_date,
                            bar.start_time,
                            prior,
                            "IN_POSITION",
                            position.entry_trigger.lower(),
                            score=position.intraday_score,
                        )

                    # ---- parity: notify core of intraday entry ----
                    _entry_oid = f"iaric-e-{symbol}-{self._order_counter}"
                    self._order_counter += 1
                    self._ensure_core_symbol(symbol, position.stop, position.route_family)
                    self._replay_core_step(
                        bar_input={
                            "bar_ts": bar.start_time,
                            "entry_request": IARICEntryRequest(
                                client_order_id=_entry_oid,
                                symbol=symbol,
                                route=position.route_family,
                                qty=position.quantity,
                                limit_price=position.entry_price,
                                stop_price=position.stop,
                            ),
                        },
                    )
                    self._replay_core_step(
                        fills=[
                            IARICFill(
                                oms_order_id=_entry_oid,
                                fill_price=position.entry_price,
                                fill_qty=position.quantity,
                                fill_time=bar.start_time,
                                commission=position.commission_entry,
                                symbol=symbol,
                                order_role="ENTRY",
                            )
                        ],
                    )

                closed_symbols: list[str] = []
                for symbol, position in list(intraday_positions.items()):
                    bars = bars_by_symbol.get(symbol)
                    if bars is None or bar_idx >= len(bars):
                        continue
                    if bar_idx < position.entry_bar_idx:
                        continue
                    bar = bars[bar_idx]
                    market = market_by_symbol[symbol]
                    state = state_by_symbol[symbol]

                    position.hold_bars += 1
                    if bar.high > position.max_favorable + 1e-9:
                        position.max_favorable = bar.high
                        position.bars_to_mfe = max(position.hold_bars, 1)
                    position.max_adverse = min(position.max_adverse, bar.low) if position.max_adverse > 0 else bar.low
                    position.highest_close = max(position.highest_close, bar.close)
                    position.close_r = position.unrealized_r(bar.close)
                    position.close_pct = _close_in_range_pct(
                        market.session_high if market.session_high is not None else bar.high,
                        market.session_low if market.session_low is not None else bar.low,
                        bar.close,
                    )

                    exit_price: float | None = None
                    exit_reason = ""
                    _v2 = settings.pb_v2_enabled
                    quick_exit_loss_r = abs(float(self._route_setting(position.route_family, "quick_exit_loss_r", "pb_opening_reclaim_quick_exit_loss_r")))
                    stale_exit_bars = int(self._route_setting(position.route_family, "stale_exit_bars", "pb_stale_exit_bars"))
                    stale_exit_min_r = float(self._route_setting(position.route_family, "stale_exit_min_r", "pb_stale_exit_min_r"))
                    partial_r = float(self._route_setting(position.route_family, "partial_r", "pb_partial_r"))
                    breakeven_r = float(self._route_setting(position.route_family, "breakeven_r", "pb_breakeven_r"))
                    trail_activate_r = float(self._route_setting(position.route_family, "trail_activate_r", "pb_trail_activate_r"))
                    if bar.low <= position.current_stop:
                        exit_price = position.current_stop
                        exit_reason = "STOP_HIT"
                    elif quick_exit_loss_r > 0 and position.hold_bars <= 2 and position.unrealized_r(bar.close) <= -quick_exit_loss_r and (market.session_vwap or bar.close) > bar.close:
                        exit_price = bar.close
                        exit_reason = "QUICK_EXIT"
                    elif stale_exit_bars > 0 and position.hold_bars >= stale_exit_bars and position.mfe_r() < stale_exit_min_r:
                        exit_price = bar.close
                        exit_reason = "STALE_EXIT"

                    # V2: EMA reversion exit on 5m bars
                    if exit_price is None and _v2 and settings.pb_v2_ema_reversion_exit:
                        p_ind = self._indicators.get(symbol)
                        p_didx = self._date_iloc.get(symbol)
                        if p_ind is not None and p_didx is not None:
                            p_iloc = p_didx.get(prev_date, -1)
                            ema10_arr = p_ind.get("ema10")
                            if ema10_arr is not None and p_iloc >= 0 and not np.isnan(ema10_arr[p_iloc]):
                                ema10_val = float(ema10_arr[p_iloc])
                                if bar.close >= ema10_val and position.unrealized_r(bar.close) > settings.pb_v2_ema_reversion_min_r:
                                    exit_price = bar.close
                                    exit_reason = "EMA_REVERSION"


                    # V2: partial profit at configured MFE
                    if exit_price is None and _v2 and not position.v2_partial_taken:
                        v2_partial_r = float(settings.pb_v2_partial_profit_trigger_r)
                        if v2_partial_r > 0 and position.mfe_r() >= v2_partial_r:
                            original_qty = position.quantity + position.partial_qty_exited
                            min_remaining = max(1, int(floor(original_qty * settings.minimum_remaining_size_pct)))
                            partial_qty = min(
                                max(1, position.quantity // 2),
                                max(position.quantity - min_remaining, 0),
                            )
                            if 1 <= partial_qty < position.quantity:
                                position.v2_partial_taken = True
                                position.partial_taken = True
                                position.partial_qty_exited += partial_qty
                                partial_commission = self._slippage.commission_per_share * partial_qty
                                # Fill at the R-level trigger price (matching V1 approach)
                                v2_trigger_price = position.entry_price + v2_partial_r * position.risk_per_share
                                partial_slip = v2_trigger_price * self._slippage.slip_bps_normal / 10_000
                                partial_fill = round(v2_trigger_price - partial_slip, 2)
                                position.realized_partial_commission += partial_commission
                                position.realized_partial_slippage += partial_slip * partial_qty
                                partial_pnl = (partial_fill - position.entry_price) * partial_qty
                                position.realized_partial_pnl += partial_pnl
                                position.quantity -= partial_qty
                                equity += partial_pnl - partial_commission
                                remainder_stop = position.entry_price + settings.pb_v2_partial_profit_remainder_stop_r * position.risk_per_share
                                position.current_stop = max(position.current_stop, remainder_stop)
                                if funnel_counters is not None:
                                    funnel_counters["partial"] = funnel_counters.get("partial", 0) + 1
                                if position.ledger_ref is not None:
                                    position.ledger_ref["partial_taken"] = True
                                # ---- parity: notify core of V2 partial exit ----
                                _partial_oid = f"iaric-p-{symbol}-{self._order_counter}"
                                self._order_counter += 1
                                self._replay_core_step(
                                    bar_input={
                                        "bar_ts": bar.start_time,
                                        "partial_exit_request": IARICPartialExitRequest(
                                            client_order_id=_partial_oid,
                                            symbol=symbol,
                                            qty=partial_qty,
                                            reason="V2_PARTIAL_PROFIT",
                                        ),
                                    },
                                    fills=[
                                        IARICFill(
                                            oms_order_id=_partial_oid,
                                            fill_price=partial_fill,
                                            fill_qty=partial_qty,
                                            fill_time=bar.start_time,
                                            commission=partial_commission,
                                            symbol=symbol,
                                            order_role="TP",
                                            exit_type="V2_PARTIAL_PROFIT",
                                        )
                                    ],
                                )
                    elif exit_price is None and not _v2 and not position.partial_taken:
                        partial_trigger = position.entry_price + partial_r * position.risk_per_share
                        if bar.high >= partial_trigger:
                            original_qty = position.quantity + position.partial_qty_exited
                            min_remaining = max(1, int(floor(original_qty * settings.minimum_remaining_size_pct)))
                            partial_qty = min(
                                max(1, int(floor(original_qty * settings.pb_partial_frac))),
                                max(position.quantity - min_remaining, 0),
                            )
                            if 1 <= partial_qty < position.quantity:
                                position.partial_taken = True
                                position.partial_qty_exited += partial_qty
                                partial_commission = self._slippage.commission_per_share * partial_qty
                                partial_slip = partial_trigger * self._slippage.slip_bps_normal / 10_000
                                partial_fill = round(partial_trigger - partial_slip, 2)
                                position.realized_partial_commission += partial_commission
                                position.realized_partial_slippage += partial_slip * partial_qty
                                partial_pnl = (partial_fill - position.entry_price) * partial_qty
                                position.realized_partial_pnl += partial_pnl
                                position.quantity -= partial_qty
                                equity += partial_pnl - partial_commission
                                if funnel_counters is not None:
                                    funnel_counters["partial"] = funnel_counters.get("partial", 0) + 1
                                if position.ledger_ref is not None:
                                    position.ledger_ref["partial_taken"] = True
                                # ---- parity: notify core of partial exit ----
                                _partial_oid = f"iaric-p-{symbol}-{self._order_counter}"
                                self._order_counter += 1
                                self._replay_core_step(
                                    bar_input={
                                        "bar_ts": bar.start_time,
                                        "partial_exit_request": IARICPartialExitRequest(
                                            client_order_id=_partial_oid,
                                            symbol=symbol,
                                            qty=partial_qty,
                                            reason="PARTIAL_PROFIT",
                                        ),
                                    },
                                    fills=[
                                        IARICFill(
                                            oms_order_id=_partial_oid,
                                            fill_price=partial_fill,
                                            fill_qty=partial_qty,
                                            fill_time=bar.start_time,
                                            commission=partial_commission,
                                            symbol=symbol,
                                            order_role="TP",
                                            exit_type="PARTIAL_PROFIT",
                                        )
                                    ],
                                )

                    # V2: 3-stage MFE protection
                    if exit_price is None and _v2:
                        mfe = position.mfe_r()
                        if mfe >= settings.pb_v2_mfe_stage3_trigger and position.mfe_stage < 3:
                            position.mfe_stage = 3
                            trail_stop = bar.high - settings.pb_v2_mfe_stage3_trail_atr * max(position.entry_atr, 0.01)
                            position.current_stop = max(position.current_stop, trail_stop)
                            position.trail_active = True
                        elif mfe >= settings.pb_v2_mfe_stage2_trigger and position.mfe_stage < 2:
                            position.mfe_stage = 2
                            position.current_stop = max(position.current_stop, position.entry_price)
                            position.breakeven_activated = True
                        elif mfe >= settings.pb_v2_mfe_stage1_trigger and position.mfe_stage < 1:
                            position.mfe_stage = 1
                            protect_stop = position.entry_price + settings.pb_v2_mfe_stage1_stop_r * position.risk_per_share
                            position.current_stop = max(position.current_stop, protect_stop)
                        # Stage 3 trailing update each bar
                        if position.mfe_stage >= 3:
                            trail_stop = bar.high - settings.pb_v2_mfe_stage3_trail_atr * max(position.entry_atr, 0.01)
                            prior_stop = position.current_stop
                            position.current_stop = max(position.current_stop, trail_stop)
                            if position.current_stop > prior_stop + 1e-9 and funnel_counters is not None:
                                funnel_counters["trailed"] = funnel_counters.get("trailed", 0) + 1
                    else:
                        # Legacy MFE protection
                        if exit_price is None:
                            protect_trigger_r = float(self._route_setting(position.route_family, "mfe_protect_trigger_r", "pb_mfe_protect_trigger_r"))
                            protect_stop_r = float(self._route_setting(position.route_family, "mfe_protect_stop_r", "pb_mfe_protect_stop_r"))
                            if protect_trigger_r > 0 and position.mfe_r() >= protect_trigger_r:
                                protect_stop = position.entry_price + protect_stop_r * position.risk_per_share
                                position.current_stop = max(position.current_stop, protect_stop)

                        if exit_price is None and position.mfe_r() >= breakeven_r:
                            position.current_stop = max(position.current_stop, position.entry_price)
                            position.breakeven_activated = True
                        if exit_price is None and position.mfe_r() >= trail_activate_r:
                            recent = bars[max(0, bar_idx - 2) : bar_idx + 1]
                            if recent:
                                higher_low = max(sample.low for sample in recent)
                                trail_stop = higher_low - settings.pb_trail_atr_mult * max(position.entry_atr, 0.01)
                                prior_stop = position.current_stop
                                position.current_stop = max(position.current_stop, trail_stop)
                                position.trail_active = position.current_stop > prior_stop + 1e-9 or position.trail_active
                                if position.trail_active and funnel_counters is not None and position.current_stop > prior_stop + 1e-9:
                                    funnel_counters["trailed"] = funnel_counters.get("trailed", 0) + 1

                    # V2: stale position tighten
                    if exit_price is None and _v2 and position.hold_bars >= settings.pb_v2_stale_bars and position.mfe_r() < settings.pb_v2_stale_mfe_thresh:
                        tighten_stop = position.entry_price - (1.0 - settings.pb_v2_stale_tighten_pct) * position.risk_per_share
                        position.current_stop = max(position.current_stop, tighten_stop)

                    # VWAP failure exit
                    if exit_price is None and self._should_exit_for_vwap_fail(position, bars, bar_idx, market):
                        exit_price = bar.close
                        exit_reason = "VWAP_FAIL"

                    if exit_price is not None:
                        trade, eq_delta = self._close_position(position, float(exit_price), bar.end_time, exit_reason)
                        trades.append(trade)
                        equity += eq_delta
                        self._attach_hybrid_trade_outcome(position, trade)
                        closed_symbols.append(symbol)
                        state.stopped_out_today = exit_reason in {"STOP_HIT", "QUICK_EXIT", "VWAP_FAIL"}
                        if state.stopped_out_today and settings.pb_pm_reentry and state.reentry_count < settings.pb_max_reentries_per_day:
                            self._invalidate_state(
                                state,
                                state.record,
                                fsm_log,
                                trade_date,
                                bar.end_time,
                                exit_reason.lower(),
                                max(bar_idx + 2, settings.pb_pm_reentry_after_bar),
                            )
                        else:
                            state.stage = "INVALIDATED"
                            state.invalid_reset_bar = max_bars + 1

                for symbol in closed_symbols:
                    sector = intraday_positions[symbol].sector
                    intraday_positions.pop(symbol, None)
                    if sector in sector_counts:
                        sector_counts[sector] = max(sector_counts[sector] - 1, 0)

                entry_acceptances: list[dict[str, Any]] = []
                for symbol, state in state_by_symbol.items():
                    if symbol in intraday_positions:
                        continue
                    bars = bars_by_symbol.get(symbol)
                    if bars is None or bar_idx >= len(bars):
                        continue
                    bar = bars[bar_idx]
                    market = market_by_symbol[symbol]
                    record = state.record
                    now_et = bar.end_time.astimezone(ET).time()
                    session_atr = session_atr_by_symbol[symbol]

                    if state.stage == "INVALIDATED":
                        if iaric_core_logic.maybe_reset_invalidated_state(state, bar_idx):
                            if fsm_log is not None:
                                self._log_fsm(fsm_log, symbol, trade_date, bar.end_time, "INVALIDATED", "WATCHING", "cooldown_reset")
                        else:
                            continue

                    if state.stage == "WATCHING":
                        can_try_opening_reclaim = (
                            self._route_enabled("OPENING_RECLAIM")
                            and state.daily_signal_score >= self._route_min_daily_signal_score("OPENING_RECLAIM")
                        )
                        if can_try_opening_reclaim:
                            step = iaric_core_logic.advance_opening_reclaim_route(
                                settings,
                                state,
                                state.item,
                                bar,
                                market,
                                bar_idx,
                                session_atr,
                                bars=bars,
                            )
                        else:
                            step = None
                        if step is not None and step.stage == "FLUSH_LOCKED":
                            self._mark_stage(record, "FLUSH_LOCKED")
                            if funnel_counters is not None:
                                funnel_counters["flush_locked"] = funnel_counters.get("flush_locked", 0) + 1
                            if record is not None:
                                record["intraday_setup_type"] = state.intraday_setup_type
                            if fsm_log is not None:
                                self._log_fsm(fsm_log, symbol, trade_date, bar.end_time, step.prior_stage, "FLUSH_LOCKED", step.reason or state.intraday_setup_type)
                            continue
                        if self._activate_delayed_confirm(
                            state,
                            market,
                            bars,
                            bar_idx,
                            session_atr,
                            trade_date,
                            record,
                            funnel_counters,
                            fsm_log,
                        ):
                            pass
                        elif self._activate_vwap_bounce(
                            state, market, bars, bar_idx, session_atr, trade_date, record, funnel_counters, fsm_log,
                        ):
                            pass
                        elif self._activate_afternoon_retest(
                            state, market, bars, bar_idx, session_atr, trade_date, record, funnel_counters, fsm_log,
                        ):
                            pass
                        else:
                            continue

                    if state.stage == "FLUSH_LOCKED":
                        step = iaric_core_logic.advance_opening_reclaim_route(
                            settings,
                            state,
                            state.item,
                            bar,
                            market,
                            bar_idx,
                            session_atr,
                            bars=bars,
                        )
                        if step is not None and step.stage == "RECLAIMING":
                            self._mark_stage(record, "RECLAIMING")
                            if funnel_counters is not None:
                                funnel_counters["reclaiming"] = funnel_counters.get("reclaiming", 0) + 1
                            if fsm_log is not None:
                                self._log_fsm(fsm_log, symbol, trade_date, bar.end_time, step.prior_stage, "RECLAIMING", step.reason or "reclaim_hit")
                        elif step is not None and step.stage == "INVALIDATED":
                            if record is not None:
                                record["intraday_invalid_reason"] = state.invalid_reason
                            self._mark_stage(record, "INVALIDATED")
                            if fsm_log is not None:
                                self._log_fsm(fsm_log, symbol, trade_date, bar.end_time, step.prior_stage, "INVALIDATED", step.reason)
                        continue

                    if state.stage == "RECLAIMING":
                        step = iaric_core_logic.advance_opening_reclaim_route(
                            settings,
                            state,
                            state.item,
                            bar,
                            market,
                            bar_idx,
                            session_atr,
                            bars=bars,
                        )
                        if step is not None and step.stage == "INVALIDATED":
                            if record is not None:
                                record["intraday_invalid_reason"] = state.invalid_reason
                            self._mark_stage(record, "INVALIDATED")
                            if fsm_log is not None:
                                self._log_fsm(fsm_log, symbol, trade_date, bar.end_time, step.prior_stage, "INVALIDATED", step.reason)
                            continue
                        if step is not None and step.stage == "READY":
                            self._mark_stage(record, "READY")
                            if funnel_counters is not None:
                                funnel_counters["ready"] = funnel_counters.get("ready", 0) + 1
                            if record is not None:
                                record["intraday_score"] = round(state.intraday_score, 2)
                                record["reclaim_bars"] = max(bar_idx - state.flush_bar_idx + 1, 1)
                                record["selection_refine_score"] = round(state.intraday_score, 2)
                                record["ready_timestamp"] = state.ready_timestamp
                                record["ready_bar_index"] = int(state.ready_bar_idx)
                                record["ready_cpr"] = round(state.ready_cpr, 4)
                                record["ready_volume_ratio"] = round(state.ready_volume_ratio, 4)
                                record["refinement_route"] = state.route_family or ("PM_REENTRY" if state.stopped_out_today else "OPENING_RECLAIM")
                                self._apply_score_components(record, state.score_components, prefix="score_component_")
                            if fsm_log is not None:
                                self._log_fsm(
                                    fsm_log,
                                    symbol,
                                    trade_date,
                                    bar.end_time,
                                    step.prior_stage,
                                    "READY",
                                    step.reason or "acceptance_complete",
                                    score=state.intraday_score,
                                )
                        continue

                    if state.stage != "READY":
                        continue
                    if now_et < settings.pb_intraday_entry_start or now_et > settings.pb_intraday_entry_end:
                        continue
                    next_bar_idx = bar_idx + 1
                    if next_bar_idx >= len(bars):
                        continue
                    step = iaric_core_logic.evaluate_ready_entry(
                        settings,
                        state,
                        state.item,
                        bar,
                        market,
                        bar_idx,
                        session_atr,
                        bars=bars,
                    )
                    if step is not None and step.stage == "INVALIDATED":
                        if record is not None:
                            record["intraday_invalid_reason"] = state.invalid_reason
                        self._mark_stage(record, "INVALIDATED")
                        if fsm_log is not None:
                            self._log_fsm(fsm_log, symbol, trade_date, bar.end_time, step.prior_stage, "INVALIDATED", step.reason)
                        continue
                    if step is not None and step.entry_feasible and record is not None:
                        record["entry_window_feasible"] = True
                        feasible_idx = record.get("entry_window_feasible_bar_index")
                        if feasible_idx is None or int(feasible_idx) < 0 or bar_idx < int(feasible_idx):
                            record["entry_window_feasible_bar_index"] = int(bar_idx)
                        record["max_feasible_intraday_score"] = max(
                            float(record.get("max_feasible_intraday_score", 0.0) or 0.0),
                            float(state.intraday_score),
                        )
                    if step is not None and step.acceptance is not None:
                        entry_acceptances.append({
                            "symbol": symbol,
                            "state": state,
                            "bar": bar,
                            "accepted_bar_idx": int(step.acceptance.accepted_bar_idx),
                            "accepted_entry_price": float(step.acceptance.accepted_entry_price),
                            "entry_trigger": step.acceptance.entry_trigger,
                            "route_family": step.acceptance.route_family,
                            "score": float(step.acceptance.score),
                            "session_atr": float(step.acceptance.session_atr),
                            "score_components": dict(step.acceptance.score_components),
                        })

                entry_acceptances.sort(
                    key=lambda row: (
                        -float(row["score"]),
                        float(row["state"].entry_rank_pct),
                        float(row["state"].entry_rsi),
                    )
                )
                for candidate in entry_acceptances:
                    state = candidate["state"]
                    symbol = str(candidate["symbol"])
                    if state.stage != "READY" or symbol in intraday_positions or symbol in open_scored_positions:
                        continue
                    record = state.record
                    prior = state.stage
                    state.stage = "ENTRY_QUEUED"
                    iaric_core_logic.apply_entry_acceptance(
                        state,
                        iaric_core_logic.IARICEntryAcceptance(
                            accepted_bar_idx=int(candidate["accepted_bar_idx"]),
                            accepted_timestamp=candidate["bar"].end_time,
                            accepted_entry_price=float(candidate["accepted_entry_price"]),
                            entry_trigger=str(candidate["entry_trigger"]),
                            route_family=str(candidate["route_family"]),
                            score=float(candidate["score"]),
                            session_atr=float(candidate["session_atr"]),
                            score_components=dict(candidate["score_components"]),
                        ),
                    )
                    if record is not None:
                        record["disposition"] = "entry_queued"
                        record["accepted_timestamp"] = state.accepted_timestamp
                        record["accepted_bar_index"] = state.accepted_bar_idx
                        record["accepted_entry_price"] = round(state.accepted_entry_price, 2)
                        record["accepted_entry_trigger"] = state.accepted_entry_trigger
                        record["accepted_route_family"] = state.accepted_route_family
                        record["accepted_intraday_score"] = round(state.accepted_score, 2)
                        record["selection_reason"] = "next_bar_open_fill"
                    self._mark_stage(record, "ENTRY_QUEUED")
                    if fsm_log is not None:
                        self._log_fsm(
                            fsm_log,
                            symbol,
                            trade_date,
                            candidate["bar"].end_time,
                            prior,
                            "ENTRY_QUEUED",
                            "next_bar_open_fill",
                            score=state.accepted_score,
                        )

            pending_dispositions = {
                "triggered",
                "candidate_pool",
                "watchlist",
                "rescue_watchlist",
                "flow_rescue_pool",
                "entry_queued",
            }
            for symbol, state in state_by_symbol.items():
                if symbol in intraday_positions:
                    continue
                record = state.record
                if record is None or record.get("disposition") == "entered":
                    continue
                if str(record.get("disposition") or "") not in pending_dispositions:
                    continue
                if state.priority_skip_count > 0 and state.stage == "READY":
                    gate = "rescue_cap_reject" if state.rescue_flow_candidate and rescue_entries_today >= settings.pb_rescue_max_per_day else "priority_reject"
                    if not record.get("blocked_by_capacity_reason"):
                        record["blocked_by_capacity_reason"] = "rescue_cap" if gate == "rescue_cap_reject" else "slot_cap"
                    self._record_rejection(record, gate, rejection_log, shadow_outcomes, funnel_counters)
                elif state.stage == "WATCHING":
                    self._record_rejection(record, "no_intraday_setup", rejection_log, shadow_outcomes, funnel_counters)
                elif state.stage in {"FLUSH_LOCKED", "RECLAIMING"}:
                    self._record_rejection(record, "never_ready", rejection_log, shadow_outcomes, funnel_counters)
                elif state.stage == "INVALIDATED":
                    self._record_rejection(
                        record,
                        state.invalid_reason or "intraday_invalidated",
                        rejection_log,
                        shadow_outcomes,
                        funnel_counters,
                    )
                elif state.stage in {"READY", "ENTRY_QUEUED"}:
                    self._record_rejection(record, "entry_window_expired", rejection_log, shadow_outcomes, funnel_counters)

            for symbol, position in list(open_scored_positions.items()):
                equity = self._manage_daily_fallback_position(
                    position,
                    trade_date,
                    carry_positions,
                    trades,
                    equity,
                    funnel_counters,
                )
                open_scored_positions.pop(symbol, None)

            for symbol, position in list(intraday_positions.items()):
                bars = bars_by_symbol[symbol]
                market = market_by_symbol[symbol]
                last_bar = bars[-1]
                ind = self._indicators.get(symbol)
                didx = self._replay._daily_didx.get(symbol)
                if ind is not None and didx is not None:
                    iloc = _iloc_for_date(didx, trade_date)
                    if iloc >= 0 and not np.isnan(ind["rsi"][iloc]):
                        position.exit_rsi = float(ind["rsi"][iloc])
                position.close_r = position.unrealized_r(last_bar.close)
                position.close_pct = _close_in_range_pct(
                    market.session_high if market.session_high is not None else last_bar.high,
                    market.session_low if market.session_low is not None else last_bar.low,
                    last_bar.close,
                )

                if settings.pb_v2_enabled:
                    # V2: inverted carry -- default is CARRY, flatten only when conditions met
                    if not has_next_backtest_day:
                        position.carry_decision_path = "no_next_day"
                        trade, eq_delta = self._close_position(position, last_bar.close, last_bar.end_time, "EOD_FLATTEN")
                        trades.append(trade)
                        equity += eq_delta
                        self._attach_hybrid_trade_outcome(position, trade)
                    else:
                        flow_last_n = self._replay.get_flow_proxy_last_n(
                            symbol, trade_date,
                            max(1, int(self._route_setting(position.route_family, "flow_reversal_lookback", "pb_flow_reversal_lookback"))),
                        )
                        should_flat, flat_reason = _should_flatten_v2(
                            position, last_bar.close, position.close_pct, regime_tier,
                            flow_last_n, settings,
                        )
                        if should_flat:
                            position.carry_decision_path = flat_reason
                            trade, eq_delta = self._close_position(position, last_bar.close, last_bar.end_time, "EOD_FLATTEN")
                            trades.append(trade)
                            equity += eq_delta
                            self._attach_hybrid_trade_outcome(position, trade)
                        else:
                            # V2 carry quality gate (reuse per-route carry params)
                            route_family = position.route_family
                            v2_close_min = float(self._route_setting(route_family, "carry_close_pct_min", "pb_carry_close_pct_min"))
                            v2_mfe_min = float(self._route_setting(route_family, "carry_mfe_gate_r", "pb_carry_mfe_gate_r"))
                            regime_carry_mult = 1.0
                            if regime_tier == "B":
                                regime_carry_mult = settings.regime_b_carry_mult
                            quality_ok = regime_carry_mult > 0 and (
                                position.close_pct >= v2_close_min
                                and position.mfe_r() >= v2_mfe_min
                            )
                            if quality_ok:
                                profit_lock_r = float(self._route_setting(route_family, "carry_profit_lock_r", "pb_v2_carry_profit_lock_r"))
                                overnight_stop = position.entry_price + max(0.0, position.close_r - profit_lock_r) * position.risk_per_share
                                position.current_stop = max(position.current_stop, overnight_stop)
                                position.carry_decision_path = "v2_carry"
                                position.carry_binary_ok = True
                                carry_positions[symbol] = position
                                if funnel_counters is not None:
                                    funnel_counters["carried"] = funnel_counters.get("carried", 0) + 1
                            else:
                                position.carry_decision_path = "v2_quality_reject"
                                trade, eq_delta = self._close_position(position, last_bar.close, last_bar.end_time, "EOD_FLATTEN")
                                trades.append(trade)
                                equity += eq_delta
                                self._attach_hybrid_trade_outcome(position, trade)
                else:
                    # Legacy carry logic
                    position.carry_score = self._compute_carry_score(position, last_bar, market, bars, len(bars) - 1)
                    binary_ok = self._binary_carry_ok(position, last_bar, market, trade_date)
                    carry_score_fallback_enabled = bool(
                        self._route_setting(position.route_family, "carry_score_fallback_enabled", "pb_carry_score_fallback")
                    )
                    carry_score_threshold = float(
                        self._route_setting(position.route_family, "carry_score_threshold", "pb_carry_score_threshold")
                    )
                    score_ok = (
                        settings.pb_carry_enabled
                        and carry_score_fallback_enabled
                        and position.carry_score >= carry_score_threshold
                    )
                    position.carry_binary_ok = bool(binary_ok)
                    position.carry_score_ok = bool(score_ok)
                    if has_next_backtest_day and position.close_r > 0 and (binary_ok or score_ok):
                        position.current_stop = max(
                            position.current_stop,
                            position.entry_price if position.close_r >= float(self._route_setting(position.route_family, "breakeven_r", "pb_breakeven_r")) else position.current_stop,
                        )
                        position.carry_decision_path = "binary" if binary_ok else "score_fallback"
                        carry_positions[symbol] = position
                        if funnel_counters is not None:
                            funnel_counters["carried"] = funnel_counters.get("carried", 0) + 1
                    else:
                        if not has_next_backtest_day:
                            position.carry_decision_path = "no_next_day"
                        elif position.close_r <= 0:
                            position.carry_decision_path = "underwater_flatten"
                        elif binary_ok or score_ok:
                            position.carry_decision_path = "eligible_but_flattened"
                        else:
                            position.carry_decision_path = "flatten"
                        trade, eq_delta = self._close_position(position, last_bar.close, last_bar.end_time, "EOD_FLATTEN")
                        trades.append(trade)
                        equity += eq_delta
                        self._attach_hybrid_trade_outcome(position, trade)

            equity_history.append(equity)

        if carry_positions and trading_dates:
            last_date = trading_dates[-1]
            ts_final = self._market_close_timestamp(last_date)
            for symbol, position in list(carry_positions.items()):
                close_price = self._replay.get_daily_close(symbol, last_date)
                if close_price is None:
                    close_price = position.entry_price
                trade, eq_delta = self._close_position(position, float(close_price), ts_final, "END_OF_BACKTEST")
                trades.append(trade)
                equity += eq_delta
                self._attach_hybrid_trade_outcome(position, trade)
                carry_positions.pop(symbol, None)
            equity_history[-1] = equity

        selection_attribution = _build_selection_attribution(candidate_ledger) if candidate_ledger is not None else None
        logger.info(
            "IARIC Tier 3 (Pullback Hybrid) complete: %d trades, final equity: $%.2f (%.1f%%)",
            len(trades),
            equity,
            (equity / cfg.initial_equity - 1) * 100,
        )
        return IARICPullbackResult(
            trades=trades,
            equity_curve=np.array(equity_history),
            timestamps=np.array([np.datetime64(ts.replace(tzinfo=None)) for ts in ts_history]),
            daily_selections=daily_selections,
            candidate_ledger=candidate_ledger,
            funnel_counters=funnel_counters,
            rejection_log=rejection_log,
            shadow_outcomes=shadow_outcomes,
            selection_attribution=selection_attribution,
            fsm_log=fsm_log,
            decision_stream=normalize_decision_stream(self._decision_events),
            trade_outcomes=trade_outcomes_from_records(trades),
        )
