"""Family-level momentum portfolio replay over live portfolio rules.

This module is intentionally a thin replay adapter around
``libs.oms.risk.portfolio_rules.PortfolioRuleChecker``.  Strategy engines
still produce the trade candidates, but portfolio approval, sizing modifiers,
directional caps, cooldowns, and drawdown tiers are evaluated through the same
live rule checker used by the momentum coordinator.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from backtests.momentum.analysis.metrics import (
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
)
from libs.oms.risk.portfolio_rules import PortfolioRuleChecker, PortfolioRulesConfig
from strategies.core.events import DecisionEvent, TradeOutcome


MOMENTUM_FAMILY_STRATEGY_IDS: tuple[str, ...] = (
    "NQ_REGIME",
    "VdubusNQ_v4",
    "NQDTC_v2.1",
    "DownturnDominator_v1",
)
PORTFOLIO_REPLAY_CONTRACT_VERSION = "completed_source_trade_replay_live_portfolio_rules.v1"


def portfolio_replay_contract(
    *,
    source_label: str = "legacy_strategy_trades",
    decision_count: int = 0,
) -> dict[str, object]:
    """Describe exactly what the portfolio replay can and cannot prove."""

    return {
        "version": PORTFOLIO_REPLAY_CONTRACT_VERSION,
        "scope": "portfolio_sizing_routing_and_live_risk_overlay",
        "candidate_source": source_label,
        "live_rule_checker": "libs.oms.risk.portfolio_rules.PortfolioRuleChecker",
        "uses_live_portfolio_rules": True,
        "uses_shared_capital_ledger": True,
        "source_strategy_execution_simulation": False,
        "decision_event_count": decision_count,
        "decision_stream_status": (
            "provided" if decision_count > 0 else "not_provided_completed_trade_replay"
        ),
        "evidence_label": "portfolio_sizing_evidence_not_full_source_execution_simulation",
        "known_limitation": (
            "The portfolio layer replays completed source trade outcomes. It cannot "
            "discover source-strategy fill, order-path, or intrabar execution defects."
        ),
        "source_strategy_parity_prerequisite": (
            "Each source momentum strategy must maintain its own live/backtest parity "
            "through shared-core or equivalent source-engine tests."
        ),
    }


@dataclass(frozen=True)
class FamilyStrategyAllocation:
    strategy_id: str
    enabled: bool = True
    base_risk_pct: float = 0.005
    daily_stop_R: float = 2.5
    max_concurrent: int = 1
    priority: int = 1
    max_contracts: int = 0


@dataclass(frozen=True)
class FamilyDynamicRiskConfig:
    enabled: bool = False
    strategy_multipliers: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    fit_to_remaining_heat: bool = False
    fit_to_remaining_directional_cap: bool = False
    fit_to_remaining_family_cap: bool = False
    min_qty: int = 1
    min_trade_risk_R: float = 0.0
    max_trade_risk_R: float = 0.0
    heat_pressure_threshold: float = 1.0
    heat_pressure_mult: float = 1.0
    same_direction_pressure_threshold: float = 1.0
    same_direction_pressure_mult: float = 1.0
    existing_position_mult: float = 1.0
    daily_loss_threshold_R: float = -999.0
    daily_loss_mult: float = 1.0

    def multiplier_for(self, strategy_id: str) -> float:
        return dict(self.strategy_multipliers).get(strategy_id, 1.0)


@dataclass(frozen=True)
class FamilySignalFilterCondition:
    field: str
    op: str
    value: object


@dataclass(frozen=True)
class FamilySignalFilterRule:
    name: str
    strategy_id: str
    conditions: tuple[FamilySignalFilterCondition, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FamilyPortfolioBacktestConfig:
    initial_equity: float = 50_000.0
    strategy_allocations: tuple[FamilyStrategyAllocation, ...] = field(default_factory=tuple)
    rules: PortfolioRulesConfig = field(default_factory=PortfolioRulesConfig)
    heat_cap_R: float = 4.75
    portfolio_daily_stop_R: float = 2.75
    portfolio_weekly_stop_R: float = 7.5
    max_total_positions: int = 5
    point_value: float = 2.0
    commission_per_side: float = 0.62
    reference_unit_risk_dollars: float = 250.0
    start_date: datetime | None = None
    end_date: datetime | None = None
    dynamic_risk: FamilyDynamicRiskConfig = field(default_factory=FamilyDynamicRiskConfig)
    signal_filter_rules: tuple[FamilySignalFilterRule, ...] = field(default_factory=tuple)

    def allocation_for(self, strategy_id: str) -> FamilyStrategyAllocation | None:
        for allocation in self.strategy_allocations:
            if allocation.strategy_id == strategy_id:
                return allocation
        return None

    def priority_for(self, strategy_id: str) -> int:
        allocation = self.allocation_for(strategy_id)
        return allocation.priority if allocation is not None else 99


@dataclass
class FamilyPortfolioTrade:
    strategy_id: str
    direction: int
    entry_time: datetime | None
    exit_time: datetime | None
    entry_price: float
    exit_price: float
    initial_stop: float
    raw_pnl_dollars: float
    raw_qty: int
    r_multiple: float
    symbol: str = "MNQ"
    mfe_r: float = 0.0
    mae_r: float = 0.0
    commission: float = 0.0
    exit_reason: str = ""
    source_label: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    portfolio_approved: bool = True
    denial_reason: str = ""
    size_multiplier: float = 1.0
    portfolio_qty: int = 0
    adjusted_pnl: float = 0.0
    risk_dollars: float = 0.0
    normalized_risk_R: float = 0.0
    equity_at_entry: float = 0.0


@dataclass(frozen=True)
class FamilyPortfolioReplayBundle:
    """Replay-ready portfolio source data with a deterministic fingerprint."""

    source_fingerprint: str
    trade_outcomes: tuple[TradeOutcome, ...]
    decisions: tuple[DecisionEvent, ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class FamilyPortfolioResult:
    trades: list[FamilyPortfolioTrade] = field(default_factory=list)
    blocked_trades: list[FamilyPortfolioTrade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    equity_timestamps: list[datetime] = field(default_factory=list)
    initial_equity: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    strategy_trade_counts: dict[str, int] = field(default_factory=dict)
    strategy_blocked_counts: dict[str, int] = field(default_factory=dict)
    rule_blocks: dict[str, int] = field(default_factory=dict)
    max_concurrent: int = 0
    replay_architecture: str = "canonical_replay_bundle_live_rule_adapter"
    action_count: int = 0
    replay_source_fingerprint: str = ""
    trade_outcome_count: int = 0
    decision_count: int = 0
    replay_bundle_metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class _OpenPosition:
    trade_idx: int
    strategy_id: str
    direction: int
    risk_R: float
    risk_dollars: float
    qty: int
    symbol: str


@dataclass
class _ReplayState:
    equity: float
    peak_equity: float
    open_positions: list[_OpenPosition] = field(default_factory=list)
    daily_pnl_R: dict[str, float] = field(default_factory=dict)
    daily_total_R: float = 0.0
    weekly_total_R: float = 0.0
    current_day: tuple[int, int, int] | None = None
    current_week: tuple[int, int] | None = None
    strategy_signals: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass(frozen=True)
class FamilyPortfolioAction:
    """Neutral portfolio-layer action produced by deterministic replay.

    Strategy engines still provide the candidate trades today.  The portfolio
    replay turns those candidates into neutral entry/exit actions, then the
    live-rule adapter decides whether and how an entry can be accepted.
    """

    timestamp: datetime
    action_type: str
    trade_idx: int
    strategy_id: str
    priority: int = 99


def make_controlled_aggressive_family_config(
    initial_equity: float = 50_000.0,
) -> FamilyPortfolioBacktestConfig:
    """Return a controlled-aggressive four-strategy seed config."""

    reference_unit_risk = initial_equity * 0.005
    priorities = (
        ("VdubusNQ_v4", 0),
        ("NQ_REGIME", 0),
        ("NQDTC_v2.1", 1),
        ("DownturnDominator_v1", 1),
    )
    rules = PortfolioRulesConfig(
        initial_equity=initial_equity,
        cooldown_session_only=True,
        nqdtc_direction_filter_enabled=True,
        nqdtc_agree_size_mult=1.25,
        nqdtc_oppose_size_mult=0.50,
        directional_cap_R=4.25,
        directional_cap_long_R=4.25,
        directional_cap_short_R=4.75,
        family_strategy_ids=MOMENTUM_FAMILY_STRATEGY_IDS,
        max_family_contracts_mnq_eq=max(8, int(initial_equity * 12.0 / (21_000.0 * 2.0))),
        strategy_priorities=priorities,
        priority_headroom_R=1.0,
        priority_reserve_threshold=1,
        reference_unit_risk_dollars=reference_unit_risk,
        dd_tiers=((0.10, 1.00), (0.15, 0.60), (0.20, 0.30), (1.00, 0.00)),
    )
    allocations = (
        FamilyStrategyAllocation("NQ_REGIME", base_risk_pct=0.0060, daily_stop_R=3.0, max_concurrent=2, priority=0),
        FamilyStrategyAllocation("VdubusNQ_v4", base_risk_pct=0.0055, daily_stop_R=2.5, max_concurrent=1, priority=0),
        FamilyStrategyAllocation("NQDTC_v2.1", base_risk_pct=0.0045, daily_stop_R=2.5, max_concurrent=1, priority=1),
        FamilyStrategyAllocation("DownturnDominator_v1", base_risk_pct=0.0040, daily_stop_R=2.0, max_concurrent=1, priority=1),
    )
    return FamilyPortfolioBacktestConfig(
        initial_equity=initial_equity,
        strategy_allocations=allocations,
        rules=rules,
        heat_cap_R=4.75,
        portfolio_daily_stop_R=2.75,
        portfolio_weekly_stop_R=7.5,
        max_total_positions=5,
        reference_unit_risk_dollars=reference_unit_risk,
    )


def make_controlled_aggressive_five_strategy_config(
    initial_equity: float = 50_000.0,
) -> FamilyPortfolioBacktestConfig:
    """Backward-compatible alias for the four-strategy family seed."""

    return make_controlled_aggressive_family_config(initial_equity)


class FamilyPortfolioBacktester:
    def __init__(self, config: FamilyPortfolioBacktestConfig):
        self.config = config

    def run(self, trades_by_strategy: dict[str, list]) -> FamilyPortfolioResult:
        return self.run_bundle(build_family_replay_bundle(trades_by_strategy))

    async def run_async(self, trades_by_strategy: dict[str, list]) -> FamilyPortfolioResult:
        return await self.run_bundle_async(build_family_replay_bundle(trades_by_strategy))

    def build_replay_bundle(
        self,
        trades_by_strategy: dict[str, list],
        *,
        decisions: tuple[DecisionEvent, ...] = (),
        source_label: str = "legacy_strategy_trades",
    ) -> FamilyPortfolioReplayBundle:
        return build_family_replay_bundle(
            trades_by_strategy,
            decisions=decisions,
            source_label=source_label,
        )

    def run_bundle(self, bundle: FamilyPortfolioReplayBundle) -> FamilyPortfolioResult:
        return asyncio.run(self.run_bundle_async(bundle))

    async def run_bundle_async(self, bundle: FamilyPortfolioReplayBundle) -> FamilyPortfolioResult:
        all_trades = self._portfolio_trades_from_bundle(bundle)
        if not all_trades:
            result = FamilyPortfolioResult(
                initial_equity=self.config.initial_equity,
                equity_curve=np.array([self.config.initial_equity]),
                replay_source_fingerprint=bundle.source_fingerprint,
                trade_outcome_count=len(bundle.trade_outcomes),
                decision_count=len(bundle.decisions),
                replay_bundle_metadata=dict(bundle.metadata),
            )
            result.metrics = self._compute_metrics(result)
            return result

        actions = self._build_actions(all_trades)
        adapter = _LiveRuleFamilyExecutionAdapter(
            engine=self,
            all_trades=all_trades,
            first_timestamp=actions[0].timestamp,
            replay_source_fingerprint=bundle.source_fingerprint,
            trade_outcome_count=len(bundle.trade_outcomes),
            decision_count=len(bundle.decisions),
            replay_bundle_metadata=dict(bundle.metadata),
        )
        return await FamilyPortfolioReplayDriver(actions, adapter).run()

    def _normalize_trades(self, trades_by_strategy: dict[str, list]) -> list[FamilyPortfolioTrade]:
        return self._portfolio_trades_from_bundle(build_family_replay_bundle(trades_by_strategy))

    def _portfolio_trades_from_bundle(
        self,
        bundle: FamilyPortfolioReplayBundle,
    ) -> list[FamilyPortfolioTrade]:
        normalized: list[FamilyPortfolioTrade] = []
        for outcome in bundle.trade_outcomes:
            allocation = self.config.allocation_for(outcome.strategy_id)
            if allocation is None or not allocation.enabled:
                continue
            portfolio_trade = _outcome_to_portfolio_trade(outcome)
            if self._in_date_range(portfolio_trade):
                normalized.append(portfolio_trade)
        return normalized

    def _build_actions(self, trades: list[FamilyPortfolioTrade]) -> list[FamilyPortfolioAction]:
        actions: list[FamilyPortfolioAction] = []
        for idx, trade in enumerate(trades):
            if trade.entry_time is not None:
                actions.append(
                    FamilyPortfolioAction(
                        timestamp=_aware_utc(trade.entry_time),
                        action_type="submit_entry",
                        trade_idx=idx,
                        strategy_id=trade.strategy_id,
                        priority=self.config.priority_for(trade.strategy_id),
                    )
                )
            if trade.exit_time is not None:
                actions.append(
                    FamilyPortfolioAction(
                        timestamp=_aware_utc(trade.exit_time),
                        action_type="settle_exit",
                        trade_idx=idx,
                        strategy_id=trade.strategy_id,
                        priority=-1,
                    )
                )
        actions.sort(key=_portfolio_action_sort_key)
        return actions

    def _in_date_range(self, trade: FamilyPortfolioTrade) -> bool:
        if trade.entry_time is None:
            return False
        entry = _aware_utc(trade.entry_time)
        if self.config.start_date is not None and entry < _aware_utc(self.config.start_date):
            return False
        if self.config.end_date is not None and entry > _aware_utc(self.config.end_date):
            return False
        return True

    async def _process_entry(
        self,
        state: _ReplayState,
        trade: FamilyPortfolioTrade,
        trade_idx: int,
        result: FamilyPortfolioResult,
        checker: PortfolioRuleChecker,
    ) -> None:
        allocation = self.config.allocation_for(trade.strategy_id)
        if allocation is None or not allocation.enabled:
            self._deny(trade, result, "strategy_disabled")
            return
        entry_context = self._entry_context(state, trade)
        trade.metadata["portfolio_entry_context"] = entry_context
        filter_reason = self._signal_filter_reason(trade)
        if filter_reason:
            self._deny(trade, result, filter_reason)
            return
        if len(state.open_positions) >= self.config.max_total_positions:
            self._deny(trade, result, "max_total_positions")
            return
        strat_open = sum(1 for pos in state.open_positions if pos.strategy_id == trade.strategy_id)
        entry_context["strategy_open_positions"] = strat_open
        if strat_open >= allocation.max_concurrent:
            self._deny(trade, result, "max_concurrent")
            return
        if state.daily_pnl_R.get(trade.strategy_id, 0.0) <= -allocation.daily_stop_R:
            self._deny(trade, result, "strategy_daily_stop")
            return
        if state.daily_total_R <= -self.config.portfolio_daily_stop_R:
            self._deny(trade, result, "portfolio_daily_stop")
            return
        if self.config.portfolio_weekly_stop_R > 0 and state.weekly_total_R <= -self.config.portfolio_weekly_stop_R:
            self._deny(trade, result, "portfolio_weekly_stop")
            return

        stop_distance = abs(trade.entry_price - trade.initial_stop)
        if stop_distance <= 0:
            self._deny(trade, result, "zero_stop_distance")
            return

        risk_budget = state.equity * allocation.base_risk_pct
        risk_per_contract = stop_distance * self.config.point_value
        base_qty = max(1, int(round(risk_budget / risk_per_contract)))
        if allocation.max_contracts > 0:
            base_qty = min(base_qty, allocation.max_contracts)
        base_risk_dollars = base_qty * risk_per_contract
        base_risk_R = base_risk_dollars / self.config.reference_unit_risk_dollars

        heat_R = sum(pos.risk_R for pos in state.open_positions)
        base_mnq_eq = base_qty * (10 if trade.symbol == "NQ" else 1)
        dynamic_info = self._apply_dynamic_risk(
            state=state,
            trade=trade,
            base_qty=base_qty,
            risk_per_contract=risk_per_contract,
            heat_R=heat_R,
        )
        base_qty = dynamic_info["qty"]
        base_risk_dollars = base_qty * risk_per_contract
        base_risk_R = base_risk_dollars / self.config.reference_unit_risk_dollars
        base_mnq_eq = base_qty * (10 if trade.symbol == "NQ" else 1)
        entry_context.update({
            "risk_budget": risk_budget,
            "risk_per_contract": risk_per_contract,
            "initial_base_qty": dynamic_info["initial_qty"],
            "base_qty": base_qty,
            "base_mnq_eq": base_mnq_eq,
            "base_risk_dollars": base_risk_dollars,
            "base_risk_R": base_risk_R,
            "dynamic_multiplier": dynamic_info["multiplier"],
            "dynamic_max_qty": dynamic_info["max_qty"],
            "dynamic_denial_reason": dynamic_info["denial_reason"],
            "heat_R": heat_R,
            "heat_cap_R": self.config.heat_cap_R,
            "family_mnq_eq": int(
                sum(
                    pos.qty * (10 if pos.symbol == "NQ" else 1)
                    for pos in state.open_positions
                )
            ),
            "family_contract_cap_mnq_eq": self.config.rules.max_family_contracts_mnq_eq,
            "same_direction_risk_R": sum(
                pos.risk_R
                for pos in state.open_positions
                if pos.direction == trade.direction
            ),
            "opposite_direction_risk_R": sum(
                pos.risk_R
                for pos in state.open_positions
                if pos.direction != trade.direction
            ),
        })
        if base_qty < 1:
            self._deny(trade, result, str(dynamic_info["denial_reason"]))
            return
        if heat_R + base_risk_R > self.config.heat_cap_R:
            self._deny(trade, result, "heat_cap")
            return

        rule_result = await checker.check_entry(
            strategy_id=trade.strategy_id,
            direction=_direction_text(trade.direction),
            new_risk_R=base_risk_R,
            symbol=trade.symbol,
            new_qty=base_qty,
            new_risk_dollars=base_risk_dollars,
        )
        if not rule_result.approved:
            self._deny(trade, result, _compact_denial(rule_result.denial_reason))
            return

        final_qty = max(1, int(round(base_qty * rule_result.size_multiplier)))
        adjusted_pnl = _scale_trade_pnl(trade, final_qty, self.config.point_value)
        risk_dollars = final_qty * risk_per_contract
        normalized_risk_R = risk_dollars / self.config.reference_unit_risk_dollars

        trade.size_multiplier = rule_result.size_multiplier
        trade.portfolio_qty = final_qty
        trade.adjusted_pnl = adjusted_pnl
        trade.risk_dollars = risk_dollars
        trade.normalized_risk_R = normalized_risk_R
        trade.equity_at_entry = state.equity
        entry_context.update({
            "final_qty": final_qty,
            "final_risk_dollars": risk_dollars,
            "final_risk_R": normalized_risk_R,
            "size_multiplier": rule_result.size_multiplier,
        })
        state.open_positions.append(
            _OpenPosition(
                trade_idx=trade_idx,
                strategy_id=trade.strategy_id,
                direction=trade.direction,
                risk_R=normalized_risk_R,
                risk_dollars=risk_dollars,
                qty=final_qty,
                symbol=trade.symbol,
            )
        )
        state.strategy_signals[trade.strategy_id] = {
            "last_entry_ts": _aware_utc(trade.entry_time),
            "last_direction": _direction_text(trade.direction),
            "signal_date": _to_et(_aware_utc(trade.entry_time)).date(),
            "chop_score": trade.metadata.get("chop_score", 0),
        }

    def _entry_context(
        self,
        state: _ReplayState,
        trade: FamilyPortfolioTrade,
    ) -> dict[str, object]:
        return {
            "equity": state.equity,
            "open_positions": len(state.open_positions),
            "portfolio_daily_R": state.daily_total_R,
            "portfolio_weekly_R": state.weekly_total_R,
            "strategy_daily_R": state.daily_pnl_R.get(trade.strategy_id, 0.0),
        }

    def _apply_dynamic_risk(
        self,
        *,
        state: _ReplayState,
        trade: FamilyPortfolioTrade,
        base_qty: int,
        risk_per_contract: float,
        heat_R: float,
    ) -> dict[str, float | int | str]:
        policy = self.config.dynamic_risk
        initial_qty = base_qty
        if not policy.enabled:
            return {
                "initial_qty": initial_qty,
                "qty": base_qty,
                "multiplier": 1.0,
                "max_qty": base_qty,
                "denial_reason": "",
            }

        multiplier = policy.multiplier_for(trade.strategy_id)
        if state.open_positions:
            multiplier *= policy.existing_position_mult
        if state.daily_total_R <= policy.daily_loss_threshold_R:
            multiplier *= policy.daily_loss_mult
        if self.config.heat_cap_R > 0:
            heat_utilization = heat_R / self.config.heat_cap_R
            if heat_utilization >= policy.heat_pressure_threshold:
                multiplier *= policy.heat_pressure_mult
        direction_cap_R = self._directional_cap_for(trade.direction)
        same_direction_risk_R = sum(
            pos.risk_R
            for pos in state.open_positions
            if pos.direction == trade.direction
        )
        if direction_cap_R > 0:
            direction_utilization = same_direction_risk_R / direction_cap_R
            if direction_utilization >= policy.same_direction_pressure_threshold:
                multiplier *= policy.same_direction_pressure_mult

        adjusted_qty = int(round(base_qty * multiplier))
        max_qty = adjusted_qty
        cap_limited = False
        if policy.max_trade_risk_R > 0:
            max_qty = min(
                max_qty,
                _qty_for_risk_R(policy.max_trade_risk_R, self.config.reference_unit_risk_dollars, risk_per_contract),
            )
        if policy.fit_to_remaining_heat:
            max_qty = min(
                max_qty,
                _qty_for_risk_R(
                    self.config.heat_cap_R - heat_R,
                    self.config.reference_unit_risk_dollars,
                    risk_per_contract,
                ),
            )
            cap_limited = True
        if policy.fit_to_remaining_directional_cap and direction_cap_R > 0:
            max_qty = min(
                max_qty,
                _qty_for_risk_R(
                    direction_cap_R - same_direction_risk_R,
                    self.config.reference_unit_risk_dollars,
                    risk_per_contract,
                ),
            )
            cap_limited = True
        if policy.fit_to_remaining_family_cap and self.config.rules.max_family_contracts_mnq_eq > 0:
            current_mnq_eq = int(
                sum(
                    pos.qty * (10 if pos.symbol == "NQ" else 1)
                    for pos in state.open_positions
                )
            )
            remaining_mnq_eq = self.config.rules.max_family_contracts_mnq_eq - current_mnq_eq
            symbol_multiplier = 10 if trade.symbol == "NQ" else 1
            max_qty = min(max_qty, remaining_mnq_eq // symbol_multiplier)
            cap_limited = True

        min_qty = max(1, policy.min_qty)
        if policy.min_trade_risk_R > 0:
            min_qty = max(
                min_qty,
                int(np.ceil(policy.min_trade_risk_R * self.config.reference_unit_risk_dollars / risk_per_contract)),
            )
        denial_reason = ""
        if max_qty < min_qty:
            denial_reason = "dynamic_capacity_floor" if cap_limited else "dynamic_min_qty"
            max_qty = 0
        return {
            "initial_qty": initial_qty,
            "qty": max(max_qty, 0),
            "multiplier": multiplier,
            "max_qty": max_qty,
            "denial_reason": denial_reason,
        }

    def _directional_cap_for(self, direction: int) -> float:
        direction_text = _direction_text(direction)
        if direction_text == "LONG" and self.config.rules.directional_cap_long_R > 0:
            return self.config.rules.directional_cap_long_R
        if direction_text == "SHORT" and self.config.rules.directional_cap_short_R > 0:
            return self.config.rules.directional_cap_short_R
        return self.config.rules.directional_cap_R

    def _signal_filter_reason(self, trade: FamilyPortfolioTrade) -> str:
        for rule in self.config.signal_filter_rules:
            if rule.strategy_id != trade.strategy_id:
                continue
            if all(_condition_matches(trade, condition) for condition in rule.conditions):
                return f"signal_filter:{rule.name}"
        return ""

    def _process_exit(
        self,
        state: _ReplayState,
        trade: FamilyPortfolioTrade,
        trade_idx: int,
        equity_points: list[tuple[datetime, float]],
    ) -> None:
        if not trade.portfolio_approved:
            return
        state.open_positions = [pos for pos in state.open_positions if pos.trade_idx != trade_idx]
        state.equity += trade.adjusted_pnl
        state.peak_equity = max(state.peak_equity, state.equity)
        realized_R = trade.adjusted_pnl / self.config.reference_unit_risk_dollars
        state.daily_pnl_R[trade.strategy_id] = state.daily_pnl_R.get(trade.strategy_id, 0.0) + realized_R
        state.daily_total_R += realized_R
        state.weekly_total_R += realized_R
        if trade.exit_time is not None:
            equity_points.append((_aware_utc(trade.exit_time), state.equity))

    def _deny(self, trade: FamilyPortfolioTrade, result: FamilyPortfolioResult, reason: str) -> None:
        trade.portfolio_approved = False
        trade.denial_reason = reason
        trade.portfolio_qty = 0
        trade.adjusted_pnl = 0.0
        result.rule_blocks[reason] = result.rule_blocks.get(reason, 0) + 1

    def _check_boundaries(self, state: _ReplayState, now: datetime) -> None:
        day = _trading_day(now)
        if day != state.current_day:
            state.daily_pnl_R.clear()
            state.daily_total_R = 0.0
            state.current_day = day
        week = _trading_week(now)
        if week != state.current_week:
            state.weekly_total_R = 0.0
            state.current_week = week

    async def _get_signal(self, state: _ReplayState, strategy_id: str) -> dict[str, object] | None:
        return state.strategy_signals.get(strategy_id)

    async def _directional_risk_R(
        self,
        state: _ReplayState,
        direction: str,
        strategy_ids: list[str] | None = None,
    ) -> float:
        ids = set(strategy_ids or MOMENTUM_FAMILY_STRATEGY_IDS)
        return float(
            sum(
                pos.risk_R
                for pos in state.open_positions
                if pos.strategy_id in ids and _direction_text(pos.direction) == direction
            )
        )

    async def _directional_risk_dollars(
        self,
        state: _ReplayState,
        direction: str,
        strategy_ids: list[str] | None = None,
    ) -> float:
        ids = set(strategy_ids or MOMENTUM_FAMILY_STRATEGY_IDS)
        return float(
            sum(
                pos.risk_dollars
                for pos in state.open_positions
                if pos.strategy_id in ids and _direction_text(pos.direction) == direction
            )
        )

    async def _family_mnq_eq(self, state: _ReplayState, strategy_ids: list[str]) -> int:
        ids = set(strategy_ids)
        return int(sum(pos.qty * (10 if pos.symbol == "NQ" else 1) for pos in state.open_positions if pos.strategy_id in ids))

    def _compute_metrics(self, result: FamilyPortfolioResult) -> dict[str, float]:
        trades = result.trades
        blocked = result.blocked_trades
        equity = result.equity_curve
        net_profit = float(sum(trade.adjusted_pnl for trade in trades))
        wins = [trade.adjusted_pnl for trade in trades if trade.adjusted_pnl > 0]
        losses = [trade.adjusted_pnl for trade in trades if trade.adjusted_pnl < 0]
        gross_win = float(sum(wins))
        gross_loss = abs(float(sum(losses)))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else (10.0 if gross_win > 0 else 0.0)
        max_dd_pct, _ = compute_max_drawdown(equity)
        years = _span_years(result.equity_timestamps)
        final_equity = float(equity[-1]) if len(equity) else self.config.initial_equity
        total_r = net_profit / self.config.reference_unit_risk_dollars
        months = max(years * 12.0, 1e-9)
        trade_count = len(trades)
        active_strategies = len({trade.strategy_id for trade in trades})
        all_count = trade_count + len(blocked)
        block_rate = len(blocked) / all_count if all_count else 0.0
        counts = _counts_by_strategy(trades)
        min_strategy_trades = min((counts.get(sid, 0) for sid in MOMENTUM_FAMILY_STRATEGY_IDS), default=0)

        # R-based drawdown from cumulative trade R-multiples (more conservative
        # than realized-only equity curve DD which only updates at trade exits)
        ref_risk = self.config.reference_unit_risk_dollars
        if trade_count > 0 and ref_risk > 0:
            realized_rs = np.array([t.adjusted_pnl / ref_risk for t in trades])
            cum_r = np.cumsum(realized_rs)
            peak_r = np.maximum.accumulate(cum_r)
            max_r_dd = float(np.max(peak_r - cum_r))
        else:
            max_r_dd = 0.0

        cagr = compute_cagr(self.config.initial_equity, final_equity, years)
        return {
            "total_trades": float(trade_count),
            "blocked_trades": float(len(blocked)),
            "block_rate": float(block_rate),
            "active_strategies": float(active_strategies),
            "min_strategy_trades": float(min_strategy_trades),
            "net_profit": net_profit,
            "net_return_pct": (final_equity / self.config.initial_equity - 1.0),
            "profit_factor": float(profit_factor),
            "win_rate": len(wins) / trade_count if trade_count else 0.0,
            "max_drawdown_pct": float(max_dd_pct),
            "max_drawdown_r": float(max_r_dd),
            "sharpe": compute_sharpe(equity, periods_per_year=252),
            "sortino": compute_sortino(equity, periods_per_year=252),
            "cagr": cagr,
            "calmar": cagr / max(max_dd_pct, 1e-9),
            "calmar_r": float(total_r / max_r_dd) if max_r_dd > 0 else 0.0,
            "total_r": float(total_r),
            "total_r_per_month": float(total_r / months),
            "trades_per_month": float(trade_count / months),
            "max_concurrent": float(result.max_concurrent),
        }


class FamilyPortfolioReplayDriver:
    """Deterministic portfolio replay driver over neutral actions."""

    def __init__(
        self,
        actions: list[FamilyPortfolioAction],
        execution_adapter: "_LiveRuleFamilyExecutionAdapter",
    ):
        self.actions = actions
        self.execution_adapter = execution_adapter

    async def run(self) -> FamilyPortfolioResult:
        for action in self.actions:
            await self.execution_adapter.handle(action)
        return self.execution_adapter.finalize()


class _LiveRuleFamilyExecutionAdapter:
    """Portfolio execution adapter that delegates approval to live rules."""

    def __init__(
        self,
        *,
        engine: FamilyPortfolioBacktester,
        all_trades: list[FamilyPortfolioTrade],
        first_timestamp: datetime,
        replay_source_fingerprint: str,
        trade_outcome_count: int,
        decision_count: int,
        replay_bundle_metadata: dict[str, object],
    ):
        self.engine = engine
        self.all_trades = all_trades
        self.state = _ReplayState(
            equity=engine.config.initial_equity,
            peak_equity=engine.config.initial_equity,
        )
        self.result = FamilyPortfolioResult(
            initial_equity=engine.config.initial_equity,
            replay_architecture="canonical_replay_bundle_live_rule_adapter",
            replay_source_fingerprint=replay_source_fingerprint,
            trade_outcome_count=trade_outcome_count,
            decision_count=decision_count,
            replay_bundle_metadata=dict(replay_bundle_metadata),
        )
        self.current_time = {"value": _aware_utc(first_timestamp)}
        self.equity_points: list[tuple[datetime, float]] = [
            (self.current_time["value"], self.state.equity)
        ]
        self.checker = PortfolioRuleChecker(
            config=engine.config.rules,
            get_strategy_signal=lambda strategy_id: engine._get_signal(self.state, strategy_id),
            get_directional_risk_R=lambda direction: engine._directional_risk_R(self.state, direction),
            get_current_equity=lambda: self.state.equity,
            get_directional_risk_R_for_strategies=(
                lambda direction, strategy_ids: engine._directional_risk_R(
                    self.state, direction, strategy_ids
                )
            ),
            get_directional_risk_dollars_for_strategies=(
                lambda direction, strategy_ids: engine._directional_risk_dollars(
                    self.state, direction, strategy_ids
                )
            ),
            get_family_aggregate_mnq_eq=lambda strategy_ids: engine._family_mnq_eq(
                self.state, strategy_ids
            ),
            now_provider=lambda: self.current_time["value"],
        )

    async def handle(self, action: FamilyPortfolioAction) -> None:
        self.result.action_count += 1
        self.current_time["value"] = _aware_utc(action.timestamp)
        self.engine._check_boundaries(self.state, self.current_time["value"])
        trade = self.all_trades[action.trade_idx]
        if action.action_type == "settle_exit":
            self.engine._process_exit(
                self.state,
                trade,
                action.trade_idx,
                self.equity_points,
            )
            return
        if action.action_type == "submit_entry":
            await self.engine._process_entry(
                self.state,
                trade,
                action.trade_idx,
                self.result,
                self.checker,
            )
            return
        raise ValueError(f"Unknown family portfolio action type: {action.action_type}")

    def finalize(self) -> FamilyPortfolioResult:
        self.result.trades = [trade for trade in self.all_trades if trade.portfolio_approved]
        self.result.blocked_trades = [
            trade for trade in self.all_trades if not trade.portfolio_approved
        ]
        self.result.equity_timestamps, self.result.equity_curve = _daily_equity(
            self.equity_points,
            self.engine.config.initial_equity,
        )
        self.result.strategy_trade_counts = _counts_by_strategy(self.result.trades)
        self.result.strategy_blocked_counts = _counts_by_strategy(self.result.blocked_trades)
        self.result.max_concurrent = _max_concurrent(self.result.trades)
        self.result.metrics = self.engine._compute_metrics(self.result)
        return self.result


def family_config_to_dict(config: FamilyPortfolioBacktestConfig) -> dict[str, object]:
    data = asdict(config)
    data["rules"] = asdict(config.rules)
    return data


def family_config_from_dict(data: dict[str, object]) -> FamilyPortfolioBacktestConfig:
    config_data = dict(data)
    rules_data = dict(config_data.pop("rules", {}))
    rules_data["dd_tiers"] = tuple(tuple(item) for item in rules_data.get("dd_tiers", ()))
    rules_data["family_strategy_ids"] = tuple(rules_data.get("family_strategy_ids", ()))
    rules_data["strategy_priorities"] = tuple(tuple(item) for item in rules_data.get("strategy_priorities", ()))
    rules_data["symbol_collision_pairs"] = tuple(tuple(item) for item in rules_data.get("symbol_collision_pairs", ()))
    rules_data["disabled_strategies"] = frozenset(rules_data.get("disabled_strategies", ()))

    allocations = tuple(
        FamilyStrategyAllocation(**dict(item))
        for item in config_data.pop("strategy_allocations", ())
    )
    filter_rules = tuple(
        FamilySignalFilterRule(
            name=str(item["name"]),
            strategy_id=str(item["strategy_id"]),
            conditions=tuple(
                FamilySignalFilterCondition(**dict(condition))
                for condition in item.get("conditions", ())
            ),
        )
        for item in config_data.pop("signal_filter_rules", ())
    )
    dynamic_data = dict(config_data.pop("dynamic_risk", {}) or {})
    dynamic_data["strategy_multipliers"] = tuple(
        tuple(item)
        for item in dynamic_data.get("strategy_multipliers", ())
    )
    for key in ("start_date", "end_date"):
        if isinstance(config_data.get(key), str):
            config_data[key] = datetime.fromisoformat(config_data[key])
    return FamilyPortfolioBacktestConfig(
        **config_data,
        rules=PortfolioRulesConfig(**rules_data),
        strategy_allocations=allocations,
        dynamic_risk=FamilyDynamicRiskConfig(**dynamic_data),
        signal_filter_rules=filter_rules,
    )


def update_allocation(
    config: FamilyPortfolioBacktestConfig,
    strategy_id: str,
    **changes: object,
) -> FamilyPortfolioBacktestConfig:
    allocations = tuple(
        replace(allocation, **changes) if allocation.strategy_id == strategy_id else allocation
        for allocation in config.strategy_allocations
    )
    return replace(config, strategy_allocations=allocations)


def build_family_replay_bundle(
    trades_by_strategy: dict[str, list],
    *,
    decisions: tuple[DecisionEvent, ...] = (),
    source_label: str = "legacy_strategy_trades",
) -> FamilyPortfolioReplayBundle:
    trade_outcomes: list[TradeOutcome] = []
    strategy_counts: dict[str, int] = {}
    converters = _family_trade_converters()
    for strategy_id, trades in trades_by_strategy.items():
        if strategy_id not in converters:
            raise KeyError(f"Unknown momentum family strategy id: {strategy_id}")
        converter = converters[strategy_id]
        for raw_trade in trades or []:
            portfolio_trade = converter(raw_trade)
            if portfolio_trade.entry_time is None:
                continue
            trade_outcomes.append(_portfolio_trade_to_outcome(portfolio_trade))
            strategy_counts[strategy_id] = strategy_counts.get(strategy_id, 0) + 1
    outcome_tuple = tuple(trade_outcomes)
    decision_tuple = tuple(decisions)
    metadata: dict[str, object] = {
        "source_label": source_label,
        "strategy_trade_counts": strategy_counts,
        "schema": "momentum_family_replay_bundle.v1",
        "replay_contract": portfolio_replay_contract(
            source_label=source_label,
            decision_count=len(decision_tuple),
        ),
    }
    return FamilyPortfolioReplayBundle(
        source_fingerprint=_replay_bundle_fingerprint(outcome_tuple, decision_tuple),
        trade_outcomes=outcome_tuple,
        decisions=decision_tuple,
        metadata=metadata,
    )


def _family_trade_converters() -> dict[str, object]:
    return {
        "NQDTC_v2.1": _from_nqdtc,
        "VdubusNQ_v4": _from_vdubus,
        "DownturnDominator_v1": _from_downturn,
        "NQ_REGIME": _from_nq_regime,
    }


def _portfolio_trade_to_outcome(trade: FamilyPortfolioTrade) -> TradeOutcome:
    if trade.entry_time is None:
        raise ValueError("Cannot create a TradeOutcome without an entry timestamp.")
    return TradeOutcome(
        strategy_id=trade.strategy_id,
        symbol=trade.symbol,
        direction=trade.direction,
        entry_ts=_aware_utc(trade.entry_time),
        exit_ts=_aware_utc(trade.exit_time) if trade.exit_time is not None else None,
        qty=max(int(trade.raw_qty or 1), 1),
        entry_price=float(trade.entry_price),
        exit_price=float(trade.exit_price),
        initial_stop=float(trade.initial_stop),
        gross_pnl=float(trade.raw_pnl_dollars + trade.commission),
        commission=float(trade.commission),
        net_pnl=float(trade.raw_pnl_dollars),
        r_multiple=float(trade.r_multiple),
        mfe_r=float(trade.mfe_r),
        mae_r=float(trade.mae_r),
        realized=trade.exit_time is not None,
        exit_reason=trade.exit_reason,
        source_label=trade.source_label,
        decision_ts=_aware_utc(trade.entry_time),
        fill_ts=_aware_utc(trade.entry_time),
        route="legacy_trade_adapter",
        metadata=dict(trade.metadata),
    )


def _outcome_to_portfolio_trade(outcome: TradeOutcome) -> FamilyPortfolioTrade:
    return FamilyPortfolioTrade(
        strategy_id=outcome.strategy_id,
        direction=int(outcome.direction),
        entry_time=outcome.entry_ts,
        exit_time=outcome.exit_ts,
        entry_price=float(outcome.entry_price),
        exit_price=float(outcome.exit_price or 0.0),
        initial_stop=float(outcome.initial_stop or 0.0),
        raw_pnl_dollars=float(outcome.net_pnl),
        raw_qty=max(int(outcome.qty or 1), 1),
        r_multiple=float(outcome.r_multiple),
        symbol=outcome.symbol or "MNQ",
        mfe_r=float(outcome.mfe_r),
        mae_r=float(outcome.mae_r),
        commission=float(outcome.commission),
        exit_reason=outcome.exit_reason,
        source_label=outcome.source_label,
        metadata=dict(outcome.metadata),
    )


def _replay_bundle_fingerprint(
    outcomes: tuple[TradeOutcome, ...],
    decisions: tuple[DecisionEvent, ...],
) -> str:
    payload = {
        "schema": "momentum_family_replay_bundle.v1",
        "trade_outcomes": [_trade_outcome_payload(outcome) for outcome in outcomes],
        "decisions": [_decision_event_payload(decision) for decision in decisions],
    }
    encoded = json.dumps(
        _stable_json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trade_outcome_payload(outcome: TradeOutcome) -> dict[str, object]:
    return {
        "strategy_id": outcome.strategy_id,
        "symbol": outcome.symbol,
        "direction": outcome.direction,
        "entry_ts": outcome.entry_ts,
        "exit_ts": outcome.exit_ts,
        "qty": outcome.qty,
        "entry_price": outcome.entry_price,
        "exit_price": outcome.exit_price,
        "initial_stop": outcome.initial_stop,
        "net_pnl": outcome.net_pnl,
        "commission": outcome.commission,
        "r_multiple": outcome.r_multiple,
        "mfe_r": outcome.mfe_r,
        "mae_r": outcome.mae_r,
        "exit_reason": outcome.exit_reason,
        "source_label": outcome.source_label,
        "metadata": outcome.metadata,
    }


def _decision_event_payload(event: DecisionEvent) -> dict[str, object]:
    return {
        "schema_version": event.schema_version,
        "event_type": event.event_type,
        "bot_id": event.bot_id,
        "strategy_id": event.strategy_id,
        "family_id": event.family_id,
        "portfolio_id": event.portfolio_id,
        "strategy_version": event.strategy_version,
        "config_version": event.config_version,
        "portfolio_config_version": event.portfolio_config_version,
        "risk_config_version": event.risk_config_version,
        "allocation_version": event.allocation_version,
        "strategy_registry_version": event.strategy_registry_version,
        "deployment_id": event.deployment_id,
        "parameter_set_id": event.parameter_set_id,
        "code_sha": event.code_sha,
        "trace_id": event.trace_id,
        "code": event.code,
        "ts": event.ts,
        "symbol": event.symbol,
        "timeframe": event.timeframe,
        "details": event.details,
        "state_ref": event.state_ref,
        "emitted_actions": event.emitted_actions,
        "bar_id": event.bar_id,
        "decision_kind": event.decision_kind,
        "sequence": event.sequence,
    }


def _stable_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _stable_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, datetime):
        return _aware_utc(value).isoformat()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "value"):
        return str(value.value)
    return value


def _from_nqdtc(trade) -> FamilyPortfolioTrade:
    return FamilyPortfolioTrade(
        strategy_id="NQDTC_v2.1",
        direction=int(getattr(trade, "direction", 0)),
        entry_time=getattr(trade, "entry_time", None),
        exit_time=getattr(trade, "exit_time", None),
        entry_price=float(getattr(trade, "entry_price", 0.0)),
        exit_price=float(getattr(trade, "exit_price", 0.0)),
        initial_stop=float(getattr(trade, "initial_stop", 0.0)),
        raw_pnl_dollars=float(getattr(trade, "pnl_dollars", 0.0)),
        raw_qty=max(int(getattr(trade, "qty", 1) or 1), 1),
        r_multiple=float(getattr(trade, "r_multiple", 0.0)),
        mfe_r=float(getattr(trade, "mfe_r", 0.0)),
        mae_r=float(getattr(trade, "mae_r", 0.0)),
        commission=float(getattr(trade, "commission", 0.0)),
        exit_reason=str(getattr(trade, "exit_reason", "")),
        source_label="nqdtc",
        metadata={
            "entry_subtype": _metadata_value(getattr(trade, "entry_subtype", "")),
            "session": _metadata_value(getattr(trade, "session", "")),
            "composite_regime": _metadata_value(getattr(trade, "composite_regime", "")),
            "chop_mode": _metadata_value(getattr(trade, "chop_mode", "")),
            "score_at_entry": float(getattr(trade, "score_at_entry", 0.0) or 0.0),
            "displacement_at_entry": float(getattr(trade, "displacement_at_entry", 0.0) or 0.0),
            "rvol_at_entry": float(getattr(trade, "rvol_at_entry", 0.0) or 0.0),
            "quality_mult": float(getattr(trade, "quality_mult", 0.0) or 0.0),
            "box_width": float(getattr(trade, "box_width", 0.0) or 0.0),
        },
    )


def _from_vdubus(trade) -> FamilyPortfolioTrade:
    return FamilyPortfolioTrade(
        strategy_id="VdubusNQ_v4",
        direction=int(getattr(trade, "direction", 0)),
        entry_time=getattr(trade, "entry_time", None),
        exit_time=getattr(trade, "exit_time", None),
        entry_price=float(getattr(trade, "entry_price", 0.0)),
        exit_price=float(getattr(trade, "exit_price", 0.0)),
        initial_stop=float(getattr(trade, "initial_stop", 0.0)),
        raw_pnl_dollars=float(getattr(trade, "pnl_dollars", 0.0)),
        raw_qty=max(int(getattr(trade, "qty", 1) or 1), 1),
        r_multiple=float(getattr(trade, "r_multiple", 0.0)),
        mfe_r=float(getattr(trade, "mfe_r", 0.0)),
        mae_r=float(getattr(trade, "mae_r", 0.0)),
        commission=float(getattr(trade, "commission", 0.0)),
        exit_reason=str(getattr(trade, "exit_reason", "")),
        source_label="vdubus",
        metadata={
            "entry_type": _metadata_value(getattr(trade, "entry_type", "")),
            "is_flip": bool(getattr(trade, "is_flip", False)),
            "is_addon": bool(getattr(trade, "is_addon", False)),
            "session": _metadata_value(getattr(trade, "session", "")),
            "sub_window": _metadata_value(getattr(trade, "sub_window", "")),
            "daily_trend": float(getattr(trade, "daily_trend", 0.0) or 0.0),
            "vol_state": _metadata_value(getattr(trade, "vol_state", "")),
            "trend_1h": float(getattr(trade, "trend_1h", 0.0) or 0.0),
            "class_mult": float(getattr(trade, "class_mult", 0.0) or 0.0),
            "overnight_sessions": float(getattr(trade, "overnight_sessions", 0.0) or 0.0),
        },
    )


def _from_downturn(trade) -> FamilyPortfolioTrade:
    return FamilyPortfolioTrade(
        strategy_id="DownturnDominator_v1",
        direction=int(getattr(trade, "direction", -1)),
        entry_time=getattr(trade, "entry_time", None),
        exit_time=getattr(trade, "exit_time", None),
        entry_price=float(getattr(trade, "entry_price", 0.0)),
        exit_price=float(getattr(trade, "exit_price", 0.0)),
        initial_stop=float(getattr(trade, "stop0", 0.0)),
        raw_pnl_dollars=float(getattr(trade, "pnl", 0.0)),
        raw_qty=max(int(getattr(trade, "qty", 1) or 1), 1),
        r_multiple=float(getattr(trade, "r_multiple", 0.0)),
        mfe_r=float(getattr(trade, "mfe", 0.0)),
        mae_r=float(getattr(trade, "mae", 0.0)),
        commission=float(getattr(trade, "commission", 0.0)),
        exit_reason=str(getattr(trade, "exit_type", "")),
        source_label="downturn",
        metadata={
            "entry_type": _metadata_value(getattr(trade, "entry_type", "")),
            "engine_tag": _metadata_value(getattr(trade, "engine_tag", "")),
            "composite_regime": _metadata_value(getattr(trade, "composite_regime_at_entry", "")),
            "vol_state": _metadata_value(getattr(trade, "vol_state_at_entry", "")),
            "in_correction_window": bool(getattr(trade, "in_correction_window", False)),
            "signal_class": _metadata_value(getattr(trade, "signal_class", "")),
        },
    )


def _from_nq_regime(trade) -> FamilyPortfolioTrade:
    side = str(getattr(trade, "side", "BUY")).upper()
    return FamilyPortfolioTrade(
        strategy_id="NQ_REGIME",
        direction=1 if side == "BUY" else -1,
        entry_time=getattr(trade, "entry_time", None),
        exit_time=getattr(trade, "exit_time", None),
        entry_price=float(getattr(trade, "entry_price", 0.0)),
        exit_price=float(getattr(trade, "exit_price", 0.0)),
        initial_stop=float(getattr(trade, "initial_stop", 0.0)),
        raw_pnl_dollars=float(getattr(trade, "pnl_dollars", 0.0)),
        raw_qty=max(int(getattr(trade, "qty", 1) or 1), 1),
        r_multiple=float(getattr(trade, "r_multiple", 0.0)),
        mfe_r=float(getattr(trade, "mfe_r", 0.0)),
        mae_r=float(getattr(trade, "mae_r", 0.0)),
        commission=float(getattr(trade, "commission", 0.0)),
        exit_reason=str(getattr(trade, "exit_reason", "")),
        source_label="nq_regime",
        metadata={
            "module": _metadata_value(getattr(trade, "module", "")),
            "setup_type": _metadata_value(getattr(trade, "setup_type", "")),
            "entry_model": _metadata_value(getattr(trade, "entry_model", "")),
            "regime": _metadata_value(getattr(trade, "regime", "")),
            "grade": _metadata_value(getattr(trade, "grade", "")),
            "setup_score": float(getattr(trade, "setup_score", 0.0) or 0.0),
            "target_room_r": float(getattr(trade, "target_room_r", 0.0) or 0.0),
            "stop_distance_points": float(getattr(trade, "stop_distance_points", 0.0) or 0.0),
            "ib_type": _metadata_value(getattr(trade, "ib_type", "")),
            "volume_multiple": float(getattr(trade, "volume_multiple", 0.0) or 0.0),
        },
    )


def _scale_trade_pnl(trade: FamilyPortfolioTrade, qty: int, point_value: float) -> float:
    if trade.raw_qty > 0 and trade.raw_pnl_dollars:
        return trade.raw_pnl_dollars / trade.raw_qty * qty
    move = (trade.exit_price - trade.entry_price) * trade.direction
    return move * point_value * qty


def _daily_equity(
    equity_points: list[tuple[datetime, float]],
    initial_equity: float,
) -> tuple[list[datetime], np.ndarray]:
    if not equity_points:
        return [], np.array([initial_equity])
    daily: dict[datetime, float] = {}
    for ts, equity in sorted(equity_points, key=lambda item: item[0]):
        day = datetime(ts.year, ts.month, ts.day)
        daily[day] = equity
    first = min(daily)
    last = max(daily)
    current = first
    values: list[float] = []
    timestamps: list[datetime] = []
    last_equity = initial_equity
    while current <= last:
        last_equity = daily.get(current, last_equity)
        timestamps.append(current)
        values.append(last_equity)
        current += timedelta(days=1)
    return timestamps, np.asarray(values, dtype=float)


def _counts_by_strategy(trades: list[FamilyPortfolioTrade]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in trades:
        counts[trade.strategy_id] = counts.get(trade.strategy_id, 0) + 1
    return counts


def _max_concurrent(trades: list[FamilyPortfolioTrade]) -> int:
    intervals: list[tuple[datetime, int]] = []
    for trade in trades:
        if trade.entry_time is not None and trade.exit_time is not None:
            intervals.append((_aware_utc(trade.entry_time), 1))
            intervals.append((_aware_utc(trade.exit_time), -1))
    intervals.sort(key=lambda item: (item[0], item[1]))
    current = 0
    max_seen = 0
    for _, delta in intervals:
        current += delta
        max_seen = max(max_seen, current)
    return max_seen


def _portfolio_action_sort_key(action: FamilyPortfolioAction) -> tuple[datetime, int, int, int]:
    action_rank = 0 if action.action_type == "settle_exit" else 1
    return action.timestamp, action_rank, action.priority, action.trade_idx


def _span_years(timestamps: list[datetime]) -> float:
    if len(timestamps) < 2:
        return 1.0
    return max((timestamps[-1] - timestamps[0]).total_seconds() / (365.25 * 24 * 3600), 1e-9)


def _aware_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_et(dt: datetime) -> datetime:
    return _aware_utc(dt).astimezone(ZoneInfo("America/New_York"))


def _trading_day(dt: datetime) -> tuple[int, int, int]:
    et = _to_et(dt)
    if et.hour >= 18:
        et += timedelta(days=1)
    return et.year, et.month, et.day


def _trading_week(dt: datetime) -> tuple[int, int]:
    et = _to_et(dt)
    if et.hour >= 18:
        et += timedelta(days=1)
    iso = et.isocalendar()
    return iso.year, iso.week


def _direction_text(direction: int) -> str:
    return "LONG" if direction > 0 else "SHORT"


def _condition_matches(
    trade: FamilyPortfolioTrade,
    condition: FamilySignalFilterCondition,
) -> bool:
    actual = _trade_feature(trade, condition.field)
    expected = condition.value
    op = condition.op
    if op == "eq":
        return actual == expected
    if op == "neq":
        return actual != expected
    if op == "in":
        return actual in set(expected if isinstance(expected, (list, tuple, set, frozenset)) else (expected,))
    if op == "not_in":
        return actual not in set(expected if isinstance(expected, (list, tuple, set, frozenset)) else (expected,))
    if op in {"lt", "lte", "gt", "gte"}:
        actual_num = _as_float(actual)
        expected_num = _as_float(expected)
        if actual_num is None or expected_num is None:
            return False
        if op == "lt":
            return actual_num < expected_num
        if op == "lte":
            return actual_num <= expected_num
        if op == "gt":
            return actual_num > expected_num
        return actual_num >= expected_num
    raise ValueError(f"Unknown signal filter op: {op}")


def _trade_feature(trade: FamilyPortfolioTrade, field_name: str) -> object:
    if field_name.startswith("metadata."):
        return trade.metadata.get(field_name.split(".", 1)[1])
    if field_name == "direction":
        return trade.direction
    if field_name == "direction_text":
        return _direction_text(trade.direction)
    if field_name == "entry_hour_et":
        return _to_et(_aware_utc(trade.entry_time)).hour if trade.entry_time is not None else None
    if field_name == "entry_weekday_et":
        return _to_et(_aware_utc(trade.entry_time)).weekday() if trade.entry_time is not None else None
    return getattr(trade, field_name)


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _qty_for_risk_R(risk_R: float, reference_unit_risk_dollars: float, risk_per_contract: float) -> int:
    if risk_R <= 0 or reference_unit_risk_dollars <= 0 or risk_per_contract <= 0:
        return 0
    return int(np.floor(risk_R * reference_unit_risk_dollars / risk_per_contract))


def _metadata_value(value: object) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _compact_denial(reason: str | None) -> str:
    if not reason:
        return "portfolio_rule"
    return reason.split(":", 1)[0].strip()
