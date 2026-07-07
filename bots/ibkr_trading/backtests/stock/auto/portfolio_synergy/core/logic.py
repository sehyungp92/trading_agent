from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import replace
from datetime import datetime
from typing import Any

import numpy as np

from backtests.stock.analysis.metrics import compute_metrics
from backtests.stock.models import Direction, TradeRecord
from libs.oms.risk.portfolio_rules import PortfolioRuleChecker, PortfolioRulesConfig

from ..phase_candidates import INITIAL_EQUITY, STRATEGY_ORDER
from .state import (
    BlockedCandidate,
    DecisionEvent,
    PortfolioAction,
    PortfolioActionType,
    PortfolioCoreState,
    PortfolioPosition,
    PortfolioReplayResult,
    ReplayCandidate,
    TradeOutcome,
)


def replay_trade_streams(
    alcb_trades: tuple[TradeRecord, ...] | list[TradeRecord],
    iaric_trades: tuple[TradeRecord, ...] | list[TradeRecord],
    effective: dict[str, Any],
) -> dict[str, float]:
    return run_portfolio_replay(alcb_trades, iaric_trades, effective).metrics


def run_portfolio_replay(
    alcb_trades: tuple[TradeRecord, ...] | list[TradeRecord],
    iaric_trades: tuple[TradeRecord, ...] | list[TradeRecord],
    effective: dict[str, Any],
) -> PortfolioReplayResult:
    return asyncio.run(_run_portfolio_replay_async(alcb_trades, iaric_trades, effective))


async def _run_portfolio_replay_async(
    alcb_trades: tuple[TradeRecord, ...] | list[TradeRecord],
    iaric_trades: tuple[TradeRecord, ...] | list[TradeRecord],
    effective: dict[str, Any],
) -> PortfolioReplayResult:
    initial_equity = float(effective.get("initial_equity", INITIAL_EQUITY))
    rules = effective["portfolio_rules"]
    reference_risk_pct = float(rules.get("reference_risk_pct", 0.006) or 0.006)
    lookback = int(effective.get("dynamic_allocation", {}).get("lookback_trades", 60) or 60)
    state = PortfolioCoreState.initial(
        initial_equity=initial_equity,
        reference_risk_pct=reference_risk_pct,
        lookback_trades=lookback,
    )

    entries: list[tuple[datetime, str, TradeRecord]] = []
    entries.extend((trade.entry_time, "ALCB_R3", trade) for trade in alcb_trades)
    entries.extend((trade.entry_time, "IARIC_V5R1", trade) for trade in iaric_trades)
    entries.sort(key=lambda item: item[0])
    live_rule_adapter = _StockPortfolioLiveRuleReplayAdapter(
        state=state,
        effective=effective,
        symbol_sector_map=_symbol_sector_map(entries),
    )

    actions: list[PortfolioAction] = []
    decisions: list[DecisionEvent] = []
    trade_outcomes: list[TradeOutcome] = []

    i = 0
    while i < len(entries):
        ts = entries[i][0]
        _close_positions(
            state,
            before=ts,
            actions=actions,
            trade_outcomes=trade_outcomes,
        )

        batch: list[tuple[str, TradeRecord]] = []
        while i < len(entries) and entries[i][0] == ts:
            batch.append((entries[i][1], entries[i][2]))
            i += 1
        candidates = [
            _build_candidate(
                strategy,
                trade,
                effective,
                state.equity,
                state.peak_equity,
                state.strategy_recent,
            )
            for strategy, trade in batch
        ]
        candidates = [candidate for candidate in candidates if candidate is not None]
        candidates.sort(key=lambda candidate: _rank_candidate(candidate, effective), reverse=True)

        for candidate in candidates:
            state.candidate_count += 1
            reason = await live_rule_adapter.check_entry(candidate)
            if reason:
                action = PortfolioAction(
                    action_type=PortfolioActionType.BLOCK_ENTRY,
                    timestamp=candidate.trade.entry_time,
                    strategy_id=candidate.strategy,
                    symbol=candidate.trade.symbol,
                    reason=reason,
                    risk_dollars=candidate.risk_dollars,
                    metadata=_candidate_metadata(candidate),
                )
                actions.append(action)
                decisions.append(
                    _decision_event(
                        state,
                        candidate,
                        decision_code="BLOCK_ENTRY",
                        reason=reason,
                        action=action,
                    )
                )
                state.blocked_candidates.append(
                    BlockedCandidate(
                        strategy=candidate.strategy,
                        symbol=candidate.trade.symbol,
                        sector=candidate.trade.sector,
                        entry_time=candidate.trade.entry_time,
                        r_multiple=candidate.r_multiple,
                        reason=reason,
                        quality=candidate.quality,
                        heat_r=candidate.heat_r,
                    )
                )
                continue

            requested_qty = max(1, int(round(float(candidate.trade.quantity) * candidate.size_mult)))
            approved_qty = max(1, int(requested_qty * float(candidate.portfolio_size_mult)))
            qty_ratio = approved_qty / requested_qty if requested_qty > 0 else 0.0
            approved_risk_dollars = candidate.risk_dollars * qty_ratio
            approved_pnl = candidate.pnl * qty_ratio
            price_scale = approved_qty / float(candidate.trade.quantity) if candidate.trade.quantity else 0.0
            scaled_commission = candidate.trade.commission * price_scale
            metadata = dict(candidate.trade.metadata or {})
            metadata.update(
                {
                    "portfolio_requested_qty": requested_qty,
                    "portfolio_approved_qty": approved_qty,
                    "portfolio_size_mult": float(candidate.portfolio_size_mult),
                }
            )
            position = PortfolioPosition(
                strategy=candidate.strategy,
                symbol=candidate.trade.symbol,
                sector=candidate.trade.sector,
                direction=candidate.trade.direction,
                entry_time=candidate.trade.entry_time,
                decision_time=candidate.trade.entry_time,
                fill_time=candidate.trade.fill_time or candidate.trade.entry_time,
                exit_time=candidate.trade.exit_time,
                risk_dollars=approved_risk_dollars,
                pnl=approved_pnl,
                r_multiple=candidate.r_multiple,
                quality=candidate.quality,
                entry_price=float(candidate.trade.entry_price),
                exit_price=float(candidate.trade.exit_price),
                quantity=float(approved_qty),
                price_scale=price_scale,
                commission=scaled_commission,
                exit_reason=candidate.trade.exit_reason,
                entry_type=candidate.trade.entry_type,
                metadata=metadata,
            )
            action = PortfolioAction(
                action_type=PortfolioActionType.SUBMIT_ENTRY,
                timestamp=position.decision_time,
                strategy_id=position.strategy,
                symbol=position.symbol,
                risk_dollars=position.risk_dollars,
                metadata=_candidate_metadata(candidate),
            )
            actions.append(action)
            decisions.append(
                _decision_event(
                    state,
                    candidate,
                    decision_code="ACCEPT_ENTRY",
                    reason="accepted",
                    action=action,
                )
            )
            state.active_positions.append(position)
            state.accepted_positions.append(position)
            state.risk_by_strategy[position.strategy] = (
                state.risk_by_strategy.get(position.strategy, 0.0) + position.risk_dollars
            )

    _close_positions(
        state,
        before=datetime.max,
        actions=actions,
        trade_outcomes=trade_outcomes,
    )
    metrics = _compute_replay_metrics(state, entries, initial_equity)
    return PortfolioReplayResult(
        metrics=metrics,
        state=state,
        decisions=tuple(decisions),
        actions=tuple(actions),
        trade_outcomes=tuple(trade_outcomes),
        replay_architecture="stock_portfolio_core_live_rule_adapter",
    )


class _StockPortfolioLiveRuleReplayAdapter:
    """Replay adapter that delegates portfolio admission to the live rule checker."""

    def __init__(
        self,
        *,
        state: PortfolioCoreState,
        effective: dict[str, Any],
        symbol_sector_map: tuple[tuple[str, str], ...],
    ) -> None:
        self._state = state
        self._effective = effective
        self._current_time: datetime | None = None
        self._base_config = self._portfolio_rules_config(symbol_sector_map)
        self._checker = PortfolioRuleChecker(
            config=self._base_config,
            get_strategy_signal=self._get_strategy_signal,
            get_directional_risk_R=self._get_directional_risk_R,
            get_current_equity=lambda: float(self._state.equity),
            get_directional_risk_R_for_strategies=self._get_directional_risk_R_for_strategies,
            get_sibling_positions_for_symbol=self._get_sibling_positions_for_symbol,
            get_directional_risk_dollars_for_strategies=(
                self._get_directional_risk_dollars_for_strategies
            ),
            get_open_position_count_for_strategies=self._get_open_position_count_for_strategies,
            get_symbol_open_risk_dollars_for_strategies=(
                self._get_symbol_open_risk_dollars_for_strategies
            ),
            get_symbols_open_risk_dollars_for_strategies=(
                self._get_symbols_open_risk_dollars_for_strategies
            ),
            get_active_risk_dollars_for_strategies=self._get_active_risk_dollars_for_strategies,
            get_completed_trade_counts_for_strategies=self._get_completed_trade_counts_for_strategies,
            get_recent_strategy_r_multiples=self._get_recent_strategy_r_multiples,
            now_provider=lambda: self._current_time or datetime.utcnow(),
        )

    async def check_entry(self, candidate: ReplayCandidate) -> str:
        self._current_time = candidate.trade.entry_time
        self._checker.update_config(
            replace(
                self._base_config,
                reference_unit_risk_dollars=self._reference_risk_dollars(),
                initial_equity=float(self._state.equity),
            )
        )
        result = await self._checker.check_entry(
            strategy_id=candidate.strategy,
            direction=_direction_text(candidate.trade.direction),
            new_risk_R=candidate.heat_r,
            symbol=candidate.trade.symbol,
            new_qty=max(1, int(round(float(candidate.trade.quantity) * candidate.size_mult))),
            new_risk_dollars=candidate.risk_dollars,
        )
        if not result.approved:
            return _legacy_block_reason(result.denial_reason or "")
        candidate.portfolio_size_mult = float(result.size_multiplier)
        return self._custom_replay_block_reason(candidate)

    def _portfolio_rules_config(
        self,
        symbol_sector_map: tuple[tuple[str, str], ...],
    ) -> PortfolioRulesConfig:
        rules = self._effective["portfolio_rules"]
        allocations = self._effective["strategy_allocations"]
        cross = self._effective.get("cross_strategy_rules", {})
        strategy_priorities = tuple(
            (strategy, int(allocations[strategy].get("priority", index)))
            for index, strategy in enumerate(STRATEGY_ORDER)
            if strategy in allocations
        )
        same_symbol_policy = str(cross.get("same_symbol_policy", "half_size"))
        collision_action = same_symbol_policy if same_symbol_policy in {"none", "block", "half_size"} else "none"
        return PortfolioRulesConfig(
            nqdtc_direction_filter_enabled=False,
            directional_cap_R=0.0,
            directional_cap_long_R=float(rules.get("max_long_heat_R", 0.0) or 0.0),
            directional_cap_short_R=0.0,
            dd_tiers=((1.0, 1.0),),
            initial_equity=float(self._state.equity),
            family_strategy_ids=tuple(STRATEGY_ORDER),
            symbol_collision_action=collision_action,
            symbol_collision_pairs=tuple(
                (str(row[0]), str(row[1]), str(row[2]))
                for row in cross.get("symbol_collision_pairs", ()) or ()
                if len(row) >= 3
            ),
            strategy_priorities=strategy_priorities,
            priority_headroom_R=0.0,
            reference_unit_risk_dollars=self._reference_risk_dollars(),
            reference_unit_risk_pct=float(rules.get("reference_risk_pct", 0.006) or 0.006),
            max_total_active_positions=int(rules.get("max_total_active_positions", 0) or 0),
            max_symbol_heat_R=float(rules.get("max_symbol_heat_R", 0.0) or 0.0),
            same_sector_heat_cap_R=float(cross.get("same_sector_heat_cap_R", 0.0) or 0.0),
            symbol_sector_map=symbol_sector_map,
            max_single_strategy_trade_share=float(
                rules.get("max_single_strategy_trade_share", 1.0) or 1.0
            ),
            strategy_trade_share_min_total=50,
            dynamic_allocation_enabled=False,
            portfolio_heat_cap_R=float(rules.get("heat_cap_R", 0.0) or 0.0),
            max_strategy_active_positions=tuple(
                (strategy, int(allocations[strategy].get("max_concurrent", 0) or 0))
                for strategy in STRATEGY_ORDER
                if strategy in allocations
            ),
            max_strategy_heat_R=tuple(
                (strategy, float(allocations[strategy].get("max_heat_R", 0.0) or 0.0))
                for strategy in STRATEGY_ORDER
                if strategy in allocations
            ),
        )

    def _reference_risk_dollars(self) -> float:
        return max(
            float(self._state.equity)
            * float(self._effective["portfolio_rules"].get("reference_risk_pct", 0.006) or 0.006),
            1.0,
        )

    def _custom_replay_block_reason(self, candidate: ReplayCandidate) -> str:
        cross = self._effective.get("cross_strategy_rules", {})
        same_symbol_policy = str(cross.get("same_symbol_policy", "half_size"))
        if same_symbol_policy != "best_rank_only":
            return ""
        same_symbol_active = [
            position
            for position in self._state.active_positions
            if position.symbol == candidate.trade.symbol
        ]
        if not same_symbol_active:
            return ""
        best_active_quality = max(position.quality for position in same_symbol_active)
        return "same_symbol_lower_rank" if candidate.quality <= best_active_quality else ""

    async def _get_strategy_signal(self, strategy_id: str) -> None:
        del strategy_id
        return None

    async def _get_directional_risk_R(self, direction: str) -> float:
        return await self._get_directional_risk_R_for_strategies(direction, list(STRATEGY_ORDER))

    async def _get_directional_risk_R_for_strategies(
        self,
        direction: str,
        strategy_ids: list[str],
    ) -> float:
        ref = self._reference_risk_dollars()
        return await self._get_directional_risk_dollars_for_strategies(direction, strategy_ids) / ref

    async def _get_directional_risk_dollars_for_strategies(
        self,
        direction: str,
        strategy_ids: list[str],
    ) -> float:
        ids = set(strategy_ids)
        return float(
            sum(
                position.risk_dollars
                for position in self._state.active_positions
                if position.strategy in ids
                and _direction_text(position.direction) == direction.upper()
            )
        )

    async def _get_open_position_count_for_strategies(self, strategy_ids: list[str]) -> int:
        ids = set(strategy_ids)
        return sum(1 for position in self._state.active_positions if position.strategy in ids)

    async def _get_sibling_positions_for_symbol(self, strategy_ids: list[str], symbol: str) -> bool:
        ids = set(strategy_ids)
        return any(
            position.strategy in ids and position.symbol == symbol
            for position in self._state.active_positions
        )

    async def _get_symbol_open_risk_dollars_for_strategies(
        self,
        strategy_ids: list[str],
        symbol: str,
    ) -> float:
        ids = set(strategy_ids)
        return float(
            sum(
                position.risk_dollars
                for position in self._state.active_positions
                if position.strategy in ids and position.symbol == symbol
            )
        )

    async def _get_symbols_open_risk_dollars_for_strategies(
        self,
        strategy_ids: list[str],
        symbols: list[str],
    ) -> float:
        ids = set(strategy_ids)
        symbol_set = set(symbols)
        return float(
            sum(
                position.risk_dollars
                for position in self._state.active_positions
                if position.strategy in ids and position.symbol in symbol_set
            )
        )

    async def _get_active_risk_dollars_for_strategies(self, strategy_ids: list[str]) -> float:
        ids = set(strategy_ids)
        return float(
            sum(
                position.risk_dollars
                for position in self._state.active_positions
                if position.strategy in ids
            )
        )

    async def _get_completed_trade_counts_for_strategies(
        self,
        strategy_ids: list[str],
    ) -> dict[str, int]:
        ids = set(strategy_ids)
        counts: dict[str, int] = {strategy: 0 for strategy in strategy_ids}
        for position in self._state.accepted_positions:
            if position.strategy in ids:
                counts[position.strategy] = counts.get(position.strategy, 0) + 1
        return counts

    async def _get_recent_strategy_r_multiples(self, strategy_id: str, lookback: int) -> list[float]:
        recent = self._state.strategy_recent.get(strategy_id)
        if not recent:
            return []
        return list(recent)[-max(1, int(lookback)):]


def _symbol_sector_map(entries: list[tuple[datetime, str, TradeRecord]]) -> tuple[tuple[str, str], ...]:
    mapping: dict[str, str] = {}
    for _, _, trade in entries:
        symbol = str(getattr(trade, "symbol", "") or "")
        sector = str(getattr(trade, "sector", "") or "")
        if symbol and sector:
            mapping[symbol] = sector
    return tuple(sorted(mapping.items()))


def _direction_text(direction: Direction) -> str:
    return "LONG" if direction == Direction.LONG or int(direction) > 0 else "SHORT"


def _legacy_block_reason(denial_reason: str) -> str:
    reason = denial_reason or "portfolio_rule_block"
    if reason.startswith("max_total_active_positions"):
        return "max_total_active_positions"
    if reason.startswith("max_strategy_active_positions"):
        return "strategy_max_concurrent"
    if reason.startswith("portfolio_heat_cap"):
        return "portfolio_heat_cap"
    if reason.startswith("strategy_heat_cap"):
        return "strategy_heat_cap"
    if reason.startswith("symbol_heat_cap"):
        return "symbol_heat_cap"
    if reason.startswith("sector_heat_cap"):
        return "sector_heat_cap"
    if reason.startswith("directional_cap"):
        return "long_heat_cap"
    if reason.startswith("strategy_trade_share_cap"):
        return "strategy_trade_share_cap"
    return reason.split(":", 1)[0]


def _decision_event(
    state: PortfolioCoreState,
    candidate: ReplayCandidate,
    *,
    decision_code: str,
    reason: str,
    action: PortfolioAction,
) -> DecisionEvent:
    state.decision_seq += 1
    return DecisionEvent(
        timestamp=candidate.trade.entry_time,
        strategy_id=candidate.strategy,
        symbol=candidate.trade.symbol,
        decision_code=decision_code,
        reason=reason,
        state_snapshot_ref=f"stock_portfolio_core:{state.decision_seq}",
        actions_emitted=(action,),
        details=_candidate_metadata(candidate),
    )


def _candidate_metadata(candidate: ReplayCandidate) -> dict[str, Any]:
    return {
        "entry_type": candidate.trade.entry_type,
        "sector": candidate.trade.sector,
        "heat_r": float(candidate.heat_r),
        "quality": float(candidate.quality),
        "r_multiple": float(candidate.r_multiple),
        "size_mult": float(candidate.size_mult),
        "portfolio_size_mult": float(candidate.portfolio_size_mult),
    }


def _compute_replay_metrics(
    state: PortfolioCoreState,
    entries: list[tuple[datetime, str, TradeRecord]],
    initial_equity: float,
) -> dict[str, float]:
    accepted = state.accepted_positions
    blocked = state.blocked_candidates
    pnl_by_strategy: dict[str, float] = defaultdict(float)
    for position in accepted:
        pnl_by_strategy[position.strategy] += position.pnl

    pnls = np.array([position.pnl for position in accepted], dtype=np.float64)
    risks = np.array([position.risk_dollars for position in accepted], dtype=np.float64)
    hold_hours = np.array(
        [
            max((position.exit_time - position.entry_time).total_seconds() / 3600.0, 0.0)
            for position in accepted
        ],
        dtype=np.float64,
    )
    commissions = np.array([p.commission for p in accepted], dtype=np.float64)
    timestamps = np.array(state.equity_times, dtype="datetime64[ns]")
    equity_curve = np.array(state.equity_points, dtype=np.float64)
    if len(timestamps) + 1 == len(equity_curve):
        equity_for_metrics = equity_curve
    else:
        equity_for_metrics = np.array([initial_equity, state.equity], dtype=np.float64)
        timestamps = np.array([], dtype="datetime64[ns]")

    perf = compute_metrics(
        pnls,
        risks,
        hold_hours,
        commissions,
        equity_for_metrics,
        timestamps,
        initial_equity,
        trade_symbols=[position.symbol for position in accepted],
    )

    months = _months_from_positions(accepted, blocked)
    total_r = float(np.sum(np.divide(pnls, np.where(risks > 0, risks, 1.0)))) if len(pnls) else 0.0
    strategy_counts = {
        strategy: sum(1 for position in accepted if position.strategy == strategy)
        for strategy in STRATEGY_ORDER
    }
    active_strategy_count = sum(1 for count in strategy_counts.values() if count > 0)
    total_risk = sum(state.risk_by_strategy.values())
    max_strategy_risk_share = (
        max(state.risk_by_strategy.values()) / total_risk
        if total_risk > 0 and state.risk_by_strategy
        else 0.0
    )
    max_strategy_trade_share = max(strategy_counts.values()) / len(accepted) if accepted else 0.0
    positive_candidates = sum(
        1
        for _, _, trade in entries
        if float(getattr(trade, "r_multiple", 0.0) or 0.0) > 0
    )
    positive_blocks = [candidate for candidate in blocked if candidate.r_multiple > 0]
    nonpositive_blocks = [candidate for candidate in blocked if candidate.r_multiple <= 0]
    blocked_r = [candidate.r_multiple for candidate in blocked]
    accepted_r = [position.r_multiple for position in accepted]
    blocked_positive_r = [candidate.r_multiple for candidate in positive_blocks]
    candidate_discrimination = _candidate_discrimination(accepted_r, blocked_r)
    daily_losses = [min(0.0, value) for value in state.daily_realized_r.values()]
    weekly_losses = [min(0.0, value) for value in state.weekly_realized_r.values()]
    max_daily_loss_r = abs(min(daily_losses)) if daily_losses else 0.0
    max_weekly_loss_r = abs(min(weekly_losses)) if weekly_losses else 0.0

    metrics = {
        "initial_equity": initial_equity,
        "final_equity": state.equity,
        "net_pnl": state.equity - initial_equity,
        "net_return_pct": (state.equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0,
        "total_trades": float(len(accepted)),
        "entry_signals_fired": float(state.candidate_count),
        "entries_accepted_by_portfolio": float(len(accepted)),
        "entries_blocked_by_portfolio": float(len(blocked)),
        "entry_accept_rate": float(len(accepted) / state.candidate_count) if state.candidate_count else 0.0,
        "active_trades_per_month": float(len(accepted) / months) if months > 0 else 0.0,
        "total_r": total_r,
        "total_r_per_month": total_r / months if months > 0 else 0.0,
        "profit_factor": float(perf.profit_factor),
        "win_rate": float(perf.win_rate),
        "expectancy_r": float(perf.expectancy),
        "sharpe": float(perf.sharpe),
        "sortino": float(perf.sortino),
        "calmar": float(perf.calmar),
        "max_drawdown_pct": float(perf.max_drawdown_pct),
        "max_drawdown_dollar": float(perf.max_drawdown_dollar),
        "active_strategy_count": float(active_strategy_count),
        "max_strategy_trade_share": float(max_strategy_trade_share),
        "max_strategy_risk_share": float(max_strategy_risk_share),
        "trade_capture_ratio": float(len(accepted) / state.candidate_count) if state.candidate_count else 0.0,
        "positive_alpha_block_rate": (
            float(len(positive_blocks) / positive_candidates) if positive_candidates > 0 else 0.0
        ),
        "blocked_positive_count": float(len(positive_blocks)),
        "blocked_nonpositive_count": float(len(nonpositive_blocks)),
        "blocked_positive_fraction": float(len(positive_blocks) / len(blocked)) if blocked else 0.0,
        "blocked_avg_r": float(np.mean(blocked_r)) if blocked_r else 0.0,
        "accepted_avg_r": float(np.mean(accepted_r)) if accepted_r else 0.0,
        "blocked_positive_avg_r": float(np.mean(blocked_positive_r)) if blocked_positive_r else 0.0,
        "candidate_discrimination": float(candidate_discrimination),
        "positive_slices": float(_positive_slices(accepted)),
        "max_daily_loss_R": float(max_daily_loss_r),
        "max_weekly_loss_R": float(max_weekly_loss_r),
        **{f"trades_{strategy}": float(count) for strategy, count in strategy_counts.items()},
        **{f"pnl_{strategy}": float(pnl_by_strategy.get(strategy, 0.0)) for strategy in STRATEGY_ORDER},
        **{
            f"risk_share_{strategy}": (
                float(state.risk_by_strategy.get(strategy, 0.0) / total_risk) if total_risk else 0.0
            )
            for strategy in STRATEGY_ORDER
        },
        **_block_reason_metrics(blocked),
    }
    return metrics


def _build_candidate(
    strategy: str,
    trade: TradeRecord,
    effective: dict[str, Any],
    equity: float,
    peak_equity: float,
    strategy_recent: dict[str, deque[float]],
) -> ReplayCandidate | None:
    allocations = effective["strategy_allocations"]
    if strategy not in allocations:
        return None
    allocation = allocations[strategy]
    if allocation.get("enabled", True) is False:
        return None

    source_risk = float(trade.risk_per_share * trade.quantity)
    if source_risk <= 0:
        return None

    drawdown_mult = _drawdown_mult(equity, peak_equity, effective)
    if drawdown_mult <= 0:
        return None
    quality = _candidate_quality(strategy, trade, effective)
    size_mult = _candidate_size_mult(strategy, trade, effective)
    dynamic_mult = _dynamic_mult(strategy, strategy_recent, effective)
    unit_risk_pct = float(allocation.get("unit_risk_pct", 0.006) or 0.006)
    target_risk = equity * unit_risk_pct * drawdown_mult * dynamic_mult * size_mult
    if target_risk <= 0:
        return None
    scale = target_risk / source_risk
    r_multiple = float(trade.r_multiple or 0.0)
    pnl = r_multiple * target_risk
    heat_r = target_risk / max(equity * float(effective["portfolio_rules"].get("reference_risk_pct", 0.006)), 1.0)
    return ReplayCandidate(
        strategy=strategy,
        trade=trade,
        risk_dollars=target_risk,
        pnl=pnl,
        r_multiple=r_multiple,
        heat_r=heat_r,
        quality=quality,
        size_mult=scale,
    )


def _block_reason(
    candidate: ReplayCandidate,
    active: list[PortfolioPosition],
    accepted: list[PortfolioPosition],
    effective: dict[str, Any],
    equity: float,
    reference_risk_pct: float,
) -> str:
    rules = effective["portfolio_rules"]
    allocation = effective["strategy_allocations"][candidate.strategy]
    cross = effective.get("cross_strategy_rules", {})
    reference_risk = max(equity * reference_risk_pct, 1.0)
    current_heat = sum(position.risk_dollars for position in active) / reference_risk
    strategy_heat = sum(position.risk_dollars for position in active if position.strategy == candidate.strategy) / reference_risk
    symbol_heat = sum(position.risk_dollars for position in active if position.symbol == candidate.trade.symbol) / reference_risk
    sector_heat = sum(position.risk_dollars for position in active if position.sector == candidate.trade.sector) / reference_risk
    long_heat = sum(
        position.risk_dollars
        for position in active
        if position.direction == Direction.LONG
    ) / reference_risk
    strategy_open = sum(1 for position in active if position.strategy == candidate.strategy)

    if len(active) >= int(rules.get("max_total_active_positions", 999)):
        return "max_total_active_positions"
    if strategy_open >= int(allocation.get("max_concurrent", 999)):
        return "strategy_max_concurrent"
    if current_heat + candidate.heat_r > float(rules.get("heat_cap_R", 999.0)):
        return "portfolio_heat_cap"
    if strategy_heat + candidate.heat_r > float(allocation.get("max_heat_R", 999.0)):
        return "strategy_heat_cap"
    if symbol_heat + candidate.heat_r > float(rules.get("max_symbol_heat_R", 999.0)):
        return "symbol_heat_cap"
    if candidate.trade.direction == Direction.LONG and long_heat + candidate.heat_r > float(rules.get("max_long_heat_R", 999.0)):
        return "long_heat_cap"
    if sector_heat + candidate.heat_r > float(cross.get("same_sector_heat_cap_R", 999.0)):
        return "sector_heat_cap"

    same_symbol_active = [position for position in active if position.symbol == candidate.trade.symbol]
    same_symbol_policy = str(cross.get("same_symbol_policy", "half_size"))
    if same_symbol_policy == "best_rank_only" and same_symbol_active:
        best_active_quality = max(position.quality for position in same_symbol_active)
        if candidate.quality <= best_active_quality:
            return "same_symbol_lower_rank"

    max_share = float(rules.get("max_single_strategy_trade_share", 1.0))
    if accepted and max_share < 1.0:
        future_count = sum(1 for position in accepted if position.strategy == candidate.strategy) + 1
        future_total = len(accepted) + 1
        if future_count / future_total > max_share and future_total > 50:
            return "strategy_trade_share_cap"

    return ""


def _close_positions(
    state: PortfolioCoreState,
    *,
    before: datetime,
    actions: list[PortfolioAction],
    trade_outcomes: list[TradeOutcome],
) -> None:
    remaining: list[PortfolioPosition] = []
    before_cmp = _naive_dt(before)
    closing = sorted(
        (position for position in state.active_positions if _naive_dt(position.exit_time) <= before_cmp),
        key=lambda item: item.exit_time,
    )
    for position in closing:
        state.equity = float(state.equity) + position.pnl
        state.peak_equity = max(float(state.peak_equity), state.equity)
        state.equity_points.append(state.equity)
        state.equity_times.append(_naive_dt(position.exit_time))
        reference_risk = max(state.equity * float(state.reference_risk_pct), 1.0)
        realized_r = position.pnl / reference_risk
        state.daily_realized_r[position.exit_time.date().isoformat()] = (
            state.daily_realized_r.get(position.exit_time.date().isoformat(), 0.0) + realized_r
        )
        iso = position.exit_time.isocalendar()
        weekly_key = f"{iso.year:04d}-W{iso.week:02d}"
        state.weekly_realized_r[weekly_key] = state.weekly_realized_r.get(weekly_key, 0.0) + realized_r
        state.strategy_recent[position.strategy].append(position.r_multiple)

        action = PortfolioAction(
            action_type=PortfolioActionType.SUBMIT_EXIT,
            timestamp=position.exit_time,
            strategy_id=position.strategy,
            symbol=position.symbol,
            reason=position.exit_reason,
            risk_dollars=position.risk_dollars,
            metadata={
                "entry_type": position.entry_type,
                "r_multiple": float(position.r_multiple),
                "quality": float(position.quality),
            },
        )
        actions.append(action)
        trade_outcomes.append(
            TradeOutcome(
                strategy_id=position.strategy,
                symbol=position.symbol,
                entry_time=position.entry_time,
                decision_time=position.decision_time,
                fill_time=position.fill_time,
                exit_time=position.exit_time,
                gross_pnl=position.pnl + position.commission,
                commission=position.commission,
                net_pnl=position.pnl,
                r_multiple=position.r_multiple,
                risk_dollars=position.risk_dollars,
                exit_reason=position.exit_reason,
                route=position.entry_type,
                metadata=dict(position.metadata),
            )
        )
    for position in state.active_positions:
        if _naive_dt(position.exit_time) > before_cmp:
            remaining.append(position)
    state.active_positions[:] = remaining


def _naive_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.replace(tzinfo=None)


def _rank_candidate(candidate: ReplayCandidate, effective: dict[str, Any]) -> float:
    mode = str(effective.get("cross_strategy_rules", {}).get("candidate_rank_mode", "diagnostic_alpha_score"))
    priority = float(effective["strategy_allocations"][candidate.strategy].get("priority", 5))
    priority_score = 1.0 / (1.0 + priority)
    if mode == "frequency_first":
        return 0.70 * priority_score + 0.30 * candidate.quality
    if mode == "expected_alpha_per_heat":
        return candidate.quality / max(candidate.heat_r, 0.10)
    if mode == "strategy_priority":
        return priority_score
    return 0.55 * candidate.quality + 0.45 * priority_score


def _candidate_quality(strategy: str, trade: TradeRecord, effective: dict[str, Any]) -> float:
    meta = trade.metadata or {}
    quality = 0.55
    if strategy == "IARIC_V5R1":
        quality = 0.72
        route = (trade.entry_type or "").upper()
        if route == "OPEN_SCORED_ENTRY":
            quality += 0.08
        elif route == "DELAYED_CONFIRM":
            quality += 0.03
        elif route in {"VWAP_BOUNCE", "AFTERNOON_RETEST"}:
            quality -= 0.06
        quality += _scaled_meta(meta, "daily_signal_score", 72.0, 92.0, 0.12)
        quality += _scaled_meta(meta, "intraday_score", 72.0, 92.0, 0.10)
        gap = _meta_float(meta, "entry_gap_pct", 0.0)
        if gap < -0.5:
            quality += 0.08
        elif gap > 0.5:
            quality -= 0.08
    else:
        quality = 0.66
        entry_type = (trade.entry_type or "").upper()
        if entry_type == "PDH_BREAKOUT":
            quality += 0.12
        elif entry_type == "OR_BREAKOUT":
            quality += 0.05
        elif entry_type == "COMBINED_BREAKOUT":
            quality += 0.02
        score = _meta_float(meta, "momentum_score", _meta_float(meta, "score", 5.0))
        quality += max(min((score - 5.0) * 0.04, 0.12), -0.12)
        rvol = _meta_float(meta, "rvol", _meta_float(meta, "entry_rvol", 2.5))
        if rvol >= 3.0:
            quality += 0.08
        if trade.sector == "Financials":
            quality -= 0.12
    return max(0.05, min(1.25, quality))


def _candidate_size_mult(strategy: str, trade: TradeRecord, effective: dict[str, Any]) -> float:
    filters = effective.get("strategy_filters", {}).get(strategy, {})
    mult = 1.0
    if strategy == "ALCB_R3":
        if trade.sector == "Financials":
            mult *= float(filters.get("financials_size_mult", 1.0) or 1.0)
        if (trade.entry_type or "").upper() == "PDH_BREAKOUT":
            mult *= float(filters.get("pdh_size_mult", 1.0) or 1.0)
        score = _meta_float(trade.metadata or {}, "momentum_score", _meta_float(trade.metadata or {}, "score", 5.0))
        has_surge = bool((trade.metadata or {}).get("bar_vol_surge", False))
        if int(round(score)) == 5 and not has_surge:
            mult *= float(filters.get("score5_no_surge_mult", 1.0) or 1.0)
    else:
        gap = _meta_float(trade.metadata or {}, "entry_gap_pct", 0.0)
        if gap > 0:
            mult *= float(filters.get("gap_up_size_mult", 1.0) or 1.0)
        if "CARRY" in (trade.exit_reason or "").upper():
            mult *= float(filters.get("carry_route_size_mult", 1.0) or 1.0)
    return max(0.0, min(1.5, mult))


def _dynamic_mult(strategy: str, strategy_recent: dict[str, deque[float]], effective: dict[str, Any]) -> float:
    dynamic = effective.get("dynamic_allocation", {})
    if not dynamic.get("enabled", False):
        return 1.0
    recent = strategy_recent.get(strategy)
    if not recent or len(recent) < 20:
        return 1.0
    avg_r = float(np.mean(recent))
    win_rate = sum(1 for value in recent if value > 0) / len(recent)
    mult = 1.0
    if avg_r > 0.20 and win_rate > 0.58:
        mult += float(dynamic.get("positive_expectancy_boost", 0.10) or 0.10)
    elif avg_r < 0.0 or win_rate < 0.45:
        mult -= float(dynamic.get("negative_expectancy_cut", 0.18) or 0.18)
    return max(float(dynamic.get("min_mult", 0.65)), min(float(dynamic.get("max_mult", 1.20)), mult))


def _drawdown_mult(equity: float, peak_equity: float, effective: dict[str, Any]) -> float:
    if peak_equity <= 0:
        return 1.0
    dd = max(0.0, (peak_equity - equity) / peak_equity)
    tiers = effective.get("portfolio_rules", {}).get("drawdown_tiers", ())
    mult = 1.0
    for threshold, tier_mult in tiers:
        if dd >= float(threshold):
            mult = float(tier_mult)
    return max(0.0, mult)


def _candidate_discrimination(accepted_r: list[float], blocked_r: list[float]) -> float:
    if not accepted_r:
        return 0.0
    if not blocked_r:
        return 1.0
    delta = float(np.mean(accepted_r) - np.mean(blocked_r))
    return max(0.0, min(1.0, 0.50 + delta / 0.60))


def _positive_slices(positions: list[PortfolioPosition]) -> int:
    if not positions:
        return 0
    ordered = sorted(positions, key=lambda item: item.entry_time)
    chunks = [chunk for chunk in np.array_split(np.array(ordered, dtype=object), 4) if len(chunk)]
    return int(sum(1 for chunk in chunks if sum(position.pnl for position in chunk) > 0))


def _months_from_positions(
    accepted: list[PortfolioPosition],
    blocked: list[BlockedCandidate],
) -> float:
    dates = [position.entry_time for position in accepted]
    dates.extend(candidate.entry_time for candidate in blocked)
    if len(dates) < 2:
        return 1.0
    start = min(dates)
    end = max(dates)
    span_days = max((end - start).total_seconds() / 86400.0, 1.0)
    return span_days / 30.4375


def _block_reason_metrics(blocked: list[BlockedCandidate]) -> dict[str, float]:
    total = len(blocked)
    if total <= 0:
        return {}
    counts: dict[str, int] = defaultdict(int)
    for candidate in blocked:
        counts[candidate.reason] += 1
    return {f"blocked_reason_{reason}": float(count) for reason, count in counts.items()}


def _scaled_meta(meta: dict[str, Any], key: str, low: float, high: float, weight: float) -> float:
    value = _meta_float(meta, key, low)
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low))) * weight


def _meta_float(meta: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(meta.get(key, default))
    except (TypeError, ValueError):
        return default
