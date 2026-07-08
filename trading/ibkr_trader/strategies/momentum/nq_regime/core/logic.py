from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from strategies.core.actions import (
    CancelAction,
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitEntry,
    SubmitProfitTarget,
    SubmitProtectiveStop,
)
from strategies.core.events import DecisionEvent
from strategies.momentum.nq_regime import config
from strategies.momentum.nq_regime.config import Grade, ModuleId, StrategyRuntimeSettings, TradeSide
from strategies.momentum.nq_regime.core.filters import active_news_event
from strategies.momentum.nq_regime.core.indicators import build_indicator_snapshot
from strategies.momentum.nq_regime.core.levels import KeyLevels, build_ib_levels
from strategies.momentum.nq_regime.core.regime import Regime, classify_regime, module_for_regime, route_candidates
from strategies.momentum.nq_regime.core.session import SessionPhase, entries_allowed, get_session_phase, should_hard_flatten
from strategies.momentum.nq_regime.core.sizing import compute_position_size
from strategies.momentum.nq_regime.core.state import BarEvent, FillEvent, OrderUpdateEvent, RegimeCoreState, clone_core_state
from strategies.momentum.nq_regime.modules import liquidity_reversion, second_wind, structural_expansion
from strategies.momentum.nq_regime.modules.base import NewsEvent, RoutingDecisionEvent, SetupCandidate
from strategies.scalp._shared.nq_contract import round_to_tick
from strategies.scalp._shared.time_utils import session_date
from strategies.scalp._shared.time_utils import to_et

_TERMINAL_STATUSES = {"cancelled", "expired", "rejected", "order_cancelled", "order_expired", "order_rejected"}


def on_bar(
    state: RegimeCoreState,
    event: BarEvent,
    *,
    scheduled_news: list[NewsEvent] | None = None,
    settings: StrategyRuntimeSettings | None = None,
) -> tuple[RegimeCoreState, list, list[DecisionEvent]]:
    settings = settings or StrategyRuntimeSettings()
    next_state = clone_core_state(state)
    actions: list = []
    events: list[DecisionEvent] = []
    _maybe_reset_day(next_state, event)
    next_state.phase = get_session_phase(event.ts)
    next_state.last_bar_ts = event.ts
    next_state.bar_index += 1
    next_state.bars_5m.append(event.bar_5m)
    next_state.bars_5m = next_state.bars_5m[-120:]
    next_state.last_decision_details["live_context"] = dict(event.live_context or {})
    if event.daily_context is not None:
        next_state.levels = event.daily_context
    if event.is_new_15m and event.bar_15m_closed is not None:
        next_state.bars_15m.append(event.bar_15m_closed)
        next_state.bars_15m = next_state.bars_15m[-120:]
    _update_ib(next_state, event)
    indicators = build_indicator_snapshot(next_state.bars_5m, next_state.bars_15m, next_state.indicators)
    next_state.indicators = indicators
    if event.is_new_15m:
        if indicators.squeeze_on:
            if next_state.second_wind_state.squeeze_duration <= 0:
                next_state.second_wind_state.squeeze_start_ts = event.ts
            next_state.second_wind_state.squeeze_duration = max(
                next_state.second_wind_state.squeeze_duration + 1,
                indicators.squeeze_duration,
            )
        elif next_state.second_wind_state.squeeze_duration > 0:
            next_state.second_wind_state.fired_ts = event.ts

    if _entry_cutoff_active(next_state) and next_state.working_entry_order_id:
        actions.append(
            CancelAction(
                symbol=settings.trade_symbol,
                target_order_id=next_state.working_entry_order_id,
                reason="entry_cutoff",
                metadata={"strategy_id": config.STRATEGY_ID, "candidate_id": next_state.last_submitted_signal_id},
            )
        )
        events.append(
            _event(
                "ENTRY_CANCEL_REQUESTED",
                event.ts,
                settings.trade_symbol,
                {"order_id": next_state.working_entry_order_id, "reason": "entry_cutoff"},
            )
        )
        if next_state.position_side is TradeSide.FLAT:
            _update_last_decision(next_state, events)
            return next_state, actions, events

    if should_hard_flatten(event.ts) and next_state.position_side is not TradeSide.FLAT and next_state.qty_open > 0:
        actions.extend(_cancel_working_exit_orders(next_state, settings.trade_symbol, reason="hard_flatten"))
        actions.append(
            FlattenPosition(
                symbol=settings.trade_symbol,
                reason="hard_flatten",
                side=next_state.position_side.exit_action_side,
                qty=next_state.qty_open,
                metadata={"strategy_id": config.STRATEGY_ID, "active_trade_id": next_state.active_trade_id},
            )
        )
        events.append(_event("HARD_FLATTEN_REQUESTED", event.ts, settings.trade_symbol, {"qty": next_state.qty_open}))
        _update_last_decision(next_state, events)
        return next_state, actions, events

    news = active_news_event(event.ts, scheduled_news)
    checkpoint = _is_regime_checkpoint(event)
    trigger = "checkpoint" if checkpoint else "event_driven"
    regime_result = classify_regime(
        next_state,
        indicators,
        event.bar_15m_closed or event.bar_5m,
        news_active=news is not None,
        trigger=trigger,
    )
    next_state.regime = regime_result.regime
    next_state.regime_scores = regime_result.scores
    next_state.active_module = module_for_regime(regime_result.regime)

    if next_state.position_side is not TradeSide.FLAT:
        management_actions, management_details = _manage_open_position(next_state, event, settings.trade_symbol)
        actions.extend(management_actions)
        events.append(_event("MANAGE_OPEN_POSITION", event.ts, settings.trade_symbol, {"regime": regime_result.regime.name, **management_details}))
        _update_last_decision(next_state, events)
        return next_state, actions, events

    if _has_working_entry(next_state):
        events.append(_event("MANAGE_OPEN_POSITION", event.ts, settings.trade_symbol, {"regime": regime_result.regime.name}))
        _update_last_decision(next_state, events)
        return next_state, actions, events

    candidates = _generate_candidates(next_state, event, settings)
    if next_state.daily_locked_out:
        for candidate in candidates:
            candidate.details["lockout"] = "daily_locked_out"
        events.append(_event("DAILY_LOCKOUT", event.ts, settings.trade_symbol, {"daily_realized_r": next_state.daily_realized_r}))
        _update_last_decision(next_state, events)
        return next_state, actions, events
    if news is not None:
        events.append(_event("NEWS_VETO", event.ts, settings.trade_symbol, {"news": news.name}))
        _update_last_decision(next_state, events)
        return next_state, actions, events

    selected, blocked, reason = route_candidates(regime_result.regime, candidates)
    routing = RoutingDecisionEvent(
        ts=event.ts,
        regime=regime_result.regime,
        regime_scores=regime_result.scores,
        selected_module=selected.module if selected else ModuleId.NONE,
        selected_candidate_id=selected.candidate_id if selected else None,
        blocked_candidates=blocked,
        reason_code=reason,
        confidence=regime_result.confidence,
    )
    next_state.routing_log.append(routing)
    next_state.routing_log = next_state.routing_log[-500:]
    events.append(
        _event(
            "ROUTING_DECISION",
            event.ts,
            settings.trade_symbol,
            {
                "regime": regime_result.regime.name,
                "selected": selected.candidate_id if selected else None,
                "selected_module": selected.module.value if selected else ModuleId.NONE.value,
                "selected_score": selected.score if selected else 0,
                "selected_grade": selected.grade.value if selected else "",
                "blocked": len(blocked),
                "reason": reason,
                "confidence": regime_result.confidence,
                "margin": regime_result.margin,
                "regime_scores": _regime_scores_payload(regime_result.scores),
                "candidate_count": len(candidates),
                "candidate_inventory": [_candidate_payload(candidate) for candidate in candidates],
                "blocked_candidates": [_blocked_candidate_payload(item) for item in blocked],
            },
        )
    )
    if selected is None:
        _update_last_decision(next_state, events)
        return next_state, actions, events
    if not entries_allowed(next_state.phase, selected.module.value):
        events.append(_event("ENTRY_BLOCKED_BY_SESSION", event.ts, settings.trade_symbol, {"module": selected.module.value, "phase": next_state.phase.value}))
        _update_last_decision(next_state, events)
        return next_state, actions, events
    qty, risk_pct = compute_position_size(
        equity=settings.initial_equity,
        grade=selected.grade,
        stop_distance=abs(selected.entry_price - selected.stop_price),
        trade_symbol=settings.trade_symbol,
        max_contracts=settings.max_contracts,
        after_loss=next_state.daily_losses > 0,
        module=selected.module,
    )
    if qty <= 0:
        events.append(_event("ENTRY_BLOCKED_BY_SIZE", event.ts, settings.trade_symbol, {"candidate": selected.candidate_id}))
        _update_last_decision(next_state, events)
        return next_state, actions, events
    selected.details["risk_pct"] = risk_pct
    selected.details["regime"] = regime_result.regime.name
    selected.details["regime_confidence"] = regime_result.confidence
    selected.details["regime_margin"] = regime_result.margin
    selected.details["active_module"] = next_state.active_module.value
    action = _entry_action(selected, qty, settings.trade_symbol)
    next_state.working_entry_order_id = action.client_order_id
    next_state.order_to_role[action.client_order_id] = "entry"
    next_state.order_to_candidate[action.client_order_id] = selected.candidate_id
    next_state.pending_candidates[selected.candidate_id] = selected
    next_state.last_submitted_signal_id = selected.candidate_id
    actions.append(action)
    events.append(_event("ENTRY_REQUESTED", event.ts, settings.trade_symbol, {"candidate": selected.candidate_id, "module": selected.module.value, "qty": qty, "score": selected.score, "grade": selected.grade.value}))
    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_fill(
    state: RegimeCoreState,
    fill: FillEvent,
    *,
    settings: StrategyRuntimeSettings | None = None,
) -> tuple[RegimeCoreState, list, list[DecisionEvent]]:
    settings = settings or StrategyRuntimeSettings()
    next_state = clone_core_state(state)
    actions: list = []
    events: list[DecisionEvent] = []
    role = next_state.order_to_role.pop(fill.oms_order_id, fill.order_role)
    candidate_id = next_state.order_to_candidate.pop(fill.oms_order_id, "")
    symbol = fill.symbol or settings.trade_symbol
    if role == "entry":
        candidate = next_state.pending_candidates.pop(candidate_id, None) if candidate_id else _candidate_from_last_signal(next_state)
        side = candidate.side if candidate else TradeSide.LONG
        stop = candidate.stop_price if candidate else fill.fill_price
        targets = candidate.targets if candidate else ()
        next_state.working_entry_order_id = None
        next_state.position_side = side
        next_state.entry_price = fill.fill_price
        next_state.entry_time = fill.fill_time
        next_state.entry_bar_index = next_state.bar_index
        next_state.stop_price = stop
        next_state.qty = fill.fill_qty
        next_state.qty_open = fill.fill_qty
        next_state.entry_module = candidate.module if candidate else ModuleId.NONE
        next_state.setup_grade = candidate.grade if candidate else Grade.INVALID
        next_state.setup_score = candidate.score if candidate else 0
        next_state.initial_risk_points = abs(fill.fill_price - stop)
        next_state.planned_targets = tuple(targets)
        next_state.active_trade_id = f"{config.STRATEGY_ID}-{fill.fill_time.strftime('%Y%m%d%H%M%S')}"
        next_state.daily_trades += 1
        if next_state.setup_grade is Grade.A_PLUS:
            next_state.daily_full_risk_trades += 1
        stop_id = f"{symbol}-nqreg-stop-{next_state.active_trade_id}"
        target_id = f"{symbol}-nqreg-t1-{next_state.active_trade_id}"
        oca_group = _stage_oca_group(next_state, 1)
        next_state.working_stop_order_id = stop_id
        target_qty = max(1, min(fill.fill_qty, int(round(fill.fill_qty * _target1_fraction(next_state)))))
        next_state.working_target_order_ids = (target_id,) if targets else ()
        next_state.order_to_role[stop_id] = "stop"
        if targets:
            next_state.order_to_role[target_id] = "target_1"
        exit_side = side.exit_action_side
        actions.append(
            SubmitProtectiveStop(
                client_order_id=stop_id,
                symbol=symbol,
                side=exit_side,
                qty=fill.fill_qty,
                stop_price=stop,
                oca_group=oca_group if targets else "",
                role="protective_stop",
                metadata={"active_trade_id": next_state.active_trade_id, "stop_for_risk": stop},
            )
        )
        if targets:
            actions.append(
                SubmitProfitTarget(
                    client_order_id=target_id,
                    symbol=symbol,
                    side=exit_side,
                    qty=target_qty,
                    limit_price=targets[0],
                    oca_group=oca_group,
                    role="target_1",
                    metadata={"active_trade_id": next_state.active_trade_id},
                )
            )
        events.append(_event("ENTRY_FILLED", fill.fill_time, symbol, {"price": fill.fill_price, "qty": fill.fill_qty, "stop": stop}))
        _update_last_decision(next_state, events)
        return next_state, actions, events

    if next_state.position_side is TradeSide.FLAT or next_state.qty_open <= 0:
        _update_last_decision(next_state, events)
        return next_state, actions, events
    exit_qty = min(fill.fill_qty, next_state.qty_open)
    if exit_qty <= 0:
        return next_state, actions, events
    point_value = settings.trade_spec.point_value
    r_mult = _realized_r(next_state, fill.fill_price, exit_qty, fill.commission, point_value)
    next_state.daily_realized_r += r_mult
    next_state.daily_realized_pnl += _realized_pnl(next_state, fill.fill_price, exit_qty, point_value) - fill.commission
    next_state.qty_open = max(0, next_state.qty_open - exit_qty)
    if role in {"target_1", "target_2", "partial"}:
        next_state.partial_taken += 1
        next_state.stop_at_be = True
        if fill.oms_order_id in next_state.working_target_order_ids:
            next_state.working_target_order_ids = tuple(
                order_id for order_id in next_state.working_target_order_ids if order_id != fill.oms_order_id
            )
        if next_state.qty_open > 0 and next_state.working_stop_order_id and config.MOVE_STOP_TO_BE_ON_T1:
            next_stage = 2 if role == "target_1" else 3
            next_state.stop_price = next_state.entry_price
            actions.append(
                ReplaceProtectiveStop(
                    symbol=symbol,
                    target_order_id=next_state.working_stop_order_id,
                    side=next_state.position_side.exit_action_side,
                    stop_price=next_state.entry_price,
                    qty=next_state.qty_open,
                    reason="target_filled_move_stop_to_be",
                    oca_group=_stage_oca_group(next_state, next_stage),
                    metadata={"active_trade_id": next_state.active_trade_id},
                )
            )
            if role == "target_1" and len(next_state.planned_targets) > 1 and next_state.qty_open > 1:
                target2_id = f"{symbol}-nqreg-t2-{next_state.active_trade_id}"
                target2_qty = max(1, min(next_state.qty_open, int(round(next_state.qty * _target2_fraction(next_state)))))
                next_state.working_target_order_ids = (target2_id,)
                next_state.order_to_role[target2_id] = "target_2"
                actions.append(
                    SubmitProfitTarget(
                        client_order_id=target2_id,
                        symbol=symbol,
                        side=next_state.position_side.exit_action_side,
                        qty=target2_qty,
                        limit_price=next_state.planned_targets[1],
                        oca_group=_stage_oca_group(next_state, 2),
                        role="target_2",
                        metadata={"active_trade_id": next_state.active_trade_id},
                    )
                )
        events.append(_event("PARTIAL_EXIT_FILLED", fill.fill_time, symbol, {"qty": exit_qty, "price": fill.fill_price, "r": r_mult}))
    else:
        if role == "stop" or r_mult < 0:
            next_state.daily_losses += 1
        events.append(_event("EXIT_FILLED", fill.fill_time, symbol, {"qty": exit_qty, "price": fill.fill_price, "role": role, "r": r_mult}))
    if next_state.qty_open <= 0:
        actions.extend(_cancel_working_exit_orders(next_state, symbol, reason=f"{role}_position_flat", exclude={fill.oms_order_id}))
        _clear_position(next_state)
    _apply_daily_lockout(next_state)
    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_order_update(
    state: RegimeCoreState,
    update: OrderUpdateEvent,
) -> tuple[RegimeCoreState, list, list[DecisionEvent]]:
    next_state = clone_core_state(state)
    events: list[DecisionEvent] = []
    status = update.status.lower()
    if status in _TERMINAL_STATUSES and update.oms_order_id:
        role = next_state.order_to_role.pop(update.oms_order_id, update.order_role)
        candidate_id = next_state.order_to_candidate.pop(update.oms_order_id, None)
        if role == "entry":
            next_state.working_entry_order_id = None
            if candidate_id:
                next_state.pending_candidates.pop(candidate_id, None)
                if next_state.last_submitted_signal_id == candidate_id:
                    next_state.last_submitted_signal_id = None
        if role == "stop" and next_state.working_stop_order_id == update.oms_order_id:
            next_state.working_stop_order_id = None
        if role.startswith("target") and update.oms_order_id in next_state.working_target_order_ids:
            next_state.working_target_order_ids = tuple(
                order_id for order_id in next_state.working_target_order_ids if order_id != update.oms_order_id
            )
        events.append(_event("ORDER_TERMINAL", update.timestamp, update.symbol, {"status": status, "role": role, "reason": update.reason}))
    _update_last_decision(next_state, events)
    return next_state, [], events


def _maybe_reset_day(state: RegimeCoreState, event: BarEvent) -> None:
    day = session_date(event.ts).isoformat()
    if state.active_session_date == day:
        return
    state.active_session_date = day
    state.ib_levels = build_ib_levels(0.0, 0.0)
    state.ib_high_working = 0.0
    state.ib_low_working = 0.0
    state.ib_locked = False
    state.ib_type = config.IBType.UNCLASSIFIED
    state.daily_trades = 0
    state.daily_full_risk_trades = 0
    state.daily_losses = 0
    state.daily_realized_r = 0.0
    state.daily_realized_pnl = 0.0
    state.daily_locked_out = False
    state.bars_5m = []
    state.bars_15m = []
    state.routing_log = []
    state.expansion_state = type(state.expansion_state)()
    state.reversion_state = type(state.reversion_state)()
    state.second_wind_state = type(state.second_wind_state)()
    if state.position_side is TradeSide.FLAT and state.qty_open <= 0:
        state.working_entry_order_id = None
        state.working_stop_order_id = None
        state.working_target_order_ids = ()
        state.order_to_role.clear()
        state.order_to_candidate.clear()
        state.pending_candidates.clear()
        state.pending_cancel_reason = None
        state.last_submitted_signal_id = None


def _update_ib(state: RegimeCoreState, event: BarEvent) -> None:
    et_time = to_et(event.ts).time()
    if config.RTH_OPEN_ET < et_time <= config.IB_END_ET:
        state.ib_high_working = max(state.ib_high_working or event.bar_5m.high, event.bar_5m.high)
        state.ib_low_working = min(state.ib_low_working or event.bar_5m.low, event.bar_5m.low)
    if not state.ib_locked and et_time >= config.IB_END_ET and state.phase is not SessionPhase.PRE_MARKET:
        state.ib_levels = build_ib_levels(state.ib_high_working, state.ib_low_working)
        state.ib_locked = state.ib_levels.range_pts > 0
        state.ib_type = state.ib_levels.ib_type
        if state.ib_locked:
            base_levels = state.levels or KeyLevels()
            state.levels = replace(base_levels, orh=state.ib_levels.high, orl=state.ib_levels.low)


def _is_regime_checkpoint(event: BarEvent) -> bool:
    et = to_et(event.ts).time()
    return (et.hour, et.minute) in {(10, 0), (11, 30), (13, 15)}


def _generate_candidates(state: RegimeCoreState, event: BarEvent, settings: StrategyRuntimeSettings) -> list[SetupCandidate]:
    candidates: list[SetupCandidate] = []
    indicators = state.indicators
    if indicators is None:
        return candidates
    if settings.enable_structural_expansion:
        item = structural_expansion.evaluate(state, event, indicators)
        if item is not None:
            candidates.append(item)
    if settings.enable_liquidity_reversion:
        item = liquidity_reversion.evaluate(state, event, indicators)
        if item is not None:
            candidates.append(item)
    if settings.enable_second_wind:
        item = second_wind.evaluate(state, event, indicators)
        if item is not None:
            candidates.append(item)
    return candidates


def _has_working_entry(state: RegimeCoreState) -> bool:
    return bool(state.working_entry_order_id)


def _manage_open_position(
    state: RegimeCoreState,
    event: BarEvent,
    symbol: str,
) -> tuple[list, dict]:
    actions: list = []
    if state.qty_open <= 0 or state.initial_risk_points <= 0:
        return actions, {}
    mfe_r = _bar_mfe_r(state, event.bar_5m)
    held_bars = max(0, state.bar_index - state.entry_bar_index) if state.entry_bar_index >= 0 else 0
    details = {"mfe_r": mfe_r, "held_bars": held_bars}

    module_exit_reason = _module_specific_exit_reason(state, event)
    if module_exit_reason:
        actions.extend(_cancel_working_exit_orders(state, symbol, reason=module_exit_reason))
        actions.append(
            FlattenPosition(
                symbol=symbol,
                reason=module_exit_reason,
                side=state.position_side.exit_action_side,
                qty=state.qty_open,
                metadata={"strategy_id": config.STRATEGY_ID, "active_trade_id": state.active_trade_id},
            )
        )
        details["management_action"] = module_exit_reason
        return actions, details

    if (
        state.entry_module is ModuleId.LIQUIDITY_REVERSION
        and config.REVERSION_TIME_STOP_ENABLED
        and held_bars >= config.REVERSION_TIME_STOP_BARS
        and mfe_r < config.REVERSION_TIME_STOP_MIN_MFE_R
    ):
        actions.extend(_cancel_working_exit_orders(state, symbol, reason="reversion_time_stop"))
        actions.append(
            FlattenPosition(
                symbol=symbol,
                reason="reversion_time_stop",
                side=state.position_side.exit_action_side,
                qty=state.qty_open,
                metadata={"strategy_id": config.STRATEGY_ID, "active_trade_id": state.active_trade_id},
            )
        )
        details["management_action"] = "reversion_time_stop"
        return actions, details

    if config.TIME_STOP_ENABLED and held_bars >= config.TIME_STOP_BARS and mfe_r < config.TIME_STOP_MIN_MFE_R:
        actions.extend(_cancel_working_exit_orders(state, symbol, reason="time_stop"))
        actions.append(
            FlattenPosition(
                symbol=symbol,
                reason="time_stop",
                side=state.position_side.exit_action_side,
                qty=state.qty_open,
                metadata={"strategy_id": config.STRATEGY_ID, "active_trade_id": state.active_trade_id},
            )
        )
        details["management_action"] = "time_stop"
        return actions, details

    if (
        state.entry_module is ModuleId.STRUCTURAL_EXPANSION
        and config.STRUCTURAL_TIME_STOP_ENABLED
        and held_bars >= config.STRUCTURAL_TIME_STOP_BARS
        and mfe_r < config.STRUCTURAL_TIME_STOP_MIN_MFE_R
    ):
        actions.extend(_cancel_working_exit_orders(state, symbol, reason="structural_time_stop"))
        actions.append(
            FlattenPosition(
                symbol=symbol,
                reason="structural_time_stop",
                side=state.position_side.exit_action_side,
                qty=state.qty_open,
                metadata={"strategy_id": config.STRATEGY_ID, "active_trade_id": state.active_trade_id},
            )
        )
        details["management_action"] = "structural_time_stop"
        return actions, details

    if (
        state.entry_module is ModuleId.SECOND_WIND
        and config.SECOND_WIND_TIME_STOP_ENABLED
        and held_bars >= config.SECOND_WIND_TIME_STOP_BARS
        and mfe_r < config.SECOND_WIND_TIME_STOP_MIN_MFE_R
    ):
        actions.extend(_cancel_working_exit_orders(state, symbol, reason="second_wind_time_stop"))
        actions.append(
            FlattenPosition(
                symbol=symbol,
                reason="second_wind_time_stop",
                side=state.position_side.exit_action_side,
                qty=state.qty_open,
                metadata={"strategy_id": config.STRATEGY_ID, "active_trade_id": state.active_trade_id},
            )
        )
        details["management_action"] = "second_wind_time_stop"
        return actions, details

    new_stop = _profit_protection_stop(state, event.bar_5m, mfe_r)
    if new_stop is not None and state.working_stop_order_id:
        state.stop_price = new_stop
        actions.append(
            ReplaceProtectiveStop(
                symbol=symbol,
                target_order_id=state.working_stop_order_id,
                side=state.position_side.exit_action_side,
                stop_price=new_stop,
                qty=state.qty_open,
                reason="profit_protection",
                oca_group=_stage_oca_group(state, max(2, state.partial_taken + 1)),
                metadata={"active_trade_id": state.active_trade_id},
            )
        )
        details["management_action"] = "profit_protection"
        details["new_stop"] = new_stop
    return actions, details


def _module_specific_exit_reason(state: RegimeCoreState, event: BarEvent) -> str:
    if state.entry_module is ModuleId.LIQUIDITY_REVERSION and config.REVERSION_VWAP_REACTION_EXIT_ENABLED:
        reason = _reversion_vwap_reaction_exit(state, event.bar_5m)
        if reason:
            return reason
    if state.entry_module is ModuleId.LIQUIDITY_REVERSION and config.REVERSION_VWAP_TOUCH_EXIT_ENABLED:
        return _reversion_vwap_touch_exit(state, event.bar_5m)
    if state.entry_module is ModuleId.STRUCTURAL_EXPANSION and config.STRUCTURAL_FAILURE_EXIT_ENABLED:
        return _structural_failure_exit(state, event)
    if state.entry_module is ModuleId.SECOND_WIND and config.SECOND_WIND_EMA_TRAIL_EXIT_ENABLED:
        return _second_wind_ema_exit(state, event)
    return ""


def _reversion_vwap_reaction_exit(state: RegimeCoreState, bar: Any) -> str:
    if len(state.planned_targets) < 2:
        return ""
    target = state.planned_targets[1]
    buffer = 2 * config.TICK_SIZE
    if state.position_side is TradeSide.LONG and bar.high >= target and bar.close < target - buffer:
        return "reversion_vwap_rejection"
    if state.position_side is TradeSide.SHORT and bar.low <= target and bar.close > target + buffer:
        return "reversion_vwap_rejection"
    return ""


def _reversion_vwap_touch_exit(state: RegimeCoreState, bar: Any) -> str:
    if len(state.planned_targets) < 2:
        return ""
    target = state.planned_targets[1]
    if state.position_side is TradeSide.LONG and bar.high >= target:
        return "reversion_vwap_touch"
    if state.position_side is TradeSide.SHORT and bar.low <= target:
        return "reversion_vwap_touch"
    return ""


def _structural_failure_exit(state: RegimeCoreState, event: BarEvent) -> str:
    if config.STRUCTURAL_FAST_FAILURE_EXIT_ENABLED and state.ib_locked:
        bar_5m = event.bar_5m
        if state.position_side is TradeSide.LONG and bar_5m.close < state.ib_levels.high:
            return "structural_fast_back_inside_ib"
        if state.position_side is TradeSide.SHORT and bar_5m.close > state.ib_levels.low:
            return "structural_fast_back_inside_ib"
    bar = event.bar_15m_closed if event.is_new_15m and event.bar_15m_closed is not None else None
    if bar is None or not state.ib_locked:
        return ""
    if state.position_side is TradeSide.LONG:
        if bar.close < state.ib_levels.high:
            return "structural_back_inside_ib"
        if state.partial_taken > 0 and state.indicators and bar.close < state.indicators.vwap:
            return "structural_vwap_lost"
    if state.position_side is TradeSide.SHORT:
        if bar.close > state.ib_levels.low:
            return "structural_back_inside_ib"
        if state.partial_taken > 0 and state.indicators and bar.close > state.indicators.vwap:
            return "structural_vwap_lost"
    return ""


def _second_wind_ema_exit(state: RegimeCoreState, event: BarEvent) -> str:
    if config.SECOND_WIND_EMA_TRAIL_REQUIRES_PARTIAL and state.partial_taken <= 0:
        return ""
    if not event.is_new_15m or event.bar_15m_closed is None or state.indicators is None:
        return ""
    ema = state.indicators.ema9_15m or state.indicators.ema20_15m
    if ema <= 0:
        return ""
    close = event.bar_15m_closed.close
    if state.position_side is TradeSide.LONG and close < ema:
        return "second_wind_ema_trail"
    if state.position_side is TradeSide.SHORT and close > ema:
        return "second_wind_ema_trail"
    return ""


def _bar_mfe_r(state: RegimeCoreState, bar: Any) -> float:
    if state.initial_risk_points <= 0:
        return 0.0
    if state.position_side is TradeSide.LONG:
        return max(0.0, (bar.high - state.entry_price) / state.initial_risk_points)
    if state.position_side is TradeSide.SHORT:
        return max(0.0, (state.entry_price - bar.low) / state.initial_risk_points)
    return 0.0


def _profit_protection_stop(state: RegimeCoreState, bar: Any, mfe_r: float) -> float | None:
    lock_r = 0.0
    floor_enabled = config.PROFIT_FLOOR_ENABLED
    floor_trigger = config.PROFIT_FLOOR_TRIGGER_R
    floor_lock = config.PROFIT_FLOOR_LOCK_R
    ratchet_enabled = config.MFE_RATCHET_ENABLED
    ratchet_trigger = config.MFE_RATCHET_TRIGGER_R
    ratchet_floor_pct = config.MFE_RATCHET_FLOOR_PCT
    if state.entry_module is ModuleId.LIQUIDITY_REVERSION:
        floor_enabled = floor_enabled or config.REVERSION_PROFIT_FLOOR_ENABLED
        floor_trigger = (
            min(floor_trigger, config.REVERSION_PROFIT_FLOOR_TRIGGER_R)
            if config.PROFIT_FLOOR_ENABLED and config.REVERSION_PROFIT_FLOOR_ENABLED
            else config.REVERSION_PROFIT_FLOOR_TRIGGER_R
            if config.REVERSION_PROFIT_FLOOR_ENABLED
            else floor_trigger
        )
        floor_lock = max(
            floor_lock,
            config.REVERSION_PROFIT_FLOOR_LOCK_R if config.REVERSION_PROFIT_FLOOR_ENABLED else floor_lock,
        )
        ratchet_enabled = ratchet_enabled or config.REVERSION_MFE_RATCHET_ENABLED
        ratchet_trigger = (
            min(ratchet_trigger, config.REVERSION_MFE_RATCHET_TRIGGER_R)
            if config.MFE_RATCHET_ENABLED and config.REVERSION_MFE_RATCHET_ENABLED
            else config.REVERSION_MFE_RATCHET_TRIGGER_R
            if config.REVERSION_MFE_RATCHET_ENABLED
            else ratchet_trigger
        )
        ratchet_floor_pct = max(
            ratchet_floor_pct,
            config.REVERSION_MFE_RATCHET_FLOOR_PCT if config.REVERSION_MFE_RATCHET_ENABLED else ratchet_floor_pct,
        )
    if state.entry_module is ModuleId.STRUCTURAL_EXPANSION:
        floor_enabled = floor_enabled or config.STRUCTURAL_PROFIT_FLOOR_ENABLED
        floor_trigger = (
            min(floor_trigger, config.STRUCTURAL_PROFIT_FLOOR_TRIGGER_R)
            if config.PROFIT_FLOOR_ENABLED and config.STRUCTURAL_PROFIT_FLOOR_ENABLED
            else config.STRUCTURAL_PROFIT_FLOOR_TRIGGER_R
            if config.STRUCTURAL_PROFIT_FLOOR_ENABLED
            else floor_trigger
        )
        floor_lock = max(floor_lock, config.STRUCTURAL_PROFIT_FLOOR_LOCK_R if config.STRUCTURAL_PROFIT_FLOOR_ENABLED else floor_lock)
        ratchet_enabled = ratchet_enabled or config.STRUCTURAL_MFE_RATCHET_ENABLED
        ratchet_trigger = (
            min(ratchet_trigger, config.STRUCTURAL_MFE_RATCHET_TRIGGER_R)
            if config.MFE_RATCHET_ENABLED and config.STRUCTURAL_MFE_RATCHET_ENABLED
            else config.STRUCTURAL_MFE_RATCHET_TRIGGER_R
            if config.STRUCTURAL_MFE_RATCHET_ENABLED
            else ratchet_trigger
        )
        ratchet_floor_pct = max(
            ratchet_floor_pct,
            config.STRUCTURAL_MFE_RATCHET_FLOOR_PCT if config.STRUCTURAL_MFE_RATCHET_ENABLED else ratchet_floor_pct,
        )
    if state.entry_module is ModuleId.SECOND_WIND:
        floor_enabled = floor_enabled or config.SECOND_WIND_PROFIT_FLOOR_ENABLED
        floor_trigger = (
            min(floor_trigger, config.SECOND_WIND_PROFIT_FLOOR_TRIGGER_R)
            if config.PROFIT_FLOOR_ENABLED and config.SECOND_WIND_PROFIT_FLOOR_ENABLED
            else config.SECOND_WIND_PROFIT_FLOOR_TRIGGER_R
            if config.SECOND_WIND_PROFIT_FLOOR_ENABLED
            else floor_trigger
        )
        floor_lock = max(floor_lock, config.SECOND_WIND_PROFIT_FLOOR_LOCK_R if config.SECOND_WIND_PROFIT_FLOOR_ENABLED else floor_lock)
        ratchet_enabled = ratchet_enabled or config.SECOND_WIND_MFE_RATCHET_ENABLED
        ratchet_trigger = (
            min(ratchet_trigger, config.SECOND_WIND_MFE_RATCHET_TRIGGER_R)
            if config.MFE_RATCHET_ENABLED and config.SECOND_WIND_MFE_RATCHET_ENABLED
            else config.SECOND_WIND_MFE_RATCHET_TRIGGER_R
            if config.SECOND_WIND_MFE_RATCHET_ENABLED
            else ratchet_trigger
        )
        ratchet_floor_pct = max(
            ratchet_floor_pct,
            config.SECOND_WIND_MFE_RATCHET_FLOOR_PCT if config.SECOND_WIND_MFE_RATCHET_ENABLED else ratchet_floor_pct,
        )
    if floor_enabled and mfe_r >= floor_trigger:
        lock_r = max(lock_r, floor_lock)
    if ratchet_enabled and mfe_r >= ratchet_trigger:
        lock_r = max(lock_r, mfe_r * ratchet_floor_pct)
    if lock_r <= 0.0 and not (floor_enabled and mfe_r >= floor_trigger):
        return None
    raw_stop = state.entry_price + state.position_side.sign * lock_r * state.initial_risk_points
    if state.position_side is TradeSide.LONG:
        capped = min(raw_stop, bar.close - config.TICK_SIZE)
        new_stop = round_to_tick(capped, config.TICK_SIZE, "down")
        if new_stop > state.stop_price + config.TICK_SIZE:
            return new_stop
    if state.position_side is TradeSide.SHORT:
        capped = max(raw_stop, bar.close + config.TICK_SIZE)
        new_stop = round_to_tick(capped, config.TICK_SIZE, "up")
        if state.stop_price <= 0 or new_stop < state.stop_price - config.TICK_SIZE:
            return new_stop
    return None


def _target1_fraction(state: RegimeCoreState) -> float:
    if state.entry_module is ModuleId.STRUCTURAL_EXPANSION:
        return config.STRUCTURAL_TARGET1_QTY_FRACTION
    if state.entry_module is ModuleId.LIQUIDITY_REVERSION:
        return config.REVERSION_TARGET1_QTY_FRACTION
    if state.entry_module is ModuleId.SECOND_WIND:
        return config.SECOND_WIND_TARGET1_QTY_FRACTION
    return config.TARGET1_QTY_FRACTION


def _target2_fraction(state: RegimeCoreState) -> float:
    if state.entry_module is ModuleId.STRUCTURAL_EXPANSION:
        return config.STRUCTURAL_TARGET2_QTY_FRACTION
    if state.entry_module is ModuleId.LIQUIDITY_REVERSION:
        return config.REVERSION_TARGET2_QTY_FRACTION
    if state.entry_module is ModuleId.SECOND_WIND:
        return config.SECOND_WIND_TARGET2_QTY_FRACTION
    return config.TARGET2_QTY_FRACTION


def _entry_cutoff_active(state: RegimeCoreState) -> bool:
    return state.phase in {SessionPhase.MANAGE_ONLY, SessionPhase.HARD_FLATTEN, SessionPhase.CLOSED}


def _candidate_from_last_signal(state: RegimeCoreState) -> SetupCandidate | None:
    if not state.last_submitted_signal_id:
        return None
    return state.pending_candidates.pop(state.last_submitted_signal_id, None)


def _entry_action(candidate: SetupCandidate, qty: int, symbol: str) -> SubmitEntry:
    order_type = (
        "MARKET"
        if candidate.entry_model
        in {
            "breakout_close",
            "reclaim_close",
            "adaptive_reclaim_close",
            "continuation_reentry",
            "sw_reclaim_close",
            "sw_micro_break_close",
            "sw_acceptance_close",
            "sw_second_leg_close",
        }
        else "STOP"
        if candidate.entry_model in {"momentum_breakout", "structure_shift"}
        else "LIMIT"
    )
    retest_like = any(token in candidate.entry_model for token in ("retest", "pullback"))
    ttl_minutes = 0 if order_type == "MARKET" else config.ENTRY_TTL_RETEST_MINUTES if retest_like else config.ENTRY_TTL_MOMENTUM_MINUTES
    return SubmitEntry(
        client_order_id=f"{symbol}-nqreg-entry-{candidate.candidate_id}",
        symbol=symbol,
        side=candidate.side.action_side,
        qty=qty,
        order_type=order_type,
        limit_price=candidate.entry_price if order_type == "LIMIT" else None,
        stop_price=candidate.entry_price if order_type == "STOP" else None,
        role="entry",
        risk_context={
            "stop_for_risk": candidate.stop_price,
            "planned_entry_price": candidate.entry_price,
            "invalidation_price": candidate.invalidation_price,
        },
        metadata={
            "candidate_id": candidate.candidate_id,
            "module": candidate.module.value,
            "grade": candidate.grade.value,
            "score": candidate.score,
            "signal_ts": candidate.timestamp.isoformat(),
            "setup_type": candidate.setup_type,
            "entry_model": candidate.entry_model,
            "level": candidate.level,
            "target_room_r": candidate.target_room_r,
            "targets": tuple(candidate.targets),
            "invalidation_price": candidate.invalidation_price,
            "candidate_details": dict(candidate.details),
            "ttl_minutes": ttl_minutes,
        },
    )


def _regime_scores_payload(scores) -> dict[str, float]:
    return {
        "expansion": float(getattr(scores, "expansion", 0.0) or 0.0),
        "reversion": float(getattr(scores, "reversion", 0.0) or 0.0),
        "pm_continuation": float(getattr(scores, "pm_continuation", 0.0) or 0.0),
        "news_distorted": float(getattr(scores, "news_distorted", 0.0) or 0.0),
        "dead_chop": float(getattr(scores, "dead_chop", 0.0) or 0.0),
    }


def _candidate_payload(candidate: SetupCandidate) -> dict:
    details = dict(candidate.details or {})
    return {
        "candidate_id": candidate.candidate_id,
        "module": candidate.module.value,
        "side": candidate.side.value,
        "setup_type": candidate.setup_type,
        "entry_model": candidate.entry_model,
        "grade": candidate.grade.value,
        "score": candidate.score,
        "valid": candidate.valid,
        "vetoes": list(candidate.vetoes),
        "target_room_r": float(candidate.target_room_r or 0.0),
        "stop_distance_points": float(abs(candidate.entry_price - candidate.stop_price)),
        "details": {
            "ib_type": details.get("ib_type", ""),
            "body_pct": details.get("body_pct", 0.0),
            "close_location": details.get("close_location", 0.0),
            "penetration": details.get("penetration", 0.0),
            "value_factors": details.get("value_factors", 0),
            "vwap_room_r": details.get("vwap_room_r", 0.0),
            "squeeze_duration": details.get("squeeze_duration", 0),
            "bias_score": details.get("bias_score", 0),
            "volume_multiple": details.get("volume_multiple", 0.0),
        },
    }


def _blocked_candidate_payload(item) -> dict:
    payload = _candidate_payload(item.candidate)
    payload["block_reason"] = item.block_reason
    return payload


def _realized_pnl(state: RegimeCoreState, exit_price: float, qty: int, point_value: float) -> float:
    return (exit_price - state.entry_price) * state.position_side.sign * qty * point_value


def _realized_r(state: RegimeCoreState, exit_price: float, qty: int, commission: float, point_value: float) -> float:
    if state.initial_risk_points <= 0 or state.qty <= 0:
        return 0.0
    pnl = _realized_pnl(state, exit_price, qty, point_value) - commission
    risk_dollars = state.initial_risk_points * state.qty * point_value
    return pnl / risk_dollars if risk_dollars > 0 else 0.0


def _clear_position(state: RegimeCoreState) -> None:
    state.position_side = TradeSide.FLAT
    state.entry_price = 0.0
    state.entry_time = None
    state.entry_bar_index = -1
    state.stop_price = 0.0
    state.qty = 0
    state.qty_open = 0
    state.entry_module = ModuleId.NONE
    state.setup_grade = Grade.INVALID
    state.setup_score = 0
    state.initial_risk_points = 0.0
    state.planned_targets = ()
    state.partial_taken = 0
    state.stop_at_be = False
    state.active_trade_id = None
    state.working_stop_order_id = None
    state.working_target_order_ids = ()


def _cancel_working_exit_orders(
    state: RegimeCoreState,
    symbol: str,
    *,
    reason: str,
    exclude: set[str] | None = None,
) -> list[CancelAction]:
    excluded = exclude or set()
    order_ids = []
    if state.working_stop_order_id:
        order_ids.append(state.working_stop_order_id)
    order_ids.extend(state.working_target_order_ids)
    return [
        CancelAction(
            symbol=symbol,
            target_order_id=order_id,
            reason=reason,
            metadata={"strategy_id": config.STRATEGY_ID, "active_trade_id": state.active_trade_id},
        )
        for order_id in dict.fromkeys(order_ids)
        if order_id and order_id not in excluded
    ]


def _stage_oca_group(state: RegimeCoreState, stage: int) -> str:
    return f"{config.STRATEGY_ID}-{state.active_trade_id or 'pending'}-stage-{stage}"


def _apply_daily_lockout(state: RegimeCoreState) -> None:
    if state.daily_trades >= config.MAX_TRADES_PER_DAY:
        state.daily_locked_out = True
    if state.daily_losses >= config.MAX_LOSSES_PER_DAY:
        state.daily_locked_out = True
    if state.daily_realized_r <= config.MAX_DAILY_REALIZED_R_LOSS:
        state.daily_locked_out = True
    if state.daily_full_risk_trades >= config.MAX_FULL_RISK_TRADES:
        state.daily_locked_out = True


def _event(code: str, ts: datetime, symbol: str, details: dict) -> DecisionEvent:
    return DecisionEvent(code=code, ts=ts or datetime.now(timezone.utc), symbol=symbol, timeframe="5m", details=dict(details))


def _update_last_decision(state: RegimeCoreState, events: list[DecisionEvent]) -> None:
    if not events:
        return
    latest = events[-1]
    state.last_decision_code = latest.code
    state.last_decision_details = dict(latest.details)
    state.last_bar_ts = latest.ts
