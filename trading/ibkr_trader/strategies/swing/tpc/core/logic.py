from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from strategies.core.actions import FlattenPosition, ReplaceProtectiveStop, SubmitAddOnEntry, SubmitPartialExit
from strategies.swing._shared.etf_core import (
    ETFCoreState,
    ETFPosition,
    SetupSnapshot,
    on_bar_common,
    on_fill_common,
    on_order_update_common,
)
from strategies.swing.tpc import STRATEGY_ID
from strategies.swing.tpc import allocator, context, gates, signals, stops
from strategies.swing.tpc.config import TPCSymbolConfig
from strategies.swing.tpc.models import Direction, PullbackType, RegimeGrade

from .state import TPCBarInput, TPCCoreState, TPCFill, TPCOrderUpdate, TPCSecondEntrySeed


def on_bar(state: TPCCoreState, bar_input: TPCBarInput, cfg: TPCSymbolConfig):
    next_state, actions, events = on_bar_common(
        state,
        bar_input,
        cfg,
        strategy_id=STRATEGY_ID,
        evaluate_setup=_evaluate_setup,
        manage_position=_manage_tpc_position,
    )
    next_state = _coerce_state(next_state)
    next_state = _expire_second_entry_seed(next_state, bar_input, cfg)
    for event in events:
        if event.code == "ENTRY_REQUESTED" and event.details.get("setup_type") == PullbackType.TYPE_C.value:
            if event.symbol in next_state.second_entry_seeds:
                next_state.second_entry_seeds = dict(next_state.second_entry_seeds)
                next_state.second_entry_seeds.pop(event.symbol, None)
    return next_state, actions, events


def on_fill(state: TPCCoreState, fill: TPCFill):
    seed = _second_entry_seed_from_fill(state, fill)
    next_state, actions, events = on_fill_common(state, fill, strategy_id=STRATEGY_ID)
    next_state = _coerce_state(next_state)
    if seed is not None:
        next_state.second_entry_seeds = dict(next_state.second_entry_seeds)
        next_state.second_entry_seeds[seed.symbol] = seed
    return next_state, actions, events


def on_order_update(state: TPCCoreState, update: TPCOrderUpdate):
    next_state, actions, events = on_order_update_common(state, update, strategy_id=STRATEGY_ID)
    return _coerce_state(next_state), actions, events


def _reject(
    collector: list[dict] | None,
    *,
    symbol: str,
    lane: str,
    blocked_by: str,
    block_reason: str = "",
    direction: Direction | None = None,
    grade: RegimeGrade | str | None = None,
    **details: object,
) -> None:
    if collector is None:
        return
    record: dict[str, object] = {
        "symbol": symbol,
        "lane": lane,
        "blocked_by": blocked_by,
        "block_reason": block_reason,
    }
    if direction is not None:
        record["direction"] = direction.value if isinstance(direction, Direction) else int(direction)
    if grade is not None:
        record["grade"] = grade.value if isinstance(grade, RegimeGrade) else str(grade)
    if details:
        record["details"] = details
    collector.append(record)


def _evaluate_setup(
    state: ETFCoreState,
    bar_input: TPCBarInput,
    cfg: TPCSymbolConfig,
    *,
    rejection_collector: list[dict] | None = None,
) -> SetupSnapshot | None:
    bar = bar_input.bar_15m
    symbol = bar_input.symbol
    if bar is None:
        _reject(rejection_collector, symbol=symbol, lane="prelude", blocked_by="bar_missing")
        return None
    if not gates.session_filter(bar.timestamp, cfg):
        _reject(rejection_collector, symbol=symbol, lane="prelude", blocked_by="session_filter_blocked",
                block_reason="outside primary trading window", ts=bar.timestamp.isoformat())
        return None
    if not gates.news_filter(bar.timestamp, cfg):
        _reject(rejection_collector, symbol=symbol, lane="prelude", blocked_by="news_filter_blocked",
                block_reason="inside news avoidance window", ts=bar.timestamp.isoformat())
        return None
    direction, grade, _reason = gates.regime_direction(bar_input, cfg)
    if direction == Direction.FLAT:
        _reject(rejection_collector, symbol=symbol, lane="prelude", blocked_by="regime_flat",
                block_reason=_reason or "regime classified as FLAT", grade=grade)
        return None
    if direction == Direction.LONG and not cfg.longs_enabled:
        _reject(rejection_collector, symbol=symbol, lane="prelude", blocked_by="longs_disabled",
                direction=direction, grade=grade)
        return None
    if direction == Direction.SHORT:
        if not cfg.shorts_enabled:
            _reject(rejection_collector, symbol=symbol, lane="prelude", blocked_by="shorts_disabled",
                    direction=direction, grade=grade)
            return None
        if cfg.shorts_require_a_plus and grade != RegimeGrade.A_PLUS:
            _reject(rejection_collector, symbol=symbol, lane="prelude", blocked_by="shorts_require_a_plus",
                    direction=direction, grade=grade)
            return None
    for lane_name, lane_cfg, pullback_timeframe in _setup_lanes(cfg):
        setup = _evaluate_setup_lane(
            state,
            bar_input,
            lane_cfg,
            direction,
            grade,
            lane_name=lane_name,
            pullback_timeframe=pullback_timeframe,
            rejection_collector=rejection_collector,
        )
        if setup is not None:
            return setup
    return None


def _setup_lanes(cfg: TPCSymbolConfig) -> list[tuple[str, TPCSymbolConfig, str]]:
    lanes = [("primary", cfg, "1h")]
    if cfg.pb30_pullback_enabled:
        lanes.append(("pb30", _pb30_lane_config(cfg, cfg.pb30_entry_order_model or cfg.entry_order_model), "30m"))
    if cfg.pb30_ema20_value_touch_enabled:
        entry_model = cfg.pb30_ema20_value_touch_entry_order_model or "ema20_30m_value_touch_market"
        lanes.append(("pb30_ema20_value_touch", _pb30_lane_config(cfg, entry_model), "30m"))
    if cfg.ema20_value_touch_entry_enabled:
        lanes.append(("ema20_value_touch", _ema20_value_touch_lane_config(cfg), "1h"))
    return lanes


def _pb30_lane_config(cfg: TPCSymbolConfig, entry_model: str) -> TPCSymbolConfig:
    min_bars = max(1, int(cfg.pb30_pullback_min_bars_30m))
    max_bars = max(min_bars + 1, int(cfg.pb30_pullback_max_bars_30m))
    updates = {
        "pullback_min_bars_1h": min_bars,
        "pullback_max_bars_1h": max_bars,
        "pullback_orderly_required": bool(cfg.pullback_orderly_required or cfg.pb30_pullback_orderly_required),
        "entry_order_model": entry_model,
    }
    if cfg.pb30_fib_a_low > 0:
        updates["fib_a_low"] = cfg.pb30_fib_a_low
    if cfg.pb30_fib_a_high > 0:
        updates["fib_a_high"] = cfg.pb30_fib_a_high
    if cfg.pb30_type_a_value_hits_min > 0:
        updates["type_a_value_hits_min"] = cfg.pb30_type_a_value_hits_min
    if cfg.pb30_confirmation_required >= 0:
        updates["confirmation_required"] = int(cfg.pb30_confirmation_required)
    if cfg.pb30_confirmation_combo_mode:
        updates["confirmation_combo_mode"] = cfg.pb30_confirmation_combo_mode
    if cfg.pb30_confirmation_required >= 0 or cfg.pb30_confirmation_combo_mode:
        updates["require_vwap_confirmation"] = False
        updates["require_structure_confirmation"] = False
        updates["require_micro_break_confirmation"] = False
        updates["require_volume_confirmation"] = False
    return replace(cfg, **updates)


def _ema20_value_touch_lane_config(cfg: TPCSymbolConfig) -> TPCSymbolConfig:
    updates = {
        "entry_order_model": cfg.ema20_value_touch_entry_order_model or "ema20_value_touch_market",
    }
    if cfg.ema20_value_touch_confirmation_required >= 0:
        updates["confirmation_required"] = int(cfg.ema20_value_touch_confirmation_required)
    if cfg.ema20_value_touch_confirmation_combo_mode:
        updates["confirmation_combo_mode"] = cfg.ema20_value_touch_confirmation_combo_mode
    if cfg.ema20_value_touch_confirmation_required >= 0 or cfg.ema20_value_touch_confirmation_combo_mode:
        updates["require_vwap_confirmation"] = False
        updates["require_structure_confirmation"] = False
        updates["require_micro_break_confirmation"] = False
        updates["require_volume_confirmation"] = False
    return replace(cfg, **updates)


def _evaluate_setup_lane(
    state: ETFCoreState,
    bar_input: TPCBarInput,
    cfg: TPCSymbolConfig,
    direction: Direction,
    grade: RegimeGrade,
    *,
    lane_name: str,
    pullback_timeframe: str,
    rejection_collector: list[dict] | None = None,
) -> SetupSnapshot | None:
    bar = bar_input.bar_15m
    symbol = bar_input.symbol
    if bar is None:
        return None
    if pullback_timeframe == "30m":
        pullback = signals.detect_pullback_30m(bar_input, direction, grade, cfg)
    else:
        pullback = signals.detect_pullback(bar_input, direction, grade, cfg)
    if pullback is None:
        _reject(rejection_collector, symbol=symbol, lane=lane_name, blocked_by="pullback_not_detected",
                block_reason=f"no qualifying pullback on {pullback_timeframe} timeframe",
                direction=direction, grade=grade, pullback_timeframe=pullback_timeframe)
        return None
    ok, confirmations = signals.check_confirmation(bar_input, direction, cfg)
    if not ok or not _confirmation_combo_allowed(confirmations, cfg):
        _reject(rejection_collector, symbol=symbol, lane=lane_name, blocked_by="confirmation_combo_failed",
                block_reason=f"required confirmations not satisfied (combo_mode={cfg.confirmation_combo_mode})",
                direction=direction, grade=grade, confirmations=list(confirmations),
                confirmation_required=cfg.confirmation_required)
        return None
    if cfg.confirmation_max_count > 0 and len(set(confirmations)) > cfg.confirmation_max_count:
        _reject(rejection_collector, symbol=symbol, lane=lane_name, blocked_by="confirmation_max_count_exceeded",
                block_reason=f"{len(set(confirmations))} > confirmation_max_count={cfg.confirmation_max_count}",
                direction=direction, grade=grade, confirmations=list(confirmations))
        return None
    second_entry_seed = _valid_second_entry_seed(state, bar_input, direction, grade, cfg) if lane_name == "primary" else None
    if second_entry_seed is not None and _second_entry_confirmation_allowed(confirmations, cfg):
        pullback = replace(pullback, pullback_type=PullbackType.TYPE_C)
    atr4 = bar_input.indicators.get("atr_4h", np.nan)
    if np.isnan(atr4) or atr4 <= 0:
        _reject(rejection_collector, symbol=symbol, lane=lane_name, blocked_by="atr_4h_unavailable",
                block_reason="atr_4h indicator missing or non-positive",
                direction=direction, grade=grade, atr_4h=float(atr4) if not np.isnan(atr4) else None)
        return None
    entry_plan = _entry_plan(bar_input, direction, cfg)
    if entry_plan is None:
        _reject(rejection_collector, symbol=symbol, lane=lane_name, blocked_by="entry_plan_invalid",
                block_reason=f"entry model {cfg.entry_order_model} could not produce a valid plan",
                direction=direction, grade=grade, entry_order_model=cfg.entry_order_model)
        return None
    entry, entry_order_type, entry_limit_price, entry_stop_price, entry_model = entry_plan
    stop = _initial_stop(bar_input, pullback, direction, entry, atr4, cfg)
    if not stops.validate_stop(stop, entry, atr4, cfg):
        _reject(rejection_collector, symbol=symbol, lane=lane_name, blocked_by="stop_validation_failed",
                block_reason=f"stop {stop} fails validation against entry {entry}",
                direction=direction, grade=grade, entry=float(entry), stop=float(stop), atr_4h=float(atr4))
        return None
    risk = abs(entry - stop)
    rr = 3.0 if risk > 0 else 0.0
    daily_levels = _daily_levels(bar_input)
    daily_has_room = gates.daily_room_filter(entry, stop, direction, daily_levels, cfg.daily_room_min_r)
    if not daily_has_room:
        _reject(rejection_collector, symbol=symbol, lane=lane_name, blocked_by="daily_room_insufficient",
                block_reason=f"daily room < {cfg.daily_room_min_r}R to next level",
                direction=direction, grade=grade, daily_room_min_r=cfg.daily_room_min_r)
        return None
    asset_context_score, asset_context_details = context.score_asset_context(bar_input, direction, cfg)
    if asset_context_score < cfg.asset_context_min_score:
        _reject(rejection_collector, symbol=symbol, lane=lane_name, blocked_by="asset_context_score_low",
                block_reason=f"asset_context_score={asset_context_score:.3f} < min={cfg.asset_context_min_score}",
                direction=direction, grade=grade, asset_context_score=float(asset_context_score),
                asset_context_min_score=float(cfg.asset_context_min_score))
        return None
    score = allocator.score_setup(
        grade,
        pullback.pullback_type,
        confirmations,
        rr,
        has_news_risk=False,
        asset_context_score=asset_context_score,
        daily_has_room=daily_has_room,
        orderly_pullback=pullback.orderly,
        score_model=cfg.score_model,
    )
    if pullback.pullback_type == PullbackType.TYPE_C:
        if cfg.type_c_requires_a_plus and grade != RegimeGrade.A_PLUS:
            _reject(rejection_collector, symbol=symbol, lane=lane_name, blocked_by="type_c_requires_a_plus",
                    block_reason="TYPE_C re-entry requires A+ regime",
                    direction=direction, grade=grade, score=float(score))
            return None
        if score < cfg.second_entry_score_min:
            _reject(rejection_collector, symbol=symbol, lane=lane_name, blocked_by="second_entry_score_min",
                    block_reason=f"TYPE_C score={score:.2f} < second_entry_score_min={cfg.second_entry_score_min}",
                    direction=direction, grade=grade, score=float(score),
                    second_entry_score_min=cfg.second_entry_score_min)
            return None
    if direction == Direction.SHORT and cfg.min_short_score > 0 and score < cfg.min_short_score:
        _reject(rejection_collector, symbol=symbol, lane=lane_name, blocked_by="min_short_score",
                block_reason=f"SHORT score={score:.2f} < min_short_score={cfg.min_short_score}",
                direction=direction, grade=grade, score=float(score), min_short_score=cfg.min_short_score)
        return None
    risk_pct = allocator.compute_risk_pct(score, pullback.pullback_type, cfg)
    if risk_pct is None:
        _reject(rejection_collector, symbol=symbol, lane=lane_name, blocked_by="risk_pct_zero",
                block_reason=f"score={score:.2f} below grade thresholds (a_min={cfg.score_a_min}, b_min={cfg.score_b_min})",
                direction=direction, grade=grade, score=float(score),
                score_a_min=cfg.score_a_min, score_b_min=cfg.score_b_min)
        return None
    qty = allocator.compute_position_size(bar_input.equity, risk_pct, entry, stop, cfg)
    if qty <= 0:
        _reject(rejection_collector, symbol=symbol, lane=lane_name, blocked_by="qty_zero",
                block_reason=f"computed qty={qty} on equity={bar_input.equity:.2f} risk_pct={risk_pct:.4f}",
                direction=direction, grade=grade, score=float(score),
                risk_pct=float(risk_pct), equity=float(bar_input.equity))
        return None
    created = bar.timestamp or datetime.now(timezone.utc)
    lane_suffix = "" if lane_name == "primary" else f"-{lane_name}"
    setup_id = f"{STRATEGY_ID}-{bar_input.symbol}{lane_suffix}-{int(created.timestamp())}"
    return SetupSnapshot(
        setup_id=setup_id,
        strategy_id=STRATEGY_ID,
        symbol=bar_input.symbol,
        direction=direction,
        grade=grade.value,
        setup_type=pullback.pullback_type.value,
        entry_model=entry_model,
        state="entry_ready",
        created_ts=created,
        entry_price=entry,
        stop_price=stop,
        qty=qty,
        score=float(score),
        risk_pct=risk_pct,
        t1_r=cfg.t1_r,
        t1_partial_pct=cfg.t1_partial_pct,
        t2_r=cfg.t2_r,
        t2_partial_pct=cfg.t2_partial_pct,
        entry_order_type=entry_order_type,
        entry_limit_price=entry_limit_price,
        entry_stop_price=entry_stop_price,
        entry_ttl_hours=cfg.entry_order_ttl_hours if entry_order_type != "MARKET" else 0.0,
        meta={
            "confirmations": confirmations,
            "depth": pullback.depth,
            "pullback_low": pullback.low,
            "pullback_high": pullback.high,
            "value_hits": pullback.value_hits,
            "rr": rr,
            "atr_4h": atr4,
            "daily_levels": daily_levels,
            "asset_context_score": asset_context_score,
            "asset_context_details": asset_context_details,
            "setup_lane": lane_name,
            "pullback_timeframe": pullback_timeframe,
            "score": float(score),
            "daily_has_room": bool(daily_has_room),
            "orderly_pullback": bool(pullback.orderly),
            "time_stop_min_mfe_r": cfg.time_stop_min_mfe_r,
            "runner_max_hold_bars_15m": cfg.runner_max_hold_bars_15m,
            "stall_exit_bars_15m": cfg.stall_exit_bars_15m,
            "stall_exit_min_mfe_r": cfg.stall_exit_min_mfe_r,
            "stall_exit_max_current_r": cfg.stall_exit_max_current_r,
            "mfe_giveback_trigger_r": cfg.mfe_giveback_trigger_r,
            "mfe_giveback_retain_frac": cfg.mfe_giveback_retain_frac,
            "mfe_giveback_lock_r": cfg.mfe_giveback_lock_r,
            "mfe_giveback_after_t1_only": cfg.mfe_giveback_after_t1_only,
            "addon_enabled": cfg.addon_enabled,
            "addon_trigger_r": cfg.addon_trigger_r,
            "addon_size_mult": cfg.addon_size_mult,
            "addon_min_score": cfg.addon_min_score,
            "addon_requires_t1": cfg.addon_requires_t1,
            "addon_require_vwap_hold": cfg.addon_require_vwap_hold,
            "addon_require_structure_hold": cfg.addon_require_structure_hold,
            "addon_max_total_risk_pct": cfg.addon_max_total_risk_pct,
            "addon_max_notional_pct": cfg.addon_max_notional_pct,
            "second_entry_source_setup_id": second_entry_seed.source_setup_id if second_entry_seed else "",
            "second_entry_source_score": second_entry_seed.source_score if second_entry_seed else 0.0,
            "second_entry_source_grade": second_entry_seed.source_grade if second_entry_seed else "",
        },
        max_hold_bars_15m=cfg.max_hold_bars_15m,
    )


def _confirmation_combo_allowed(confirmations: list[str], cfg: TPCSymbolConfig) -> bool:
    names = {str(item) for item in confirmations}
    has_vwap = any("vwap" in item for item in names)
    has_structure = any(("higher_low" in item or "lower_high" in item or "micro_break" in item) for item in names)
    has_micro = any("micro_break" in item for item in names)
    has_volume = any("volume" in item for item in names)
    if cfg.require_vwap_confirmation and not has_vwap:
        return False
    if cfg.require_structure_confirmation and not has_structure:
        return False
    if cfg.require_micro_break_confirmation and not has_micro:
        return False
    if cfg.require_volume_confirmation and not has_volume:
        return False
    if cfg.confirmation_combo_mode == "structure_vwap":
        return has_vwap and has_structure
    if cfg.confirmation_combo_mode == "micro_vwap":
        return has_vwap and has_micro
    if cfg.confirmation_combo_mode == "preferred":
        return has_vwap and has_structure and (has_micro or has_volume)
    if cfg.confirmation_combo_mode == "structure_or_vwap":
        return has_structure or has_vwap
    return True


def _entry_plan(
    bar_input: TPCBarInput,
    direction: Direction,
    cfg: TPCSymbolConfig,
) -> tuple[float, str, float, float, str] | None:
    bar = bar_input.bar_15m
    if bar is None:
        return None
    model = cfg.entry_order_model
    tick = cfg.tick_size
    if model in {"structure_stop", "adaptive_structure_stop", "structure_stop_market"}:
        if direction == Direction.LONG:
            stop_price = _round_tick(bar.high + tick, tick)
        else:
            stop_price = _round_tick(bar.low - tick, tick)
        if model == "structure_stop_market":
            return stop_price, "STOP", 0.0, float(stop_price), "structure_stop_market"
        atr15 = max(float(bar_input.indicators.get("atr_15m", 0.0) or 0.0), tick)
        limit_mult = cfg.entry_stop_limit_atr_mult
        if model == "adaptive_structure_stop":
            signal_range_mult = max((bar.high - bar.low) / atr15, 1.0)
            limit_mult = min(
                max(limit_mult * signal_range_mult, cfg.entry_adaptive_stop_limit_min_atr_mult),
                cfg.entry_adaptive_stop_limit_max_atr_mult,
            )
        limit_offset = limit_mult * atr15
        limit_price = stop_price + limit_offset if direction == Direction.LONG else stop_price - limit_offset
        return stop_price, "STOP_LIMIT", float(limit_price), float(stop_price), model
    if model in {"vwap_retest_limit", "midpoint_retest_limit"}:
        if model == "vwap_retest_limit":
            limit = float(bar_input.indicators.get("vwap_15m", np.nan))
        else:
            limit = (bar.high + bar.low) / 2.0
        if not np.isfinite(limit):
            return None
        if direction == Direction.LONG and not (bar.low <= limit <= bar.close):
            return None
        if direction == Direction.SHORT and not (bar.close <= limit <= bar.high):
            return None
        return float(limit), "LIMIT", float(limit), 0.0, model
    if model in {
        "ema20_value_touch_market",
        "ema20_value_touch_limit",
        "ema20_30m_value_touch_market",
        "ema20_30m_value_touch_limit",
    }:
        level_key = "ema20_30m" if model.startswith("ema20_30m") else "ema20_1h"
        level = float(bar_input.indicators.get(level_key, np.nan))
        if not np.isfinite(level) or not (bar.low <= level <= bar.high):
            return None
        if model.endswith("_limit"):
            limit = _round_tick(level, tick)
            return float(limit), "LIMIT", float(limit), 0.0, model
        return bar.close, "MARKET", 0.0, 0.0, model
    return bar.close, "MARKET", 0.0, 0.0, "confirmation_close"


def _initial_stop(
    bar_input: TPCBarInput,
    pullback: signals.PullbackCandidate,
    direction: Direction,
    entry: float,
    atr4: float,
    cfg: TPCSymbolConfig,
) -> float:
    pullback_stop = stops.compute_initial_stop(pullback, direction, atr4, cfg)
    bar = bar_input.bar_15m
    if bar is None or cfg.initial_stop_source == "pullback":
        return _enforce_min_stop_distance(entry, pullback_stop, direction, atr4, cfg)
    buffer = cfg.signal_stop_buffer_atr_mult * atr4
    signal_stop = bar.low - buffer if direction == Direction.LONG else bar.high + buffer
    if cfg.initial_stop_source == "signal":
        return _enforce_min_stop_distance(entry, signal_stop, direction, atr4, cfg)
    if cfg.initial_stop_source == "hybrid_tighter":
        stop = max(pullback_stop, signal_stop) if direction == Direction.LONG else min(pullback_stop, signal_stop)
    elif cfg.initial_stop_source == "hybrid_wider":
        stop = min(pullback_stop, signal_stop) if direction == Direction.LONG else max(pullback_stop, signal_stop)
    else:
        stop = pullback_stop
    return _enforce_min_stop_distance(entry, stop, direction, atr4, cfg)


def _enforce_min_stop_distance(
    entry: float,
    stop: float,
    direction: Direction,
    atr4: float,
    cfg: TPCSymbolConfig,
) -> float:
    min_distance = max(0.0, float(cfg.min_stop_atr_mult) * float(atr4))
    if min_distance <= 0 or not np.isfinite(min_distance):
        return stop
    current_distance = abs(float(entry) - float(stop))
    if current_distance >= min_distance:
        return stop
    return float(entry) - min_distance if direction == Direction.LONG else float(entry) + min_distance


def _manage_tpc_position(
    state: ETFCoreState,
    bar_input: TPCBarInput,
    cfg: TPCSymbolConfig,
    position: ETFPosition,
) -> list:
    del state
    bar = bar_input.bar_15m
    if bar is None:
        return []
    actions: list = []
    direction = position.direction
    risk = max(position.risk_per_share, 1e-9)
    close = bar.close
    t1_r = float(position.meta.get("t1_r", cfg.t1_r))
    t2_r = float(position.meta.get("t2_r", cfg.t2_r))
    current_r = _current_r(position, close)
    t1_hit = close >= position.entry_price + t1_r * risk if direction == Direction.LONG else close <= position.entry_price - t1_r * risk
    t2_hit = close >= position.entry_price + t2_r * risk if direction == Direction.LONG else close <= position.entry_price - t2_r * risk

    if not position.t1_done and t1_hit and "t1" not in position.pending_exit_roles:
        qty = _partial_qty(position.qty_open, float(position.meta.get("t1_partial_pct", cfg.t1_partial_pct)))
        if qty > 0:
            position.pending_exit_roles.add("t1")
            actions.append(
                SubmitPartialExit(
                    client_order_id=f"{position.setup_id}-t1",
                    symbol=position.symbol,
                    side="SELL" if direction == Direction.LONG else "BUY",
                    qty=qty,
                    order_type="MARKET",
                    metadata={"reason": "T1", "setup_id": position.setup_id},
                )
            )
            _raise_stop_to_r(actions, position, cfg.t1_stop_r, reason="t1_profit_lock")

    if position.t1_done and not position.t2_done and t2_hit and "t2" not in position.pending_exit_roles:
        qty = _partial_qty(position.qty_open, float(position.meta.get("t2_partial_pct", cfg.t2_partial_pct)))
        if qty > 0:
            position.pending_exit_roles.add("t2")
            actions.append(
                SubmitPartialExit(
                    client_order_id=f"{position.setup_id}-t2",
                    symbol=position.symbol,
                    side="SELL" if direction == Direction.LONG else "BUY",
                    qty=qty,
                    order_type="MARKET",
                    metadata={"reason": "T2", "setup_id": position.setup_id},
                )
            )

    _apply_profit_floor(actions, position, cfg)
    _apply_structure_trail(actions, position, bar_input, cfg)
    _apply_mfe_giveback(actions, position, cfg, current_r)
    _maybe_submit_addon(actions, position, bar_input, cfg, current_r)

    max_hold = int(position.meta.get("max_hold_bars_15m", 0) or 0)
    min_mfe = float(position.meta.get("time_stop_min_mfe_r", 0.0) or 0.0)
    if max_hold > 0 and position.bars_held_15m >= max_hold and not position.t1_done and position.mfe_r < min_mfe:
        actions.append(
            FlattenPosition(
                symbol=position.symbol,
                side="SELL" if direction == Direction.LONG else "BUY",
                qty=position.qty_open,
                reason="TIME_STOP",
                metadata={"setup_id": position.setup_id},
            )
        )
    stall_bars = int(position.meta.get("stall_exit_bars_15m", cfg.stall_exit_bars_15m) or 0)
    stall_min_mfe = float(position.meta.get("stall_exit_min_mfe_r", cfg.stall_exit_min_mfe_r) or 0.0)
    stall_max_current = float(position.meta.get("stall_exit_max_current_r", cfg.stall_exit_max_current_r) or 0.0)
    if (
        not _has_flatten(actions)
        and stall_bars > 0
        and position.bars_held_15m >= stall_bars
        and position.mfe_r >= stall_min_mfe
        and current_r <= stall_max_current
    ):
        actions.append(
            FlattenPosition(
                symbol=position.symbol,
                side="SELL" if direction == Direction.LONG else "BUY",
                qty=position.qty_open,
                reason="STALL_EXIT",
                metadata={"setup_id": position.setup_id},
            )
        )
    runner_max_hold = int(position.meta.get("runner_max_hold_bars_15m", cfg.runner_max_hold_bars_15m) or 0)
    if runner_max_hold > 0 and position.t1_done and position.bars_held_15m >= runner_max_hold and not _has_flatten(actions):
        actions.append(
            FlattenPosition(
                symbol=position.symbol,
                side="SELL" if direction == Direction.LONG else "BUY",
                qty=position.qty_open,
                reason="RUNNER_TIME_STOP",
                metadata={"setup_id": position.setup_id},
            )
        )
    return actions


def _current_r(position: ETFPosition, close: float) -> float:
    risk = max(position.risk_per_share, 1e-9)
    if position.direction == Direction.LONG:
        return (close - position.entry_price) / risk
    return (position.entry_price - close) / risk


def _has_flatten(actions: list) -> bool:
    return any(isinstance(action, FlattenPosition) for action in actions)


def _maybe_submit_addon(
    actions: list,
    position: ETFPosition,
    bar_input: TPCBarInput,
    cfg: TPCSymbolConfig,
    current_r: float,
) -> None:
    if _has_flatten(actions) or not bool(position.meta.get("addon_enabled", cfg.addon_enabled)):
        return
    if position.meta.get("addon_done") or position.meta.get("addon_pending") or position.qty_open <= 0:
        return
    if bool(position.meta.get("addon_requires_t1", cfg.addon_requires_t1)) and not position.t1_done:
        return
    trigger_r = float(position.meta.get("addon_trigger_r", cfg.addon_trigger_r) or 0.0)
    if trigger_r <= 0 or current_r < trigger_r or position.mfe_r < trigger_r:
        return
    min_score = float(position.meta.get("addon_min_score", cfg.addon_min_score) or 0.0)
    if position.score < min_score:
        return
    if not _addon_confirmation_holds(position, bar_input, cfg):
        return
    qty = _addon_qty(position, bar_input, cfg)
    if qty <= 0:
        return
    bar = bar_input.bar_15m
    if bar is None:
        return
    position.meta["addon_pending"] = True
    actions.append(
        SubmitAddOnEntry(
            client_order_id=f"{position.setup_id}-addon-{position.bars_held_15m}",
            symbol=position.symbol,
            side="BUY" if position.direction == Direction.LONG else "SELL",
            qty=qty,
            order_type="MARKET",
            price=bar.close,
            risk_context={
                "stop_for_risk": position.current_stop,
                "current_r": current_r,
                "mfe_r": position.mfe_r,
            },
            metadata={
                "setup_id": position.setup_id,
                "reason": "ADDON_CONTINUATION",
                "score": position.score,
                "current_r": current_r,
                "mfe_r": position.mfe_r,
            },
        )
    )


def _addon_confirmation_holds(position: ETFPosition, bar_input: TPCBarInput, cfg: TPCSymbolConfig) -> bool:
    bar = bar_input.bar_15m
    if bar is None:
        return False
    direction = position.direction
    if bool(position.meta.get("addon_require_vwap_hold", cfg.addon_require_vwap_hold)):
        vwap = float(bar_input.indicators.get("vwap_15m", np.nan))
        if not np.isfinite(vwap):
            vwap = float(bar_input.indicators.get("vwap_30m", np.nan))
        if not np.isfinite(vwap):
            return False
        if direction == Direction.LONG and bar.close <= vwap:
            return False
        if direction == Direction.SHORT and bar.close >= vwap:
            return False
    if bool(position.meta.get("addon_require_structure_hold", cfg.addon_require_structure_hold)):
        bars = bar_input.bars_15m
        if bars is None or len(bars) < 5:
            return False
        ema20 = float(bar_input.indicators.get("ema20_15m", np.nan))
        prev_high = float(np.nanmax(bars.highs[-5:-1]))
        prev_low = float(np.nanmin(bars.lows[-5:-1]))
        if direction == Direction.LONG:
            if np.isfinite(ema20) and bar.close <= ema20:
                return False
            if not np.isfinite(prev_high) or bar.close <= prev_high:
                return False
        else:
            if np.isfinite(ema20) and bar.close >= ema20:
                return False
            if not np.isfinite(prev_low) or bar.close >= prev_low:
                return False
    return True


def _addon_qty(position: ETFPosition, bar_input: TPCBarInput, cfg: TPCSymbolConfig) -> int:
    bar = bar_input.bar_15m
    if bar is None or bar.close <= 0:
        return 0
    size_mult = max(float(position.meta.get("addon_size_mult", cfg.addon_size_mult) or 0.0), 0.0)
    if size_mult <= 0:
        return 0
    qty = max(1, int(round(position.qty_initial * size_mult)))
    cap_pct = float(position.meta.get("addon_max_notional_pct", cfg.addon_max_notional_pct) or 0.0)
    if cap_pct <= 0:
        cap_pct = float(cfg.max_position_notional_pct or 0.0)
    if cap_pct > 0 and bar_input.equity > 0:
        notional_room = max(0.0, bar_input.equity * cap_pct - position.qty_open * bar.close)
        qty = min(qty, int(notional_room // bar.close))
    risk_cap_pct = float(position.meta.get("addon_max_total_risk_pct", cfg.addon_max_total_risk_pct) or 0.0)
    if risk_cap_pct > 0 and bar_input.equity > 0:
        risk_per_share = max(abs(bar.close - position.current_stop), 1e-9)
        current_risk = max(0.0, position.qty_open * risk_per_share)
        risk_room = max(0.0, bar_input.equity * risk_cap_pct - current_risk)
        qty = min(qty, int(risk_room // risk_per_share))
    return max(qty, 0)


def _apply_mfe_giveback(
    actions: list,
    position: ETFPosition,
    cfg: TPCSymbolConfig,
    current_r: float,
) -> None:
    if _has_flatten(actions):
        return
    trigger_r = float(position.meta.get("mfe_giveback_trigger_r", cfg.mfe_giveback_trigger_r) or 0.0)
    if trigger_r <= 0 or position.mfe_r < trigger_r:
        return
    after_t1_only = bool(position.meta.get("mfe_giveback_after_t1_only", cfg.mfe_giveback_after_t1_only))
    if after_t1_only and not position.t1_done:
        return
    retain_frac = min(max(float(position.meta.get("mfe_giveback_retain_frac", cfg.mfe_giveback_retain_frac) or 0.0), 0.0), 1.0)
    lock_r = float(position.meta.get("mfe_giveback_lock_r", cfg.mfe_giveback_lock_r) or 0.0)
    floor_r = max(lock_r, position.mfe_r * retain_frac)
    if current_r > floor_r:
        return
    actions.append(
        FlattenPosition(
            symbol=position.symbol,
            side="SELL" if position.direction == Direction.LONG else "BUY",
            qty=position.qty_open,
            reason="MFE_GIVEBACK",
            metadata={
                "setup_id": position.setup_id,
                "mfe_r": position.mfe_r,
                "current_r": current_r,
                "floor_r": floor_r,
            },
        )
    )


def _second_entry_seed_from_fill(state: TPCCoreState, fill: TPCFill) -> TPCSecondEntrySeed | None:
    role = str(fill.order_role or "").lower()
    exit_type = str(fill.exit_type or "").upper()
    if role != "stop" and exit_type != "STOP":
        return None
    position = state.positions.get(fill.symbol)
    if position is None or position.t1_done or position.setup_type == PullbackType.TYPE_C.value:
        return None
    pullback_low = float(position.meta.get("pullback_low", position.initial_stop) or position.initial_stop)
    pullback_high = float(position.meta.get("pullback_high", position.initial_stop) or position.initial_stop)
    return TPCSecondEntrySeed(
        symbol=position.symbol,
        source_setup_id=position.setup_id,
        direction=int(position.direction),
        pullback_low=pullback_low,
        pullback_high=pullback_high,
        stop_time=fill.fill_time or datetime.now(timezone.utc),
        source_grade=position.grade,
        source_score=float(position.score),
    )


def _valid_second_entry_seed(
    state: ETFCoreState,
    bar_input: TPCBarInput,
    direction: Direction,
    grade: RegimeGrade,
    cfg: TPCSymbolConfig,
) -> TPCSecondEntrySeed | None:
    if not cfg.type_c_enabled or cfg.type_c_mode not in {"real_reentry", "shallow_or_reentry"}:
        return None
    seed = getattr(state, "second_entry_seeds", {}).get(bar_input.symbol)
    if seed is None:
        return None
    if int(seed.direction) != int(direction):
        return None
    if cfg.type_c_requires_a_plus and grade != RegimeGrade.A_PLUS:
        return None
    if cfg.second_entry_min_source_score > 0 and seed.source_score < cfg.second_entry_min_source_score:
        return None
    if cfg.second_entry_requires_source_a_plus and str(seed.source_grade).lower() != RegimeGrade.A_PLUS.value:
        return None
    if not _second_entry_wait_elapsed(seed, bar_input, cfg):
        return None
    if not _second_entry_structure_intact(seed, bar_input, direction, cfg):
        return None
    return seed


def _expire_second_entry_seed(state: TPCCoreState, bar_input: TPCBarInput, cfg: TPCSymbolConfig) -> TPCCoreState:
    if not state.second_entry_seeds or bar_input is None or bar_input.bar_15m is None:
        return state
    seed = state.second_entry_seeds.get(bar_input.symbol)
    if seed is None or cfg.second_entry_max_wait_bars_15m <= 0:
        return state
    elapsed = _bars_since_seed(seed, bar_input)
    if elapsed <= cfg.second_entry_max_wait_bars_15m:
        return state
    state.second_entry_seeds = dict(state.second_entry_seeds)
    state.second_entry_seeds.pop(bar_input.symbol, None)
    return state


def _second_entry_wait_elapsed(seed: TPCSecondEntrySeed, bar_input: TPCBarInput, cfg: TPCSymbolConfig) -> bool:
    elapsed = _bars_since_seed(seed, bar_input)
    if elapsed < max(int(cfg.second_entry_min_wait_bars_15m), 0):
        return False
    max_wait = int(cfg.second_entry_max_wait_bars_15m)
    return max_wait <= 0 or elapsed <= max_wait


def _bars_since_seed(seed: TPCSecondEntrySeed, bar_input: TPCBarInput) -> int:
    bar = bar_input.bar_15m
    if bar is None:
        return 0
    try:
        return max(0, int((bar.timestamp - seed.stop_time).total_seconds() // 900))
    except TypeError:
        stop_time = seed.stop_time
        if stop_time.tzinfo is None and bar.timestamp.tzinfo is not None:
            stop_time = stop_time.replace(tzinfo=bar.timestamp.tzinfo)
        return max(0, int((bar.timestamp - stop_time).total_seconds() // 900))


def _second_entry_structure_intact(
    seed: TPCSecondEntrySeed,
    bar_input: TPCBarInput,
    direction: Direction,
    cfg: TPCSymbolConfig,
) -> bool:
    bars = bar_input.bars_1h
    if bars is None or len(bars) == 0:
        return False
    atr4 = float(bar_input.indicators.get("atr_4h", 0.0) or 0.0)
    buffer = max(0.0, float(cfg.second_entry_structure_buffer_atr_mult) * atr4)
    since_stop = [idx for idx, ts in enumerate(bars.times) if _timestamp_gte(ts, seed.stop_time)]
    if since_stop:
        start = since_stop[0]
    else:
        start = max(0, len(bars) - max(int(cfg.pullback_max_bars_1h), 1))
    if direction == Direction.LONG:
        return float(np.nanmin(bars.lows[start:])) >= seed.pullback_low - buffer
    return float(np.nanmax(bars.highs[start:])) <= seed.pullback_high + buffer


def _second_entry_confirmation_allowed(confirmations: list[str], cfg: TPCSymbolConfig) -> bool:
    if cfg.second_entry_require_vwap and not _has_vwap_confirmation(confirmations):
        return False
    if cfg.second_entry_require_structure and not _has_structure_confirmation(confirmations):
        return False
    return True


def _has_vwap_confirmation(confirmations: list[str]) -> bool:
    return any("vwap" in str(item) for item in confirmations)


def _has_structure_confirmation(confirmations: list[str]) -> bool:
    return any(
        "higher_low" in str(item) or "lower_high" in str(item) or "micro_break" in str(item)
        for item in confirmations
    )


def _timestamp_gte(left: datetime, right: datetime) -> bool:
    try:
        return left >= right
    except TypeError:
        if left.tzinfo is not None and right.tzinfo is None:
            right = right.replace(tzinfo=left.tzinfo)
        elif left.tzinfo is None and right.tzinfo is not None:
            left = left.replace(tzinfo=right.tzinfo)
        return left >= right


def _apply_profit_floor(actions: list, position: ETFPosition, cfg: TPCSymbolConfig) -> None:
    floor_r: float | None = None
    for threshold, floor in cfg.profit_floor_ladder or ():
        if position.mfe_r >= float(threshold):
            floor_r = float(floor)
    if floor_r is not None:
        _raise_stop_to_r(actions, position, floor_r, reason="profit_floor")


def _apply_structure_trail(actions: list, position: ETFPosition, bar_input: TPCBarInput, cfg: TPCSymbolConfig) -> None:
    if position.t2_done and cfg.trail_after_t2_1h_bars > 0 and bar_input.bars_1h is not None:
        window = bar_input.bars_1h
        lookback = min(cfg.trail_after_t2_1h_bars, len(window))
    elif position.t1_done and cfg.trail_after_t1_30m_bars > 0 and bar_input.bars_30m is not None:
        window = bar_input.bars_30m
        lookback = min(cfg.trail_after_t1_30m_bars, len(window))
    else:
        window = None
        lookback = 0
    if window is None or lookback < 2:
        return
    if position.direction == Direction.LONG:
        stop = float(np.nanmin(window.lows[-lookback:]))
        if cfg.trail_use_vwap_after_t1:
            vwap = float(bar_input.indicators.get("vwap_30m", np.nan))
            if np.isfinite(vwap):
                stop = max(stop, vwap)
    else:
        stop = float(np.nanmax(window.highs[-lookback:]))
        if cfg.trail_use_vwap_after_t1:
            vwap = float(bar_input.indicators.get("vwap_30m", np.nan))
            if np.isfinite(vwap):
                stop = min(stop, vwap)
    _raise_stop_to_price(actions, position, stop, reason="structure_trail")


def _raise_stop_to_r(actions: list, position: ETFPosition, floor_r: float, *, reason: str) -> None:
    risk = max(position.risk_per_share, 1e-9)
    if position.direction == Direction.LONG:
        stop = position.entry_price + floor_r * risk
    else:
        stop = position.entry_price - floor_r * risk
    _raise_stop_to_price(actions, position, stop, reason=reason)


def _raise_stop_to_price(actions: list, position: ETFPosition, stop: float, *, reason: str) -> None:
    if not np.isfinite(stop) or not position.stop_order_id:
        return
    improves = (
        position.direction == Direction.LONG and stop > position.current_stop
    ) or (
        position.direction == Direction.SHORT and stop < position.current_stop
    )
    if not improves:
        return
    position.current_stop = float(stop)
    actions.append(
        ReplaceProtectiveStop(
            symbol=position.symbol,
            target_order_id=position.stop_order_id,
            side="SELL" if position.direction == Direction.LONG else "BUY",
            stop_price=float(stop),
            qty=position.qty_open,
            reason=reason,
        )
    )


def _partial_qty(qty_open: int, frac: float) -> int:
    if qty_open <= 1:
        return qty_open
    qty = max(1, int(round(qty_open * frac)))
    return min(qty, qty_open - 1)


def _round_tick(value: float, tick: float) -> float:
    if tick <= 0:
        return value
    return round(round(value / tick) * tick, 10)


def _daily_levels(bar_input: TPCBarInput) -> list[float]:
    bars = bar_input.bars_daily
    if bars is None or len(bars) < 2:
        return []
    highs = bars.highs[-20:]
    lows = bars.lows[-20:]
    levels = [
        float(bars.highs[-1]),
        float(bars.lows[-1]),
        float(np.nanmax(highs)),
        float(np.nanmin(lows)),
    ]
    return [level for level in levels if np.isfinite(level) and level > 0]


def _coerce_state(state: ETFCoreState) -> TPCCoreState:
    if isinstance(state, TPCCoreState):
        return state
    return TPCCoreState(
        setups=state.setups,
        positions=state.positions,
        pending_orders=state.pending_orders,
        daily_loss_r=state.daily_loss_r,
        weekly_loss_r=state.weekly_loss_r,
        failed_entries=state.failed_entries,
        last_bar_ts=state.last_bar_ts,
        last_decision_code=state.last_decision_code,
        last_decision_details=state.last_decision_details,
        second_entry_seeds=getattr(state, "second_entry_seeds", {}),
    )
