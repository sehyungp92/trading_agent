"""Translate TPC core events into rich InstrumentationKit calls.

Pure-functional helper. The TPC engine subclass owns caches and dispatches
each ``DecisionEvent`` produced by ``core/logic.py`` through these routes.
Backtest replay does not import this module.

All ``kit.*`` calls are wrapped with logged exception handling: the kit must
never block trading, but failures are surfaced to the logger instead of
silently swallowed.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from libs.oms.instrumentation.runtime_refs import fill_runtime_refs
from strategies.swing._shared.etf_core import (
    ETFCoreState,
    ETFFill,
    ETFOrderUpdate,
    ETFPosition,
    SetupSnapshot,
)
from strategies.swing.tpc.config import TPCSymbolConfig

_log = logging.getLogger(__name__)

_EXIT_REASON_MAP: dict[str, str] = {
    "T1": "TAKE_PROFIT_T1",
    "T2": "TAKE_PROFIT_T2",
    "TIME_STOP": "TIME_STOP_NO_PROGRESS",
    "STALL_EXIT": "TIME_STOP_STALL",
    "RUNNER_TIME_STOP": "TIME_STOP_RUNNER",
    "MFE_GIVEBACK": "TRAILING_GIVEBACK",
}

_STOP_ADJ_MAP: dict[str, tuple[str, str]] = {
    "t1_profit_lock": ("BREAKEVEN", "T1_HIT"),
    "profit_floor": ("PROFIT_FLOOR", "R_LADDER"),
    "structure_trail": ("STRUCTURE_TRAIL", "STRUCTURE_BREAK"),
    "partial_resize": ("PARTIAL_RESIZE", "QTY_REDUCTION"),
    "mfe_giveback": ("MFE_GIVEBACK", "MFE_RETRACE"),
}

_GRADE_STRENGTH: dict[str, float] = {
    "a_plus": 0.9,
    "a": 0.7,
    "b": 0.5,
    "c": 0.3,
    "flat": 0.0,
}

_ENTRY_GATES: tuple[str, ...] = (
    "session", "news", "regime", "pullback", "confirmation",
    "entry_plan", "stop_validation", "daily_room", "asset_context", "score_min",
)


def _safe_kit_call(kit: Any, method: str, **kwargs: Any) -> None:
    fn = getattr(kit, method, None)
    if fn is None:
        return
    try:
        fn(**kwargs)
    except Exception:  # noqa: BLE001 — instrumentation must never block trading
        _log.exception("TPC instrumentation kit.%s failed", method)


def _epoch_seconds(ts: datetime | None) -> int | None:
    if ts is None:
        return None
    try:
        return int(ts.timestamp())
    except Exception:  # noqa: BLE001
        return None


def param_set_id(cfg: TPCSymbolConfig) -> str:
    payload = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.md5(encoded).hexdigest()[:8]


def signal_strength_proxy(score: float | None, grade: str | None, cfg: TPCSymbolConfig) -> float:
    if score is not None and score > 0:
        denom = max(int(cfg.score_a_min), 1)
        return min(float(score) / denom, 1.0)
    if grade:
        return _GRADE_STRENGTH.get(str(grade).lower(), 0.0)
    return 0.0


def exit_reason_for(event_code: str, action_reason: str | None) -> str:
    if event_code == "STOP_FILLED":
        return "STOP_LOSS"
    key = (action_reason or "").upper()
    return _EXIT_REASON_MAP.get(key, key or "EXIT")


def stop_adjustment_taxonomy(action_reason: str) -> tuple[str, str]:
    return _STOP_ADJ_MAP.get(
        action_reason,
        (action_reason.upper() or "OTHER", action_reason or "UNKNOWN"),
    )


def _t1_target_price(position: ETFPosition) -> float | None:
    target = (position.meta or {}).get("target_price")
    try:
        target_f = float(target) if target is not None else 0.0
    except (TypeError, ValueError):
        return None
    return target_f if target_f > 0 else None


def route_entry(
    kit: Any,
    setup: SetupSnapshot,
    fill: ETFFill,
    cfg: TPCSymbolConfig,
    state: ETFCoreState,
) -> None:
    if kit is None or setup is None:
        return
    side = "LONG" if int(setup.direction) == 1 else "SHORT"
    fill_price = float(fill.fill_price or setup.entry_price)
    fill_qty = float(fill.fill_qty or setup.qty)
    score = float(setup.score)
    strength = signal_strength_proxy(score, setup.grade, cfg)
    meta = setup.meta or {}
    confirmations = list(meta.get("confirmations") or [])
    lane = str(meta.get("setup_lane") or "primary")
    pullback_timeframe = str(meta.get("pullback_timeframe") or "1h")
    active = [*_ENTRY_GATES[:3], f"lane_{lane}", *_ENTRY_GATES[3:]]
    strategy_params = {
        "param_set_id": param_set_id(cfg),
        "config": asdict(cfg) if is_dataclass(cfg) else dict(cfg),
        "lane": lane,
        "pullback_timeframe": pullback_timeframe,
        "score_model": cfg.score_model,
        "t1_r": cfg.t1_r,
        "t2_r": cfg.t2_r,
        "t1_partial_pct": cfg.t1_partial_pct,
        "t2_partial_pct": cfg.t2_partial_pct,
        "risk_pct": setup.risk_pct,
        "entry_model": setup.entry_model,
        "grade": setup.grade,
        "setup_type": setup.setup_type,
    }
    signal_factors = [
        {"factor_name": "grade", "factor_value": setup.grade, "threshold": "A_PLUS",
         "contribution": "regime_quality"},
        {"factor_name": "setup_type", "factor_value": setup.setup_type, "threshold": "TYPE_A",
         "contribution": "pullback_class"},
        {"factor_name": "score", "factor_value": score, "threshold": cfg.score_a_min,
         "contribution": "composite_quality"},
        {"factor_name": "confirmations_count", "factor_value": len(set(confirmations)),
         "threshold": cfg.confirmation_required, "contribution": "signal_confluence"},
        {"factor_name": "asset_context_score", "factor_value": meta.get("asset_context_score"),
         "threshold": cfg.asset_context_min_score, "contribution": "cross_asset_alignment"},
        {"factor_name": "depth_atr", "factor_value": meta.get("depth"),
         "contribution": "pullback_depth"},
        {"factor_name": "rr_planned", "factor_value": meta.get("rr"), "contribution": "reward_risk"},
        {"factor_name": "daily_has_room", "factor_value": meta.get("daily_has_room"),
         "contribution": "higher_tf_room"},
        {"factor_name": "orderly_pullback", "factor_value": meta.get("orderly_pullback"),
         "contribution": "pullback_quality"},
    ]
    sizing_inputs = {
        "target_risk_pct": setup.risk_pct,
        "account_equity": float(getattr(state, "equity", 0.0)) or None,
        "atr_4h": meta.get("atr_4h"),
        "risk_per_share": setup.risk_per_share,
        "sizing_model": "tpc_score_band",
    }
    lane_distribution: dict[str, int] = {}
    for pos in state.positions.values():
        pos_lane = str((pos.meta or {}).get("setup_lane") or "primary")
        lane_distribution[pos_lane] = lane_distribution.get(pos_lane, 0) + 1
    portfolio_state_at_entry = {
        "num_positions": len(state.positions),
        "symbols_held": sorted(state.positions.keys()),
        "tpc_lane_distribution": lane_distribution,
    }
    bar_id = (
        f"{setup.symbol}-{_epoch_seconds(setup.created_ts)}"
        if setup.created_ts is not None else None
    )
    _safe_kit_call(
        kit, "log_entry",
        trade_id=setup.setup_id,
        pair=setup.symbol,
        side=side,
        entry_price=fill_price,
        position_size=fill_qty,
        position_size_quote=fill_price * fill_qty,
        entry_signal=setup.setup_type,
        entry_signal_id=setup.setup_id,
        entry_signal_strength=strength,
        active_filters=active,
        passed_filters=active,
        strategy_params=strategy_params,
        signal_factors=signal_factors,
        sizing_inputs=sizing_inputs,
        portfolio_state_at_entry=portfolio_state_at_entry,
        expected_entry_price=setup.entry_price,
        exchange_timestamp=fill.fill_time,
        bar_id=bar_id,
        **fill_runtime_refs(
            fill.oms_order_id,
            fill.runtime_payload,
            fill_qty=fill_qty,
        ),
    )


def route_exit(
    kit: Any,
    pre_position: ETFPosition,
    fill: ETFFill,
    event_code: str,
    event_reason: str | None,
) -> None:
    if kit is None or pre_position is None:
        return
    exit_price = float(fill.fill_price or 0.0)
    entry_price = float(pre_position.entry_price)
    direction = 1 if int(pre_position.direction) == 1 else -1
    pnl_per_share = (exit_price - entry_price) * direction
    pnl_pct = pnl_per_share / entry_price if entry_price else 0.0

    # Sign-correct MFE/MAE pct from price (not from R-magnitude). MFE is
    # always favourable, MAE always adverse, but pct keeps directional sign
    # so downstream analytics see "+2.5%" vs "-1.1%" consistently.
    mfe_pct: float | None = None
    mae_pct: float | None = None
    if entry_price:
        if pre_position.mfe_price:
            mfe_pct = (float(pre_position.mfe_price) - entry_price) / entry_price * direction
        if pre_position.mae_price:
            mae_pct = (float(pre_position.mae_price) - entry_price) / entry_price * direction

    reason_upper = (event_reason or "").upper()
    expected: float | None = None
    if event_code == "STOP_FILLED" or reason_upper == "STOP":
        expected = float(pre_position.current_stop)
    elif reason_upper == "T1":
        expected = _t1_target_price(pre_position)

    _safe_kit_call(
        kit, "log_exit",
        trade_id=pre_position.setup_id,
        exit_price=exit_price,
        exit_reason=exit_reason_for(event_code, event_reason),
        fees_paid=float(getattr(fill, "commission", 0.0) or 0.0),
        exchange_timestamp=fill.fill_time,
        expected_exit_price=expected,
        mfe_price=float(pre_position.mfe_price) if pre_position.mfe_price else None,
        mae_price=float(pre_position.mae_price) if pre_position.mae_price else None,
        mfe_pct=mfe_pct,
        mae_pct=mae_pct,
        mfe_r=float(pre_position.mfe_r),
        mae_r=float(pre_position.mae_r),
        pnl_pct=float(pnl_pct),
        **fill_runtime_refs(
            fill.oms_order_id,
            fill.runtime_payload,
            fill_qty=fill.fill_qty,
            is_exit=True,
        ),
    )


def route_stop_adjustment(
    kit: Any,
    setup_id: str,
    symbol: str,
    old_stop: float,
    new_stop: float,
    action_reason: str,
    position: ETFPosition | None,
) -> None:
    if kit is None or old_stop == new_stop:
        return
    adjustment_type, trigger = stop_adjustment_taxonomy(action_reason)
    metadata: dict[str, Any] = {}
    if position is not None:
        metadata = {
            "mfe_r": float(position.mfe_r),
            "bars_held_15m": int(position.bars_held_15m),
            "setup_lane": (position.meta or {}).get("setup_lane"),
            "pullback_type": position.setup_type,
        }
    _safe_kit_call(
        kit, "log_stop_adjustment",
        trade_id=setup_id,
        symbol=symbol,
        old_stop=float(old_stop),
        new_stop=float(new_stop),
        adjustment_type=adjustment_type,
        trigger=trigger,
        metadata=metadata,
    )


def route_missed(
    kit: Any,
    rejection: dict[str, Any],
    cfg: TPCSymbolConfig,
    bar_ts: datetime | None = None,
) -> None:
    if kit is None or not rejection:
        return
    symbol = str(rejection.get("symbol") or "")
    lane = str(rejection.get("lane") or "primary")
    direction = str(rejection.get("direction") or "FLAT").upper()
    side = direction if direction in {"LONG", "SHORT"} else "FLAT"
    grade = rejection.get("grade")
    grade_str = grade if isinstance(grade, str) else ""
    details = rejection.get("details") if isinstance(rejection.get("details"), dict) else {}
    score = details.get("score")
    blocked_by = str(rejection.get("blocked_by") or "unknown")
    block_reason = str(rejection.get("block_reason") or "")
    strength = signal_strength_proxy(score, grade_str, cfg)
    epoch = _epoch_seconds(bar_ts)
    signal_id = f"{symbol}-{epoch if epoch is not None else 'bar'}-{lane}"
    bar_id = f"{symbol}-{epoch}" if epoch is not None else None
    strategy_params = {
        "lane": lane,
        "current_grade": grade_str,
        "candidate_score": score,
        "details": details,
    }
    _safe_kit_call(
        kit, "log_missed",
        pair=symbol,
        side=side,
        signal=f"TPC_{lane}",
        signal_id=signal_id,
        signal_strength=float(strength),
        blocked_by=blocked_by,
        block_reason=block_reason,
        strategy_params=strategy_params,
        market_regime=grade_str,
        exchange_timestamp=bar_ts,
        bar_id=bar_id,
    )


def route_filter_decisions(
    kit: Any,
    bar_input: Any,
    cfg: TPCSymbolConfig,
    rejections: list[dict[str, Any]] | None = None,
    entry_setup: SetupSnapshot | None = None,
) -> None:
    """Emit one ``on_filter_decision`` per rejected gate, plus one ``passed``
    record for every active gate when an entry is requested.

    Without an entry, only rejected gates produce records — no synthetic
    "passed" stream is invented (would be expensive and noisy with no signal).
    """
    if kit is None or bar_input is None or bar_input.bar_15m is None:
        return
    bar = bar_input.bar_15m
    epoch = _epoch_seconds(bar.timestamp)
    bar_id = f"{bar_input.symbol}-{epoch}" if epoch is not None else None
    pair = bar_input.symbol

    for rejection in rejections or ():
        blocked_by = str(rejection.get("blocked_by") or "unknown")
        lane = str(rejection.get("lane") or "primary")
        grade = rejection.get("grade") if isinstance(rejection.get("grade"), str) else ""
        details = rejection.get("details") if isinstance(rejection.get("details"), dict) else {}
        score = details.get("score") if isinstance(details, dict) else None
        threshold = details.get("threshold")
        actual = details.get("actual") or score
        _safe_kit_call(
            kit, "on_filter_decision",
            pair=pair,
            filter_name=blocked_by,
            passed=False,
            threshold=float(threshold) if isinstance(threshold, (int, float)) else 0.0,
            actual_value=float(actual) if isinstance(actual, (int, float)) else 0.0,
            signal_name=f"TPC_{lane}",
            signal_strength=signal_strength_proxy(score, grade, cfg),
            strategy_id="TPC",
            exchange_timestamp=bar.timestamp,
            bar_id=bar_id,
        )

    if entry_setup is not None:
        meta = entry_setup.meta or {}
        lane = str(meta.get("setup_lane") or "primary")
        signal_name = f"{lane}_{entry_setup.setup_type}"
        strength = signal_strength_proxy(entry_setup.score, entry_setup.grade, cfg)
        for gate in _ENTRY_GATES:
            _safe_kit_call(
                kit, "on_filter_decision",
                pair=pair,
                filter_name=gate,
                passed=True,
                threshold=0.0,
                actual_value=1.0,
                signal_name=signal_name,
                signal_strength=strength,
                strategy_id="TPC",
                exchange_timestamp=bar.timestamp,
                bar_id=bar_id,
            )


def route_indicator_snapshot(
    kit: Any,
    bar_input: Any,
    state: ETFCoreState,
    cfg: TPCSymbolConfig,
    decision: str,
    setup: SetupSnapshot | None,
) -> None:
    if kit is None or bar_input is None or bar_input.bar_15m is None:
        return
    indicators = {
        k: float(v) for k, v in (bar_input.indicators or {}).items()
        if isinstance(v, (int, float))
    }
    if setup is not None:
        lane = (setup.meta or {}).get("setup_lane", "primary")
        signal_name = f"{lane}_{setup.setup_type}"
        signal_strength = signal_strength_proxy(setup.score, setup.grade, cfg)
    else:
        signal_name = decision or "NO_SIGNAL"
        signal_strength = 0.0
    bar = bar_input.bar_15m
    epoch = _epoch_seconds(bar.timestamp)
    bar_id = f"{bar_input.symbol}-{epoch}" if epoch is not None else None
    _safe_kit_call(
        kit, "on_indicator_snapshot",
        pair=bar_input.symbol,
        indicators=indicators,
        signal_name=signal_name,
        signal_strength=float(signal_strength),
        decision=decision or "NO_SIGNAL",
        strategy_id="TPC",
        exchange_timestamp=bar.timestamp,
        bar_id=bar_id,
    )


def route_order_event(
    kit: Any,
    update: ETFOrderUpdate,
    event: Any,
) -> None:
    """Translate ORDER_TERMINAL / ADDON_ORDER_TERMINAL events to ``on_order_event``."""
    if kit is None or update is None:
        return
    details = getattr(event, "details", None) or {}
    setup_id = str(details.get("setup_id") or "")
    status = str(details.get("status") or update.status or "")
    role = str(getattr(update, "order_role", "") or "")
    _safe_kit_call(
        kit, "on_order_event",
        order_id=str(update.oms_order_id or ""),
        pair=str(update.symbol or ""),
        side="",
        order_type=role,
        status=status,
        requested_qty=0.0,
        filled_qty=0.0,
        related_trade_id=setup_id,
        strategy_id="TPC",
        order_action=event.code if hasattr(event, "code") else "TERMINAL",
        exchange_timestamp=update.timestamp,
    )
