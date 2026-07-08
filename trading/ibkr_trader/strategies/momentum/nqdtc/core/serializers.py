from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from strategies.momentum.nqdtc.models import Direction, EntrySubtype, ExitTier, PositionState, Session, TPLevel, WorkingOrder

from .state import NQDTCCoreState


def snapshot_state(state: NQDTCCoreState) -> dict[str, Any]:
    return {
        "symbol": state.symbol,
        "bar_count_5m": state.bar_count_5m,
        "last_decision_code": state.last_decision_code,
        "last_decision_details": dict(state.last_decision_details),
        "last_bar_ts": state.last_bar_ts.isoformat() if state.last_bar_ts else None,
        "position": _snapshot_position(state.position),
        "working_orders": [_snapshot_working_order(order) for order in state.working_orders],
    }


def restore_state(snapshot: Mapping[str, Any]) -> NQDTCCoreState:
    return NQDTCCoreState(
        symbol=str(snapshot.get("symbol", "")),
        position=_restore_position(snapshot.get("position", {})),
        working_orders=[_restore_working_order(order) for order in snapshot.get("working_orders", [])],
        bar_count_5m=int(snapshot.get("bar_count_5m", 0)),
        last_decision_code=str(snapshot.get("last_decision_code", "IDLE")),
        last_decision_details=dict(snapshot.get("last_decision_details", {})),
        last_bar_ts=_restore_datetime(snapshot.get("last_bar_ts")),
    )


def _snapshot_position(position: PositionState) -> dict[str, Any]:
    return {
        "open": position.open,
        "symbol": position.symbol,
        "direction": position.direction.value,
        "entry_subtype": position.entry_subtype.value,
        "entry_price": position.entry_price,
        "stop_price": position.stop_price,
        "initial_stop_price": position.initial_stop_price,
        "qty": position.qty,
        "qty_open": position.qty_open,
        "R_dollars": position.R_dollars,
        "risk_pct": position.risk_pct,
        "quality_mult": position.quality_mult,
        "final_risk_pct": position.final_risk_pct,
        "exit_tier": position.exit_tier.value,
        "profit_funded": position.profit_funded,
        "tp_levels": [_snapshot_tp_level(level) for level in position.tp_levels],
        "runner_active": position.runner_active,
        "chandelier_trail": position.chandelier_trail,
        "mm_level": position.mm_level,
        "mm_reached": position.mm_reached,
        "stop_oms_order_id": position.stop_oms_order_id,
        "bars_since_entry_30m": position.bars_since_entry_30m,
        "highest_since_entry": position.highest_since_entry,
        "lowest_since_entry": position.lowest_since_entry,
        "peak_mfe_r": position.peak_mfe_r,
        "peak_mae_r": position.peak_mae_r,
        "hold_ref": position.hold_ref,
        "box_high_at_entry": position.box_high_at_entry,
        "box_low_at_entry": position.box_low_at_entry,
        "box_mid_at_entry": position.box_mid_at_entry,
        "entry_session": position.entry_session.value,
        "tp1_only_cap": position.tp1_only_cap,
        "stale_bridge_extended": position.stale_bridge_extended,
        "stale_bridge_extra_bars": position.stale_bridge_extra_bars,
        "bars_since_tp1": position.bars_since_tp1,
        "peak_r_initial": position.peak_r_initial,
        "early_be_triggered": position.early_be_triggered,
        "stop_source": position.stop_source,
    }


def _restore_position(data: Mapping[str, Any]) -> PositionState:
    position = PositionState()
    if not data:
        return position
    position.open = bool(data.get("open", False))
    position.symbol = str(data.get("symbol", "NQ"))
    position.direction = Direction(data.get("direction", 0))
    position.entry_subtype = EntrySubtype(data.get("entry_subtype", EntrySubtype.A_RETEST.value))
    position.entry_price = float(data.get("entry_price", 0.0))
    position.stop_price = float(data.get("stop_price", 0.0))
    position.initial_stop_price = float(data.get("initial_stop_price", 0.0))
    position.qty = int(data.get("qty", 0))
    position.qty_open = int(data.get("qty_open", 0))
    position.R_dollars = float(data.get("R_dollars", 0.0))
    position.risk_pct = float(data.get("risk_pct", 0.0))
    position.quality_mult = float(data.get("quality_mult", 1.0))
    position.final_risk_pct = float(data.get("final_risk_pct", 0.0))
    position.exit_tier = ExitTier(data.get("exit_tier", ExitTier.NEUTRAL.value))
    position.profit_funded = bool(data.get("profit_funded", False))
    position.tp_levels = [_restore_tp_level(level) for level in data.get("tp_levels", [])]
    position.runner_active = bool(data.get("runner_active", False))
    position.chandelier_trail = float(data.get("chandelier_trail", 0.0))
    position.mm_level = float(data.get("mm_level", 0.0))
    position.mm_reached = bool(data.get("mm_reached", False))
    position.stop_oms_order_id = str(data.get("stop_oms_order_id", ""))
    position.bars_since_entry_30m = int(data.get("bars_since_entry_30m", 0))
    position.highest_since_entry = float(data.get("highest_since_entry", 0.0))
    position.lowest_since_entry = float(data.get("lowest_since_entry", 0.0))
    position.peak_mfe_r = float(data.get("peak_mfe_r", 0.0))
    position.peak_mae_r = float(data.get("peak_mae_r", 0.0))
    position.hold_ref = float(data.get("hold_ref", 0.0))
    position.box_high_at_entry = float(data.get("box_high_at_entry", 0.0))
    position.box_low_at_entry = float(data.get("box_low_at_entry", 0.0))
    position.box_mid_at_entry = float(data.get("box_mid_at_entry", 0.0))
    position.entry_session = Session(data.get("entry_session", Session.RTH.value))
    position.tp1_only_cap = bool(data.get("tp1_only_cap", False))
    position.stale_bridge_extended = bool(data.get("stale_bridge_extended", False))
    position.stale_bridge_extra_bars = int(data.get("stale_bridge_extra_bars", 0))
    position.bars_since_tp1 = int(data.get("bars_since_tp1", -1))
    position.peak_r_initial = float(data.get("peak_r_initial", 0.0))
    position.early_be_triggered = bool(data.get("early_be_triggered", False))
    position.stop_source = str(data.get("stop_source", "INITIAL"))
    return position


def _snapshot_tp_level(level: TPLevel) -> dict[str, Any]:
    return {
        "r_target": level.r_target,
        "pct": level.pct,
        "qty": level.qty,
        "filled": level.filled,
        "oms_order_id": level.oms_order_id,
    }


def _restore_tp_level(data: Mapping[str, Any]) -> TPLevel:
    return TPLevel(
        r_target=float(data.get("r_target", 0.0)),
        pct=float(data.get("pct", 0.0)),
        qty=int(data.get("qty", 0)),
        filled=bool(data.get("filled", False)),
        oms_order_id=str(data.get("oms_order_id", "")),
    )


def _snapshot_working_order(order: WorkingOrder) -> dict[str, Any]:
    return {
        "oms_order_id": order.oms_order_id,
        "subtype": order.subtype.value,
        "direction": order.direction.value,
        "price": order.price,
        "qty": order.qty,
        "submitted_bar_idx": order.submitted_bar_idx,
        "ttl_bars": order.ttl_bars,
        "oca_group": order.oca_group,
        "is_limit": order.is_limit,
        "rescue_attempted": order.rescue_attempted,
        "quality_mult": order.quality_mult,
        "stop_for_risk": order.stop_for_risk,
        "expected_fill_price": order.expected_fill_price,
        "disp_norm": order.disp_norm,
    }


def _restore_working_order(data: Mapping[str, Any]) -> WorkingOrder:
    return WorkingOrder(
        oms_order_id=str(data.get("oms_order_id", "")),
        subtype=EntrySubtype(data.get("subtype", EntrySubtype.A_RETEST.value)),
        direction=Direction(data.get("direction", 0)),
        price=float(data.get("price", 0.0)),
        qty=int(data.get("qty", 0)),
        submitted_bar_idx=int(data.get("submitted_bar_idx", 0)),
        ttl_bars=int(data.get("ttl_bars", 3)),
        oca_group=str(data.get("oca_group", "")),
        is_limit=bool(data.get("is_limit", False)),
        rescue_attempted=bool(data.get("rescue_attempted", False)),
        quality_mult=float(data.get("quality_mult", 1.0)),
        stop_for_risk=float(data.get("stop_for_risk", 0.0)),
        expected_fill_price=float(data.get("expected_fill_price", 0.0)),
        disp_norm=float(data.get("disp_norm", 0.0)),
    )


def _restore_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
