from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from strategies.momentum.downturn.models import ActivePosition, CompositeRegime, EngineTag, VolState, WorkingEntry

from .state import DownturnCoreState


def snapshot_state(state: DownturnCoreState) -> dict[str, Any]:
    return {
        "symbol": state.symbol,
        "bar_count_5m": state.bar_count_5m,
        "bars_since_last_entry": state.bars_since_last_entry,
        "last_decision_code": state.last_decision_code,
        "last_decision_details": dict(state.last_decision_details),
        "last_bar_ts": state.last_bar_ts.isoformat() if state.last_bar_ts else None,
        "position": _snapshot_position(state.position),
        "working_entries": [_snapshot_working_entry(entry) for entry in state.working_entries],
    }


def restore_state(snapshot: Mapping[str, Any]) -> DownturnCoreState:
    return DownturnCoreState(
        symbol=str(snapshot.get("symbol", "")),
        position=_restore_position(snapshot.get("position")),
        working_entries=[_restore_working_entry(entry) for entry in snapshot.get("working_entries", [])],
        bar_count_5m=int(snapshot.get("bar_count_5m", 0)),
        bars_since_last_entry=int(snapshot.get("bars_since_last_entry", 999)),
        last_decision_code=str(snapshot.get("last_decision_code", "IDLE")),
        last_decision_details=dict(snapshot.get("last_decision_details", {})),
        last_bar_ts=_restore_datetime(snapshot.get("last_bar_ts")),
    )


def _snapshot_position(position: ActivePosition | None) -> dict[str, Any] | None:
    if position is None:
        return None
    return {
        "engine_tag": position.engine_tag.value,
        "signal_class": position.signal_class,
        "trade_id": position.trade_id,
        "entry_price": position.entry_price,
        "stop0": position.stop0,
        "qty": position.qty,
        "remaining_qty": position.remaining_qty,
        "entry_oms_order_id": position.entry_oms_order_id,
        "stop_oms_order_id": position.stop_oms_order_id,
        "entry_time": position.entry_time.isoformat() if position.entry_time else None,
        "hold_bars_5m": position.hold_bars_5m,
        "hold_bars_1h": position.hold_bars_1h,
        "hold_bars_30m": position.hold_bars_30m,
        "hold_bars_4h": position.hold_bars_4h,
        "mfe_price": position.mfe_price,
        "mae_price": position.mae_price,
        "r_at_peak": position.r_at_peak,
        "chandelier_stop": position.chandelier_stop,
        "be_triggered": position.be_triggered,
        "exit_trigger": position.exit_trigger,
        "tp_schedule": [list(level) for level in position.tp_schedule],
        "tp_idx": position.tp_idx,
        "scaled_out": position.scaled_out,
        "composite_regime": position.composite_regime.value,
        "vol_state": position.vol_state.value,
        "in_correction": position.in_correction,
        "predator": position.predator,
        "commission": position.commission,
    }


def _restore_position(data: Mapping[str, Any] | None) -> ActivePosition | None:
    if not data:
        return None
    return ActivePosition(
        engine_tag=EngineTag(data.get("engine_tag", "fade")),
        signal_class=str(data.get("signal_class", "")),
        trade_id=str(data.get("trade_id", "")),
        entry_price=float(data.get("entry_price", 0.0)),
        stop0=float(data.get("stop0", 0.0)),
        qty=int(data.get("qty", 0)),
        remaining_qty=int(data.get("remaining_qty", 0)),
        entry_oms_order_id=str(data.get("entry_oms_order_id", "")),
        stop_oms_order_id=str(data.get("stop_oms_order_id", "")),
        entry_time=_restore_datetime(data.get("entry_time")),
        hold_bars_5m=int(data.get("hold_bars_5m", 0)),
        hold_bars_1h=int(data.get("hold_bars_1h", 0)),
        hold_bars_30m=int(data.get("hold_bars_30m", 0)),
        hold_bars_4h=int(data.get("hold_bars_4h", 0)),
        mfe_price=float(data.get("mfe_price", 0.0)),
        mae_price=float(data.get("mae_price", 0.0)),
        r_at_peak=float(data.get("r_at_peak", 0.0)),
        chandelier_stop=float(data.get("chandelier_stop", 0.0)),
        be_triggered=bool(data.get("be_triggered", False)),
        exit_trigger=str(data.get("exit_trigger", "")),
        tp_schedule=[tuple(level) for level in data.get("tp_schedule", [])],
        tp_idx=int(data.get("tp_idx", 0)),
        scaled_out=bool(data.get("scaled_out", False)),
        composite_regime=CompositeRegime(data.get("composite_regime", "neutral")),
        vol_state=VolState(data.get("vol_state", "normal")),
        in_correction=bool(data.get("in_correction", False)),
        predator=bool(data.get("predator", False)),
        commission=float(data.get("commission", 0.0)),
    )


def _snapshot_working_entry(entry: WorkingEntry) -> dict[str, Any]:
    return {
        "oms_order_id": entry.oms_order_id,
        "engine_tag": entry.engine_tag.value,
        "signal_class": entry.signal_class,
        "entry_price": entry.entry_price,
        "stop0": entry.stop0,
        "qty": entry.qty,
        "submitted_bar_idx": entry.submitted_bar_idx,
        "ttl_bars": entry.ttl_bars,
        "composite_regime": entry.composite_regime.value,
        "vol_state": entry.vol_state.value,
        "in_correction": entry.in_correction,
        "predator": entry.predator,
        "tp_schedule": [list(level) for level in entry.tp_schedule],
        "signal_strength": entry.signal_strength,
    }


def _restore_working_entry(data: Mapping[str, Any]) -> WorkingEntry:
    return WorkingEntry(
        oms_order_id=str(data.get("oms_order_id", "")),
        engine_tag=EngineTag(data.get("engine_tag", "fade")),
        signal_class=str(data.get("signal_class", "")),
        entry_price=float(data.get("entry_price", 0.0)),
        stop0=float(data.get("stop0", 0.0)),
        qty=int(data.get("qty", 0)),
        submitted_bar_idx=int(data.get("submitted_bar_idx", 0)),
        ttl_bars=int(data.get("ttl_bars", 72)),
        composite_regime=CompositeRegime(data.get("composite_regime", "neutral")),
        vol_state=VolState(data.get("vol_state", "normal")),
        in_correction=bool(data.get("in_correction", False)),
        predator=bool(data.get("predator", False)),
        tp_schedule=[tuple(level) for level in data.get("tp_schedule", [])],
        signal_strength=float(data.get("signal_strength", 0.5)),
    )


def _restore_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
