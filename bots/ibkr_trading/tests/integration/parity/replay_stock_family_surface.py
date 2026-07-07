from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from tests.integration.parity.replay_candidates import (
    ReplayDecisionTimeline,
    _last_source_close,
    entry_candidate_specs as _entry_candidate_specs,
)
from tests.integration.parity.replay_family_surface_common import (
    _accepted_status,
    _candidate_risk_pct_by_strategy,
    _decision_summary,
    _family_decision,
    _initial_position_stop,
    _initial_positions,
    _portfolio_reason,
    _run_blocking,
)
from tests.integration.parity.source_inputs import parse_time


def _run_stock_family_surface(fixture: Mapping[str, Any], out: ReplayDecisionTimeline) -> dict[str, Any]:
    from backtests.stock.auto.portfolio_synergy.core.logic import run_portfolio_replay
    from backtests.stock.auto.portfolio_synergy.evaluator import build_effective_portfolio_config
    from backtests.stock.models import Direction, TradeRecord

    alcb: list[TradeRecord] = []
    iaric: list[TradeRecord] = []
    candidates = _entry_candidate_specs(fixture, out)
    for trade in _initial_stock_position_trades(fixture, TradeRecord, Direction):
        if str(trade.strategy) == "ALCB_v1":
            alcb.append(trade)
        else:
            iaric.append(trade)
    for candidate in candidates:
        trade = _stock_trade_from_candidate(fixture, candidate, TradeRecord, Direction)
        if str(candidate["order"]["strategy_id"]) == "ALCB_v1":
            alcb.append(trade)
        else:
            iaric.append(trade)
    effective = build_effective_portfolio_config(
        _stock_portfolio_mutations(fixture, candidates),
        initial_equity=float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0)),
    )
    result = _run_blocking(lambda: run_portfolio_replay(alcb, iaric, effective))
    decisions = _stock_family_decisions(candidates, result)
    return {
        "adapter": "stock_portfolio_replay",
        **_decision_summary(decisions, family_surface="stock_portfolio_replay"),
        "portfolio_state": {
            "metrics": {
                "entries_accepted_by_portfolio": result.metrics.get("entries_accepted_by_portfolio", 0.0),
                "entries_blocked_by_portfolio": result.metrics.get("entries_blocked_by_portfolio", 0.0),
            },
            "rules": effective.get("portfolio_rules", {}),
            "cross_strategy_rules": effective.get("cross_strategy_rules", {}),
        },
    }


def _stock_trade_from_candidate(fixture: Mapping[str, Any], candidate: Mapping[str, Any], trade_cls: Any, direction_cls: Any) -> Any:
    order = candidate["order"]
    symbol = str(order["symbol"])
    direction = direction_cls.LONG if str(order["side"]).upper() == "BUY" else direction_cls.SHORT
    entry_time = candidate["entry_time"]
    return trade_cls(
        strategy=str(order["strategy_id"]),
        symbol=symbol,
        direction=direction,
        entry_time=entry_time,
        exit_time=entry_time + timedelta(minutes=1),
        entry_price=float(candidate["entry_price"]),
        exit_price=float(candidate["entry_price"]),
        quantity=float(candidate["qty"]),
        pnl=0.0,
        r_multiple=0.0,
        risk_per_share=max(abs(float(candidate["entry_price"]) - float(candidate["stop_price"])), 0.01),
        commission=float(candidate.get("commission", 0.0)),
        slippage=0.0,
        entry_type=_stock_entry_type(fixture, symbol),
        sector=_stock_sector(fixture, symbol),
        fill_time=entry_time,
        metadata={
            "parity_generated_candidate": True,
            "parity_original_qty": int(float(order.get("qty", candidate["qty"]) or candidate["qty"])),
            "parity_fill_qty": int(float(candidate.get("fill_qty", candidate["qty"]) or candidate["qty"])),
        },
    )


def _initial_stock_position_trades(fixture: Mapping[str, Any], trade_cls: Any, direction_cls: Any) -> list[Any]:
    trades = []
    clock = parse_time(fixture["clock_start"])
    for pos in _initial_positions(fixture):
        symbol = str(pos.get("symbol") or pos.get("instrument_symbol") or "")
        qty = abs(float(pos.get("net_qty", pos.get("qty", 0.0)) or 0.0))
        if not symbol or qty <= 0:
            continue
        entry_price = float(pos.get("avg_price", pos.get("entry_price", _last_source_close(fixture, symbol, 1.0))) or 0.0)
        stop = _initial_position_stop(fixture, pos, entry_price)
        direction = direction_cls.LONG if float(pos.get("net_qty", pos.get("qty", 0.0)) or 0.0) > 0 else direction_cls.SHORT
        trades.append(
            trade_cls(
                strategy=str(pos.get("strategy_id", "")),
                symbol=symbol,
                direction=direction,
                entry_time=clock - timedelta(minutes=1),
                exit_time=clock + timedelta(days=365),
                entry_price=entry_price,
                exit_price=entry_price,
                quantity=qty,
                pnl=0.0,
                r_multiple=0.0,
                risk_per_share=max(abs(entry_price - stop), 0.01),
                commission=0.0,
                slippage=0.0,
                entry_type="INITIAL_POSITION",
                sector=_stock_sector(fixture, symbol),
                fill_time=clock - timedelta(minutes=1),
                metadata={"initial_position": True},
            )
        )
    return trades


def _stock_portfolio_mutations(fixture: Mapping[str, Any], candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    equity = float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0))
    payload = dict(((fixture.get("family_config", {}) or {}).get("portfolio_rules", {}) or {}))
    risk_pct_by_strategy = _candidate_risk_pct_by_strategy(fixture, candidates, equity)
    for pos in _initial_positions(fixture):
        sid = str(pos.get("strategy_id", ""))
        symbol = str(pos.get("symbol") or pos.get("instrument_symbol") or "")
        risk = float(pos.get("open_risk_dollars", 0.0) or 0.0)
        if sid and symbol and risk > 0:
            risk_pct_by_strategy.setdefault(sid, risk / max(equity, 1.0))
    return {
        "portfolio_rules": {
            "reference_risk_pct": float(payload.get("reference_unit_risk_pct", 0.00648)),
            "heat_cap_R": float(payload.get("portfolio_heat_cap_R", payload.get("directional_cap_R", 6.5))),
            "max_total_active_positions": int(payload.get("max_total_active_positions", 12)),
            "max_symbol_heat_R": float(payload.get("max_symbol_heat_R", 2.2)),
            "max_long_heat_R": float(payload.get("directional_cap_long_R", payload.get("directional_cap_R", 6.25))),
            "max_single_strategy_trade_share": float(payload.get("max_single_strategy_trade_share", 1.0)),
            "drawdown_tiers": tuple(tuple(float(part) for part in row) for row in payload.get("dd_tiers", ((1.0, 1.0),)) or ()),
        },
        "strategy_allocations": {
            "IARIC_V5R1": {
                "unit_risk_pct": float(risk_pct_by_strategy.get("IARIC_v1", 0.0080)),
                "max_heat_R": float(payload.get("max_strategy_heat_R", {}).get("IARIC_v1", 4.6)) if isinstance(payload.get("max_strategy_heat_R"), Mapping) else 4.6,
                "max_concurrent": int(payload.get("max_strategy_active_positions", {}).get("IARIC_v1", 9)) if isinstance(payload.get("max_strategy_active_positions"), Mapping) else 9,
                "priority": 0,
            },
            "ALCB_R3": {
                "unit_risk_pct": float(risk_pct_by_strategy.get("ALCB_v1", 0.0065)),
                "max_heat_R": float(payload.get("max_strategy_heat_R", {}).get("ALCB_v1", 3.25)) if isinstance(payload.get("max_strategy_heat_R"), Mapping) else 3.25,
                "max_concurrent": int(payload.get("max_strategy_active_positions", {}).get("ALCB_v1", 6)) if isinstance(payload.get("max_strategy_active_positions"), Mapping) else 6,
                "priority": 1,
            },
        },
        "dynamic_allocation": {"enabled": False},
        "cross_strategy_rules": {
            "same_symbol_policy": str(payload.get("symbol_collision_action", "half_size")),
            "symbol_collision_pairs": _stock_replay_collision_pairs(payload.get("symbol_collision_pairs", ())),
            "same_sector_heat_cap_R": float(payload.get("same_sector_heat_cap_R", 3.8)),
        },
    }


def _stock_replay_collision_pairs(value: Any) -> list[tuple[str, str, str]]:
    strategy_map = {"IARIC_v1": "IARIC_V5R1", "ALCB_v1": "ALCB_R3"}
    pairs = []
    for row in value or ():
        if len(row) < 3:
            continue
        pairs.append((strategy_map.get(str(row[0]), str(row[0])), strategy_map.get(str(row[1]), str(row[1])), str(row[2])))
    return pairs


def _stock_family_decisions(candidates: list[Mapping[str, Any]], result: Any) -> list[dict[str, Any]]:
    generated_ids = {
        (
            _stock_replay_strategy(str(candidate["order"]["strategy_id"])),
            str(candidate["order"]["symbol"]),
            candidate["entry_time"],
        )
        for candidate in candidates
    }
    accepted_by_key = {
        _stock_position_key(pos): pos
        for pos in result.state.accepted_positions
        if _stock_position_key(pos) in generated_ids
    }
    blocked_by_key = {
        _stock_blocked_key(item): item
        for item in result.state.blocked_candidates
        if _stock_blocked_key(item) in generated_ids
    }
    decisions = []
    for candidate in candidates:
        key = (
            _stock_replay_strategy(str(candidate["order"]["strategy_id"])),
            str(candidate["order"]["symbol"]),
            candidate["entry_time"],
        )
        if key in blocked_by_key:
            reason = str(getattr(blocked_by_key[key], "reason", "") or "portfolio_rule")
            decisions.append(_family_decision(candidate, approved_qty=0, status="rejected", reason=_portfolio_reason(reason)))
        elif key in accepted_by_key:
            pos = accepted_by_key[key]
            metadata = getattr(pos, "metadata", {}) or {}
            approved_qty = int(float(metadata.get("portfolio_approved_qty", getattr(pos, "quantity", candidate["qty"])) or candidate["qty"]))
            decisions.append(_family_decision(candidate, approved_qty=approved_qty, status=_accepted_status(candidate, approved_qty), reason=""))
        else:
            raise AssertionError(f"stock family replay produced no decision for candidate {candidate['candidate_key']}")
    return decisions


def _stock_position_key(position: Any) -> tuple[str, str, Any]:
    return (str(getattr(position, "strategy", "")), str(getattr(position, "symbol", "")), getattr(position, "entry_time", None))


def _stock_blocked_key(item: Any) -> tuple[str, str, Any]:
    return (str(getattr(item, "strategy", "")), str(getattr(item, "symbol", "")), getattr(item, "entry_time", None))


def _stock_replay_strategy(strategy_id: str) -> str:
    return {"IARIC_v1": "IARIC_V5R1", "ALCB_v1": "ALCB_R3"}.get(strategy_id, strategy_id)


def _stock_sector(fixture: Mapping[str, Any], symbol: str) -> str:
    for item in (((fixture.get("artifacts", {}) or {}).get("iaric", {}) or {}).get("watchlist", []) or []):
        if str(item.get("symbol", "")).upper() == symbol.upper():
            return str(item.get("sector", "Technology"))
    return "Technology"


def _stock_entry_type(fixture: Mapping[str, Any], symbol: str) -> str:
    for item in (((fixture.get("artifacts", {}) or {}).get("iaric", {}) or {}).get("watchlist", []) or []):
        if str(item.get("symbol", "")).upper() == symbol.upper():
            triggers = item.get("trigger_types", []) or []
            return str(triggers[0]) if triggers else "OPENING_RECLAIM"
    return "OPENING_RECLAIM"


run_stock_family_surface = _run_stock_family_surface
