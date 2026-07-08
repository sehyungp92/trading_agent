from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from typing import Any

from tests.integration.parity.portfolio_rules import portfolio_rules_config_from_fixture
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
    _max_generated_qty,
    _portfolio_reason,
    _run_blocking,
)
from tests.integration.parity.source_inputs import parse_time, plain, point_value, strategy_ids


def _run_momentum_family_surface(fixture: Mapping[str, Any], out: ReplayDecisionTimeline) -> dict[str, Any]:
    from backtests.momentum.engine.family_portfolio_engine import (
        FamilyDynamicRiskConfig,
        FamilyPortfolioBacktester,
        FamilyStrategyAllocation,
        FamilyPortfolioTrade,
        make_controlled_aggressive_family_config,
    )

    trades_by_strategy: dict[str, list[FamilyPortfolioTrade]] = {sid: [] for sid in strategy_ids(fixture)}
    candidates = _entry_candidate_specs(fixture, out)
    initial_trades = _initial_momentum_position_trades(fixture, FamilyPortfolioTrade)
    for trade in initial_trades:
        trades_by_strategy.setdefault(trade.strategy_id, []).append(trade)
    for candidate in candidates:
        trade = _momentum_trade_from_candidate(candidate, FamilyPortfolioTrade)
        trades_by_strategy.setdefault(trade.strategy_id, []).append(trade)
    cfg = _momentum_family_config(
        fixture,
        make_controlled_aggressive_family_config(
            float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0))
        ),
        FamilyStrategyAllocation,
        FamilyDynamicRiskConfig,
        candidates,
        initial_trades,
    )
    result = _run_blocking(lambda: FamilyPortfolioBacktester(cfg).run(trades_by_strategy))
    decisions = _momentum_family_decisions(
        candidates,
        accepted=result.trades,
        blocked=result.blocked_trades,
    )
    return {
        "adapter": "momentum_family_portfolio_backtester",
        **_decision_summary(decisions, family_surface="momentum_family_portfolio_backtester"),
        "portfolio_state": {
            "metrics": {
                "entries_accepted_by_portfolio": result.metrics.get("entries_accepted_by_portfolio", 0.0),
                "entries_blocked_by_portfolio": result.metrics.get("entries_blocked_by_portfolio", 0.0),
            },
            "rules": plain(cfg.rules),
        },
    }


def _momentum_trade_from_candidate(candidate: Mapping[str, Any], trade_cls: Any) -> Any:
    order = candidate["order"]
    entry_time = candidate["entry_time"]
    direction = 1 if str(order["side"]).upper() == "BUY" else -1
    return trade_cls(
        strategy_id=str(order["strategy_id"]),
        direction=direction,
        entry_time=entry_time,
        exit_time=entry_time + timedelta(minutes=1),
        entry_price=float(candidate["entry_price"]),
        exit_price=float(candidate["entry_price"]),
        initial_stop=float(candidate["stop_price"]),
        raw_pnl_dollars=0.0,
        raw_qty=max(int(candidate["qty"]), 1),
        r_multiple=0.0,
        symbol=str(order["symbol"]),
        commission=float(candidate.get("commission", 0.0)),
        source_label="parity_fixture_source_replay",
        metadata={"parity_generated_candidate": True},
    )


def _initial_momentum_position_trades(fixture: Mapping[str, Any], trade_cls: Any) -> list[Any]:
    trades = []
    clock = parse_time(fixture["clock_start"])
    for pos in _initial_positions(fixture):
        symbol = str(pos.get("symbol") or pos.get("instrument_symbol") or "")
        qty = abs(float(pos.get("net_qty", pos.get("qty", 0.0)) or 0.0))
        if not symbol or qty <= 0:
            continue
        entry_price = float(pos.get("avg_price", pos.get("entry_price", _last_source_close(fixture, symbol, 1.0))) or 0.0)
        stop = _initial_position_stop(fixture, pos, entry_price)
        direction = 1 if float(pos.get("net_qty", pos.get("qty", 0.0)) or 0.0) > 0 else -1
        trades.append(
            trade_cls(
                strategy_id=str(pos.get("strategy_id", "")),
                direction=direction,
                entry_time=clock - timedelta(minutes=1),
                exit_time=clock + timedelta(days=365),
                entry_price=entry_price,
                exit_price=entry_price,
                initial_stop=stop,
                raw_pnl_dollars=0.0,
                raw_qty=max(int(round(qty)), 1),
                r_multiple=0.0,
                symbol=symbol,
                commission=0.0,
                source_label="initial_repository_state",
                metadata={"initial_position": True},
            )
        )
    return trades


def _momentum_family_config(
    fixture: Mapping[str, Any],
    base_config: Any,
    allocation_cls: Any,
    dynamic_cls: Any,
    candidates: list[Mapping[str, Any]],
    initial_trades: list[Any],
) -> Any:
    equity = float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0))
    family_cfg = fixture.get("family_config", {}) or {}
    rules = portfolio_rules_config_from_fixture(fixture) or base_config.rules
    risk_by_strategy = _candidate_risk_pct_by_strategy(fixture, candidates, equity)
    for trade in initial_trades:
        risk = abs(float(trade.entry_price) - float(trade.initial_stop)) * point_value(fixture, trade.symbol) * max(int(trade.raw_qty), 1)
        risk_by_strategy.setdefault(str(trade.strategy_id), max(risk / max(equity, 1.0), 0.0001))
    max_strategy_active = dict(getattr(rules, "max_strategy_active_positions", ()) or ())
    allocations = []
    for item in family_cfg.get("strategies", []) or []:
        sid = str(item.get("id", ""))
        if not sid:
            continue
        allocations.append(
            allocation_cls(
                sid,
                base_risk_pct=float(risk_by_strategy.get(sid, float(item.get("unit_risk_dollars", 500.0)) / max(equity, 1.0))),
                daily_stop_R=float(item.get("daily_stop_R", 2.5)),
                max_concurrent=int(max_strategy_active.get(sid, 1)),
                priority=int(item.get("priority", 99)),
                max_contracts=max(1, _max_generated_qty(candidates, sid)),
            )
        )
    return replace(
        base_config,
        initial_equity=equity,
        strategy_allocations=tuple(allocations),
        rules=rules,
        heat_cap_R=float(family_cfg.get("heat_cap_R", base_config.heat_cap_R)),
        portfolio_daily_stop_R=float(family_cfg.get("portfolio_daily_stop_R", base_config.portfolio_daily_stop_R)),
        portfolio_weekly_stop_R=float(family_cfg.get("portfolio_weekly_stop_R", base_config.portfolio_weekly_stop_R)),
        reference_unit_risk_dollars=float(getattr(rules, "reference_unit_risk_dollars", 0.0) or base_config.reference_unit_risk_dollars),
        dynamic_risk=dynamic_cls(
            enabled=any(
                bool(getattr(rules, field_name, False))
                for field_name in ("fit_to_remaining_heat", "fit_to_remaining_directional_cap", "fit_to_remaining_family_cap")
            ),
            strategy_multipliers=tuple(getattr(rules, "strategy_size_multipliers", ()) or ()),
            fit_to_remaining_heat=bool(getattr(rules, "fit_to_remaining_heat", False)),
            fit_to_remaining_directional_cap=bool(getattr(rules, "fit_to_remaining_directional_cap", False)),
            fit_to_remaining_family_cap=bool(getattr(rules, "fit_to_remaining_family_cap", False)),
            min_qty=int(getattr(rules, "min_qty", 1) or 1),
            max_trade_risk_R=float(getattr(rules, "max_trade_risk_R", 0.0) or 0.0),
            heat_pressure_threshold=float(getattr(rules, "heat_pressure_threshold", 1.0) or 1.0),
            heat_pressure_mult=float(getattr(rules, "heat_pressure_mult", 1.0) or 1.0),
            same_direction_pressure_threshold=float(getattr(rules, "same_direction_pressure_threshold", 1.0) or 1.0),
            same_direction_pressure_mult=float(getattr(rules, "same_direction_pressure_mult", 1.0) or 1.0),
            existing_position_mult=float(getattr(rules, "existing_position_mult", 1.0) or 1.0),
        ),
    )


def _momentum_family_decisions(
    candidates: list[Mapping[str, Any]],
    *,
    accepted: list[Any],
    blocked: list[Any],
) -> list[dict[str, Any]]:
    generated_ids = {
        (
            str(candidate["order"]["strategy_id"]),
            str(candidate["order"]["symbol"]),
            candidate["entry_time"],
        )
        for candidate in candidates
    }
    accepted_by_key = {
        _trade_key(trade): trade
        for trade in accepted
        if _trade_key(trade) in generated_ids
    }
    blocked_by_key = {
        _trade_key(trade): trade
        for trade in blocked
        if _trade_key(trade) in generated_ids
    }
    decisions = []
    for candidate in candidates:
        key = (
            str(candidate["order"]["strategy_id"]),
            str(candidate["order"]["symbol"]),
            candidate["entry_time"],
        )
        if key in blocked_by_key:
            reason = str(getattr(blocked_by_key[key], "denial_reason", "") or "portfolio_rule")
            decisions.append(_family_decision(candidate, approved_qty=0, status="rejected", reason=_portfolio_reason(reason)))
        elif key in accepted_by_key:
            trade = accepted_by_key[key]
            approved_qty = int(float(getattr(trade, "portfolio_qty", getattr(trade, "raw_qty", candidate["qty"])) or candidate["qty"]))
            decisions.append(_family_decision(candidate, approved_qty=approved_qty, status=_accepted_status(candidate, approved_qty), reason=""))
        else:
            raise AssertionError(f"momentum family replay produced no decision for candidate {candidate['candidate_key']}")
    return decisions


def _trade_key(trade: Any) -> tuple[str, str, Any]:
    return (str(getattr(trade, "strategy_id", "")), str(getattr(trade, "symbol", "")), getattr(trade, "entry_time", None))


run_momentum_family_surface = _run_momentum_family_surface
