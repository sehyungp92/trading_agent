from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.integration.parity.replay_candidates import (
    ReplayDecisionTimeline,
    entry_candidate_specs as _entry_candidate_specs,
)
from tests.integration.parity.replay_family_surface_common import (
    _decision_summary,
    _family_decision,
    _initial_positions,
    _portfolio_reason,
    _run_blocking,
)
from tests.integration.parity.source_inputs import overlay_rebalance_payload, point_value


def _run_swing_family_surface(fixture: Mapping[str, Any], out: ReplayDecisionTimeline) -> dict[str, Any]:
    from backtests.swing.config_unified import StrategySlot, UnifiedBacktestConfig
    from backtests.swing.engine.unified_portfolio_engine import (
        SwingFamilyReplayCandidate,
        SwingFamilyReplayExposure,
        replay_swing_family_candidates,
    )
    from strategies.swing.overlay.config import OverlayConfig
    from strategies.swing.overlay.engine import OverlayEngine

    payload = overlay_rebalance_payload(fixture)
    candidates = _entry_candidate_specs(fixture, out)
    config = _swing_unified_config(fixture, payload, StrategySlot, UnifiedBacktestConfig)
    replay_candidates = [
        SwingFamilyReplayCandidate(
            candidate_key=str(candidate["candidate_key"]),
            strategy_id=str(candidate["order"]["strategy_id"]),
            symbol=str(candidate["order"]["symbol"]),
            direction="LONG" if str(candidate["order"]["side"]).upper() == "BUY" else "SHORT",
            risk_dollars=_candidate_risk_dollars(fixture, candidate),
            qty=int(candidate["qty"]),
            entry_time=candidate.get("entry_time"),
        )
        for candidate in candidates
    ]
    replay_decisions = _run_blocking(
        lambda: replay_swing_family_candidates(
            config,
            replay_candidates,
            initial_exposures=_swing_initial_exposures(
                fixture,
                SwingFamilyReplayExposure,
                initial_equity=float(config.initial_equity),
            ),
            initial_equity=float(config.initial_equity),
            current_equity=float(config.initial_equity),
        )
    )
    decisions = _swing_family_decisions(candidates, replay_decisions)
    if not payload.get("rebalance_due") or not payload.get("symbols"):
        return {
            "adapter": "swing_unified_overlay_replay_adapter",
            "overlay": {},
            **_decision_summary(decisions, family_surface="swing_unified_overlay_replay_adapter"),
        }
    symbols = [str(symbol) for symbol in payload["symbols"]]
    weights = {str(symbol): float(weight) for symbol, weight in (payload.get("target_weights") or {}).items()}
    ema_overrides = {
        str(symbol): tuple(int(part) for part in periods)
        for symbol, periods in (payload.get("ema_overrides") or {}).items()
    }
    config.overlay_symbols = symbols
    config.overlay_ema_overrides = ema_overrides or {symbol: (10, 21) for symbol in symbols}
    config.overlay_max_pct = float(payload.get("max_equity_pct", 0.85))
    config.overlay_weights = weights or None
    overlay_config = OverlayConfig(
        enabled=True,
        symbols=symbols,
        max_equity_pct=config.overlay_max_pct,
        ema_overrides=config.overlay_ema_overrides,
        weights=config.overlay_weights,
    )
    overlay_engine = OverlayEngine(
        ib_session=None,
        equity=float(payload.get("equity", 0.0) or 0.0),
        config=overlay_config,
        get_deployed_capital=lambda: _swing_deployed_capital_from_initial_positions(fixture),
        disable_scheduler=True,
    )
    overlay_engine._shares.update(
        {
            str(symbol): int(qty)
            for symbol, qty in (payload.get("starting_holdings") or {}).items()
        }
    )
    plan = overlay_engine.build_rebalance_plan_from_bars(
        payload.get("daily_bars", {}) or {},
        equity=float(payload.get("equity", 0.0) or 0.0),
        min_bars=0,
    )
    overlay_state = overlay_engine.apply_rebalance_plan_dry_run(
        plan,
        timestamp=str(payload.get("timestamp", "")),
        reason=str(payload.get("rebalance_reason", "fixture")),
    )
    return {
        "adapter": "swing_unified_overlay_replay_adapter",
        **_decision_summary(decisions, family_surface="swing_unified_overlay_replay_adapter"),
        "overlay": {
            "positions": dict(overlay_state.get("positions", {})),
            "signals": dict(overlay_state.get("signals", {})),
            "last_rebalance_date": str(overlay_state.get("last_rebalance_date", "")),
            "last_decision_code": str(overlay_state.get("last_decision_code", "")),
            "rebalances_completed": int(overlay_state.get("rebalances_completed", 0) or 0),
        },
    }


def _swing_unified_config(
    fixture: Mapping[str, Any],
    payload: Mapping[str, Any],
    slot_cls: Any,
    config_cls: Any,
) -> Any:
    account_equity = float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0))
    family_cfg = fixture.get("family_config", {}) or {}
    rules = family_cfg.get("portfolio_rules", {}) or {}
    configured = {
        str(item.get("id")): item
        for item in family_cfg.get("strategies", []) or []
        if item.get("id")
    }

    def _slot(strategy_id: str, default_priority: int, default_pct: float, default_heat: float, default_stop: float, default_working: int) -> Any:
        row = configured.get(strategy_id, {})
        unit_risk = float(row.get("unit_risk_dollars", account_equity * default_pct) or (account_equity * default_pct))
        return slot_cls(
            strategy_id=strategy_id,
            priority=int(row.get("priority", default_priority)),
            unit_risk_pct=unit_risk / account_equity if account_equity > 0 else default_pct,
            max_heat_R=float(row.get("max_heat_R", default_heat)),
            daily_stop_R=float(row.get("daily_stop_R", default_stop)),
            max_working_orders=int(row.get("max_working_orders", default_working)),
        )

    dd_tiers = tuple(
        (float(row[0]), float(row[1]))
        for row in (rules.get("dd_tiers") or ((0.04, 0.90), (0.07, 0.70), (0.10, 0.50), (0.14, 0.25), (0.18, 0.00)))
    )
    return config_cls(
        initial_equity=account_equity,
        atrss_symbols=[],
        helix_symbols=[],
        tpc_symbols=[],
        heat_cap_R=float(rules.get("directional_cap_R", family_cfg.get("heat_cap_R", 5.5))),
        portfolio_daily_stop_R=float(family_cfg.get("portfolio_daily_stop_R", 3.75)),
        portfolio_constraints_enabled=True,
        dynamic_risk_enabled=False,
        drawdown_risk_tiers=dd_tiers,
        atrss=_slot("ATRSS", 0, 0.0165, 2.15, 2.25, 4),
        helix=_slot("AKC_HELIX", 2, 0.013, 2.10, 2.5, 4),
        tpc=_slot("TPC", 1, 0.005, 4.0, 2.0, 3),
        overlay_enabled=True,
        overlay_symbols=[str(symbol) for symbol in payload.get("symbols", [])],
        overlay_ema_overrides={
            str(symbol): tuple(int(part) for part in periods)
            for symbol, periods in (payload.get("ema_overrides") or {}).items()
        },
        overlay_max_pct=float(payload.get("max_equity_pct", 0.85)),
        overlay_weights=dict(payload.get("target_weights") or {}) or None,
    )


def _swing_family_decisions(
    candidates: list[Mapping[str, Any]],
    replay_decisions: list[Any],
) -> list[dict[str, Any]]:
    by_key = {str(decision.candidate_key): decision for decision in replay_decisions}
    decisions = []
    for candidate in candidates:
        key = str(candidate["candidate_key"])
        decision = by_key.get(key)
        if decision is None:
            raise AssertionError(f"swing family replay produced no decision for candidate {key}")
        status = str(decision.status).lower()
        reason = _portfolio_reason(str(decision.reason)) if status == "rejected" else ""
        decisions.append(
            _family_decision(
                candidate,
                approved_qty=int(decision.approved_qty),
                status=status,
                reason=reason,
            )
        )
    return decisions


def _candidate_risk_dollars(
    fixture: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> float:
    action = candidate.get("action")
    raw = getattr(action, "risk_context", {}) or {}
    if raw.get("risk_dollars") not in (None, ""):
        return float(raw["risk_dollars"])
    qty = int(float(candidate.get("qty", 0) or 0))
    entry_price = float(candidate.get("entry_price", 0.0) or 0.0)
    stop_price = float(candidate.get("stop_price", entry_price) or entry_price)
    symbol = str(candidate.get("order", {}).get("symbol", ""))
    return qty * abs(entry_price - stop_price) * point_value(fixture, symbol)


def _swing_initial_exposures(
    fixture: Mapping[str, Any],
    exposure_cls: Any,
    *,
    initial_equity: float,
) -> list[Any]:
    exposures: list[Any] = []
    for pos in _initial_positions(fixture):
        strategy_id = str(pos.get("strategy_id", ""))
        if strategy_id not in {"ATRSS", "AKC_HELIX", "TPC"}:
            continue
        qty = abs(int(float(pos.get("net_qty", pos.get("qty", 0.0)) or 0.0)))
        risk_dollars = float(pos.get("open_risk_dollars", 0.0) or 0.0)
        if qty <= 0 or risk_dollars <= 0.0:
            continue
        unit = _fixture_unit_risk_dollars(fixture, strategy_id, initial_equity)
        risk_R = float(pos.get("open_risk_R", 0.0) or 0.0)
        if risk_R <= 0.0 and unit > 0.0:
            risk_R = risk_dollars / unit
        direction = "LONG" if float(pos.get("net_qty", pos.get("qty", 0.0)) or 0.0) > 0 else "SHORT"
        exposures.append(
            exposure_cls(
                strategy_id=strategy_id,
                symbol=str(pos.get("symbol") or pos.get("instrument_symbol") or ""),
                direction=direction,
                risk_dollars=risk_dollars,
                risk_R=risk_R,
                qty=qty,
            )
        )
    return exposures


def _swing_deployed_capital_from_initial_positions(fixture: Mapping[str, Any]) -> float:
    equity = float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0))
    deployed = 0.0
    for pos in _initial_positions(fixture):
        strategy_id = str(pos.get("strategy_id", ""))
        if strategy_id not in {"ATRSS", "AKC_HELIX", "TPC"}:
            continue
        risk_dollars = float(pos.get("open_risk_dollars", 0.0) or 0.0)
        unit = _fixture_unit_risk_dollars(fixture, strategy_id, equity)
        risk_pct = unit / equity if equity > 0 else 0.0
        if risk_dollars > 0.0 and risk_pct > 0.0:
            deployed += risk_dollars / risk_pct
    return deployed


def _fixture_unit_risk_dollars(
    fixture: Mapping[str, Any],
    strategy_id: str,
    initial_equity: float,
) -> float:
    for item in (fixture.get("family_config", {}) or {}).get("strategies", []) or []:
        if str(item.get("id", "")) == strategy_id:
            value = float(item.get("unit_risk_dollars", 0.0) or 0.0)
            if value > 0.0:
                return value
    defaults = {"ATRSS": 0.0165, "AKC_HELIX": 0.013, "TPC": 0.005}
    return float(initial_equity) * defaults.get(strategy_id, 0.005)


run_swing_family_surface = _run_swing_family_surface
