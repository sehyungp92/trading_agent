from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from libs.oms.models.instrument import Instrument
from tests.integration.parity.live_oms import settle_callbacks as _settle_callbacks
from tests.integration.parity.source_inputs import (
    iaric_artifact,
    iaric_minute_bars,
    iaric_quote,
    iaric_state_snapshot,
    nq_bar_data,
    nq_daily_context,
    nq_live_context,
    overlay_rebalance_payload as _overlay_rebalance_payload,
    parse_time,
    plain,
    source_bars,
    strategy_ids,
    tpc_bar_input,
    tpc_symbol_config,
)


def _instantiate_live_engines(
    fixture: Mapping[str, Any],
    oms: Any,
    instruments: Mapping[str, Instrument],
    instrumentation_dir: str,
) -> dict[str, Any]:
    configured_ids = set(strategy_ids(fixture))
    surface = str(fixture.get("surface", "")).upper()
    clock = lambda: parse_time(fixture["clock_start"])
    account = fixture.get("account_state", {}) or {}
    engines: dict[str, Any] = {}
    if "TPC" in configured_ids or surface == "TPC":
        from strategies.swing.tpc.engine import TPCEngine

        tpc_symbols = {str(row["symbol"]) for row in fixture.get("bars", []) if str(row.get("timeframe", "")).lower() == "15m"}
        if not tpc_symbols:
            tpc_symbols = {"QQQ"}
        engines["TPC"] = TPCEngine(
            ib_session=None,
            oms_service=oms,
            instruments={symbol: instruments[symbol] for symbol in tpc_symbols if symbol in instruments},
            config={symbol: tpc_symbol_config(fixture, symbol) for symbol in tpc_symbols},
            equity=float(account.get("equity", 100_000.0)),
            state_dir=instrumentation_dir,
            bar_input_provider=lambda symbol, _request_kind: tpc_bar_input(fixture, symbol),
            disable_scheduler=True,
            clock=clock,
        )
    if "NQ_REGIME" in configured_ids or surface == "NQ_REGIME":
        from strategies.momentum.nq_regime.config import StrategyRuntimeSettings
        from strategies.momentum.nq_regime.engine import NQRegimeEngine

        nq_overrides = dict((fixture.get("strategy_config", {}) or {}).get("config_overrides", {}) or {})
        engines["NQ_REGIME"] = NQRegimeEngine(
            ib_session=None,
            oms_service=oms,
            instruments=dict(instruments),
            equity=float(account.get("equity", 100_000.0)),
            instrumentation=None,
            state_dir=instrumentation_dir,
            disable_scheduler=True,
            clock=clock,
        )
        engines["NQ_REGIME"]._settings = StrategyRuntimeSettings(
            initial_equity=float(account.get("equity", 100_000.0)),
            max_contracts=int(nq_overrides.get("max_contracts", 5)),
            enable_liquidity_reversion=False,
            enable_second_wind=False,
        )
    if "IARIC_v1" in configured_ids or surface == "IARIC":
        from strategies.stock.iaric.config import StrategySettings
        from strategies.stock.iaric.diagnostics import JsonlDiagnostics
        from strategies.stock.iaric.engine import IARICEngine

        artifact = iaric_artifact(fixture)
        config_overrides = dict((fixture.get("strategy_config", {}) or {}).get("config_overrides", {}) or {})
        engines["IARIC_v1"] = IARICEngine(
            oms_service=oms,
            artifact=artifact,
            account_id=str(account.get("account_id", "ACCT-PARITY")),
            nav=float(account.get("equity", 100_000.0)),
            settings=StrategySettings(
                diagnostics_dir=instrumentation_dir,
                state_dir=instrumentation_dir,
                base_risk_fraction=float(config_overrides.get("base_risk_fraction", StrategySettings().base_risk_fraction)),
            ),
            diagnostics=JsonlDiagnostics(Path(instrumentation_dir), enabled=False),
            disable_background_tasks=True,
        )
    return engines


async def _start_engines(engines: Mapping[str, Any]) -> None:
    for engine in engines.values():
        start = getattr(engine, "start", None)
        if start is not None:
            await start()


async def _hydrate_live_state(fixture: Mapping[str, Any], engines: Mapping[str, Any]) -> None:
    initial = fixture.get("initial_strategy_state", {}) or {}
    if "TPC" in engines:
        from strategies.swing.tpc.core.serializers import restore_state

        engines["TPC"]._state = restore_state(initial.get("TPC", {}))
        tpc_symbols = {str(row["symbol"]) for row in fixture.get("bars", []) if str(row.get("timeframe", "")).lower() == "15m"}
        if tpc_symbols:
            engines["TPC"]._bar_input_provider = lambda symbol, _request_kind: tpc_bar_input(fixture, symbol)
            for symbol in tpc_symbols:
                engines["TPC"]._config[symbol] = tpc_symbol_config(fixture, symbol)
    if "NQ_REGIME" in engines:
        from strategies.momentum.nq_regime.config import StrategyRuntimeSettings

        await engines["NQ_REGIME"].hydrate(initial.get("NQ_REGIME", {}))
        overrides = dict((fixture.get("strategy_config", {}) or {}).get("config_overrides", {}) or {})
        engines["NQ_REGIME"]._settings = StrategyRuntimeSettings(
            initial_equity=float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0)),
            max_contracts=int(overrides.get("max_contracts", 5)),
            enable_liquidity_reversion=False,
            enable_second_wind=False,
        )
    if "IARIC_v1" in engines:
        engines["IARIC_v1"].hydrate_state(iaric_state_snapshot(fixture, "IARIC_v1"))
    if "OVERLAY" in engines:
        payload = _overlay_rebalance_payload(fixture)
        shares = {
            str(symbol): int(qty)
            for symbol, qty in (payload.get("starting_holdings") or {}).items()
        }
        if shares:
            engines["OVERLAY"]._shares.update(shares)


def _compact_overlay_state(engine: Any) -> dict[str, Any]:
    if engine is None:
        return {}
    positions = engine.get_positions() if hasattr(engine, "get_positions") else getattr(engine, "_shares", {})
    signals = engine.get_signals() if hasattr(engine, "get_signals") else getattr(engine, "_last_signals", {})
    return {
        "positions": {str(symbol): int(qty) for symbol, qty in sorted((positions or {}).items())},
        "signals": {str(symbol): bool(value) for symbol, value in sorted((signals or {}).items())},
        "last_rebalance_date": getattr(engine, "_last_rebalance_date", "") or "",
        "last_decision_code": getattr(engine, "_last_decision_code", ""),
        "rebalances_completed": int(getattr(engine, "_rebalances_completed", 0) or 0),
    }


def _compact_engine_state(engine: Any, strategy_id_hint: str = "") -> dict[str, Any]:
    strategy_id = str(getattr(engine, "strategy_id", "") or strategy_id_hint or "")
    if strategy_id == "OVERLAY" or type(engine).__name__ == "OverlayEngine":
        return _compact_overlay_state(engine)
    if strategy_id == "NQ_REGIME":
        core = getattr(engine, "_state", None)
        return {
            "position_side": str(getattr(getattr(core, "position_side", None), "value", getattr(core, "position_side", ""))),
            "entry_price": getattr(core, "entry_price", 0.0),
            "stop_price": getattr(core, "stop_price", 0.0),
            "qty_open": getattr(core, "qty_open", 0),
            "daily_trades": getattr(core, "daily_trades", 0),
            "last_decision_code": getattr(core, "last_decision_code", ""),
        }
    if strategy_id == "TPC":
        core = getattr(engine, "_state", None)
        return {
            "setups": sorted(getattr(core, "setups", {}).keys()),
            "positions": sorted(getattr(core, "positions", {}).keys()),
            "pending_count": len(getattr(core, "pending_orders", {}) or {}),
        }
    if strategy_id in {"ATRSS", "AKC_HELIX"}:
        state = {
            "strategy_id": strategy_id,
            "last_bar_ts": getattr(engine, "_last_bar_ts", None),
            "last_decision_code": getattr(engine, "_last_decision_code", "IDLE"),
            "last_decision_details": dict(getattr(engine, "_last_decision_details", {}) or {}),
        }
        if strategy_id == "ATRSS":
            state.update(
                {
                    "position_count": len(getattr(engine, "positions", {}) or {}),
                    "pending_order_count": len(getattr(engine, "pending_orders", {}) or {}),
                    "risk_halted": bool(getattr(engine, "_risk_halted", False)),
                }
            )
        if strategy_id == "AKC_HELIX":
            state.update(
                {
                    "active_setup_count": len(getattr(engine, "active_setups", {}) or {}),
                    "pending_setup_count": len(getattr(engine, "pending_setups", {}) or {}),
                    "queued_setup_count": len(getattr(engine, "queued_setups", {}) or {}),
                    "risk_halted": bool(getattr(engine, "_risk_halted", False)),
                }
            )
        return state
    if strategy_id in {"NQDTC_v2.1", "VdubusNQ_v4", "DownturnDominator_v1"}:
        state = {
            "strategy_id": strategy_id,
            "last_bar_ts": getattr(engine, "_last_bar_ts", None),
            "last_decision_code": getattr(engine, "_last_decision_code", "IDLE"),
            "last_decision_details": dict(getattr(engine, "_last_decision_details", {}) or {}),
        }
        if strategy_id == "NQDTC_v2.1":
            position = getattr(engine, "_position", None)
            state["bar_count_5m"] = int(getattr(engine, "_bar_count_5m", 0) or 0)
            state["working_order_count"] = len(getattr(engine, "_working_orders", []) or [])
            state["position_open"] = bool(getattr(position, "open", False))
        if strategy_id == "VdubusNQ_v4":
            state["bar_idx"] = int(getattr(engine, "_bar_idx", 0) or 0)
            state["position_count"] = len(getattr(engine, "positions", []) or [])
            state["working_entry_count"] = len(getattr(engine, "working_entries", {}) or {})
        if strategy_id == "DownturnDominator_v1":
            state["bar_count_5m"] = int(getattr(engine, "_bar_count_5m", 0) or 0)
            state["bars_since_last_entry"] = int(getattr(engine, "_bars_since_last_entry", 0) or 0)
            state["working_entry_count"] = len(getattr(engine, "_working_entries", []) or [])
            state["position_open"] = bool(getattr(engine, "_position_open", False))
        return state
    if strategy_id == "IARIC_v1" or type(engine).__name__.startswith("IARIC"):
        symbols = {}
        for symbol, state in sorted(getattr(engine, "_symbols", {}).items()):
            position = getattr(state, "position", None)
            symbols[symbol] = {
                "stage": getattr(state, "stage", ""),
                "route_family": getattr(state, "route_family", ""),
                "in_position": bool(getattr(state, "in_position", False)),
                "risk_per_share": getattr(state, "risk_per_share", 0.0),
                "position": (
                    {
                        "qty_open": getattr(position, "qty_open", 0),
                        "entry_price": getattr(position, "entry_price", 0.0),
                        "current_stop": getattr(position, "current_stop", 0.0),
                    }
                    if position is not None
                    else None
                ),
            }
        return {"symbols": symbols, "last_decision_code": getattr(engine, "_last_decision_code", "")}
    if strategy_id == "ALCB_v1":
        snapshot = getattr(engine, "snapshot_state", None)
        snap = plain(snapshot()) if callable(snapshot) else {}
        return {
            "positions": snap.get("positions", {}),
            "pending_entries": snap.get("pending_entries", {}),
            "last_decision_code": snap.get("last_decision_code", "IDLE"),
            "last_decision_details": snap.get("last_decision_details", {}),
            "last_bar_ts": snap.get("last_bar_ts"),
        }
    snapshot = getattr(engine, "snapshot_state", None)
    if callable(snapshot):
        return plain(snapshot())
    health = getattr(engine, "health_status", None)
    if callable(health):
        raw = plain(health())
        return {
            key: raw[key]
            for key in (
                "strategy_id",
                "last_decision_code",
                "last_decision_details",
                "last_bar_ts",
                "position_open",
            )
            if key in raw
        }
    return {}



async def drive_layer2_live_inputs(fixture: Mapping[str, Any], engines: Mapping[str, Any]) -> None:
    if "TPC" in engines:
        await engines["TPC"]._cycle_once(request_kind="fixture")
    if "NQ_REGIME" in engines:
        for row in source_bars(fixture, "NQ", "5m") or source_bars(fixture, "MNQ", "5m"):
            await engines["NQ_REGIME"].on_bar(
                nq_bar_data(row),
                daily_context=nq_daily_context(fixture),
                live_context=nq_live_context(fixture),
                scheduled_news=[],
            )
    if "IARIC_v1" in engines:
        engine = engines["IARIC_v1"]
        symbols = sorted({str(row.get("symbol", "")).upper() for row in fixture.get("bars", []) if row.get("symbol")})
        for symbol in symbols:
            engine.on_quote(symbol, iaric_quote(fixture, symbol))
            for bar in iaric_minute_bars(fixture, symbol):
                engine.on_bar(symbol, bar)
                await _settle_callbacks()


instantiate_live_engines = _instantiate_live_engines
start_engines = _start_engines
hydrate_live_state = _hydrate_live_state
compact_engine_state = _compact_engine_state
compact_overlay_state = _compact_overlay_state
