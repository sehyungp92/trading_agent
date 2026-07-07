from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from tests.integration.parity.source_inputs import (
    IDLE_MARKET_INPUT_ARTIFACT_KEYS,
    idle_market_input,
    strategy_ids,
)


IdleReplayAdapter = Callable[[Mapping[str, Any], str, Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]


def replay_idle_market_children(fixture: Mapping[str, Any], out: Any) -> None:
    validate_idle_replay_registry()
    for strategy_id in strategy_ids(fixture):
        if strategy_id not in IDLE_MARKET_INPUT_ARTIFACT_KEYS:
            continue
        out.strategy_state[strategy_id] = run_idle_market_core(fixture, strategy_id)


def run_idle_market_core(fixture: Mapping[str, Any], strategy_id: str) -> dict[str, Any]:
    market_input = idle_market_input(fixture, strategy_id)
    initial = (fixture.get("initial_strategy_state", {}) or {}).get(strategy_id, {})
    adapter = IDLE_REPLAY_ADAPTERS.get(strategy_id)
    if adapter is None:
        if strategy_id in IDLE_MARKET_INPUT_ARTIFACT_KEYS:
            raise AssertionError(f"{strategy_id} is missing an idle replay adapter")
        return idle_replay_strategy_state(fixture, strategy_id)
    return adapter(fixture, strategy_id, market_input, _core_snapshot(initial))


def idle_replay_strategy_state(fixture: Mapping[str, Any], strategy_id: str) -> dict[str, Any]:
    last_bar_ts = None
    if strategy_id in IDLE_MARKET_INPUT_ARTIFACT_KEYS:
        last_bar_ts = idle_market_input(fixture, strategy_id)["timestamp"]
    if strategy_id == "ALCB_v1":
        return {
            "positions": {},
            "pending_entries": {},
            "last_decision_code": "IDLE",
            "last_decision_details": {},
            "last_bar_ts": last_bar_ts,
        }
    if strategy_id == "DownturnDominator_v1":
        return {
            "strategy_id": strategy_id,
            "last_decision_code": "IDLE",
            "last_decision_details": {},
            "last_bar_ts": last_bar_ts,
            "position_open": False,
        }
    if strategy_id == "OVERLAY":
        return {}
    return {
        "strategy_id": strategy_id,
        "last_decision_code": "IDLE",
        "last_decision_details": {},
        "last_bar_ts": last_bar_ts,
    }


def _replay_nqdtc_idle(
    _fixture: Mapping[str, Any],
    strategy_id: str,
    market_input: Mapping[str, Any],
    initial: Mapping[str, Any],
) -> dict[str, Any]:
    from strategies.momentum.nqdtc.core.logic import on_bar
    from strategies.momentum.nqdtc.core.serializers import restore_state, snapshot_state

    state = restore_state(initial)
    next_state, actions, _events = on_bar(
        state,
        bar_count_5m=int(getattr(state, "bar_count_5m", 0) or 0) + _idle_timeframe_count(market_input, "5m"),
        bar_ts=market_input["timestamp"],
        idle_market_bars=list(market_input.get("bars", []) or []),
        idle_market_symbol=str(market_input.get("symbol", "")),
        idle_market_timeframe=str(market_input.get("timeframe", "")),
    )
    _assert_no_idle_actions(strategy_id, actions)
    return _compact_idle_market_snapshot(strategy_id, snapshot_state(next_state))


def _replay_vdub_idle(
    _fixture: Mapping[str, Any],
    strategy_id: str,
    market_input: Mapping[str, Any],
    initial: Mapping[str, Any],
) -> dict[str, Any]:
    from strategies.momentum.vdub.core.logic import on_bar
    from strategies.momentum.vdub.core.serializers import restore_state, snapshot_state

    state = restore_state(initial or {})
    state.bar_idx = int(getattr(state, "bar_idx", 0) or 0) + _idle_timeframe_count(market_input, "15m")
    next_state, actions, _events = on_bar(
        state,
        bar_ts=market_input["timestamp"],
        idle_market_bars=list(market_input.get("bars", []) or []),
        idle_market_symbol=str(market_input.get("symbol", "")),
        idle_market_timeframe=str(market_input.get("timeframe", "")),
    )
    _assert_no_idle_actions(strategy_id, actions)
    return _compact_idle_market_snapshot(strategy_id, snapshot_state(next_state))


def _replay_downturn_idle(
    _fixture: Mapping[str, Any],
    strategy_id: str,
    market_input: Mapping[str, Any],
    initial: Mapping[str, Any],
) -> dict[str, Any]:
    from strategies.momentum.downturn.core.logic import on_bar
    from strategies.momentum.downturn.core.serializers import restore_state, snapshot_state

    state = restore_state(initial)
    state.bars_since_last_entry = (
        int(getattr(state, "bars_since_last_entry", 0) or 0)
        + _idle_timeframe_count(market_input, "5m")
    )
    next_state, actions, _events = on_bar(
        state,
        bar_count_5m=int(getattr(state, "bar_count_5m", 0) or 0) + _idle_timeframe_count(market_input, "5m"),
        bar_ts=market_input["timestamp"],
        idle_market_bars=list(market_input.get("bars", []) or []),
        idle_market_symbol=str(market_input.get("symbol", "")),
        idle_market_timeframe=str(market_input.get("timeframe", "")),
    )
    _assert_no_idle_actions(strategy_id, actions)
    return _compact_idle_market_snapshot(strategy_id, snapshot_state(next_state))


def _replay_alcb_idle(
    _fixture: Mapping[str, Any],
    strategy_id: str,
    market_input: Mapping[str, Any],
    initial: Mapping[str, Any],
) -> dict[str, Any]:
    from strategies.stock.alcb.core.logic import on_bar
    from strategies.stock.alcb.core.serializers import restore_state, snapshot_state

    next_state, actions, _events = on_bar(
        restore_state(initial or {}),
        bar_ts=market_input["timestamp"],
        idle_market_bars=list(market_input.get("bars", []) or []),
        idle_market_symbol=str(market_input.get("symbol", "")),
        idle_market_timeframe=str(market_input.get("timeframe", "")),
    )
    _assert_no_idle_actions(strategy_id, actions)
    return _compact_idle_market_snapshot(strategy_id, snapshot_state(next_state))


def _replay_atrss_idle(
    _fixture: Mapping[str, Any],
    strategy_id: str,
    market_input: Mapping[str, Any],
    initial: Mapping[str, Any],
) -> dict[str, Any]:
    from strategies.swing.atrss.core.logic import on_bar
    from strategies.swing.atrss.core.serializers import restore_state, snapshot_state

    next_state, actions, _events = on_bar(
        restore_state(initial or {}),
        bar_ts=market_input["timestamp"],
        idle_market_bars=list(market_input.get("bars", []) or []),
        idle_market_symbol=str(market_input.get("symbol", "")),
        idle_market_timeframe=str(market_input.get("timeframe", "")),
    )
    _assert_no_idle_actions(strategy_id, actions)
    return _compact_idle_market_snapshot(strategy_id, snapshot_state(next_state))


def _replay_akc_helix_idle(
    _fixture: Mapping[str, Any],
    strategy_id: str,
    market_input: Mapping[str, Any],
    initial: Mapping[str, Any],
) -> dict[str, Any]:
    from strategies.swing.akc_helix.core.logic import on_bar
    from strategies.swing.akc_helix.core.serializers import restore_state, snapshot_state

    next_state, actions, _events = on_bar(
        restore_state(initial or {}),
        bar_ts=market_input["timestamp"],
        idle_market_bars=list(market_input.get("bars", []) or []),
        idle_market_symbol=str(market_input.get("symbol", "")),
        idle_market_timeframe=str(market_input.get("timeframe", "")),
    )
    _assert_no_idle_actions(strategy_id, actions)
    return _compact_idle_market_snapshot(strategy_id, snapshot_state(next_state))


IDLE_REPLAY_ADAPTERS: dict[str, IdleReplayAdapter] = {
    "NQDTC_v2.1": _replay_nqdtc_idle,
    "VdubusNQ_v4": _replay_vdub_idle,
    "DownturnDominator_v1": _replay_downturn_idle,
    "ALCB_v1": _replay_alcb_idle,
    "ATRSS": _replay_atrss_idle,
    "AKC_HELIX": _replay_akc_helix_idle,
}


def validate_idle_replay_registry() -> None:
    missing = sorted(set(IDLE_MARKET_INPUT_ARTIFACT_KEYS) - set(IDLE_REPLAY_ADAPTERS))
    if missing:
        raise AssertionError(f"missing idle replay adapter(s): {', '.join(missing)}")


def _core_snapshot(initial: Any) -> Mapping[str, Any]:
    if isinstance(initial, Mapping):
        core = initial.get("core", initial)
        if isinstance(core, Mapping):
            return core
    return {}


def _assert_no_idle_actions(strategy_id: str, actions: list[Any]) -> None:
    if actions:
        raise AssertionError(f"{strategy_id} idle market replay input generated actions: {actions}")


def _idle_timeframe_count(market_input: Mapping[str, Any], timeframe: str) -> int:
    grouped = market_input.get("bars_by_timeframe", {}) or {}
    rows = grouped.get(timeframe) or grouped.get(timeframe.lower()) or market_input.get("bars", [])
    return max(1, len(rows or []))


def _compact_idle_market_snapshot(
    strategy_id: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if strategy_id == "ALCB_v1":
        return {
            "positions": dict(snapshot.get("positions", {}) or {}),
            "pending_entries": dict(snapshot.get("pending_entries", {}) or {}),
            "last_decision_code": str(snapshot.get("last_decision_code", "IDLE")),
            "last_decision_details": dict(snapshot.get("last_decision_details", {}) or {}),
            "last_bar_ts": snapshot.get("last_bar_ts"),
        }
    state = {
        "strategy_id": strategy_id,
        "last_bar_ts": snapshot.get("last_bar_ts"),
        "last_decision_code": str(snapshot.get("last_decision_code", "IDLE")),
        "last_decision_details": dict(snapshot.get("last_decision_details", {}) or {}),
    }
    if strategy_id == "NQDTC_v2.1":
        state["bar_count_5m"] = int(snapshot.get("bar_count_5m", 0) or 0)
        state["working_order_count"] = len(snapshot.get("working_orders", []) or [])
        state["position_open"] = bool((snapshot.get("position", {}) or {}).get("open", False))
    if strategy_id == "VdubusNQ_v4":
        state["bar_idx"] = int(snapshot.get("bar_idx", 0) or 0)
        state["position_count"] = len(snapshot.get("positions", []) or [])
        state["working_entry_count"] = len(snapshot.get("working_entries", {}) or {})
    if strategy_id == "DownturnDominator_v1":
        state["bar_count_5m"] = int(snapshot.get("bar_count_5m", 0) or 0)
        state["bars_since_last_entry"] = int(snapshot.get("bars_since_last_entry", 0) or 0)
        state["working_entry_count"] = len(snapshot.get("working_entries", []) or [])
        state["position_open"] = bool(snapshot.get("position"))
    if strategy_id == "ATRSS":
        state["position_count"] = len(snapshot.get("positions", {}) or {})
        state["pending_order_count"] = len(snapshot.get("pending_orders", {}) or {})
        state["risk_halted"] = bool(snapshot.get("risk_halted", False))
    if strategy_id == "AKC_HELIX":
        state["active_setup_count"] = len(snapshot.get("active_setups", {}) or {})
        state["pending_setup_count"] = len(snapshot.get("pending_setups", {}) or {})
        state["queued_setup_count"] = len(snapshot.get("queued_setups", {}) or {})
        state["risk_halted"] = bool(snapshot.get("risk_halted", False))
    return state
