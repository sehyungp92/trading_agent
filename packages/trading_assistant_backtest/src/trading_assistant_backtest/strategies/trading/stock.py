"""Stock-family bridge backed by the production ``trading`` live-shadow harness."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from trading_assistant_backtest.contract_models import DecisionParityReport, MonthlyRunManifest
from trading_assistant_backtest.replay.decision_trace import DecisionTraceEvent
from trading_assistant_backtest.replay.parity import decision_parity_report_from_traces
from trading_assistant_backtest.strategies.live_clone import validate_pinned_head

PLUGIN_ID = "trading-stock-family"
DECISION_API_VERSION = "trading_stock_live_shadow_decision_api_v1"


def build_trading_stock_decision_parity_report(
    manifest: MonthlyRunManifest,
    *,
    candidate_id: str,
    fixture_paths: Iterable[str | Path],
    live_repo_path: str | Path,
    live_repo_commit_sha: str = "",
    backtest_adapter_commit_sha: str = "",
) -> DecisionParityReport:
    """Build a parity report from the trading repo's stock live-shadow fixtures."""

    repo_path = Path(live_repo_path)
    if live_repo_commit_sha:
        checkout_errors = validate_pinned_head(repo_path, live_repo_commit_sha)
        if checkout_errors:
            raise ValueError("; ".join(checkout_errors))

    live_events: list[DecisionTraceEvent] = []
    adapter_events: list[DecisionTraceEvent] = []
    evidence_paths: list[str] = []
    with tempfile.TemporaryDirectory(prefix="trading-stock-parity-") as scratch:
        runtime_repo_path = _runtime_repo_copy(repo_path, Path(scratch) / "trading")
        with _trading_import_path(runtime_repo_path):
            for fixture_path in fixture_paths:
                path = Path(fixture_path)
                fixture = json.loads(path.read_text(encoding="utf-8"))
                evidence_paths.append(str(path))
                live, adapter = _decision_traces_from_fixture(fixture, path)
                live_events.extend(live)
                adapter_events.extend(adapter)

    return decision_parity_report_from_traces(
        manifest,
        candidate_id=candidate_id,
        live_events=live_events,
        adapter_events=adapter_events,
        evidence_paths=evidence_paths,
        live_repo_commit_sha=live_repo_commit_sha,
        backtest_adapter_commit_sha=backtest_adapter_commit_sha,
    )


def _decision_traces_from_fixture(
    fixture: Mapping[str, Any],
    path: Path,
) -> tuple[list[DecisionTraceEvent], list[DecisionTraceEvent]]:
    from tests.integration.parity.harness import run_layer2_contract, run_layer3_family_contract

    surface = str(fixture.get("surface") or "")
    if surface == "IARIC":
        contract = asyncio.run(run_layer2_contract("IARIC", path))
        return (
            _trace_events(contract.live, fixture, surface="stock:IARIC_v1"),
            _trace_events(contract.replay, fixture, surface="stock:IARIC_v1"),
        )
    if surface == "stock_family":
        family = asyncio.run(run_layer3_family_contract("stock", path))
        live_events: list[DecisionTraceEvent] = []
        adapter_events: list[DecisionTraceEvent] = []
        for child in family.children:
            live_events.extend(_trace_events(child.live, fixture, surface=child.surface))
            adapter_events.extend(_trace_events(child.replay, fixture, surface=child.surface))
        live_events.append(_family_event(family.live_family_state, fixture, surface="stock:family"))
        adapter_events.append(
            _family_event(family.replay_family_state, fixture, surface="stock:family")
        )
        return live_events, adapter_events
    raise ValueError(f"unsupported trading stock parity surface: {surface!r}")


def _trace_events(
    trace: Any,
    fixture: Mapping[str, Any],
    *,
    surface: str,
) -> list[DecisionTraceEvent]:
    key = f"{surface}:{fixture.get('surface', 'fixture')}:{trace.source_fingerprint}"
    ts = _fixture_ts(fixture)
    state = trace.state_snapshot if isinstance(trace.state_snapshot, Mapping) else {}
    strategy_state = state.get("strategy_state") if isinstance(state, Mapping) else {}
    family_state = state.get("family_state") if isinstance(state, Mapping) else {}
    positions = list(state.get("positions") or []) if isinstance(state, Mapping) else []
    entry_orders = [row for row in trace.order_intents if row.get("order_role") == "ENTRY"]
    stop_orders = [row for row in trace.order_intents if row.get("order_role") == "STOP"]
    return [
        _event(
            ts,
            "signals",
            key,
            {
                "last_decision_code": _state_value(strategy_state, "last_decision_code"),
                "order_intent_count": len(trace.order_intents),
                "source_fingerprint": trace.source_fingerprint,
                "surface": surface,
                "trade_count": len(trace.trade_ledger),
            },
        ),
        _event(
            ts,
            "filters",
            key,
            {
                "blocked_reasons": state.get("blocked_reasons", {}),
                "family": fixture.get("family", ""),
                "portfolio_surface": _family_section(family_state, "portfolio_surface"),
                "state_hydrated": bool(state),
            },
        ),
        _event(ts, "entries", key, {"orders": entry_orders or [{"action": "no_entry"}]}),
        _event(
            ts,
            "exits",
            key,
            {
                "terminal_events": trace.terminal_events,
                "trade_ledger": _exit_projection(trace.trade_ledger),
            },
        ),
        _event(
            ts,
            "stops",
            key,
            {
                "orders": stop_orders or [{"action": "no_stop_order"}],
                "position_stops": _position_stops(strategy_state),
            },
        ),
        _event(
            ts,
            "sizing",
            key,
            {
                "entry_quantities": [row.get("qty", 0) for row in entry_orders],
                "positions": _position_sizes(positions),
            },
        ),
        _event(
            ts,
            "risk_caps",
            key,
            {
                "portfolio_risk": state.get("portfolio_risk", []),
                "portfolio_rules": state.get("portfolio_rules", []),
                "risk_exposure": _family_section(family_state, "risk_exposure"),
                "strategy_risk": state.get("strategy_risk", []),
            },
        ),
        _event(
            ts,
            "order_intent",
            key,
            {"orders": trace.order_intents or [{"action": "no_order"}]},
        ),
    ]


def _family_event(
    family_state: Mapping[str, Any] | None,
    fixture: Mapping[str, Any],
    *,
    surface: str,
) -> DecisionTraceEvent:
    state = dict(family_state or {})
    return _event(
        _fixture_ts(fixture),
        "risk_caps",
        f"{surface}:{fixture.get('surface', 'fixture')}",
        {
            "configured_strategy_ids": state.get("configured_strategy_ids", []),
            "portfolio_surface": state.get("portfolio_surface", {}),
            "risk_exposure": state.get("risk_exposure", {}),
            "risk_state": state.get("risk_state", {}),
        },
    )


def _exit_projection(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected = []
    for row in rows:
        projected.append(
            {
                "strategy_id": row.get("strategy_id", ""),
                "symbol": row.get("symbol", ""),
                "exit_time": row.get("exit_time"),
                "exit_price": row.get("exit_price"),
                "exit_reason": row.get("exit_reason", ""),
                "net_pnl": row.get("net_pnl", 0),
            }
        )
    return projected or [{"action": "no_closed_trade"}]


def _position_stops(strategy_state: Any) -> list[dict[str, Any]]:
    symbols = strategy_state.get("symbols", {}) if isinstance(strategy_state, Mapping) else {}
    rows = []
    for symbol, state in dict(symbols or {}).items():
        position = state.get("position", {}) if isinstance(state, Mapping) else {}
        if isinstance(position, Mapping):
            rows.append(
                {
                    "symbol": symbol,
                    "current_stop": position.get("current_stop"),
                    "risk_per_share": position.get("risk_per_share"),
                }
            )
    return rows


def _position_sizes(positions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "strategy_id": row.get("strategy_id", ""),
            "symbol": row.get("symbol", ""),
            "net_qty": row.get("net_qty", row.get("qty", 0)),
            "avg_price": row.get("avg_price", row.get("entry_price", 0)),
        }
        for row in positions
    ]


def _state_value(state: Any, key: str) -> Any:
    return state.get(key, "") if isinstance(state, Mapping) else ""


def _family_section(state: Any, key: str) -> Any:
    return state.get(key, {}) if isinstance(state, Mapping) else {}


def _runtime_repo_copy(source: Path, destination: Path) -> Path:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache"),
    )
    return destination


@contextmanager
def _trading_import_path(repo_path: Path) -> Iterator[None]:
    text = str(repo_path.resolve())
    already_present = text in sys.path
    if not already_present:
        sys.path.insert(0, text)
    prior_modules = _pop_modules(
        (
            "backtests",
            "config",
            "ib_insync",
            "libs",
            "src",
            "strategies",
            "tests",
            "tests.integration",
            "utils",
        )
    )
    try:
        yield
    finally:
        _pop_modules(
        (
            "backtests",
            "config",
            "ib_insync",
            "libs",
            "src",
            "strategies",
            "tests",
                "tests.integration",
                "utils",
            )
        )
        sys.modules.update(prior_modules)
        if not already_present:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


def _pop_modules(prefixes: tuple[str, ...]) -> dict[str, Any]:
    modules = {
        name: module
        for name, module in list(sys.modules.items())
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    for name in modules:
        sys.modules.pop(name, None)
    return modules


def _fixture_ts(fixture: Mapping[str, Any]) -> datetime:
    raw = fixture.get("clock_start") or "2026-05-20T14:30:00+00:00"
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _event(
    ts: datetime,
    dimension: str,
    key: str,
    payload: Mapping[str, Any],
) -> DecisionTraceEvent:
    return DecisionTraceEvent(ts=ts, dimension=dimension, key=key, payload=_clean(payload))


def _clean(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in sorted(value.items()) if item is not None}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    return value

