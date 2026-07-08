from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from libs.oms.models.instrument import Instrument
from libs.oms.services.factory import build_oms_service
from tests.integration.parity.fake_ibkr import FakeIBKRExecutionAdapter
from tests.integration.parity.broker_matching import order_matches as _order_matches
from tests.integration.parity.oms_state import (
    blocked_reasons_from_repo_events as _blocked_reasons_from_repo_events,
    drain_queue as _drain_queue,
    ledger_from_repo as _ledger_from_repo,
    plain_dataclass as _plain_dataclass,
    portfolio_rules_state as _portfolio_rules_state,
)
from tests.integration.parity.oms_hydration import (
    build_instruments_from_fixture,
    hydrate_repository_from_fixture,
)
from tests.integration.parity.source_inputs import parse_time


async def _build_single_oms(
    fixture: Mapping[str, Any],
    adapter: FakeIBKRExecutionAdapter,
    instrumentation_dir: str,
    *,
    event_clock,
) -> Any:
    strategy_cfg = fixture.get("strategy_config", {}) or {}
    strategy_id = str(strategy_cfg.get("strategy_id") or fixture.get("surface"))
    account = fixture.get("account_state", {}) or {}
    return await build_oms_service(
        adapter=adapter,
        strategy_id=strategy_id,
        unit_risk_dollars=float(strategy_cfg.get("unit_risk_dollars", 1_000.0)),
        daily_stop_R=float(strategy_cfg.get("daily_stop_R", 10.0)),
        heat_cap_R=float(strategy_cfg.get("heat_cap_R", 20.0)),
        portfolio_daily_stop_R=float(strategy_cfg.get("portfolio_daily_stop_R", 20.0)),
        portfolio_weekly_stop_R=float(strategy_cfg.get("portfolio_weekly_stop_R", 50.0)),
        db_pool=None,
        family_id=str(strategy_cfg.get("family", fixture.get("family", "unknown"))),
        get_current_equity=lambda: float(account.get("equity", 100_000.0)),
        recon_interval_s=3600.0,
        instrumentation_data_dir=instrumentation_dir,
        event_clock=event_clock,
    )


async def _apply_broker_script_to_repos(
    fixture: Mapping[str, Any],
    adapters: list[FakeIBKRExecutionAdapter],
    repos: list[Any],
) -> None:
    for event_spec in fixture.get("broker_event_script", []):
        candidates: list[tuple[FakeIBKRExecutionAdapter, Any, Mapping[str, Any]]] = []
        for adapter in adapters:
            for repo in repos:
                submitted = await _submitted_for_match(adapter, repo, event_spec["order_match"])
                if submitted is not None:
                    candidates.append((adapter, repo, submitted))
        if len(candidates) != 1:
            raise AssertionError(
                f"broker event expected exactly one submitted live order match, got {len(candidates)}: "
                f"{event_spec['order_match']}"
            )
        adapter, repo, submitted = candidates[0]
        await _emit_broker_event(adapter, repo, submitted, event_spec)


async def _get_order_any(repos: list[Any], oms_order_id: str) -> Any:
    for repo in repos:
        order = await repo.get_order(oms_order_id)
        if order is not None:
            return order
    return None


async def _submitted_for_match(
    adapter: FakeIBKRExecutionAdapter,
    repo: Any,
    match: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    sequence = int(match.get("sequence", 1))
    matches: list[Mapping[str, Any]] = []
    for item in adapter.submitted:
        order = await repo.get_order(item["oms_order_id"])
        if order is None or not _order_matches(order, match):
            continue
        matches.append(item)
    if len(matches) < sequence:
        return None
    if len(matches) > sequence and "sequence" not in match:
        raise AssertionError(f"broker event matched multiple submitted live orders: {match}")
    return matches[sequence - 1]




async def _emit_broker_event(
    adapter: FakeIBKRExecutionAdapter,
    repo: Any,
    submitted: Mapping[str, Any],
    event_spec: Mapping[str, Any],
) -> None:
    broker_order_id = submitted["ref"].broker_order_id
    event_type = str(event_spec.get("event", "fill")).lower()
    if event_type == "fill":
        exec_id = str(event_spec.get("exec_id", f"EXEC-{submitted['oms_order_id']}"))
        adapter.emit_fill(
            broker_order_id,
            exec_id=exec_id,
            price=float(event_spec.get("price", submitted.get("limit_price") or 0.0)),
            qty=float(event_spec.get("qty", submitted.get("qty", 0))),
            commission=float(event_spec.get("commission", 0.0)),
            fill_time=parse_time(event_spec.get("timestamp")),
        )
        await _await(lambda: repo.fill_exists(exec_id))
    elif event_type == "status":
        adapter.emit_status(
            broker_order_id,
            str(event_spec.get("status", "Submitted")),
            remaining=float(event_spec.get("remaining", 0.0)),
        )
    elif event_type == "reject":
        adapter.emit_reject(
            broker_order_id,
            str(event_spec.get("reason", "rejected")),
            int(event_spec.get("error_code", 0)),
            bool(event_spec.get("retryable", False)),
        )


def _build_instruments(fixture: Mapping[str, Any]) -> dict[str, Instrument]:
    return build_instruments_from_fixture(fixture)


async def _hydrate_repositories(
    fixture: Mapping[str, Any],
    repos: list[Any],
    instruments: Mapping[str, Instrument],
) -> None:
    initial = fixture.get("initial_repository_state", {}) or {}
    if not initial or not repos:
        return
    target = repos[0]
    await hydrate_repository_from_fixture(fixture, target, instruments)










async def _settle_callbacks() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def _await(predicate) -> None:
    for _ in range(100):
        result = predicate()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return
        await asyncio.sleep(0.01)
    result = predicate()
    if asyncio.iscoroutine(result):
        result = await result
    assert result




build_single_oms = _build_single_oms
apply_broker_script_to_repos = _apply_broker_script_to_repos
get_order_any = _get_order_any
build_instruments = _build_instruments
hydrate_repositories = _hydrate_repositories
ledger_from_repo = _ledger_from_repo
portfolio_rules_state = _portfolio_rules_state
blocked_reasons_from_repo_events = _blocked_reasons_from_repo_events
drain_queue = _drain_queue
settle_callbacks = _settle_callbacks
plain_dataclass = _plain_dataclass
