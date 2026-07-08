from __future__ import annotations

from collections.abc import Mapping
from tempfile import TemporaryDirectory
from typing import Any

from libs.oms.models.instrument_registry import InstrumentRegistry
from tests.integration.parity.fake_ibkr import FakeIBKRExecutionAdapter
from tests.integration.parity.live_family import (
    coordinator_engines as _coordinator_engines,
    coordinator_oms_services as _coordinator_oms_services,
    drive_overlay_rebalance,
    start_family_coordinator as _start_family_coordinator,
)
from tests.integration.parity.live_idle import drive_idle_market_children
from tests.integration.parity.live_layer2 import (
    drive_layer2_live_inputs,
    hydrate_live_state as _hydrate_live_state,
    instantiate_live_engines as _instantiate_live_engines,
    start_engines as _start_engines,
)
from tests.integration.parity.live_oms import (
    apply_broker_script_to_repos as _apply_broker_script_to_repos,
    build_instruments as _build_instruments,
    build_single_oms as _build_single_oms,
    drain_queue as _drain_queue,
    get_order_any as _get_order_any,
    hydrate_repositories as _hydrate_repositories,
    ledger_from_repo as _ledger_from_repo,
    settle_callbacks as _settle_callbacks,
)
from tests.integration.parity.live_shadow_contract import ParityTrace
from tests.integration.parity.live_state import state_from_repos as _state_from_repos
from tests.integration.parity.normalizers import (
    normalize_oms_events,
    normalize_order_intents,
    normalize_state_snapshot,
    normalize_trade_ledger,
)
from tests.integration.parity.runtime_source import runtime_source_fingerprint
from tests.integration.parity.source_inputs import family_resolver, instrument_ticks, parse_time


async def run_layer2_live_trace(fixture: Mapping[str, Any]) -> ParityTrace:
    return await _run_live_trace(fixture, family_mode=False)


async def run_layer3_family_live_trace(fixture: Mapping[str, Any]) -> ParityTrace:
    return await _run_live_trace(fixture, family_mode=True)


async def _run_live_trace(
    fixture: Mapping[str, Any],
    *,
    family_mode: bool,
) -> ParityTrace:
    source_hash = _live_source_fingerprint(fixture)
    InstrumentRegistry.clear()
    instruments = _build_instruments(fixture)
    ticks = instrument_ticks(fixture)
    family_for_strategy = family_resolver(fixture)
    adapter = FakeIBKRExecutionAdapter(auto_ack=False)
    adapters: list[FakeIBKRExecutionAdapter] = [adapter]
    event_clock = lambda: parse_time(fixture["clock_start"])
    oms_services: list[Any] = []
    engines: dict[str, Any] = {}
    coordinator: Any = None

    with TemporaryDirectory(prefix="parity-oms-") as instrumentation_dir:
        if family_mode:
            coordinator, adapters = await _start_family_coordinator(
                fixture,
                instrumentation_dir,
                instruments,
                event_clock=event_clock,
            )
            oms_services = _coordinator_oms_services(coordinator)
            oms = oms_services[0]
            engines = _coordinator_engines(coordinator)
        else:
            oms = await _build_single_oms(fixture, adapter, instrumentation_dir, event_clock=event_clock)
            oms_services.append(oms)
        repos = list(dict.fromkeys(service._handler._repo for service in oms_services))
        event_queues = [service.stream_all_events() for service in oms_services]

        try:
            if not family_mode:
                await oms.start()
                engines = _instantiate_live_engines(fixture, oms, instruments, instrumentation_dir)
                await _start_engines(engines)
                await _hydrate_repositories(fixture, repos, instruments)
            await _hydrate_live_state(fixture, engines)
            await drive_layer2_live_inputs(fixture, engines)
            await drive_idle_market_children(fixture, engines)
            await drive_overlay_rebalance(fixture, coordinator)
            await _settle_callbacks()

            await _apply_broker_script_to_repos(fixture, adapters, repos)
            await _settle_callbacks()

            raw_events = [event for queue in event_queues for event in _drain_queue(queue)]
            submitted_orders = [
                await _get_order_any(repos, item["oms_order_id"])
                for current_adapter in adapters
                for item in current_adapter.submitted
            ]
            ledger_rows = [
                row
                for repo in repos
                for row in await _ledger_from_repo(repo, family_for_strategy)
            ]
            state = await _state_from_repos(repos, oms_services, fixture, engines, coordinator)
        finally:
            if family_mode and coordinator is not None:
                await coordinator.stop()
            else:
                for service in reversed(oms_services):
                    await service.stop()
            InstrumentRegistry.clear()

    return ParityTrace(
        producer="live_oms",
        source_fingerprint=source_hash,
        order_intents=normalize_order_intents(
            submitted_orders,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        terminal_events=normalize_oms_events(
            raw_events,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        trade_ledger=normalize_trade_ledger(
            ledger_rows,
            family_for_strategy=family_for_strategy,
            instrument_ticks=ticks,
        ),
        state_snapshot=normalize_state_snapshot(state),
    )


def _live_source_fingerprint(fixture: Mapping[str, Any]) -> str:
    return runtime_source_fingerprint(fixture)
