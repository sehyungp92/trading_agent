from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from libs.oms.models.instrument import Instrument
from libs.oms.persistence.in_memory import InMemoryRepository
from libs.oms.services.factory import build_multi_strategy_oms, build_oms_service
from strategies.contracts import CoordinatorRuntimeOverrides, RuntimeContext
from tests.integration.parity.fake_ibkr import FakeIBKRExecutionAdapter
from tests.integration.parity.family_surface_names import family_surface_adapter_name as _family_surface_adapter_name
from tests.integration.parity.live_oms import hydrate_repositories
from tests.integration.parity.portfolio_rules import portfolio_rules_config_from_fixture
from tests.integration.parity.source_inputs import (
    iaric_artifact,
    idle_market_input,
    overlay_rebalance_payload as _overlay_rebalance_payload,
    parse_time,
    strategy_ids,
)


async def _start_family_coordinator(
    fixture: Mapping[str, Any],
    instrumentation_dir: str,
    instruments: Mapping[str, Instrument],
    *,
    event_clock,
) -> tuple[Any, list[FakeIBKRExecutionAdapter]]:
    adapters: list[FakeIBKRExecutionAdapter] = []
    shared_repo = InMemoryRepository()

    def adapter_factory(**_: Any) -> FakeIBKRExecutionAdapter:
        adapter = FakeIBKRExecutionAdapter(auto_ack=False)
        adapters.append(adapter)
        return adapter

    coordinator = _instantiate_family_coordinator(
        fixture,
        instrumentation_dir,
        shared_repo,
        adapter_factory=adapter_factory,
        event_clock=event_clock,
    )
    await hydrate_repositories(fixture, [shared_repo], instruments)
    await coordinator.start()
    return coordinator, adapters


def _instantiate_family_coordinator(
    fixture: Mapping[str, Any],
    instrumentation_dir: str,
    shared_repo: InMemoryRepository | None = None,
    *,
    adapter_factory,
    event_clock,
) -> Any:
    family = str(fixture.get("family", "") or (fixture.get("family_config", {}) or {}).get("family", ""))
    account = fixture.get("account_state", {}) or {}
    shared_repo = shared_repo or InMemoryRepository()

    async def build_single_override(**kwargs: Any) -> Any:
        kwargs.setdefault("db_pool", None)
        kwargs.setdefault("repository", shared_repo)
        kwargs.setdefault("recon_interval_s", 3600.0)
        kwargs.setdefault("instrumentation_data_dir", instrumentation_dir)
        kwargs.setdefault("event_clock", event_clock)
        return await build_oms_service(**kwargs)

    async def build_multi_override(**kwargs: Any) -> Any:
        kwargs.setdefault("db_pool", None)
        kwargs.setdefault("repository", shared_repo)
        kwargs.setdefault("recon_interval_s", 3600.0)
        kwargs.setdefault("instrumentation_data_dir", instrumentation_dir)
        kwargs.setdefault("event_clock", event_clock)
        return await build_multi_strategy_oms(**kwargs)

    ctx = RuntimeContext(
        manifest=SimpleNamespace(),
        registry=_fixture_registry(fixture),
        portfolio=SimpleNamespace(
            capital=SimpleNamespace(
                paper_initial_equity=float(account.get("equity", 100_000.0)),
                family_allocations={family: 1.0},
            )
        ),
        contracts={"account_id": str(account.get("account_id", "ACCT-PARITY"))},
        runtime_overrides=CoordinatorRuntimeOverrides(
            adapter_factory=adapter_factory,
            calendar_factory=_FixtureCalendar,
            equity_provider=lambda: float(account.get("equity", 100_000.0)),
            build_oms_service=build_single_override,
            build_multi_strategy_oms=build_multi_override,
            strategy_ids=tuple(strategy_ids(fixture)),
            stock_artifact_provider=lambda: _stock_artifacts(fixture),
            overlay_rebalance_provider=lambda: _overlay_rebalance_payload(fixture),
            portfolio_rules_provider=lambda: portfolio_rules_config_from_fixture(fixture),
            state_dir_overrides={
                strategy_id: Path(instrumentation_dir) / strategy_id
                for strategy_id in strategy_ids(fixture)
            },
            instrumentation_data_dir=Path(instrumentation_dir),
            disable_background_tasks=True,
            disable_instrumentation=True,
            disable_market_data=True,
        ),
    )
    if family == "swing":
        from strategies.swing.coordinator import SwingFamilyCoordinator

        return SwingFamilyCoordinator(ctx)
    if family == "momentum":
        from strategies.momentum.coordinator import MomentumFamilyCoordinator

        return MomentumFamilyCoordinator(ctx)
    if family == "stock":
        from strategies.stock.coordinator import StockFamilyCoordinator

        return StockFamilyCoordinator(ctx)
    return None


def _coordinator_oms_services(coordinator: Any) -> list[Any]:
    if coordinator is None:
        return []
    services = list(getattr(coordinator, "_oms_services", []) or [])
    shared = getattr(coordinator, "_oms", None)
    if shared is not None and shared not in services:
        services.append(shared)
    return services


def _coordinator_engines(coordinator: Any) -> dict[str, Any]:
    engines: dict[str, Any] = {}
    for item in getattr(coordinator, "_engines", []) or []:
        if isinstance(item, tuple) and len(item) == 2:
            engines[str(item[0])] = item[1]
    strategy_ids = list(getattr(coordinator, "_strategy_ids", []) or [])
    for sid, engine in zip(strategy_ids, getattr(coordinator, "_engines", []) or [], strict=False):
        if not isinstance(engine, tuple):
            engines[str(sid)] = engine
    engines.update(getattr(coordinator, "_engine_map", {}) or {})
    overlay = getattr(coordinator, "_overlay_engine", None)
    if overlay is not None:
        engines["OVERLAY"] = overlay
    return engines


class _FixtureCalendar:
    def is_trading_day(self, _value: Any) -> bool:
        return True

    def is_market_open(self, *_args: Any, **_kwargs: Any) -> bool:
        return True

    def is_entry_blocked(self, *_args: Any, **_kwargs: Any) -> bool:
        return False


class _FixtureRegistry:
    def __init__(self, fixture: Mapping[str, Any]) -> None:
        family = str(fixture.get("family", "") or (fixture.get("family_config", {}) or {}).get("family", ""))
        self.strategies = {
            strategy_id: SimpleNamespace(
                strategy_id=strategy_id,
                system_id=strategy_id,
                family=family,
                enabled=True,
                paper_mode=False,
                engine_config={},
            )
            for strategy_id in strategy_ids(fixture)
        }

    def enabled_strategies(self, *, live: bool = False) -> list[Any]:
        return [
            manifest
            for manifest in self.strategies.values()
            if manifest.enabled and (not live or not manifest.paper_mode)
        ]


def _fixture_registry(fixture: Mapping[str, Any]) -> _FixtureRegistry:
    return _FixtureRegistry(fixture)


def _stock_artifacts(fixture: Mapping[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    ids = set(strategy_ids(fixture))
    if "IARIC_v1" in ids:
        artifacts["IARIC_v1"] = iaric_artifact(fixture)
    if "ALCB_v1" in ids:
        artifacts["ALCB_v1"] = _empty_alcb_artifact(fixture)
    return artifacts


def _empty_alcb_artifact(fixture: Mapping[str, Any]) -> Any:
    from strategies.stock.alcb.models import (
        Campaign,
        CampaignState,
        CandidateArtifact,
        CandidateItem,
        RegimeSnapshot,
        ResearchDailyBar,
    )

    clock = parse_time(fixture["clock_start"])
    market_input = idle_market_input(fixture, "ALCB_v1")
    symbol = str(market_input.get("symbol", "")).upper()
    bars = list(market_input.get("bars", []) or [])
    last = bars[-1] if bars else {}
    daily_bars = [
        ResearchDailyBar(
            trade_date=parse_time(row.get("timestamp") or row.get("time") or fixture["clock_start"]).date(),
            open=float(row.get("open", 0.0) or 0.0),
            high=float(row.get("high", 0.0) or 0.0),
            low=float(row.get("low", 0.0) or 0.0),
            close=float(row.get("close", 0.0) or 0.0),
            volume=float(row.get("volume", 0.0) or 0.0),
        )
        for row in bars
    ]
    regime = RegimeSnapshot(
        score=0.75,
        tier="B",
        risk_multiplier=1.0,
        price_ok=True,
        breadth_ok=True,
        vol_ok=True,
        credit_ok=True,
        market_regime="BULL",
    )
    item = CandidateItem(
        symbol=symbol,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        tick_size=0.01,
        point_value=1.0,
        sector="Technology",
        adv20_usd=25_000_000.0,
        median_spread_pct=0.001,
        selection_score=80,
        selection_detail={"fixture": 80},
        stock_regime="BULL",
        market_regime="BULL",
        sector_regime="BULL",
        daily_trend_sign=1,
        relative_strength_percentile=80.0,
        accumulation_score=1.0,
        ttm_squeeze_bonus=0,
        average_30m_volume=max(float(last.get("volume", 600_000.0) or 600_000.0), 1.0),
        median_30m_volume=max(float(last.get("volume", 600_000.0) or 600_000.0), 1.0),
        tradable_flag=True,
        direction_bias="LONG",
        price=float(last.get("close", 0.0) or 0.0),
        earnings_risk_flag=False,
        campaign=Campaign(symbol=symbol, state=CampaignState.COMPRESSION, campaign_id=1),
        daily_bars=daily_bars,
    )
    return CandidateArtifact(
        trade_date=clock.date(),
        generated_at=clock,
        regime=regime,
        items=[item],
        tradable=[item],
        overflow=[],
        long_candidates=[item],
        short_candidates=[],
    )





async def drive_overlay_rebalance(fixture: Mapping[str, Any], coordinator: Any) -> None:
    if coordinator is None or not hasattr(coordinator, "run_overlay_rebalance_once"):
        return
    payload = _overlay_rebalance_payload(fixture)
    if payload.get("rebalance_due") and payload.get("symbols"):
        await coordinator.run_overlay_rebalance_once(payload)


start_family_coordinator = _start_family_coordinator
coordinator_oms_services = _coordinator_oms_services
coordinator_engines = _coordinator_engines
family_surface_adapter_name = _family_surface_adapter_name
