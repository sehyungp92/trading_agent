from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tests.integration.parity.fixtures import load_parity_fixture
from tests.integration.parity.live_family import (
    coordinator_engines as _coordinator_engines,
    coordinator_oms_services as _coordinator_oms_services,
    start_family_coordinator as _start_family_coordinator,
)
from tests.integration.parity.live_oms import build_instruments as _build_instruments
from tests.integration.parity.source_inputs import (
    overlay_rebalance_payload as _overlay_rebalance_payload,
    parse_time,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "parity" / "layer3"


@pytest.mark.parity_nightly
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fixture_name", "expected_engines"),
    [
        ("swing_family_overlay_rebalance.json", {"ATRSS", "AKC_HELIX", "TPC"}),
        ("momentum_family_shared_risk.json", {"NQDTC_v2.1", "NQ_REGIME", "VdubusNQ_v4", "DownturnDominator_v1"}),
        ("stock_family_collision.json", {"IARIC_v1", "ALCB_v1"}),
    ],
)
async def test_family_coordinators_start_offline_with_runtime_overrides(
    fixture_name: str,
    expected_engines: set[str],
) -> None:
    fixture = load_parity_fixture(FIXTURE_ROOT / fixture_name)
    with TemporaryDirectory(prefix="parity-coordinator-") as state_dir:
        coordinator, adapters = await _start_family_coordinator(
            fixture,
            state_dir,
            _build_instruments(fixture),
            event_clock=lambda: parse_time(fixture["clock_start"]),
        )
        try:
            assert adapters
            assert _coordinator_oms_services(coordinator)
            assert expected_engines <= set(_coordinator_engines(coordinator))
            assert getattr(coordinator, "_heartbeat_task", None) is None
            assert getattr(coordinator, "_market_data_task", None) is None
            assert getattr(getattr(coordinator, "_ctx", None), "session", None) is None
        finally:
            await coordinator.stop()


@pytest.mark.parity_nightly
@pytest.mark.asyncio
async def test_stock_overrides_do_not_read_local_ibkr_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    class PoisonIBKRConfig:
        def __init__(self, config_dir: Path) -> None:
            calls.append(Path(config_dir))
            self.profile = SimpleNamespace(account_id="LOCAL-POISON")
            self.contracts = {}
            self.routes = {}

    import libs.broker_ibkr.config.loader as loader

    monkeypatch.setattr(loader, "IBKRConfig", PoisonIBKRConfig)
    fixture = load_parity_fixture(FIXTURE_ROOT / "stock_family_collision.json")
    with TemporaryDirectory(prefix="parity-stock-overrides-") as state_dir:
        coordinator, _adapters = await _start_family_coordinator(
            fixture,
            state_dir,
            _build_instruments(fixture),
            event_clock=lambda: parse_time(fixture["clock_start"]),
        )
        try:
            assert calls == []
            accounts = {
                getattr(engine, "_account_id", None)
                for engine in _coordinator_engines(coordinator).values()
                if hasattr(engine, "_account_id")
            }
            assert accounts == {"ACCT-PARITY"}
        finally:
            await coordinator.stop()


@pytest.mark.parity_nightly
@pytest.mark.asyncio
async def test_swing_overlay_parity_rebalance_subtracts_deployed_capital() -> None:
    fixture = load_parity_fixture(FIXTURE_ROOT / "swing_family_overlay_rebalance.json")
    fixture["initial_repository_state"] = {
        "positions": [
            {
                "strategy_id": "TPC",
                "account_id": "ACCT-PARITY",
                "symbol": "QQQ",
                "net_qty": 1,
                "avg_price": 100.0,
                "open_risk_dollars": 250.0,
                "open_risk_R": 0.5,
                "last_update_at": "2026-05-20T14:29:00+00:00",
            }
        ]
    }
    with TemporaryDirectory(prefix="parity-swing-overlay-") as state_dir:
        coordinator, _adapters = await _start_family_coordinator(
            fixture,
            state_dir,
            _build_instruments(fixture),
            event_clock=lambda: parse_time(fixture["clock_start"]),
        )
        try:
            result = await coordinator.run_overlay_rebalance_once(_overlay_rebalance_payload(fixture))
            assert result["positions"]["QQQ"] == 408
            assert result["last_decision_details"]["deployed_capital"] == 50_000
            assert result["last_decision_details"]["net_equity"] == 50_000
        finally:
            await coordinator.stop()
