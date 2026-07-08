import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from pydantic import ValidationError

from apps.runtime.runtime import RuntimeShell
from libs.config.loader import load_portfolio_config, load_strategy_registry
from libs.config.models import PortfolioCapitalConfig
from libs.config.registry import build_registry_artifact


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def test_strategy_registry_loads_expected_inventory() -> None:
    registry = load_strategy_registry(CONFIG_DIR)

    assert len(registry.connection_groups) == 1
    assert len(registry.strategies) == 11
    assert len(registry.enabled_strategies()) == 9
    assert registry.strategies["ATRSS"].connection_group == "default"
    assert registry.strategies["AKC_HELIX"].connection_group == "default"
    assert "NQ_REGIME" in registry.strategies
    assert "US_ORB_v1" not in registry.strategies
    assert "S5_PB" not in registry.strategies
    assert "S5_DUAL" not in registry.strategies


def test_strategy_registry_connection_group_is_env_driven(monkeypatch) -> None:
    monkeypatch.setenv("IB_HOST", "10.0.0.12")
    monkeypatch.setenv("IB_PORT", "4001")
    monkeypatch.setenv("IB_ACCOUNT_ID", "U1234567")

    registry = load_strategy_registry(CONFIG_DIR)
    group = registry.connection_groups["default"]

    assert group.host == "10.0.0.12"
    assert group.port == 4001
    assert group.account_id == "U1234567"


def test_registry_artifact_contains_all_strategies() -> None:
    registry = load_strategy_registry(CONFIG_DIR)

    artifact = build_registry_artifact(registry)

    assert len(artifact["strategies"]) == 11
    assert {item["strategy_id"] for item in artifact["strategies"]} >= {
        "ATRSS",
        "IARIC_v1",
        "NQ_REGIME",
        "NQDTC_v2.1",
        "SCALP_IVB_AUCTION",
        "SCALP_PO3_REVERSAL",
    }
    by_id = {item["strategy_id"]: item for item in artifact["strategies"]}
    assert "US_ORB_v1" not in by_id
    assert {item["strategy_id"] for item in artifact["strategies"]}.isdisjoint({"S5_PB", "S5_DUAL"})


def test_checked_in_registry_artifact_matches_config() -> None:
    registry = load_strategy_registry(CONFIG_DIR)
    artifact_path = CONFIG_DIR.parent / "data" / "strategy-registry.json"

    checked_in_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert checked_in_artifact == build_registry_artifact(registry)


def test_portfolio_account_urd_config() -> None:
    portfolio = load_portfolio_config(CONFIG_DIR)

    assert portfolio.risk.account_urd_dollars == 200.0


def test_portfolio_allocation_check_equity_is_preflight_only() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "portfolio.yaml").read_text(encoding="utf-8"))
    portfolio = load_portfolio_config(CONFIG_DIR)

    assert raw["capital"]["allocation_check_equity"] == 100_000.0
    assert "initial_equity" not in raw["capital"]
    assert portfolio.capital.allocation_check_equity == 100_000.0
    assert not hasattr(portfolio.capital, "initial_equity")


def test_portfolio_capital_rejects_legacy_initial_equity() -> None:
    with pytest.raises(ValidationError):
        PortfolioCapitalConfig.model_validate({"initial_equity": 100_000.0})


def test_portfolio_paper_equity_starts_as_30k_account_split_by_family() -> None:
    portfolio = load_portfolio_config(CONFIG_DIR)

    assert portfolio.capital.paper_initial_equity == 30_000.0
    for family in ("swing", "momentum", "stock"):
        assert (
            portfolio.capital.paper_initial_equity
            * portfolio.capital.family_allocations[family]
        ) == pytest.approx(10_000.0)


def test_runtime_preflight_flags_mode_port_mismatch(monkeypatch) -> None:
    monkeypatch.setattr("apps.runtime.runtime.get_environment", lambda: "live")
    shell = RuntimeShell(CONFIG_DIR)

    checks = shell.run_preflight()

    by_name = {check.name: check for check in checks}
    assert "ib-mode-port:default" in by_name
    assert not by_name["ib-mode-port:default"].ok


def test_runtime_preflight_flags_mode_account_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("IB_ACCOUNT_ID", "U1234567")
    monkeypatch.setattr("apps.runtime.runtime.get_environment", lambda: "paper")
    shell = RuntimeShell(CONFIG_DIR)

    checks = shell.run_preflight()

    by_name = {check.name: check for check in checks}
    assert "ib-mode-account:default" in by_name
    assert not by_name["ib-mode-account:default"].ok


def test_runtime_preflight_rejects_placeholder_account(monkeypatch) -> None:
    monkeypatch.setenv("IB_ACCOUNT_ID", "DU_PLACEHOLDER")
    monkeypatch.setattr("apps.runtime.runtime.get_environment", lambda: "paper")
    shell = RuntimeShell(CONFIG_DIR)

    checks = shell.run_preflight()

    by_name = {check.name: check for check in checks}
    assert not by_name["ib-mode-account:default"].ok
    assert not by_name["stock-account-config:default"].ok


def test_runtime_preflight_accepts_live_account_and_port(monkeypatch) -> None:
    monkeypatch.setenv("IB_PORT", "4001")
    monkeypatch.setenv("IB_ACCOUNT_ID", "U1234567")
    monkeypatch.setattr("apps.runtime.runtime.get_environment", lambda: "live")
    shell = RuntimeShell(CONFIG_DIR)

    checks = shell.run_preflight()

    by_name = {check.name: check for check in checks}
    assert by_name["ib-mode-port:default"].ok
    assert by_name["ib-mode-account:default"].ok


def test_trading_assistant_momentum_membership_is_current() -> None:
    cfg = yaml.safe_load((CONFIG_DIR / "strategies.yaml").read_text(encoding="utf-8"))
    strategies = cfg["strategies"]

    assert "NQ_REGIME" in strategies
    assert "AKC_Helix_v40" not in strategies
    assert strategies["NQDTC_v2.1"]["module_path"] == "strategies.momentum.nqdtc.plugin"


def test_trading_assistant_stock_membership_is_current() -> None:
    cfg = yaml.safe_load((CONFIG_DIR / "strategies.yaml").read_text(encoding="utf-8"))
    strategies = cfg["strategies"]

    assert "IARIC_v1" in strategies
    assert "ALCB_v1" in strategies
    assert "US_ORB_v1" not in strategies


def test_dashboard_fallback_strategy_config_matches_enabled_runtime_roster() -> None:
    registry = load_strategy_registry(CONFIG_DIR)
    src = (CONFIG_DIR.parent / "apps" / "dashboard" / "src" / "lib" / "types.ts").read_text(
        encoding="utf-8"
    )

    enabled_non_scalp = {
        strategy.strategy_id
        for strategy in registry.strategies.values()
        if strategy.enabled and strategy.family != "scalp"
    }
    for strategy_id in enabled_non_scalp:
        assert strategy_id in src
    assert "US_ORB_v1" not in src


def test_runtime_preflight_flags_stock_readiness_failures_on_scaffold_config(monkeypatch) -> None:
    monkeypatch.delenv("IB_ACCOUNT_ID", raising=False)
    shell = RuntimeShell(CONFIG_DIR)

    checks = shell.run_preflight()

    assert checks
    by_name = {check.name: check for check in checks}
    assert "stock-account-config:default" in by_name
    assert not by_name["stock-account-config:default"].ok


@pytest.mark.asyncio
async def test_runtime_run_filters_paper_mode_only_in_live(monkeypatch) -> None:
    class DummyRegistry:
        def __init__(self) -> None:
            self.connection_groups = {}
            self.calls: list[bool] = []

        def enabled_strategies(self, *, live: bool = False):
            self.calls.append(live)
            return []

    shell = RuntimeShell(CONFIG_DIR)
    shell.registry = DummyRegistry()
    shell.portfolio = object()
    shell.contracts = object()
    shell.routes = object()
    shell.event_calendar = object()
    monkeypatch.setattr(shell, "load", lambda: None)
    monkeypatch.setattr(shell, "_run_sync_preflight_or_raise", lambda: [])
    monkeypatch.setattr(shell, "_run_async_preflight", AsyncMock(return_value=[]))

    monkeypatch.setattr("apps.runtime.runtime.get_environment", lambda: "paper")
    await shell.run(once=True, connect_ib=False)
    assert shell.registry.calls == [False]

    shell.registry.calls.clear()
    monkeypatch.setattr("apps.runtime.runtime.get_environment", lambda: "live")
    await shell.run(once=True, connect_ib=False)
    assert shell.registry.calls == [True]


@pytest.mark.asyncio
async def test_runtime_run_invokes_sync_preflight_for_paper_startup(monkeypatch) -> None:
    class DummyRegistry:
        connection_groups = {}

        @staticmethod
        def enabled_strategies(*, live: bool = False):
            return []

    calls = []
    shell = RuntimeShell(CONFIG_DIR)
    shell.registry = DummyRegistry()
    shell.portfolio = object()
    shell.contracts = object()
    shell.routes = object()
    shell.event_calendar = object()
    monkeypatch.setattr(shell, "load", lambda: None)
    monkeypatch.setattr(shell, "_run_sync_preflight_or_raise", lambda: calls.append("sync") or [])
    monkeypatch.setattr(shell, "_run_async_preflight", AsyncMock(return_value=[]))
    monkeypatch.setattr("apps.runtime.runtime.get_environment", lambda: "paper")

    await shell.run(once=True, connect_ib=False)

    assert calls == ["sync"]


def test_effective_config_preflight_checks_source_hashes(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "trading" / "ibkr_trader" / "config"
    source_root.mkdir(parents=True)
    strategies = source_root / "strategies.yaml"
    portfolio = source_root / "portfolio.yaml"
    strategies.write_text("strategies: {}\n", encoding="utf-8")
    portfolio.write_text("risk:\n  heat_cap_R: 4\n", encoding="utf-8")

    shell = RuntimeShell(CONFIG_DIR)
    shell.load()
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    payload = {
        "bot_id": "ibkr",
        "effective_config_hash": "a" * 64,
        "materialized_config_hash": "b" * 64,
        "materialized_config": _materialized_from_shell(shell),
        "source_files": [
            {
                "role": "ibkr_strategies",
                "path": "trading/ibkr_trader/config/strategies.yaml",
                "sha256": shell._file_sha256(strategies),
            },
            {
                "role": "ibkr_portfolio",
                "path": "trading/ibkr_trader/config/portfolio.yaml",
                "sha256": shell._file_sha256(portfolio),
            },
        ],
    }
    effective_config = tmp_path / "strategies.effective.json"
    effective_config.write_text(json.dumps(payload), encoding="utf-8")

    checks = shell._effective_config_preflight_checks(effective_config)
    assert all(check.ok for check in checks)

    portfolio.write_text("risk:\n  heat_cap_R: 9\n", encoding="utf-8")
    checks = shell._effective_config_preflight_checks(effective_config)
    by_name = {check.name: check for check in checks}
    assert not by_name["effective-config-source:ibkr_portfolio"].ok


def test_effective_config_preflight_checks_loaded_runtime_objects(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "trading" / "ibkr_trader" / "config"
    source_root.mkdir(parents=True)
    strategies = source_root / "strategies.yaml"
    portfolio = source_root / "portfolio.yaml"
    strategies.write_text("strategies: {}\n", encoding="utf-8")
    portfolio.write_text("risk:\n  heat_cap_R: 4\n", encoding="utf-8")

    shell = RuntimeShell(CONFIG_DIR)
    shell.load()
    materialized = _materialized_from_shell(shell)
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    payload = {
        "bot_id": "ibkr",
        "effective_config_hash": "a" * 64,
        "materialized_config_hash": "b" * 64,
        "materialized_config": materialized,
        "source_files": [
            {
                "role": "ibkr_strategies",
                "path": "trading/ibkr_trader/config/strategies.yaml",
                "sha256": shell._file_sha256(strategies),
            },
            {
                "role": "ibkr_portfolio",
                "path": "trading/ibkr_trader/config/portfolio.yaml",
                "sha256": shell._file_sha256(portfolio),
            },
        ],
    }
    effective_config = tmp_path / "strategies.effective.json"
    effective_config.write_text(json.dumps(payload), encoding="utf-8")

    shell.registry.connection_groups["default"].client_id += 100
    checks = shell._effective_config_preflight_checks(effective_config)
    by_name = {check.name: check for check in checks}

    assert not by_name["effective-config-loaded-runtime:ibkr"].ok


def test_effective_config_preflight_rejects_extra_loaded_strategy(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "trading" / "ibkr_trader" / "config"
    source_root.mkdir(parents=True)
    strategies = source_root / "strategies.yaml"
    portfolio = source_root / "portfolio.yaml"
    strategies.write_text("strategies: {}\n", encoding="utf-8")
    portfolio.write_text("risk:\n  heat_cap_R: 4\n", encoding="utf-8")

    shell = RuntimeShell(CONFIG_DIR)
    shell.load()
    materialized = _materialized_from_shell(shell)
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    payload = {
        "bot_id": "ibkr",
        "effective_config_hash": "a" * 64,
        "materialized_config_hash": "b" * 64,
        "materialized_config": materialized,
        "source_files": [
            {
                "role": "ibkr_strategies",
                "path": "trading/ibkr_trader/config/strategies.yaml",
                "sha256": shell._file_sha256(strategies),
            },
            {
                "role": "ibkr_portfolio",
                "path": "trading/ibkr_trader/config/portfolio.yaml",
                "sha256": shell._file_sha256(portfolio),
            },
        ],
    }
    effective_config = tmp_path / "strategies.effective.json"
    effective_config.write_text(json.dumps(payload), encoding="utf-8")

    shell.registry.strategies["UNAPPROVED_EXTRA"] = shell.registry.strategies["ATRSS"]
    checks = shell._effective_config_preflight_checks(effective_config)
    by_name = {check.name: check for check in checks}

    assert not by_name["effective-config-loaded-runtime:ibkr"].ok
    assert "extra=['UNAPPROVED_EXTRA']" in by_name["effective-config-loaded-runtime:ibkr"].detail


def test_instrumentation_readiness_posts_relay_acceptance_heartbeat(tmp_path, monkeypatch) -> None:
    import trading_contracts.relay_acceptance as relay_acceptance

    calls = []
    monkeypatch.delenv("INSTRUMENTATION_RELAY_URL", raising=False)
    monkeypatch.setenv("RELAY_API_KEY", "assistant-read-key")
    monkeypatch.delenv("ASSISTANT_RELAY_API_KEY", raising=False)
    monkeypatch.delenv("ALLOW_BOT_RELAY_EXACT_ACK", raising=False)
    monkeypatch.delenv("ASSISTANT_RELAY_EXACT_ACK_API_KEY", raising=False)
    monkeypatch.setenv("INSTRUMENTATION_HMAC_SECRET", "0123456789abcdef0123456789ABCDEF")
    monkeypatch.setattr(
        relay_acceptance,
        "probe_relay_acceptance",
        lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(ok=True, event_id="relay-heartbeat-ibkr", error=""),
    )

    shell = RuntimeShell(CONFIG_DIR)
    shell._effective_config_hash = "c" * 64
    checks = shell._instrumentation_readiness_checks(
        "stock",
        {
            "bot_id": "stock_trader",
            "data_dir": str(tmp_path / "data"),
            "sidecar": {
                "relay_url": "https://assistant-relay.internal/events",
                "hmac_secret_env": "INSTRUMENTATION_HMAC_SECRET",
                "buffer_dir": str(tmp_path / "buffer"),
            },
        },
        require_instrumentation=True,
    )
    by_name = {check.name: check for check in checks}

    assert by_name["instrumentation-relay-acceptance:stock"].ok
    assert calls == [
        {
            "relay_url": "https://assistant-relay.internal/events",
            "hmac_secret": "0123456789abcdef0123456789ABCDEF",
            "bot_id": "stock_trader",
            "runtime_instance_id": "ibkr-stock:cccccccccccc",
            "effective_config_hash": "c" * 64,
            "deployment_id": "ibkr:stock:cccccccccccccccc",
            "source": "ibkr-stock-runtime-preflight",
            "timeout_seconds": 10.0,
            "confirm_health": True,
            "require_exact_ack": False,
            "relay_api_key": "",
        }
    ]


def _materialized_from_shell(shell: RuntimeShell) -> dict:
    return {
        "connection_groups": {
            key: value.model_dump(mode="json")
            for key, value in sorted(shell.registry.connection_groups.items())
        },
        "portfolio": shell.portfolio.model_dump(mode="json"),
        "strategies": [
            {"strategy_id": key, "effective_config": value.model_dump(mode="json")}
            for key, value in sorted(shell.registry.strategies.items())
        ],
    }


@pytest.mark.asyncio
async def test_runtime_run_allows_dev_mixed_family_stock_readiness_warnings(monkeypatch) -> None:
    class DummyRegistry:
        connection_groups = {}

        @staticmethod
        def enabled_strategies(*, live: bool = False):
            return [
                SimpleNamespace(family="swing"),
                SimpleNamespace(family="stock"),
            ]

    shell = RuntimeShell(CONFIG_DIR)
    shell.registry = DummyRegistry()
    shell.portfolio = object()
    shell.contracts = object()
    shell.routes = object()
    shell.event_calendar = object()
    monkeypatch.setattr(shell, "load", lambda: None)
    monkeypatch.setattr("apps.runtime.runtime.get_environment", lambda: "dev")
    monkeypatch.setattr(
        shell,
        "_run_async_preflight",
        AsyncMock(return_value=[
            SimpleNamespace(
                name="stock-account-config:default",
                ok=False,
                detail="account_id is unresolved placeholder ${IB_ACCOUNT_ID}",
            ),
            SimpleNamespace(
                name="stock-artifact-readiness:IARIC_v1",
                ok=False,
                detail="watchlist unavailable",
            ),
        ]),
    )

    await shell.run(once=True, connect_ib=False)


@pytest.mark.asyncio
async def test_runtime_run_hard_fails_paper_mixed_family_stock_readiness(monkeypatch) -> None:
    class DummyRegistry:
        connection_groups = {}

        @staticmethod
        def enabled_strategies(*, live: bool = False):
            return [
                SimpleNamespace(family="swing"),
                SimpleNamespace(family="stock"),
            ]

    shell = RuntimeShell(CONFIG_DIR)
    shell.registry = DummyRegistry()
    shell.portfolio = object()
    shell.contracts = object()
    shell.routes = object()
    shell.event_calendar = object()
    monkeypatch.setattr(shell, "load", lambda: None)
    monkeypatch.setattr(shell, "_run_sync_preflight_or_raise", lambda: [])
    monkeypatch.setattr("apps.runtime.runtime.get_environment", lambda: "paper")
    monkeypatch.setattr(
        shell,
        "_run_async_preflight",
        AsyncMock(return_value=[
            SimpleNamespace(
                name="stock-artifact-readiness:IARIC_v1",
                ok=False,
                detail="watchlist unavailable",
            ),
        ]),
    )

    with pytest.raises(RuntimeError, match="Preflight failed: 1 critical check"):
        await shell.run(once=True, connect_ib=False)


@pytest.mark.asyncio
async def test_runtime_run_still_hard_fails_stock_only_startup_when_readiness_missing(monkeypatch) -> None:
    class DummyRegistry:
        connection_groups = {}

        @staticmethod
        def enabled_strategies(*, live: bool = False):
            return [SimpleNamespace(family="stock")]

    shell = RuntimeShell(CONFIG_DIR)
    shell.registry = DummyRegistry()
    shell.portfolio = object()
    shell.contracts = object()
    shell.routes = object()
    shell.event_calendar = object()
    monkeypatch.setattr(shell, "load", lambda: None)
    monkeypatch.setattr(
        shell,
        "_run_async_preflight",
        AsyncMock(return_value=[
            SimpleNamespace(
                name="stock-account-config:default",
                ok=False,
                detail="account_id is unresolved placeholder ${IB_ACCOUNT_ID}",
            ),
        ]),
    )

    with pytest.raises(RuntimeError, match="Preflight failed: 1 critical check"):
        await shell.run(once=True, connect_ib=False)
