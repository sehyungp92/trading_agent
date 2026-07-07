"""Tests for non-blocking live parity warnings."""

from types import SimpleNamespace

from crypto_trader.core.execution_adapter import ExecutionCapabilities
from crypto_trader.core.models import OrderType, Side
from crypto_trader.core.runtime_types import OrderIntent
from crypto_trader.live.config import LiveConfig
from crypto_trader.live.parity_warnings import (
    collect_live_parity_warnings,
    ParityWarning,
    should_block_live_startup,
    validate_order_intent_capabilities,
)
from crypto_trader.portfolio.config import PortfolioConfig


def _warning_ids(portfolio_config: PortfolioConfig) -> set[str]:
    warnings = collect_live_parity_warnings(LiveConfig(rate_limit_per_sec=5.0), portfolio_config)
    return {warning.warning_id for warning in warnings}


def test_symbol_collision_warning_for_cap_or_allow() -> None:
    assert "symbol_collision_not_blocked" in _warning_ids(
        PortfolioConfig(symbol_collision="cap")
    )
    assert "symbol_collision_not_blocked" in _warning_ids(
        PortfolioConfig(symbol_collision="allow")
    )


def test_no_symbol_collision_warning_for_block() -> None:
    assert "symbol_collision_not_blocked" not in _warning_ids(
        PortfolioConfig(symbol_collision="block")
    )


def test_oms_and_metadata_warnings_are_non_fatal() -> None:
    warnings = collect_live_parity_warnings(
        LiveConfig(rate_limit_per_sec=5.0),
        PortfolioConfig(symbol_collision="block"),
    )

    ids = {warning.warning_id for warning in warnings}
    assert "durable_oms_unavailable" in ids
    assert "exchange_metadata_not_enforced" in ids
    assert all(warning.severity == "warning" for warning in warnings)


def test_warnings_serialize_to_dict() -> None:
    warning = collect_live_parity_warnings(
        LiveConfig(rate_limit_per_sec=0),
        PortfolioConfig(symbol_collision="block"),
    )[-1]

    payload = warning.to_dict()
    assert payload["warning_id"] == "invalid_rate_limit_for_parity"
    assert payload["severity"] == "warning"
    assert payload["message"]
    assert payload["mitigation"]


def test_strategy_ttl_order_warning_when_capabilities_do_not_emulate_ttl() -> None:
    trend_cfg = SimpleNamespace(
        entry=SimpleNamespace(
            mode="confirm_preferred",
            max_bars_after_confirmation=2,
            entry_on_break=False,
            entry_on_close=True,
        )
    )

    warnings = collect_live_parity_warnings(
        LiveConfig(rate_limit_per_sec=5.0),
        PortfolioConfig(symbol_collision="block"),
        durable_oms_available=True,
        exchange_metadata_enforced=True,
        strategy_configs={"trend": trend_cfg},
        capabilities=ExecutionCapabilities(ttl=False),
    )

    warning = next(w for w in warnings if w.warning_id == "trend_ttl_orders_unsupported_live")
    assert warning.severity == "error"
    assert "ttl_bars" in warning.message


def test_strategy_ttl_order_warning_is_clear_with_live_ttl_emulation() -> None:
    trend_cfg = SimpleNamespace(
        entry=SimpleNamespace(
            mode="confirm_preferred",
            max_bars_after_confirmation=2,
            entry_on_break=False,
            entry_on_close=True,
        )
    )

    warnings = collect_live_parity_warnings(
        LiveConfig(rate_limit_per_sec=5.0),
        PortfolioConfig(symbol_collision="block"),
        durable_oms_available=True,
        exchange_metadata_enforced=True,
        strategy_configs={"trend": trend_cfg},
    )

    assert "trend_ttl_orders_unsupported_live" not in {w.warning_id for w in warnings}


def test_order_intent_capability_errors_cover_unsupported_live_surfaces() -> None:
    intent = OrderIntent(
        intent_id="intent_1",
        strategy_id="trend",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.STOP_LIMIT,
        qty=0.1,
        reduce_only=True,
        ttl_bars=2,
        oca_group="oca_1",
        bracket_group="bracket_1",
    )

    ids = {
        warning.warning_id
        for warning in validate_order_intent_capabilities(
            intent,
            capabilities=ExecutionCapabilities(ttl=False, reduce_only=True),
        )
    }

    assert ids == {
        "stop_limit_not_supported_live",
        "ttl_not_supported_live",
        "oca_not_supported_live",
        "bracket_not_supported_live",
    }


def test_live_startup_policy_blocks_errors_even_on_testnet() -> None:
    warnings = [ParityWarning(
        warning_id="ttl_not_supported_live",
        severity="error",
        message="unsupported",
        mitigation="fix config",
    )]

    assert should_block_live_startup(warnings, LiveConfig(is_testnet=True)) is True


def test_native_oca_requirement_blocks_when_live_adapter_has_no_native_support() -> None:
    warnings = collect_live_parity_warnings(
        LiveConfig(rate_limit_per_sec=5.0, require_native_oca=True),
        PortfolioConfig(symbol_collision="block"),
        durable_oms_available=True,
        exchange_metadata_enforced=True,
        capabilities=ExecutionCapabilities(reduce_only=True, ttl=True, oca=False),
    )

    warning = next(w for w in warnings if w.warning_id == "native_oca_required_but_unavailable")
    assert warning.severity == "error"
    assert should_block_live_startup(warnings, LiveConfig(is_testnet=True)) is True


def test_live_startup_policy_keeps_soft_testnet_warnings_non_blocking() -> None:
    warnings = [ParityWarning(
        warning_id="exchange_metadata_not_enforced",
        severity="warning",
        message="metadata missing",
        mitigation="set asset_meta_path",
    )]

    assert should_block_live_startup(warnings, LiveConfig(is_testnet=True)) is False
    assert should_block_live_startup(warnings, LiveConfig(is_testnet=False)) is True
    assert should_block_live_startup(
        warnings,
        LiveConfig(is_testnet=True, strict_live_parity=True),
    ) is True
