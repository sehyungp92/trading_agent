"""Non-blocking live/backtest parity warnings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crypto_trader.core.execution_adapter import (
    ExecutionCapabilities,
    unsupported_order_intent_reasons,
)
from crypto_trader.core.runtime_types import OrderIntent
from crypto_trader.live.config import LiveConfig
from crypto_trader.live.execution_adapter import HyperliquidExecutionAdapter
from crypto_trader.portfolio.config import PortfolioConfig


@dataclass(frozen=True, slots=True)
class ParityWarning:
    """A live parity warning that should be surfaced but not block startup."""

    warning_id: str
    message: str
    mitigation: str
    severity: str = "warning"

    def to_dict(self) -> dict[str, str]:
        return {
            "warning_id": self.warning_id,
            "severity": self.severity,
            "message": self.message,
            "mitigation": self.mitigation,
        }


def collect_live_parity_warnings(
    live_config: LiveConfig,
    portfolio_config: PortfolioConfig | None = None,
    *,
    durable_oms_available: bool = False,
    exchange_metadata_enforced: bool = False,
    strategy_configs: dict[str, Any] | None = None,
    capabilities: ExecutionCapabilities | None = None,
) -> list[ParityWarning]:
    """Collect known live/backtest parity warnings without changing behavior."""
    warnings: list[ParityWarning] = []

    if portfolio_config is not None and portfolio_config.symbol_collision != "block":
        warnings.append(ParityWarning(
            warning_id="symbol_collision_not_blocked",
            message=(
                "Portfolio symbol_collision is not 'block'; live may allow shared-symbol "
                "risk that is not yet covered by a strict parity gate."
            ),
            mitigation="Use symbol_collision='block' for strict live/backtest parity.",
        ))

    if not durable_oms_available:
        warnings.append(ParityWarning(
            warning_id="durable_oms_unavailable",
            message="Live OMS ownership, fill watermarks, and lifecycle state are not durable yet.",
            mitigation="Implement Phase 4 OMS persistence before relying on restart parity.",
        ))

    if not exchange_metadata_enforced:
        warnings.append(ParityWarning(
            warning_id="exchange_metadata_not_enforced",
            message="Exchange precision/margin metadata is not enforced as a required live input yet.",
            mitigation="Require a shared asset metadata cache before production parity gates.",
        ))

    if live_config.rate_limit_per_sec <= 0:
        warnings.append(ParityWarning(
            warning_id="invalid_rate_limit_for_parity",
            message="Live rate_limit_per_sec is non-positive, so request pacing cannot match assumptions.",
            mitigation="Set a positive live rate_limit_per_sec.",
        ))

    caps = capabilities or HyperliquidExecutionAdapter.capabilities
    if live_config.require_native_oca and not caps.oca:
        warnings.append(ParityWarning(
            warning_id="native_oca_required_but_unavailable",
            severity="error",
            message=(
                "Native OCA/OCO was required by live config, but the live adapter "
                "has not implemented verified exchange-side group submit, sibling "
                "cancellation reports, and restart sync."
            ),
            mitigation=(
                "Disable require_native_oca for paper broker-managed fallback, "
                "or implement and test venue-native OCA before mainnet/strict startup."
            ),
        ))

    warnings.extend(validate_live_capabilities(
        strategy_configs=strategy_configs,
        capabilities=caps,
    ))

    return warnings


def should_block_live_startup(
    warnings: list[ParityWarning],
    live_config: LiveConfig,
) -> bool:
    """Return whether live/paper startup must stop for parity findings."""
    if any(warning.severity == "error" for warning in warnings):
        return True
    return bool(warnings) and (live_config.strict_live_parity or not live_config.is_testnet)


def validate_live_capabilities(
    *,
    strategy_configs: dict[str, Any] | None = None,
    capabilities: ExecutionCapabilities | None = None,
) -> list[ParityWarning]:
    """Validate known live strategy surfaces against adapter capabilities."""
    caps = capabilities or HyperliquidExecutionAdapter.capabilities
    warnings: list[ParityWarning] = []

    for strategy_id, strategy_config in (strategy_configs or {}).items():
        if not caps.ttl and _strategy_can_emit_ttl_order(strategy_id, strategy_config):
            warnings.append(ParityWarning(
                warning_id=f"{strategy_id}_ttl_orders_unsupported_live",
                severity="error",
                message=(
                    f"Strategy '{strategy_id}' can emit STOP entries with ttl_bars, "
                    "but the live adapter rejects TTL order intents."
                ),
                mitigation=(
                    "Use close/market entries for live, disable break-entry TTL paths, "
                    "or implement explicit live TTL emulation before enabling this config."
                ),
            ))

    return warnings


def validate_order_intent_capabilities(
    intent: OrderIntent,
    *,
    capabilities: ExecutionCapabilities | None = None,
) -> list[ParityWarning]:
    """Return adapter capability errors for a concrete order intent."""
    caps = capabilities or HyperliquidExecutionAdapter.capabilities
    details = {
        "stop_limit_not_supported_live": (
            "STOP_LIMIT orders are not supported by the live adapter.",
            "Use LIMIT/STOP alternatives or implement live stop-limit emulation.",
        ),
        "reduce_only_not_enforced_live": (
            "Reduce-only order intents are not enforced by the live adapter.",
            "Use exchange-native reduce-only support or explicit local emulation.",
        ),
        "oca_not_supported_live": (
            "OCA order groups are not supported by the live adapter.",
            "Use exchange-native OCA/OCO support or explicit local emulation.",
        ),
        "bracket_not_supported_live": (
            "Attached bracket order groups are not supported by the live adapter.",
            "Use exchange-native brackets or explicit local bracket management.",
        ),
        "ttl_not_supported_live": (
            "TTL order intents are not supported by the live adapter.",
            "Use non-TTL entries or implement explicit live TTL cancellation.",
        ),
    }
    return [
        ParityWarning(
            warning_id=reason,
            severity="error",
            message=message,
            mitigation=mitigation,
        )
        for reason in unsupported_order_intent_reasons(intent, caps)
        for message, mitigation in [details[reason]]
    ]


def _strategy_can_emit_ttl_order(strategy_id: str, strategy_config: Any) -> bool:
    entry = getattr(strategy_config, "entry", None)
    if entry is None:
        return False

    if strategy_id == "trend":
        mode = str(getattr(entry, "mode", "legacy") or "legacy").lower()
        stop_possible = (
            mode in {
                "break",
                "confirm_preferred",
                "hybrid_grade",
                "reentry_break",
                "reentry_confirm_preferred",
            }
            or (
                mode == "legacy"
                and bool(getattr(entry, "entry_on_break", False))
                and not bool(getattr(entry, "entry_on_close", False))
            )
        )
        return stop_possible and getattr(entry, "max_bars_after_confirmation", None) is not None

    if strategy_id == "momentum":
        mode = str(getattr(entry, "mode", "legacy") or "legacy").lower()
        stop_possible = (
            mode in {"break", "confirmation_specific"}
            or (
                mode == "legacy"
                and bool(getattr(entry, "entry_on_break", False))
                and not bool(getattr(entry, "entry_on_close", False))
            )
        )
        return stop_possible and getattr(entry, "max_bars_after_confirmation", None) is not None

    if strategy_id == "breakout":
        return (
            bool(getattr(entry, "model2_entry_on_break", False))
            and getattr(entry, "max_bars_after_signal", None) is not None
        )

    return False
