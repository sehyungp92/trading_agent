"""Shared structural plugin registry for monthly execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

CRYPTO_TREND_PLUGIN_ID = "crypto-trend-v1"
CRYPTO_TREND_DECISION_API_VERSION = "crypto_trader_trend_decision_api_v1"
CRYPTO_MOMENTUM_PLUGIN_ID = "crypto-momentum-v1"
CRYPTO_MOMENTUM_DECISION_API_VERSION = "crypto_trader_momentum_decision_api_v1"
CRYPTO_BREAKOUT_PLUGIN_ID = "crypto-breakout-v1"
CRYPTO_BREAKOUT_DECISION_API_VERSION = "crypto_trader_breakout_decision_api_v1"
K_STOCK_PLUGIN_ID = "k-stock-olr-kalcb"
K_STOCK_DECISION_API_VERSION = "k_stock_olr_kalcb_artifact_replay_decision_api_v1"
TRADING_STOCK_PLUGIN_ID = "trading-stock-family"
TRADING_STOCK_DECISION_API_VERSION = "trading_stock_live_shadow_decision_api_v1"
TRADING_MOMENTUM_PLUGIN_ID = "trading-momentum-family"
TRADING_MOMENTUM_DECISION_API_VERSION = "trading_momentum_live_shadow_decision_api_v1"
TRADING_SWING_PLUGIN_ID = "trading-swing-family"
TRADING_SWING_DECISION_API_VERSION = "trading_swing_live_shadow_decision_api_v1"

StructuralParityBuilder = Callable[..., Any]


def _crypto_trend_builder(*args: Any, **kwargs: Any) -> Any:
    from trading_assistant_backtest.strategies.crypto.trend import (
        build_crypto_trend_decision_parity_report,
    )

    return build_crypto_trend_decision_parity_report(*args, **kwargs)


def _crypto_momentum_builder(*args: Any, **kwargs: Any) -> Any:
    from trading_assistant_backtest.strategies.crypto.momentum import (
        build_crypto_momentum_decision_parity_report,
    )

    return build_crypto_momentum_decision_parity_report(*args, **kwargs)


def _crypto_breakout_builder(*args: Any, **kwargs: Any) -> Any:
    from trading_assistant_backtest.strategies.crypto.breakout import (
        build_crypto_breakout_decision_parity_report,
    )

    return build_crypto_breakout_decision_parity_report(*args, **kwargs)


def _k_stock_builder(*args: Any, **kwargs: Any) -> Any:
    from trading_assistant_backtest.strategies.krx.olr_kalcb import (
        build_k_stock_olr_kalcb_decision_parity_report,
    )

    return build_k_stock_olr_kalcb_decision_parity_report(*args, **kwargs)


def _trading_stock_builder(*args: Any, **kwargs: Any) -> Any:
    from trading_assistant_backtest.strategies.trading.stock import (
        build_trading_stock_decision_parity_report,
    )

    return build_trading_stock_decision_parity_report(*args, **kwargs)


def _trading_momentum_builder(*args: Any, **kwargs: Any) -> Any:
    from trading_assistant_backtest.strategies.trading.momentum import (
        build_trading_momentum_decision_parity_report,
    )

    return build_trading_momentum_decision_parity_report(*args, **kwargs)


def _trading_swing_builder(*args: Any, **kwargs: Any) -> Any:
    from trading_assistant_backtest.strategies.trading.swing import (
        build_trading_swing_decision_parity_report,
    )

    return build_trading_swing_decision_parity_report(*args, **kwargs)


STRUCTURAL_PARITY_BUILDERS: dict[str, tuple[str, StructuralParityBuilder]] = {
    CRYPTO_TREND_PLUGIN_ID: (CRYPTO_TREND_DECISION_API_VERSION, _crypto_trend_builder),
    CRYPTO_MOMENTUM_PLUGIN_ID: (
        CRYPTO_MOMENTUM_DECISION_API_VERSION,
        _crypto_momentum_builder,
    ),
    CRYPTO_BREAKOUT_PLUGIN_ID: (
        CRYPTO_BREAKOUT_DECISION_API_VERSION,
        _crypto_breakout_builder,
    ),
    K_STOCK_PLUGIN_ID: (K_STOCK_DECISION_API_VERSION, _k_stock_builder),
    TRADING_STOCK_PLUGIN_ID: (
        TRADING_STOCK_DECISION_API_VERSION,
        _trading_stock_builder,
    ),
    TRADING_MOMENTUM_PLUGIN_ID: (
        TRADING_MOMENTUM_DECISION_API_VERSION,
        _trading_momentum_builder,
    ),
    TRADING_SWING_PLUGIN_ID: (
        TRADING_SWING_DECISION_API_VERSION,
        _trading_swing_builder,
    ),
}

CRYPTO_PLUGIN_IDS = frozenset(
    {
        CRYPTO_TREND_PLUGIN_ID,
        CRYPTO_MOMENTUM_PLUGIN_ID,
        CRYPTO_BREAKOUT_PLUGIN_ID,
    }
)

BRIDGE_IDS_BY_SCOPE = {
    "crypto_trader_portfolio": (
        "crypto_trend_v1",
        "crypto_momentum_v1",
        "crypto_breakout_v1",
    ),
    "k_stock_olr_kalcb": ("k_stock_olr_kalcb",),
    "trading_stock_family": ("trading_stock_family",),
    "trading_momentum_family": ("trading_momentum_family",),
    "trading_swing_family": ("trading_swing_family",),
}

BRIDGE_ID_BY_PLUGIN_ID = {
    CRYPTO_TREND_PLUGIN_ID: "crypto_trend_v1",
    CRYPTO_MOMENTUM_PLUGIN_ID: "crypto_momentum_v1",
    CRYPTO_BREAKOUT_PLUGIN_ID: "crypto_breakout_v1",
    K_STOCK_PLUGIN_ID: "k_stock_olr_kalcb",
    TRADING_STOCK_PLUGIN_ID: "trading_stock_family",
    TRADING_MOMENTUM_PLUGIN_ID: "trading_momentum_family",
    TRADING_SWING_PLUGIN_ID: "trading_swing_family",
}


def bridge_ids_for_scope(scope_id: str) -> tuple[str, ...]:
    return BRIDGE_IDS_BY_SCOPE.get(scope_id, ())


def bridge_id_for_plugin(plugin_id: str, fallback: str) -> str:
    return BRIDGE_ID_BY_PLUGIN_ID.get(plugin_id, fallback)


def scope_id_for_plugin(plugin_id: str, fallback: str) -> str:
    if plugin_id in CRYPTO_PLUGIN_IDS:
        return "crypto_trader_portfolio"
    return BRIDGE_ID_BY_PLUGIN_ID.get(plugin_id, fallback)
