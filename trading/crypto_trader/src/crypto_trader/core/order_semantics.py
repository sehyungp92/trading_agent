"""Shared order tag semantics for live and simulated execution."""

from __future__ import annotations

from datetime import datetime

from crypto_trader.core.models import Order, Side

ENTRY_ORDER_TAGS = frozenset({"entry"})

EXIT_ORDER_TAGS = frozenset({
    "exit",
    "close",
    "tp",
    "stop",
    "protective_stop",
    "breakeven_stop",
    "proof_lock_stop",
    "trailing_stop",
    "tp1",
    "tp2",
    "time_stop",
    "soft_time_stop",
    "hard_time_stop",
    "quick_exit",
    "invalidation",
    "ema_failsafe",
    "scratch_exit",
    "mfe_lock_exit",
})

STOP_LOSS_TRIGGER_TAGS = frozenset({
    "stop",
    "protective_stop",
    "breakeven_stop",
    "proof_lock_stop",
    "trailing_stop",
})

EXIT_OCA_POLICY = "broker_managed_cancel_siblings_on_terminal_close"
NATIVE_OCA_POLICY = "native_cancel_siblings_on_terminal_close"

_OCA_ROLES_BY_TAG = {
    "stop": "stop_loss",
    "protective_stop": "stop_loss",
    "breakeven_stop": "stop_loss",
    "proof_lock_stop": "stop_loss",
    "trailing_stop": "stop_loss",
    "tp": "take_profit",
    "tp1": "take_profit",
    "tp2": "take_profit",
    "time_stop": "time_exit",
    "soft_time_stop": "time_exit",
    "hard_time_stop": "time_exit",
    "invalidation": "invalidation",
    "close": "manual_close",
    "exit": "manual_close",
    "quick_exit": "invalidation",
    "scratch_exit": "invalidation",
    "mfe_lock_exit": "take_profit",
    "ema_failsafe": "invalidation",
}


def is_entry_order(order: Order) -> bool:
    return order.tag in ENTRY_ORDER_TAGS


def is_exit_order(order: Order) -> bool:
    return order.tag in EXIT_ORDER_TAGS


def exit_oca_group(
    strategy_id: str,
    symbol: str,
    *,
    position_instance_id: str | None = None,
    entry_root_id: str | None = None,
) -> str | None:
    """Return the deterministic local OCA id for sibling exits.

    The seed intentionally uses a stable position/entry root. Per-exit ids are
    rejected by callers because they create one group per sibling.
    """
    strategy_id = str(strategy_id or "").strip()
    symbol = str(symbol or "").strip()
    root = str(position_instance_id or entry_root_id or "").strip()
    if not strategy_id or not symbol or not root:
        return None
    return f"{strategy_id}:{symbol}:{root}:exit_oca"


def entry_position_instance_id(
    strategy_id: str,
    symbol: str,
    side: Side | str,
    timestamp: datetime,
) -> str:
    direction = side.value if isinstance(side, Side) else str(side)
    return f"{strategy_id}:{symbol}:{direction}:{int(timestamp.timestamp() * 1000)}"


def is_exit_oca_member(order: Order) -> bool:
    return is_exit_order(order) and bool(order.oca_group or order.metadata.get("oca_group"))


def oca_role_for_order(order: Order) -> str:
    return _OCA_ROLES_BY_TAG.get(order.tag, "manual_close")


def validate_strategy_scoped_oca_group(
    oca_group: str | None,
    *,
    strategy_id: str,
    symbol: str,
) -> str:
    """Return an empty string when an explicit OCA group is strategy scoped."""
    group = str(oca_group or "").strip()
    if not group:
        return "oca_group_empty"
    prefix = f"{strategy_id}:{symbol}:"
    if not group.startswith(prefix):
        return "oca_group_not_strategy_symbol_scoped"
    if not group.endswith(":exit_oca"):
        return "oca_group_missing_exit_suffix"
    root = group.removeprefix(prefix).removesuffix(":exit_oca")
    if not root:
        return "oca_group_missing_stable_root"
    if root.startswith("intent:") or root.startswith("client_order:") or root.startswith("order:"):
        return "oca_group_uses_per_exit_root"
    return ""


def stamp_exit_order_oca(
    order: Order,
    *,
    strategy_id: str,
    position_instance_id: str | None = None,
    entry_root_id: str | None = None,
    native_oca_required: bool = False,
) -> str | None:
    """Stamp reduce-only and local OCA metadata on an exit order.

    Hyperliquid currently has no adapter-verified native OCA support in this
    codebase, so the default policy is explicitly broker-managed fallback. The
    top-level field and metadata are kept in sync for canonical intent and OMS
    consistency.
    """
    if not is_exit_order(order):
        return None

    order.metadata["reduce_only"] = True
    order.metadata["exit_only"] = True

    known_position_instance_id = str(
        position_instance_id
        or order.metadata.get("position_instance_id")
        or ""
    ).strip()
    root = str(
        known_position_instance_id
        or entry_root_id
        or order.metadata.get("entry_root_id")
        or order.metadata.get("entry_intent_id")
        or ""
    ).strip()
    explicit = str(order.oca_group or order.metadata.get("oca_group") or "").strip()
    if explicit:
        invalid_reason = validate_strategy_scoped_oca_group(
            explicit,
            strategy_id=strategy_id,
            symbol=order.symbol,
        )
        if invalid_reason:
            order.metadata["oca_group_invalid_reason"] = invalid_reason
            return None
        group = explicit
    else:
        group = exit_oca_group(strategy_id, order.symbol, entry_root_id=root)
        if group is None:
            order.metadata["oca_group_unstamped_reason"] = "missing_stable_position_or_entry_root"
            return None

    order.oca_group = group
    order.metadata["oca_group"] = group
    order.metadata.setdefault("oca_role", oca_role_for_order(order))
    order.metadata.setdefault("oca_policy", EXIT_OCA_POLICY)
    order.metadata.setdefault("native_oca_required", bool(native_oca_required))
    order.metadata.setdefault("oca_root", root)
    if known_position_instance_id:
        order.metadata.setdefault("position_instance_id", known_position_instance_id)
    return group
