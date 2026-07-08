"""Portfolio coordinator — BrokerProxy and multi-strategy orchestration."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable

import structlog

from crypto_trader.core.broker import BrokerAdapter
from crypto_trader.core.models import Bar, Fill, Order, OrderStatus, Position, Side
from crypto_trader.core.order_semantics import (
    entry_position_instance_id,
    is_entry_order,
    is_exit_order,
    stamp_exit_order_oca,
)
from crypto_trader.instrumentation.lineage import (
    ALLOCATION_CONFIG_KEYS,
    RISK_CONFIG_KEYS,
    stable_hash,
    subset_keys,
)
from crypto_trader.portfolio.manager import PortfolioManager

log = structlog.get_logger()


class BrokerProxy:
    """Wraps a real broker, intercepting entry orders for portfolio approval.

    - Entry orders (tag="entry"): check with PortfolioManager → approved? forward
      with size_multiplier applied : reject the order.
    - Exit/stop/TP orders: pass through unconditionally.
    - All other BrokerAdapter methods delegate directly.

    Implements the BrokerAdapter protocol so strategies see a uniform interface.
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        manager: PortfolioManager,
        strategy_id: str,
        coordinator: "StrategyCoordinator | None" = None,
        use_manager_equity: bool = False,
        event_callback: Callable[[str, dict], None] | None = None,
    ) -> None:
        self._broker = broker
        self._manager = manager
        self.strategy_id = strategy_id
        self._coordinator = coordinator
        self._use_manager_equity = use_manager_equity
        self._event_callback = event_callback
        self._decision_context = None
        self._broker_id_by_client_id: dict[str, str] = {}
        self._client_id_by_broker_id: dict[str, str] = {}

    def begin_decision_context(self, context) -> None:
        self._decision_context = context
        begin_fn = getattr(self._broker, "begin_decision_context", None)
        if callable(begin_fn):
            begin_fn(context)

    def end_decision_context(self, context) -> None:
        end_fn = getattr(self._broker, "end_decision_context", None)
        if callable(end_fn):
            end_fn(context)
        if self._decision_context is context:
            self._decision_context = None

    def submit_order(self, order: Order) -> str:
        """Submit an order, intercepting entries for portfolio approval."""
        # Stamp every order, including stops/targets, so live and backtest
        # adapters can keep order visibility and fill routing strategy-scoped.
        client_order_id = order.order_id
        order.metadata["strategy_id"] = self.strategy_id
        if client_order_id:
            order.metadata.setdefault("client_order_id", client_order_id)
        decision_context = self._decision_context
        if decision_context is not None:
            order.metadata.setdefault("decision_id", getattr(decision_context, "decision_id", ""))
            metadata = getattr(decision_context, "metadata", {})
            if isinstance(metadata, dict):
                order.metadata.setdefault("bar_id", metadata.get("bar_id", ""))
            decision_time = getattr(decision_context, "decision_time", None)
            if hasattr(decision_time, "isoformat"):
                decision_time = decision_time.isoformat()
            order.metadata.setdefault("decision_time", decision_time)

        order.metadata.setdefault("order_qty", order.qty)
        if is_exit_order(order):
            stamp_exit_order_oca(
                order,
                strategy_id=self.strategy_id,
                position_instance_id=self._stable_exit_position_instance_id(order),
                entry_root_id=self._stable_exit_root(order),
                native_oca_required=False,
            )
            invalid_oca_reason = str(order.metadata.get("oca_group_invalid_reason") or "")
            if invalid_oca_reason:
                order.status = OrderStatus.REJECTED
                canonical_recorded = self._record_order_contract_rejection(
                    order,
                    reject_reason=invalid_oca_reason,
                    metadata={"rejection_stage": "order_semantics"},
                )
                if decision_context is not None and not canonical_recorded:
                    record_order = getattr(decision_context, "record_order", None)
                    if callable(record_order):
                        record_order()
                return order.order_id

        if is_entry_order(order):
            order.metadata.setdefault(
                "intent_id",
                self._preview_intent_id(order, decision_context),
            )
            direction = order.side
            risk_R = order.metadata.get("risk_R", 1.0)

            result = self._manager.check_entry(
                strategy_id=self.strategy_id,
                symbol=order.symbol,
                direction=direction,
                new_risk_R=risk_R,
            )
            portfolio_rule_event_id = self._contextual_decision_event_id(
                "portfolio_rule",
                order,
                result.rule_event_id,
            )
            risk_decision_id = self._contextual_decision_event_id(
                "risk_decision",
                order,
                result.risk_decision_id,
            )
            order.metadata["portfolio_rule_event_id"] = portfolio_rule_event_id
            order.metadata["risk_decision_id"] = risk_decision_id
            order.metadata["rule_evaluation_id"] = result.rule_event_id
            self._emit_portfolio_rule_event(
                order,
                result,
                portfolio_rule_event_id=portfolio_rule_event_id,
                risk_decision_id=risk_decision_id,
            )

            if not result.approved:
                log.info(
                    "portfolio.entry_blocked",
                    strategy=self.strategy_id,
                    symbol=order.symbol,
                    reason=result.denial_reason,
                )
                order.status = OrderStatus.REJECTED
                canonical_recorded = self._record_portfolio_rejection(
                    order,
                    result,
                    portfolio_rule_event_id=portfolio_rule_event_id,
                    risk_decision_id=risk_decision_id,
                )
                if decision_context is not None and not canonical_recorded:
                    record_order = getattr(decision_context, "record_order", None)
                    if callable(record_order):
                        record_order()
                self._emit_risk_decision_event(
                    order,
                    result,
                    original_risk_R=risk_R,
                    portfolio_rule_event_id=portfolio_rule_event_id,
                    risk_decision_id=risk_decision_id,
                )
                self._emit_order_rejected_by_portfolio(
                    order,
                    result,
                    portfolio_rule_event_id=portfolio_rule_event_id,
                    risk_decision_id=risk_decision_id,
                )
                return order.order_id

            # Apply size multiplier from drawdown tiers
            if result.size_multiplier != 1.0:
                order.metadata.setdefault("original_qty", order.qty)
                order.qty = order.qty * result.size_multiplier
                order.metadata["risk_R"] = risk_R * result.size_multiplier
                order.metadata["order_qty"] = order.qty
                order.metadata["portfolio_size_multiplier"] = result.size_multiplier
                log.debug(
                    "portfolio.size_adjusted",
                    strategy=self.strategy_id,
                    multiplier=result.size_multiplier,
                    new_qty=order.qty,
                )
            self._emit_risk_decision_event(
                order,
                result,
                original_risk_R=risk_R,
                portfolio_rule_event_id=portfolio_rule_event_id,
                risk_decision_id=risk_decision_id,
            )

        result_id = self._broker.submit_order(order)
        visible_order_id = client_order_id or result_id

        # Register order ownership for fill routing (works with any broker)
        if order.status != OrderStatus.REJECTED and self._coordinator is not None:
            for tracking_id in self._tracking_order_ids(result_id, visible_order_id, order):
                self._coordinator.register_order(tracking_id, self.strategy_id, order)

        if (
            order.status != OrderStatus.REJECTED
            and client_order_id
            and result_id
            and client_order_id != result_id
        ):
            self._broker_id_by_client_id[client_order_id] = result_id
            self._client_id_by_broker_id[result_id] = client_order_id

        return visible_order_id

    def _record_portfolio_rejection(
        self,
        order: Order,
        result,
        *,
        portfolio_rule_event_id: str,
        risk_decision_id: str,
    ) -> bool:
        if getattr(type(self._broker), "record_rejected_order", None) is None:
            return False
        record_fn = getattr(self._broker, "record_rejected_order", None)
        if not callable(record_fn):
            return False
        try:
            record_fn(
                order,
                reject_reason=result.denial_reason or "portfolio_rule_rejected",
                metadata={
                    "rejection_stage": "portfolio_rule",
                    "blocking_rule": result.blocking_rule,
                    "portfolio_rule_event_id": portfolio_rule_event_id,
                    "risk_decision_id": risk_decision_id,
                    "rule_evaluation_id": result.rule_event_id,
                },
            )
            return True
        except Exception:
            log.exception("portfolio.canonical_rejection_record_failed")
            return False

    def _record_order_contract_rejection(
        self,
        order: Order,
        *,
        reject_reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        record_fn = getattr(self._broker, "record_rejected_order", None)
        if not callable(record_fn):
            return False
        try:
            record_fn(
                order,
                reject_reason=reject_reason,
                metadata={
                    "strategy_id": self.strategy_id,
                    **dict(metadata or {}),
                },
            )
            return True
        except Exception:
            log.exception("portfolio.contract_rejection_record_failed")
            return False

    def _stable_exit_root(self, order: Order) -> str:
        for key in ("position_instance_id", "entry_root_id", "entry_intent_id"):
            value = str(order.metadata.get(key) or "")
            if value:
                return value
        for risk in getattr(self._manager.state, "open_risks", []):
            if risk.strategy_id != self.strategy_id or risk.symbol != order.symbol:
                continue
            for value in (
                risk.position_instance_id,
                risk.risk_id,
                risk.intent_id,
                risk.client_order_id,
                risk.order_id,
                risk.exchange_order_id,
            ):
                if value:
                    return str(value)
        return ""

    def _stable_exit_position_instance_id(self, order: Order) -> str:
        value = str(order.metadata.get("position_instance_id") or "")
        if value:
            return value
        for risk in getattr(self._manager.state, "open_risks", []):
            if risk.strategy_id == self.strategy_id and risk.symbol == order.symbol:
                return str(risk.position_instance_id or "")
        return ""

    def _emit_portfolio_rule_event(
        self,
        order: Order,
        result,
        *,
        portfolio_rule_event_id: str,
        risk_decision_id: str,
    ) -> None:
        portfolio_config = self._portfolio_config_payload()
        config_versions = self._portfolio_config_versions(portfolio_config)
        payload = {
            "event_type": "portfolio_rule",
            "portfolio_rule_event_id": portfolio_rule_event_id,
            "rule_event_id": portfolio_rule_event_id,
            "risk_decision_id": risk_decision_id,
            "rule_evaluation_id": result.rule_event_id,
            "strategy_id": self.strategy_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "direction": order.side.value,
            "decision_id": order.metadata.get("decision_id", ""),
            "bar_id": order.metadata.get("bar_id", ""),
            "intent_id": order.metadata.get("intent_id", ""),
            "client_order_id": order.metadata.get("client_order_id", order.order_id),
            "requested_risk_R": result.request.get("requested_risk_R"),
            "approved": result.approved,
            "action": self._risk_action(result),
            "denial_reason": result.denial_reason,
            "blocking_rule": result.blocking_rule,
            "size_multiplier": result.size_multiplier,
            "adjusted_risk_R": (result.request.get("requested_risk_R") or 0.0) * result.size_multiplier if result.approved else 0.0,
            "rule_evaluations": list(result.rule_evaluations),
            "evaluations": list(result.rule_evaluations),
            "state_before": dict(result.state_before),
            "state_after_preview": dict(result.state_after_preview),
            "allocation": result.allocation,
            "request": dict(result.request),
            "portfolio_config": portfolio_config,
            **config_versions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._emit_event("portfolio_rule", payload)

    def _emit_risk_decision_event(
        self,
        order: Order,
        result,
        *,
        original_risk_R: float,
        portfolio_rule_event_id: str,
        risk_decision_id: str,
    ) -> None:
        payload = {
            "risk_decision_id": risk_decision_id,
            "portfolio_rule_event_id": portfolio_rule_event_id,
            "rule_evaluation_id": result.rule_event_id,
            "strategy_id": self.strategy_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "direction": order.side.value,
            "decision_id": order.metadata.get("decision_id", ""),
            "bar_id": order.metadata.get("bar_id", ""),
            "intent_id": order.metadata.get("intent_id", ""),
            "order_id": order.metadata.get("client_order_id", order.order_id),
            "client_order_id": order.metadata.get("client_order_id", order.order_id),
            "approved": result.approved,
            "action": self._risk_action(result),
            "reason": result.denial_reason or "",
            "original_risk_R": original_risk_R,
            "effective_risk_R": order.metadata.get("risk_R", original_risk_R),
            "requested_risk_R": original_risk_R,
            "approved_risk_R": order.metadata.get("risk_R", original_risk_R) if result.approved else 0.0,
            "original_qty": order.metadata.get("original_qty"),
            "effective_qty": order.qty,
            "requested_qty": order.metadata.get("original_qty", order.qty),
            "approved_qty": order.qty if result.approved else 0.0,
            "size_multiplier": result.size_multiplier,
            "rule_event_id": portfolio_rule_event_id,
            "portfolio_state_before": dict(result.state_before),
            "portfolio_state_after_preview": dict(result.state_after_preview),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._emit_event("risk_decision", payload)

    @staticmethod
    def _risk_action(result) -> str:
        if not result.approved:
            return "block"
        if result.size_multiplier != 1.0:
            return "scale"
        return "allow"

    def _emit_order_rejected_by_portfolio(
        self,
        order: Order,
        result,
        *,
        portfolio_rule_event_id: str,
        risk_decision_id: str,
    ) -> None:
        payload = {
            "order_event_id": stable_hash({
                "kind": "portfolio_reject",
                "order_id": order.order_id,
                "portfolio_rule_event_id": portfolio_rule_event_id,
                "decision_id": order.metadata.get("decision_id", ""),
                "intent_id": order.metadata.get("intent_id", ""),
            }),
            "event_kind": "rejected",
            "rejection_stage": "portfolio_rule",
            "strategy_id": self.strategy_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "client_order_id": order.metadata.get("client_order_id", order.order_id),
            "decision_id": order.metadata.get("decision_id", ""),
            "bar_id": order.metadata.get("bar_id", ""),
            "intent_id": order.metadata.get("intent_id", ""),
            "portfolio_rule_event_id": portfolio_rule_event_id,
            "risk_decision_id": risk_decision_id,
            "reject_reason": result.denial_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._emit_event("order", payload)

    def _preview_intent_id(self, order: Order, decision_context: object | None) -> str:
        existing = str(order.metadata.get("intent_id") or "")
        if existing:
            return existing
        decision_id = str(
            order.metadata.get("decision_id")
            or getattr(decision_context, "decision_id", "")
            or "manual"
        )
        if decision_context is not None:
            seq = int(getattr(decision_context, "order_count", 0) or 0) + 1
            return f"{self.strategy_id}:{order.symbol}:{decision_id}:intent:{seq}"
        client_order_id = str(order.metadata.get("client_order_id") or order.order_id or "")
        if client_order_id:
            return client_order_id
        return f"{self.strategy_id}:{order.symbol}:{decision_id}:intent:1"

    @staticmethod
    def _contextual_decision_event_id(kind: str, order: Order, evaluation_id: str) -> str:
        seed = {
            "kind": kind,
            "evaluation_id": evaluation_id,
            "decision_id": order.metadata.get("decision_id", ""),
            "bar_id": order.metadata.get("bar_id", ""),
            "intent_id": order.metadata.get("intent_id", ""),
            "client_order_id": order.metadata.get("client_order_id", order.order_id),
        }
        return stable_hash(seed)

    def _portfolio_config_payload(self) -> dict:
        config = getattr(self._manager, "config", None)
        if config is None:
            return {}
        to_dict = getattr(config, "to_dict", None)
        return to_dict() if callable(to_dict) else dict(getattr(config, "__dict__", {}))

    @staticmethod
    def _portfolio_config_versions(portfolio_config: dict) -> dict[str, str]:
        return {
            "portfolio_config_version": stable_hash(portfolio_config),
            "risk_config_version": stable_hash(subset_keys(portfolio_config, RISK_CONFIG_KEYS)),
            "allocation_version": stable_hash(subset_keys(portfolio_config, ALLOCATION_CONFIG_KEYS)),
        }

    def _emit_event(self, event_type: str, payload: dict) -> None:
        if self._event_callback is None:
            return
        try:
            self._event_callback(event_type, payload)
        except Exception:
            log.exception("portfolio.event_callback_failed", event_type=event_type)

    def _tracking_order_ids(self, result_id: str, visible_order_id: str, order: Order) -> list[str]:
        ids = [
            result_id,
            visible_order_id,
            order.metadata.get("exchange_order_id"),
            order.metadata.get("broker_order_id"),
            order.metadata.get("client_order_id"),
        ]
        local_to_oid = getattr(self._broker, "_local_to_oid", None)
        if isinstance(local_to_oid, dict) and (exchange_oid := local_to_oid.get(result_id)):
            ids.append(str(exchange_oid))
        return list(dict.fromkeys(str(oid) for oid in ids if oid))

    def cancel_order(self, order_id: str) -> bool:
        broker_order_id = self._broker_id_by_client_id.get(order_id, order_id)
        return self._broker.cancel_order(broker_order_id)

    def cancel_all(self, symbol: str = "") -> int:
        cancelled = 0
        for order in self.get_open_orders(symbol):
            if self.cancel_order(order.order_id):
                cancelled += 1
        return cancelled

    def expire_ttl_orders_for_bar(self, bar: Bar) -> list:
        expire_fn = getattr(self._broker, "expire_ttl_orders_for_bar", None)
        if callable(expire_fn):
            return expire_fn(bar)
        return []

    def drain_immediate_fill_syncs(self) -> None:
        drain = getattr(self._broker, "drain_immediate_fill_syncs", None)
        if callable(drain):
            drain()

    def get_position(self, symbol: str) -> Position | None:
        return self._broker.get_position(symbol)

    def get_positions(self) -> list[Position]:
        return self._broker.get_positions()

    def get_open_orders(self, symbol: str = "") -> list[Order]:
        orders: list[Order] = []
        for order in self._broker.get_open_orders(symbol):
            if not self._owns_order(order):
                continue
            orders.append(self._strategy_visible_order(order))
        return orders

    def get_equity(self) -> float:
        if self._use_manager_equity:
            return self._manager.state.equity
        return self._broker.get_equity()

    def get_fills_since(self, since: datetime) -> list[Fill]:
        return self._broker.get_fills_since(since)

    def get_portfolio_snapshot(self, symbol: str, direction: Side) -> dict[str, float | int]:
        """Capture a compact pre-entry portfolio snapshot for instrumentation."""
        state = self._manager.state
        return {
            "heat_R": state.total_heat_R(),
            "heat_cap_R": self._manager.config.heat_cap_R,
            "open_risk_count": state.total_positions(),
            "directional_risk_R": state.directional_risk_R(direction),
            "symbol_risk_R": state.symbol_risk_R(symbol, direction),
            "portfolio_daily_pnl_R": state.portfolio_daily_pnl_R,
            "strategy_daily_pnl_R": state.strategy_daily_pnl_R(self.strategy_id),
        }

    def _owns_order(self, order: Order) -> bool:
        owner = order.metadata.get("strategy_id")
        if owner is None and self._coordinator is not None:
            owner = self._coordinator.get_strategy_for_order(order.order_id)
        return owner == self.strategy_id

    def _strategy_visible_order(self, order: Order) -> Order:
        client_order_id = (
            order.metadata.get("client_order_id")
            or self._client_id_by_broker_id.get(order.order_id)
        )
        if not client_order_id or client_order_id == order.order_id:
            return order
        metadata = dict(order.metadata)
        metadata.setdefault("broker_order_id", order.order_id)
        return replace(order, order_id=str(client_order_id), metadata=metadata)

    # Delegate SimBroker-specific methods for backtest compatibility
    def __getattr__(self, name: str) -> Any:
        return getattr(self._broker, name)


class StrategyCoordinator:
    """Orchestrates multiple strategies sharing one broker + one portfolio manager.

    Creates BrokerProxy per strategy, tracks position book, routes fills.

    Fill routing: uses _order_owners dict (populated by BrokerProxy on submit).
    Entry registration: done on entry fills (tag="entry").
    Exit registration: done via on_trade_closed() when a PositionClosedEvent fires.
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        manager: PortfolioManager,
        event_callback: Callable[[str, dict], None] | None = None,
    ) -> None:
        self._broker = broker
        self._manager = manager
        self._event_callback = event_callback
        self._proxies: dict[str, BrokerProxy] = {}
        self._order_metadata: dict[str, dict] = {}
        self._order_owners: dict[str, str] = {}  # order_id → strategy_id

    def get_proxy(self, strategy_id: str, *, use_manager_equity: bool = False) -> BrokerProxy:
        """Get or create a BrokerProxy for a strategy."""
        if strategy_id not in self._proxies:
            self._proxies[strategy_id] = BrokerProxy(
                broker=self._broker,
                manager=self._manager,
                strategy_id=strategy_id,
                coordinator=self,
                use_manager_equity=use_manager_equity,
                event_callback=self._event_callback,
            )
        elif use_manager_equity:
            self._proxies[strategy_id]._use_manager_equity = True
        return self._proxies[strategy_id]

    def register_order(
        self,
        order_id: str,
        strategy_id: str,
        order: Order | None = None,
    ) -> None:
        """Track which strategy submitted an order."""
        self._order_owners[order_id] = strategy_id
        metadata = (
            dict(order.metadata)
            if order is not None
            else dict(self._order_metadata.get(order_id, {}))
        )
        metadata.setdefault("strategy_id", strategy_id)
        if order is not None:
            metadata.setdefault("order_qty", order.qty)
            metadata.setdefault("order_id", order.order_id)
            metadata.setdefault("client_order_id", order.metadata.get("client_order_id", order.order_id))
        self._order_metadata[order_id] = metadata

    def get_strategy_for_order(self, order_id: str) -> str | None:
        """Look up which strategy submitted an order."""
        # Primary: our own tracking (works with any broker)
        if order_id in self._order_owners:
            return self._order_owners[order_id]
        if order_id in self._order_metadata:
            owner = self._order_metadata[order_id].get("strategy_id")
            if owner:
                return str(owner)
        # Fallback: broker._orders (for HyperliquidBroker)
        all_orders = getattr(self._broker, '_orders', {})
        order = all_orders.get(order_id)
        if order and "strategy_id" in order.metadata:
            return order.metadata["strategy_id"]
        return None

    def on_fill(self, fill: Fill) -> str | None:
        """Route a fill to update portfolio state. Returns strategy_id or None.

        Only handles entry registration. Exit registration is handled by
        on_trade_closed() to avoid double-counting.
        """
        strategy_id = self.get_strategy_for_order(fill.order_id)
        if strategy_id is None:
            return None

        if fill.tag == "entry":
            metadata = self._order_metadata_for_fill(fill)
            risk_R = self._get_fill_risk_R(fill, metadata)
            order_qty = _float_or_default(
                metadata.get("order_qty") or metadata.get("original_qty"),
                0.0,
            )
            risk_id = self._fill_risk_id(fill, metadata)
            self._manager.register_entry(
                strategy_id=strategy_id,
                symbol=fill.symbol,
                direction=fill.side,
                risk_R=risk_R,
                entry_time=fill.timestamp,
                risk_id=risk_id,
                position_instance_id=str(
                    metadata.get("position_instance_id")
                    or fill.raw.get("position_instance_id")
                    or entry_position_instance_id(strategy_id, fill.symbol, fill.side, fill.timestamp)
                ),
                intent_id=str(metadata.get("intent_id") or ""),
                client_order_id=str(metadata.get("client_order_id") or ""),
                order_id=str(metadata.get("order_id") or fill.order_id),
                exchange_order_id=fill.exchange_order_id,
                order_qty=order_qty,
                fill_qty=fill.qty,
                fill_id=_fill_ledger_id(fill),
            )

        return strategy_id

    def on_trade_closed(
        self,
        strategy_id: str,
        symbol: str,
        pnl_R: float,
        *,
        trade: Any | None = None,
        risk_id: str = "",
        order_refs: set[str] | None = None,
    ) -> None:
        """Called when a complete trade (round-trip) closes."""
        refs = set(order_refs or set())
        refs.update(_trade_order_refs(trade))
        self._manager.register_exit(
            strategy_id=strategy_id,
            symbol=symbol,
            pnl_R=pnl_R,
            risk_id=risk_id,
            order_refs=refs or None,
        )

    def _get_fill_risk_R(self, fill: Fill, metadata: dict | None = None) -> float:
        """Extract risk_R from the order that generated a fill."""
        metadata = metadata if metadata is not None else self._order_metadata_for_fill(fill)
        risk_R = _float_or_default(metadata.get("risk_R"), 1.0)
        order_qty = _float_or_default(
            metadata.get("order_qty") or metadata.get("original_qty"),
            0.0,
        )
        if order_qty > 0:
            return round(risk_R * min(max(fill.qty, 0.0), order_qty) / order_qty, 12)
        return risk_R

    def _order_metadata_for_fill(self, fill: Fill) -> dict:
        """Return metadata for a fill's order id or exchange id."""
        for order_id in (fill.order_id, fill.exchange_order_id):
            if order_id and order_id in self._order_metadata:
                return self._order_metadata[order_id]
        if fill.order_id in self._order_metadata:
            return self._order_metadata[fill.order_id]
        # Try HyperliquidBroker's _orders dict
        all_orders = getattr(self._broker, '_orders', {})
        order = all_orders.get(fill.order_id)
        if order:
            metadata = dict(order.metadata)
            metadata.setdefault("order_qty", order.qty)
            metadata.setdefault("order_id", order.order_id)
            return metadata
        # Try SimBroker's pending/deferred orders
        for lst_name in ('_pending_orders', '_deferred_orders'):
            for o in getattr(self._broker, lst_name, []):
                if o.order_id == fill.order_id:
                    metadata = dict(o.metadata)
                    metadata.setdefault("order_qty", o.qty)
                    metadata.setdefault("order_id", o.order_id)
                    return metadata
        return {}

    def _fill_risk_id(self, fill: Fill, metadata: dict) -> str:
        for key in ("position_instance_id", "intent_id", "client_order_id", "order_id"):
            value = str(metadata.get(key) or "")
            if value:
                return value
        return fill.order_id or fill.exchange_order_id


def _float_or_default(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _fill_ledger_id(fill: Fill) -> str:
    if fill.exchange_fill_id:
        return fill.exchange_fill_id
    return "|".join(
        str(value)
        for value in (
            fill.order_id,
            fill.exchange_order_id,
            fill.symbol,
            fill.qty,
            fill.fill_price,
            fill.timestamp.isoformat(),
        )
        if value
    )


def _trade_order_refs(trade: Any | None) -> set[str]:
    context = getattr(trade, "instrumentation_context", None)
    if not isinstance(context, dict):
        return set()
    refs: set[str] = set()
    for key in ("entry_order_ids", "client_order_ids", "exchange_order_ids"):
        raw = context.get(key)
        if isinstance(raw, (list, tuple, set)):
            refs.update(str(value) for value in raw if value)
        elif raw:
            refs.add(str(raw))
    return refs
