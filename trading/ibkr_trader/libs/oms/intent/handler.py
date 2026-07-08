"""Intent handler for processing strategy requests."""
import asyncio
import uuid
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from libs.market_data.futures_roll import (
    roll_blackout_reason,
    with_contract_expiry_for_order,
)
from libs.instrumentation.lineage import stable_hash
from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import lineage_to_payload

from ..models.intent import Intent, IntentType, IntentReceipt, IntentResult, PreapprovedFamilyDecision
from ..models.order import OMSOrder, OrderRole, OrderStatus
from ..engine.state_machine import transition

if TYPE_CHECKING:
    from ..risk.gateway import RiskGateway
    from ..execution.router import ExecutionRouter
    from ..persistence.repository import OMSRepository
    from ..events.bus import EventBus

logger = logging.getLogger(__name__)

_MAX_IDEMP_CACHE = 5000
_IDEMP_PRUNE_BATCH = 1000


class IntentHandler:
    """Processes strategy intents. Validates, risk-checks, routes."""

    def __init__(
        self,
        risk: "RiskGateway",
        router: "ExecutionRouter",
        repo: "OMSRepository",
        bus: "EventBus",
        default_account_id: str = "",
    ):
        self._risk = risk
        self._router = router
        self._repo = repo
        self._bus = bus
        # OMS-7: configured IB account injected by the factory. When a strategy's
        # OMSOrder has account_id="" (the default for swing/momentum builders
        # don't set it), the handler stamps this value before persistence so
        # DB attribution and reconciliation can keyed-by-account.
        self._default_account_id = default_account_id
        self._idempotency: OrderedDict[str, str] = OrderedDict()  # client_order_id -> oms_order_id
        # C1: per-client_order_id locks to prevent duplicate orders across concurrent tasks
        self._idemp_locks: dict[str, asyncio.Lock] = {}
        # 1C: serialize entry risk-check to persist to prevent concurrent race in shared OMS
        self._entry_lock = asyncio.Lock()

    def _prune_idemp_cache(self) -> None:
        """Evict oldest entries when cache exceeds max size.

        DB fallback at get_order_id_by_client_order_id handles cache misses.
        """
        if len(self._idempotency) <= _MAX_IDEMP_CACHE:
            return
        for _ in range(_IDEMP_PRUNE_BATCH):
            if not self._idempotency:
                break
            key, _ = self._idempotency.popitem(last=False)
            self._idemp_locks.pop(key, None)

    def _current_oms_lineage(self, strategy_id: str = ""):
        provider = getattr(self._risk, "_current_oms_lineage", None)
        if callable(provider):
            try:
                if strategy_id:
                    return provider(strategy_id)
                return provider()
            except TypeError:
                try:
                    return provider()
                except Exception:
                    logger.debug("OMS lineage provider failed", exc_info=True)
                    return None
            except Exception:
                logger.debug("OMS lineage provider failed", exc_info=True)
                return None
        return getattr(self._risk, "_oms_lineage", None)

    def _stamp_order_correlation(self, order: OMSOrder, intent_id: str) -> None:
        risk_ctx = getattr(order, "risk_context", None)
        if risk_ctx is None:
            return
        risk_ctx.intent_id = intent_id
        if not getattr(risk_ctx, "trace_id", ""):
            risk_ctx.trace_id = stable_hash(
                "trace_",
                {
                    "intent_id": intent_id,
                    "client_order_id": getattr(order, "client_order_id", ""),
                    "oms_order_id": getattr(order, "oms_order_id", ""),
                    "strategy_id": getattr(order, "strategy_id", ""),
                },
            )
        if not getattr(risk_ctx, "signal_id", ""):
            risk_ctx.signal_id = (
                getattr(order, "client_order_id", "")
                or getattr(risk_ctx, "risk_budget_tag", "")
                or intent_id
            )
        if not getattr(risk_ctx, "exchange_timestamp", None):
            risk_ctx.exchange_timestamp = (
                getattr(order, "created_at", None)
                or getattr(order, "submitted_at", None)
                or datetime.now(timezone.utc)
            )
        if not getattr(risk_ctx, "lineage_context", None):
            risk_ctx.lineage_context = lineage_to_payload(
                self._current_oms_lineage(getattr(order, "strategy_id", ""))
            )

    def _enrich_risk_decision_payload(self, payload: dict) -> dict:
        strategy_id = str(payload.get("strategy_id") or "")
        return enrich_payload(
            payload,
            lineage=self._current_oms_lineage(strategy_id) or payload.get("lineage"),
            event_type="risk_decision",
            scope="oms",
        )

    async def submit(self, intent: Intent) -> IntentReceipt:
        intent_id = str(uuid.uuid4())

        if intent.intent_type == IntentType.NEW_ORDER:
            return await self._handle_new_order(intent, intent_id)
        elif intent.intent_type == IntentType.PREAPPROVED_ORDER:
            return await self._handle_new_order(intent, intent_id, preapproved=True)
        elif intent.intent_type == IntentType.CANCEL_ORDER:
            return await self._handle_cancel(intent, intent_id)
        elif intent.intent_type == IntentType.REPLACE_ORDER:
            return await self._handle_replace(intent, intent_id)
        elif intent.intent_type == IntentType.FLATTEN:
            return await self._handle_flatten(intent, intent_id)
        else:
            self._emit_intent_risk_decision(
                intent,
                intent_id,
                decision="deny",
                reason="Unknown intent type",
            )
            return IntentReceipt(
                IntentResult.DENIED, intent_id, denial_reason="Unknown intent type"
            )

    def _emit_intent_risk_decision(
        self,
        intent: Intent,
        intent_id: str,
        *,
        decision: str,
        reason: str = "",
        order: OMSOrder | None = None,
        preapproved: bool = False,
    ) -> dict:
        order = order or intent.order
        risk_ctx = getattr(order, "risk_context", None) if order is not None else None
        instrument = getattr(order, "instrument", None) if order is not None else None
        strategy_id = (
            getattr(order, "strategy_id", "")
            if order is not None
            else getattr(intent, "strategy_id", "")
        )
        oms_order_id = getattr(order, "oms_order_id", "") if order is not None else ""
        side = getattr(getattr(order, "side", None), "value", "") if order is not None else ""
        role = getattr(getattr(order, "role", None), "value", "") if order is not None else ""
        order_type = getattr(getattr(order, "order_type", None), "value", "") if order is not None else ""
        requested_qty = int(getattr(order, "qty", 0) or 0) if order is not None else 0
        gateway_context = dict(getattr(risk_ctx, "gateway_decision_context", {}) or {})
        approved = decision in {"approve", "scale", "route"}
        payload = {
            **gateway_context,
            "intent_id": intent_id,
            "strategy_id": strategy_id,
            "family_id": getattr(self._risk, "_family_id", ""),
            "oms_order_id": oms_order_id,
            "client_order_id": getattr(order, "client_order_id", "") if order is not None else "",
            "symbol": getattr(instrument, "symbol", "") if instrument is not None else "",
            "side": side,
            "role": role,
            "order_type": order_type,
            "decision": decision,
            "reason": reason,
            "requested_qty": requested_qty,
            "approved_qty": requested_qty if approved else 0,
            "requested_risk_dollars": float(getattr(risk_ctx, "risk_dollars", 0.0) or 0.0) if risk_ctx else 0.0,
            "approved_risk_dollars": float(getattr(risk_ctx, "risk_dollars", 0.0) or 0.0) if approved and risk_ctx else 0.0,
            "requested_risk_R": 0.0,
            "approved_risk_R": 0.0,
            "portfolio_size_mult": getattr(risk_ctx, "portfolio_size_mult", 1.0) if risk_ctx else 1.0,
            "preapproved": preapproved,
            "portfolio_decision_ref": getattr(risk_ctx, "portfolio_decision_ref", "") if risk_ctx else "",
            "signal_id": getattr(risk_ctx, "signal_id", "") if risk_ctx else "",
            "bar_id": getattr(risk_ctx, "bar_id", "") if risk_ctx else "",
            "exchange_timestamp": getattr(risk_ctx, "exchange_timestamp", None) if risk_ctx else None,
        }
        payload["risk_decision_ref"] = stable_hash(
            "risk_decision_",
            {
                "intent_id": intent_id,
                "oms_order_id": oms_order_id,
                "decision": decision,
                "reason": reason,
            },
        )
        if risk_ctx is not None:
            risk_ctx.intent_id = intent_id
            risk_ctx.risk_decision_ref = payload["risk_decision_ref"]
        payload = self._enrich_risk_decision_payload(payload)
        try:
            self._bus.emit_risk_decision(strategy_id, oms_order_id, payload)
        except Exception as exc:
            logger.debug("Risk decision event emission failed: %s", exc)
        return payload

    async def _handle_new_order(
        self, intent: Intent, intent_id: str, *, preapproved: bool = False
    ) -> IntentReceipt:
        order = intent.order
        if not order:
            self._emit_intent_risk_decision(
                intent,
                intent_id,
                decision="deny",
                reason="No order in intent",
            )
            return IntentReceipt(
                IntentResult.DENIED, intent_id, denial_reason="No order in intent"
            )
        if order.risk_context is not None:
            self._stamp_order_correlation(order, intent_id)
        if preapproved:
            denial = self._validate_preapproved_family_decision(
                intent.preapproved_family_decision,
                order,
            )
            if denial:
                self._emit_intent_risk_decision(
                    intent,
                    intent_id,
                    decision="deny",
                    reason=denial,
                    order=order,
                    preapproved=True,
                )
                return IntentReceipt(
                    IntentResult.DENIED, intent_id, denial_reason=denial
                )

        # OMS-7: stamp the configured account_id on orders that didn't set
        # one. Swing+momentum builders historically left this blank, so DB
        # attribution and reconciliation lost the account scope; stock
        # already sets it explicitly. The factory passes the configured
        # IBKRConfig.profile.account_id into this handler.
        if not order.account_id and self._default_account_id:
            order.account_id = self._default_account_id

        now_utc = datetime.now(timezone.utc)
        if order.instrument is not None:
            order.instrument = with_contract_expiry_for_order(
                order.instrument,
                order_role=order.role.value,
                as_of=now_utc,
            )

        if order.role == OrderRole.ENTRY and order.instrument is not None:
            denial = roll_blackout_reason(order.instrument, as_of=now_utc)
            if denial:
                logger.warning(
                    "Denying entry during futures roll blackout: strategy=%s symbol=%s reason=%s",
                    order.strategy_id,
                    getattr(order.instrument, "symbol", ""),
                    denial,
                )
                self._emit_intent_risk_decision(
                    intent,
                    intent_id,
                    decision="deny",
                    reason=denial,
                    order=order,
                    preapproved=preapproved,
                )
                return IntentReceipt(
                    IntentResult.DENIED,
                    intent_id,
                    denial_reason=denial,
                )

        # M1: Validate qty > 0
        if order.qty <= 0:
            self._emit_intent_risk_decision(
                intent,
                intent_id,
                decision="deny",
                reason="Order qty must be > 0",
                order=order,
                preapproved=preapproved,
            )
            return IntentReceipt(
                IntentResult.DENIED, intent_id, denial_reason="Order qty must be > 0"
            )

        # M2: For EXIT orders, validate qty doesn't exceed open position
        if order.role in (OrderRole.EXIT, OrderRole.STOP):
            positions = await self._repo.get_positions(
                order.strategy_id,
                order.instrument.symbol if order.instrument else None,
            )
            open_qty = sum(abs(p.net_qty) for p in positions)
            if open_qty > 0 and order.qty > open_qty:
                reason = f"Exit qty {order.qty} exceeds open position {open_qty}"
                self._emit_intent_risk_decision(
                    intent,
                    intent_id,
                    decision="deny",
                    reason=reason,
                    order=order,
                    preapproved=preapproved,
                )
                return IntentReceipt(
                    IntentResult.DENIED,
                    intent_id,
                    denial_reason=reason,
                )

        # C1: Idempotency check under per-key lock to prevent race between
        # cache lookup and DB fallback across concurrent async tasks.
        if order.client_order_id:
            if order.client_order_id not in self._idemp_locks:
                self._idemp_locks[order.client_order_id] = asyncio.Lock()
            async with self._idemp_locks[order.client_order_id]:
                existing_id = self._idempotency.get(order.client_order_id)
                if not existing_id:
                    existing_id = await self._repo.get_order_id_by_client_order_id(
                        order.strategy_id, order.client_order_id
                    )
                    if existing_id:
                        self._idempotency[order.client_order_id] = existing_id
                if existing_id:
                    return IntentReceipt(
                        IntentResult.ACCEPTED, intent_id, oms_order_id=existing_id
                    )
                # Register idempotency early (inside lock) to block concurrent duplicates
                self._idempotency[order.client_order_id] = order.oms_order_id
                self._prune_idemp_cache()

        # Set timestamps
        order.created_at = now_utc
        order.remaining_qty = order.qty
        self._stamp_order_correlation(order, intent_id)
        original_qty = order.qty

        # 1C: Serialize ENTRY risk-check to persist to prevent concurrent entries
        # from both passing heat cap before either persists (swing shared OMS).
        # Exits/stops skip the lock since RiskGateway auto-approves non-ENTRY orders.
        use_entry_lock = order.role == OrderRole.ENTRY

        def _rollback_idempotency() -> None:
            if order.client_order_id:
                self._idempotency.pop(order.client_order_id, None)
                self._idemp_locks.pop(order.client_order_id, None)

        def _apply_portfolio_multiplier() -> None:
            if not order.risk_context or order.risk_context.portfolio_size_mult == 1.0:
                return
            mult = order.risk_context.portfolio_size_mult
            original_qty = order.qty
            order.qty = max(1, int(order.qty * mult))
            order.remaining_qty = order.qty
            if order.instrument is not None:
                order.risk_context.risk_dollars = (
                    order.qty
                    * abs(
                        order.risk_context.planned_entry_price
                        - order.risk_context.stop_for_risk
                    )
                    * order.instrument.point_value
                )
            elif original_qty > 0:
                order.risk_context.risk_dollars *= order.qty / original_qty
            logger.info(
                "Portfolio size mult %.2fx: qty %d -> %d for %s",
                mult, original_qty, order.qty, order.strategy_id,
            )

        def _risk_decision_payload(decision: str, reason: str = "") -> dict:
            risk_ctx = order.risk_context
            requested_risk_dollars = 0.0
            approved_risk_dollars = 0.0
            requested_risk_R = 0.0
            approved_risk_R = 0.0
            unit_risk = 0.0
            if risk_ctx is not None:
                unit_risk = float(getattr(risk_ctx, "unit_risk_dollars", 0.0) or 0.0)
                approved_risk_dollars = float(getattr(risk_ctx, "risk_dollars", 0.0) or 0.0)
                if order.qty > 0:
                    requested_risk_dollars = approved_risk_dollars * (original_qty / order.qty)
                else:
                    requested_risk_dollars = approved_risk_dollars
                if unit_risk > 0:
                    requested_risk_R = requested_risk_dollars / unit_risk
                    approved_risk_R = approved_risk_dollars / unit_risk
            side = getattr(getattr(order, "side", None), "value", "")
            role = getattr(getattr(order, "role", None), "value", "")
            order_type = getattr(getattr(order, "order_type", None), "value", "")
            symbol = order.instrument.symbol if order.instrument else ""
            approved = decision in {"approve", "scale", "route"}
            gateway_context = dict(getattr(risk_ctx, "gateway_decision_context", {}) or {})
            payload = {
                **gateway_context,
                "intent_id": intent_id,
                "strategy_id": order.strategy_id,
                "family_id": getattr(self._risk, "_family_id", ""),
                "oms_order_id": order.oms_order_id,
                "client_order_id": order.client_order_id,
                "symbol": symbol,
                "side": side,
                "role": role,
                "order_type": order_type,
                "decision": decision,
                "reason": reason,
                "requested_qty": original_qty,
                "approved_qty": order.qty if approved else 0,
                "requested_risk_dollars": requested_risk_dollars,
                "approved_risk_dollars": approved_risk_dollars if approved else 0.0,
                "requested_risk_R": requested_risk_R,
                "approved_risk_R": approved_risk_R if approved else 0.0,
                "portfolio_size_mult": getattr(risk_ctx, "portfolio_size_mult", 1.0) if risk_ctx else 1.0,
                "preapproved": preapproved,
                "portfolio_decision_ref": getattr(risk_ctx, "portfolio_decision_ref", "") if risk_ctx else "",
                "signal_id": getattr(risk_ctx, "signal_id", "") if risk_ctx else "",
                "bar_id": getattr(risk_ctx, "bar_id", "") if risk_ctx else "",
                "exchange_timestamp": getattr(risk_ctx, "exchange_timestamp", None) if risk_ctx else None,
            }
            payload["risk_decision_ref"] = stable_hash(
                "risk_decision_",
                {
                    "intent_id": intent_id,
                    "oms_order_id": order.oms_order_id,
                    "decision": decision,
                    "reason": reason,
                },
            )
            return payload

        def _stamp_risk_decision(decision: str, reason: str = "") -> dict:
            payload = _risk_decision_payload(decision, reason)
            if order.risk_context is not None and payload.get("risk_decision_ref"):
                order.risk_context.risk_decision_ref = payload["risk_decision_ref"]
            return payload

        def _emit_risk_decision(decision: str, reason: str = "", payload: dict | None = None) -> None:
            try:
                payload = payload or _stamp_risk_decision(decision, reason)
                payload = self._enrich_risk_decision_payload(payload)
                self._bus.emit_risk_decision(
                    order.strategy_id,
                    order.oms_order_id,
                    payload,
                )
            except Exception as exc:
                logger.debug("Risk decision event emission failed: %s", exc)

        async def _risk_check_and_route():
            if preapproved:
                denial = await self._risk.check_preapproved_entry(order)
            else:
                denial = await self._risk.check_entry(
                    order,
                    skip_account_gate=order.role == OrderRole.ENTRY,
                )
            if denial:
                _rollback_idempotency()
                order.status = OrderStatus.REJECTED
                decision_payload = _stamp_risk_decision("deny", denial)
                await self._repo.save_order_and_event(
                    order,
                    "RISK_DENIED",
                    {"reason": denial},
                )
                _emit_risk_decision("deny", denial, payload=decision_payload)
                self._bus.emit_risk_denial(order.strategy_id, order.oms_order_id, denial)
                return IntentReceipt(
                    IntentResult.DENIED, intent_id, denial_reason=denial
                )

            if not preapproved:
                _apply_portfolio_multiplier()

            # Approve and persist
            if order.role == OrderRole.ENTRY:
                async with self._repo.transaction() as conn:
                    account_denial = await self._risk.check_account_gate(order, conn=conn)
                    if account_denial:
                        _rollback_idempotency()
                        order.status = OrderStatus.REJECTED
                        decision_payload = _stamp_risk_decision("deny", account_denial)
                        await self._repo.save_order_and_event(
                            order,
                            "RISK_DENIED",
                            {"reason": account_denial},
                            conn=conn,
                        )
                        _emit_risk_decision("deny", account_denial, payload=decision_payload)
                        self._bus.emit_risk_denial(
                            order.strategy_id,
                            order.oms_order_id,
                            account_denial,
                        )
                        return IntentReceipt(
                            IntentResult.DENIED,
                            intent_id,
                            denial_reason=account_denial,
                        )
                    decision = "scale" if (
                        order.risk_context is not None
                        and order.risk_context.portfolio_size_mult != 1.0
                    ) else "approve"
                    decision_payload = _stamp_risk_decision(decision)
                    order.status = OrderStatus.RISK_APPROVED
                    await self._repo.save_order_and_event(
                        order,
                        "RISK_APPROVED",
                        {},
                        conn=conn,
                    )
            else:
                decision = "scale" if (
                    order.risk_context is not None
                    and order.risk_context.portfolio_size_mult != 1.0
                ) else "approve"
                decision_payload = _stamp_risk_decision(decision)
                order.status = OrderStatus.RISK_APPROVED
                await self._repo.save_order_and_event(order, "RISK_APPROVED", {})

            _emit_risk_decision(decision, payload=decision_payload)

            # Route to execution
            await self._router.route(order)
            return None  # success

        if use_entry_lock:
            async with self._entry_lock:
                receipt = await _risk_check_and_route()
        else:
            receipt = await _risk_check_and_route()

        if receipt is not None:
            return receipt

        self._bus.emit_order_event(order)
        return IntentReceipt(
            IntentResult.ACCEPTED, intent_id, oms_order_id=order.oms_order_id
        )

    @staticmethod
    def _validate_preapproved_family_decision(
        decision: PreapprovedFamilyDecision | None,
        order: OMSOrder,
    ) -> str | None:
        if decision is None:
            return "PREAPPROVED_ORDER requires preapproved_family_decision"
        status = str(decision.status).lower()
        if status not in {"accepted", "reduced"}:
            return f"Invalid preapproved family decision status: {decision.status}"
        if not decision.candidate_key:
            return "Preapproved family decision missing candidate_key"
        if not decision.family_surface:
            return "Preapproved family decision missing family_surface"
        if decision.original_qty <= 0:
            return "Preapproved family decision original_qty must be > 0"
        if decision.approved_qty <= 0:
            return "Preapproved family decision approved_qty must be > 0"
        if decision.approved_qty > decision.original_qty:
            return "Preapproved family decision approved_qty exceeds original_qty"
        if status == "accepted" and decision.approved_qty != decision.original_qty:
            return "Accepted preapproved decision must preserve quantity"
        if status == "reduced" and decision.approved_qty >= decision.original_qty:
            return "Reduced preapproved decision must reduce quantity"
        if order.role != OrderRole.ENTRY:
            return "PREAPPROVED_ORDER is only valid for ENTRY orders"
        symbol = order.instrument.symbol if order.instrument is not None else ""
        if str(order.strategy_id) != str(decision.strategy_id):
            return "Preapproved family decision strategy_id mismatch"
        if str(symbol).upper() != str(decision.symbol).upper():
            return "Preapproved family decision symbol mismatch"
        if str(order.side.value).upper() != str(decision.side).upper():
            return "Preapproved family decision side mismatch"
        if str(order.role.value).upper() != str(decision.role).upper():
            return "Preapproved family decision role mismatch"
        if int(order.qty) != int(decision.approved_qty):
            return "Preapproved family decision approved_qty mismatch"
        if decision.sequence <= 0:
            return "Preapproved family decision sequence must be > 0"
        return None

    async def _handle_cancel(self, intent: Intent, intent_id: str) -> IntentReceipt:
        """Cancel a working order."""
        order = await self._repo.get_order(intent.target_oms_order_id)
        if not order:
            return IntentReceipt(
                IntentResult.DENIED, intent_id, denial_reason="Order not found"
            )
        if order.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.DONE,
        }:
            return IntentReceipt(
                IntentResult.DENIED,
                intent_id,
                denial_reason=f"Order in terminal state: {order.status}",
            )

        order.status = OrderStatus.CANCEL_REQUESTED
        await self._repo.save_order(order)
        await self._router.cancel(order)
        return IntentReceipt(
            IntentResult.ACCEPTED, intent_id, oms_order_id=order.oms_order_id
        )

    async def _handle_replace(self, intent: Intent, intent_id: str) -> IntentReceipt:
        """Replace (modify) a working order."""
        order = await self._repo.get_order(intent.target_oms_order_id)
        if not order:
            return IntentReceipt(
                IntentResult.DENIED, intent_id, denial_reason="Order not found"
            )

        order.status = OrderStatus.REPLACE_REQUESTED
        await self._repo.save_order(order)
        await self._router.replace(
            order, intent.new_qty, intent.new_limit_price, intent.new_stop_price
        )
        return IntentReceipt(
            IntentResult.ACCEPTED, intent_id, oms_order_id=order.oms_order_id
        )

    async def _handle_flatten(self, intent: Intent, intent_id: str) -> IntentReceipt:
        """Flatten positions for a strategy, optionally filtered by instrument."""
        # 1. Snapshot working orders BEFORE creating flatten exits
        working = await self._repo.get_working_orders(
            intent.strategy_id, intent.instrument_symbol
        )

        # 2. Submit flatten exits
        flatten_order_ids: list[str] = []
        positions = await self._repo.get_positions(
            intent.strategy_id, intent.instrument_symbol
        )
        for pos in positions:
            if pos.net_qty != 0:
                order = await self._router.flatten(pos)
                if order is not None:
                    flatten_order_ids.append(order.oms_order_id)

        # 3. Cancel pre-existing working orders only (not the new flatten exits)
        for order in working:
            if transition(order, OrderStatus.CANCEL_REQUESTED):
                await self._repo.save_order(order)
                await self._router.cancel(order)
            elif transition(order, OrderStatus.CANCELLED):
                # ROUTED/ACKED go directly to CANCELLED
                order.last_update_at = datetime.now(timezone.utc)
                await self._repo.save_order(order)
                await self._router.cancel(order)

        return IntentReceipt(
            IntentResult.ACCEPTED, intent_id,
            oms_order_id=flatten_order_ids[0] if flatten_order_ids else None,
        )
