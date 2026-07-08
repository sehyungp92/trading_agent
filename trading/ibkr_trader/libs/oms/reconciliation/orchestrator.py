"""Reconciliation orchestrator."""
import logging
import os
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from libs.broker_ibkr.models.types import BrokerOrderRef, BrokerOrderStatus, OrderStatusEvent
from libs.broker_ibkr.reconciler.sync import ReconcilerSync, Discrepancy
from libs.broker_ibkr.reconciler.discrepancy_policy import DiscrepancyAction, DiscrepancyPolicy
from .authority import ReconciliationAuthorityScope

if TYPE_CHECKING:
    from ..engine.fill_processor import FillProcessor
    from ..persistence.repository import OMSRepository
    from ..events.bus import EventBus

logger = logging.getLogger(__name__)


class ReconciliationOrchestrator:
    """Startup + periodic reconciliation.
    Calls broker_ibkr reconciler and applies results to OMS state.
    """

    def __init__(
        self,
        adapter,
        repo: "OMSRepository",
        bus: "EventBus",
        halt_trading: Optional[Callable[[str], Awaitable[None]]] = None,
        fill_processor: Optional["FillProcessor"] = None,
        offline_fill_importer: Optional[Callable[[str, object], Awaitable[bool]]] = None,
        lifecycle_event_writer: Optional[Callable[[dict], object]] = None,
        family_id: str = "unknown",
        owner_id: str = "",
        reconciliation_authoritative: bool = True,
        authority: object | None = None,
    ):
        self._adapter = adapter  # IBKRExecutionAdapter
        self._repo = repo
        self._bus = bus
        self._halt_trading = halt_trading
        # OMS-3: production passes offline_fill_importer so startup broker
        # executions run through the same side-effect path as live fills.
        self._offline_fill_importer = offline_fill_importer
        self._fill_processor = fill_processor
        self._lifecycle_event_writer = lifecycle_event_writer
        self.family_id = family_id or "unknown"
        self.owner_id = owner_id or self.family_id
        self.reconciliation_authoritative = bool(reconciliation_authoritative)
        self._authority = authority
        self._policy = DiscrepancyPolicy.from_environment()
        self._reconciler = ReconcilerSync(self._policy)

    @property
    def is_authoritative(self) -> bool:
        return self.reconciliation_authoritative

    async def _claim_mutation_authority(
        self,
        *,
        recon_kind: str,
        discrepancies: list[Discrepancy],
    ) -> bool:
        if self._authority is None:
            return True
        client_id = self._managed_client_id()
        scope = ReconciliationAuthorityScope(
            broker=str(getattr(self._adapter, "broker", "") or "IBKR"),
            account_id=self._managed_account_id() or "unknown",
            client_id=client_id if client_id is not None else -1,
            family_id=self.family_id,
            recon_kind=recon_kind,
        )
        lease = await self._authority.acquire(
            scope,
            self.owner_id,
            ttl_seconds=120.0,
        )
        if lease is not None and self._authority.is_authoritative(lease):
            return True
        self._emit_lifecycle(
            "reconciliation_skipped",
            phase="apply_discrepancy",
            status="authority_lease_unavailable",
            details={
                "reason": "authority_lease_unavailable",
                "broker": scope.broker,
                "account_id": scope.account_id,
                "client_id": scope.client_id,
                "recon_kind": scope.recon_kind,
                "discrepancy_count": len(discrepancies),
            },
            discrepancies=discrepancies,
        )
        logger.info(
            "Reconciliation authority lease unavailable: owner=%s family=%s kind=%s",
            self.owner_id,
            self.family_id,
            recon_kind,
        )
        return False

    @staticmethod
    def _int_or_none(value: object) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _discrepancy_payload(d: Discrepancy) -> dict:
        details = {}
        for key, value in d.details.items():
            if is_dataclass(value):
                details[key] = asdict(value)
            else:
                details[key] = value
        return {
            "type": d.type,
            "action": getattr(d.action, "value", str(d.action)),
            "details": details,
        }

    def _emit_lifecycle(
        self,
        lifecycle_action: str,
        *,
        phase: str,
        status: str,
        details: dict | None = None,
        discrepancies: list[Discrepancy] | None = None,
    ) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lifecycle_action": lifecycle_action,
            "phase": phase,
            "status": status,
            "source": "reconciliation",
            "details": {
                "family_id": self.family_id,
                "owner_id": self.owner_id,
                "reconciliation_authoritative": self.reconciliation_authoritative,
                **(details or {}),
            },
            "discrepancies": [
                self._discrepancy_payload(d) for d in list(discrepancies or [])
            ],
        }
        if self._lifecycle_event_writer is not None:
            try:
                self._lifecycle_event_writer(payload)
            except Exception as exc:
                logger.warning("Reconciliation lifecycle writer failed: %s", exc)
        try:
            strategy_id = str((details or {}).get("strategy_id") or "")
            self._bus.emit_reconciliation_event(payload, strategy_id=strategy_id)
        except Exception as exc:
            logger.warning("Reconciliation event bus emit failed: %s", exc)

    async def _working_order_context(self) -> tuple[list, set[int], dict[str, str]]:
        """Return working OMS orders, broker IDs, and exact repair refs."""
        oms_working_orders = await self._repo.get_all_working_orders()
        oms_working_broker_ids: set[int] = set()
        known_order_refs: dict[str, str] = {}
        duplicate_refs: set[str] = set()

        for order in oms_working_orders:
            broker_order_id = self._int_or_none(order.broker_order_id)
            if broker_order_id is not None:
                oms_working_broker_ids.add(broker_order_id)
            for ref in (order.client_order_id, order.oms_order_id):
                ref = (ref or "").strip()
                if not ref:
                    continue
                existing = known_order_refs.get(ref)
                if existing is not None and existing != order.oms_order_id:
                    duplicate_refs.add(ref)
                    known_order_refs.pop(ref, None)
                    continue
                if ref not in duplicate_refs:
                    known_order_refs[ref] = order.oms_order_id

        return oms_working_orders, oms_working_broker_ids, known_order_refs

    def _managed_account_id(self) -> str:
        return str(getattr(self._adapter, "account_id", "") or "").strip()

    def _managed_client_id(self) -> int | None:
        client_id = getattr(self._adapter, "client_id", None)
        try:
            return int(client_id) if client_id is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _owned_order_ref_prefixes() -> tuple[str, ...]:
        raw = os.environ.get("OMS_OWNED_ORDER_REF_PREFIXES", "")
        return tuple(
            prefix.strip()
            for prefix in raw.split(",")
            if prefix.strip()
        )

    async def _halt_for_discrepancy(self, d: Discrepancy, reason: str | None = None) -> None:
        reason = reason or f"Reconciliation: {d.type} - {d.details}"
        logger.error("CRITICAL DISCREPANCY - halting: %s %s", d.type, d.details)
        if self._halt_trading is not None:
            await self._halt_trading(reason)
        self._bus.emit_risk_halt("", reason)

    @staticmethod
    def _target_status_from_broker_event(order_event: OrderStatusEvent):
        from ..models.order import OrderStatus

        if order_event.filled_qty > 0 and order_event.remaining_qty <= 0:
            return OrderStatus.FILLED
        if order_event.filled_qty > 0:
            return OrderStatus.PARTIALLY_FILLED
        if order_event.status == BrokerOrderStatus.PENDING_SUBMIT:
            return OrderStatus.ROUTED
        if order_event.status == BrokerOrderStatus.PRE_SUBMITTED:
            return OrderStatus.ACKED
        if order_event.status == BrokerOrderStatus.SUBMITTED:
            return OrderStatus.WORKING
        if order_event.status == BrokerOrderStatus.PENDING_CANCEL:
            return OrderStatus.CANCEL_REQUESTED
        if order_event.status == BrokerOrderStatus.CANCELLED:
            return OrderStatus.CANCELLED
        if order_event.status == BrokerOrderStatus.FILLED:
            return OrderStatus.FILLED
        if order_event.status == BrokerOrderStatus.INACTIVE:
            return OrderStatus.REJECTED
        return None

    @staticmethod
    def _advance_order_status(order, target_status) -> None:
        from ..engine.state_machine import TRANSITIONS, transition
        from ..models.order import OrderStatus

        if target_status is None or order.status == target_status:
            return
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
            return

        paths = {
            OrderStatus.ROUTED: (OrderStatus.ROUTED,),
            OrderStatus.ACKED: (OrderStatus.ROUTED, OrderStatus.ACKED),
            OrderStatus.WORKING: (OrderStatus.ROUTED, OrderStatus.ACKED, OrderStatus.WORKING),
            OrderStatus.PARTIALLY_FILLED: (
                OrderStatus.ROUTED, OrderStatus.ACKED, OrderStatus.WORKING, OrderStatus.PARTIALLY_FILLED,
            ),
            OrderStatus.FILLED: (
                OrderStatus.ROUTED, OrderStatus.ACKED, OrderStatus.WORKING, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
            ),
            OrderStatus.CANCEL_REQUESTED: (OrderStatus.CANCEL_REQUESTED,),
            OrderStatus.CANCELLED: (OrderStatus.ROUTED, OrderStatus.ACKED, OrderStatus.CANCELLED),
            OrderStatus.REJECTED: (OrderStatus.REJECTED,),
        }
        for candidate in paths.get(target_status, (target_status,)):
            if order.status == candidate:
                continue
            if candidate in TRANSITIONS.get(order.status, set()):
                transition(order, candidate)
            if order.status == target_status:
                return

    async def _repair_order_mapping(self, d: Discrepancy) -> None:
        """Repair a broker order that exactly matches a live OMS ref."""
        order_event = d.details.get("order")
        oms_order_id = d.details.get("oms_order_id")
        if order_event is None or not oms_order_id:
            await self._halt_for_discrepancy(d, f"Reconciliation repair missing details: {d.details}")
            return

        order = await self._repo.get_order(oms_order_id)
        if order is None:
            await self._halt_for_discrepancy(
                d,
                f"Reconciliation repair ref resolved to missing OMS order: {oms_order_id}",
            )
            return

        broker_order_id = self._int_or_none(order_event.broker_order_id)
        if broker_order_id is None:
            await self._halt_for_discrepancy(
                d,
                f"Reconciliation repair has invalid broker order id: {order_event.broker_order_id}",
            )
            return
        perm_id = self._int_or_none(order_event.perm_id) or 0

        self._adapter.cache.register_order(
            order.oms_order_id,
            broker_order_id,
            perm_id,
        )
        if order_event.status in {BrokerOrderStatus.PRE_SUBMITTED, BrokerOrderStatus.SUBMITTED}:
            self._adapter.cache.mark_acked(order.oms_order_id)

        order.broker_order_id = broker_order_id
        order.perm_id = perm_id
        order.broker_order_ref = BrokerOrderRef(
            broker_order_id=broker_order_id,
            perm_id=perm_id,
            con_id=0,
        )
        order.filled_qty = max(float(order.filled_qty or 0.0), float(order_event.filled_qty or 0.0))
        order.remaining_qty = float(order_event.remaining_qty or 0.0)
        if order_event.avg_fill_price:
            order.avg_fill_price = float(order_event.avg_fill_price)
        order.last_update_at = datetime.now(timezone.utc)
        self._advance_order_status(order, self._target_status_from_broker_event(order_event))

        await self._repo.save_order(order)
        self._bus.emit_order_event(order)
        logger.info(
            "Repaired OMS/broker mapping: oms_order_id=%s broker_order_id=%s order_ref=%s",
            order.oms_order_id,
            order_event.broker_order_id,
            d.details.get("order_ref", ""),
        )

    async def _build_oms_position_map(self) -> dict[int, float]:
        """Build con_id -> net_qty map from OMS positions, aggregating across strategies."""
        oms_positions = await self._repo.get_all_positions()
        oms_position_map: dict[int, float] = defaultdict(float)
        broker_contracts = self._adapter.cache.contracts
        for pos in oms_positions:
            for con_id, spec in broker_contracts.items():
                if spec.symbol == pos.instrument_symbol:
                    oms_position_map[con_id] += pos.net_qty
                    break
        return oms_position_map

    async def startup_reconciliation(self) -> None:
        """MANDATORY: run before accepting any intents.
        1. Load OMS state from DB
        2. Pull broker snapshots
        3. Reconcile
        4. Apply discrepancy actions
        """
        if not self.reconciliation_authoritative:
            logger.info(
                "Skipping startup reconciliation for non-authoritative owner=%s family=%s",
                self.owner_id,
                self.family_id,
            )
            self._emit_lifecycle(
                "reconciliation_skipped",
                phase="startup_reconciliation",
                status="non_authoritative_skipped",
                details={"reason": "non_authoritative"},
            )
            return

        logger.info("Starting reconciliation...")

        # OMS-3: import broker executions that are missing locally. Production
        # uses the live fill callback pipeline; FillProcessor is a compatibility
        # fallback for older tests/callers.
        async def _fill_importer(oms_order_id: str, exec_report) -> bool:
            if self._offline_fill_importer is not None:
                imported = await self._offline_fill_importer(oms_order_id, exec_report)
                if imported:
                    self._emit_lifecycle(
                        "inferred_fill",
                        phase="startup_reconciliation",
                        status="imported",
                        details={
                            "oms_order_id": oms_order_id,
                            "exec_id": getattr(exec_report, "exec_id", ""),
                            "qty": getattr(exec_report, "qty", 0.0),
                            "price": getattr(exec_report, "price", 0.0),
                        },
                    )
                return imported
            if self._fill_processor is None:
                return False
            ts = getattr(exec_report, "fill_time", None) or datetime.now(timezone.utc)
            commission = getattr(exec_report, "commission", 0.0) or 0.0
            try:
                imported = await self._fill_processor.process_fill(
                    oms_order_id=oms_order_id,
                    broker_fill_id=exec_report.exec_id,
                    price=float(exec_report.price),
                    qty=float(exec_report.qty),
                    timestamp=ts,
                    fees=float(commission),
                )
                if imported:
                    self._emit_lifecycle(
                        "inferred_fill",
                        phase="startup_reconciliation",
                        status="imported",
                        details={
                            "oms_order_id": oms_order_id,
                            "exec_id": getattr(exec_report, "exec_id", ""),
                            "qty": getattr(exec_report, "qty", 0.0),
                            "price": getattr(exec_report, "price", 0.0),
                        },
                    )
                return imported
            except Exception as e:
                logger.exception(
                    "Offline fill import failed for exec_id=%s oms_order_id=%s: %s",
                    exec_report.exec_id, oms_order_id, e,
                )
                raise

        # Rebuild broker->OMS order mappings first so fills/cancels remain routable
        # after a process restart.
        try:
            await self._adapter.rebuild_cache(
                self._repo.get_order_id_by_broker_order_id,
                fill_exists_check=self._repo.fill_exists,
                fill_importer=_fill_importer,
            )
        except Exception as e:
            logger.warning("Cache rebuild before reconciliation failed: %s", e)

        # Fetch broker state
        broker_orders = await self._adapter.request_open_orders()
        broker_positions = await self._adapter.request_positions()
        broker_executions = await self._adapter.request_executions()

        logger.info(
            f"Broker state: {len(broker_orders)} orders, "
            f"{len(broker_positions)} positions, "
            f"{len(broker_executions)} executions"
        )

        # C3 fix: Compare against OMS DB state
        _oms_working_orders, oms_working_broker_ids, known_order_refs = (
            await self._working_order_context()
        )

        # Reconcile orders
        order_discrepancies = self._reconciler.reconcile_orders(
            broker_orders,
            oms_working_broker_ids,
            known_order_refs=known_order_refs,
            managed_account_id=self._managed_account_id(),
            managed_client_id=self._managed_client_id(),
            owned_order_ref_prefixes=self._owned_order_ref_prefixes(),
        )

        # Gather OMS position quantities by con_id
        oms_position_map = await self._build_oms_position_map()

        # Reconcile positions
        position_discrepancies = self._reconciler.reconcile_positions(
            broker_positions, oms_position_map
        )

        all_discrepancies = order_discrepancies + position_discrepancies

        if all_discrepancies:
            logger.warning(f"Reconciliation found {len(all_discrepancies)} discrepancies")
            self._emit_lifecycle(
                "allocation_drift",
                phase="startup_reconciliation",
                status="detected",
                details={
                    "order_discrepancy_count": len(order_discrepancies),
                    "position_discrepancy_count": len(position_discrepancies),
                },
                discrepancies=all_discrepancies,
            )
            await self._apply_discrepancies(
                all_discrepancies,
                recon_kind="startup",
            )
        else:
            logger.info("Reconciliation complete: no discrepancies found")
            self._emit_lifecycle(
                "allocation_unfreeze",
                phase="startup_reconciliation",
                status="clean",
                details={
                    "broker_order_count": len(broker_orders),
                    "broker_position_count": len(broker_positions),
                    "broker_execution_count": len(broker_executions),
                },
            )

    async def _apply_discrepancies(
        self,
        discrepancies: list[Discrepancy],
        *,
        recon_kind: str = "manual",
    ) -> None:
        """Apply policy-driven actions for each discrepancy."""
        if not self.reconciliation_authoritative:
            self._emit_lifecycle(
                "reconciliation_skipped",
                phase="apply_discrepancy",
                status="non_authoritative_skipped",
                details={
                    "reason": "mutating_actions_require_authority",
                    "discrepancy_count": len(discrepancies),
                },
                discrepancies=discrepancies,
            )
            logger.info(
                "Non-authoritative reconciler skipped %d discrepancy action(s): owner=%s family=%s",
                len(discrepancies),
                self.owner_id,
                self.family_id,
            )
            return
        if not await self._claim_mutation_authority(
            recon_kind=recon_kind,
            discrepancies=discrepancies,
        ):
            return
        for d in discrepancies:
            logger.warning(f"Discrepancy: type={d.type}, action={d.action.value}, details={d.details}")
            self._emit_lifecycle(
                "allocation_drift",
                phase="apply_discrepancy",
                status="detected",
                details=self._discrepancy_payload(d),
                discrepancies=[d],
            )

            if d.action == DiscrepancyAction.HALT_AND_ALERT:
                self._emit_lifecycle(
                    "allocation_freeze",
                    phase="apply_discrepancy",
                    status="halted",
                    details=self._discrepancy_payload(d),
                    discrepancies=[d],
                )
                await self._halt_for_discrepancy(d)
                continue
            if d.action == DiscrepancyAction.REPAIR_MAPPING:
                await self._repair_order_mapping(d)
                self._emit_lifecycle(
                    "admin_correction",
                    phase="apply_discrepancy",
                    status="applied",
                    details=self._discrepancy_payload(d),
                    discrepancies=[d],
                )
                continue
            if d.action == DiscrepancyAction.IMPORT:
                self._emit_lifecycle(
                    "allocation_freeze",
                    phase="apply_discrepancy",
                    status="unsafe_import_blocked",
                    details=self._discrepancy_payload(d),
                    discrepancies=[d],
                )
                await self._halt_for_discrepancy(
                    d,
                    f"Reconciliation unsafe unknown tagged broker order: {d.details}",
                )
                continue
            if d.action == DiscrepancyAction.ADJUST_POSITION:
                self._emit_lifecycle(
                    "drift_assignment",
                    phase="apply_discrepancy",
                    status="assigned",
                    details=self._discrepancy_payload(d),
                    discrepancies=[d],
                )
                self._emit_lifecycle(
                    "allocation_freeze",
                    phase="apply_discrepancy",
                    status="position_mismatch_halted",
                    details=self._discrepancy_payload(d),
                    discrepancies=[d],
                )
                await self._halt_for_discrepancy(
                    d,
                    "Reconciliation position mismatch: "
                    f"broker_qty={d.details.get('broker_qty')}, "
                    f"oms_qty={d.details.get('oms_qty')} for "
                    f"{d.details.get('symbol', d.details.get('con_id'))}",
                )
                continue

            if d.action == DiscrepancyAction.MARK_CANCELLED:
                # OMS thinks order is working but broker doesn't have it
                oms_broker_id = d.details.get("oms_broker_order_id")
                if oms_broker_id is not None:
                    oms_id = self._adapter.cache.lookup_oms_id(oms_broker_id)
                    if not oms_id:
                        oms_id = await self._repo.get_order_id_by_broker_order_id(oms_broker_id)
                    if oms_id:
                        order = await self._repo.get_order(oms_id)
                        if order:
                            from ..models.order import OrderStatus
                            from ..engine.state_machine import transition
                            if transition(order, OrderStatus.CANCELLED):
                                await self._repo.save_order(order)
                                self._bus.emit_order_event(order)
                                self._emit_lifecycle(
                                    "admin_correction",
                                    phase="apply_discrepancy",
                                    status="marked_cancelled",
                                    details={
                                        **self._discrepancy_payload(d),
                                        "oms_order_id": oms_id,
                                    },
                                    discrepancies=[d],
                                )
                                logger.info(f"Marked missing order as cancelled: {oms_id}")

            elif d.action == DiscrepancyAction.CANCEL:
                # Unknown orphan order at broker — cancel it
                order_event = d.details.get("order")
                if order_event:
                    try:
                        await self._adapter.cancel_order(
                            order_event.broker_order_id, order_event.perm_id
                        )
                        self._emit_lifecycle(
                            "admin_correction",
                            phase="apply_discrepancy",
                            status="cancelled_orphan",
                            details=self._discrepancy_payload(d),
                            discrepancies=[d],
                        )
                        logger.info(f"Cancelled orphan broker order: {order_event.broker_order_id}")
                    except Exception as e:
                        logger.error(f"Failed to cancel orphan order: {e}")

    async def periodic_reconciliation(self) -> None:
        """Run every 60-180 seconds. Verifies open orders and positions."""
        if not self.reconciliation_authoritative:
            self._emit_lifecycle(
                "reconciliation_skipped",
                phase="periodic_reconciliation",
                status="non_authoritative_skipped",
                details={"reason": "non_authoritative"},
            )
            return
        broker_orders = await self._adapter.request_open_orders()
        broker_positions = await self._adapter.request_positions()

        # Lightweight comparison
        _oms_working_orders, oms_working_broker_ids, known_order_refs = (
            await self._working_order_context()
        )

        order_discrepancies = self._reconciler.reconcile_orders(
            broker_orders,
            oms_working_broker_ids,
            known_order_refs=known_order_refs,
            managed_account_id=self._managed_account_id(),
            managed_client_id=self._managed_client_id(),
            owned_order_ref_prefixes=self._owned_order_ref_prefixes(),
        )

        oms_position_map = await self._build_oms_position_map()
        position_discrepancies = self._reconciler.reconcile_positions(
            broker_positions, oms_position_map
        )

        all_discrepancies = order_discrepancies + position_discrepancies

        if all_discrepancies:
            logger.warning(f"Periodic recon: {len(all_discrepancies)} discrepancies ({len(order_discrepancies)} order, {len(position_discrepancies)} position)")
            self._emit_lifecycle(
                "allocation_drift",
                phase="periodic_reconciliation",
                status="detected",
                details={
                    "order_discrepancy_count": len(order_discrepancies),
                    "position_discrepancy_count": len(position_discrepancies),
                },
                discrepancies=all_discrepancies,
            )
            await self._apply_discrepancies(
                all_discrepancies,
                recon_kind="periodic",
            )
        else:
            logger.debug(
                f"Periodic recon: {len(broker_orders)} orders, "
                f"{len(broker_positions)} positions — OK"
            )

    async def on_reconnect_reconciliation(self) -> None:
        """Immediate recon after reconnection."""
        logger.info("Running post-reconnect reconciliation")
        await self.startup_reconciliation()
