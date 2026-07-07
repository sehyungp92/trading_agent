"""Reconciliation sync logic."""
import logging
from dataclasses import dataclass
from ..models.types import OrderStatusEvent, PositionSnapshot
from .discrepancy_policy import DiscrepancyAction, DiscrepancyPolicy

logger = logging.getLogger(__name__)


@dataclass
class Discrepancy:
    type: str  # "unknown_order", "missing_order", "position_mismatch", etc.
    action: DiscrepancyAction
    details: dict


class ReconcilerSync:
    """Compares broker snapshots against OMS expectations.

    Returns list of discrepancies with policy-driven actions.
    """

    def __init__(self, policy: DiscrepancyPolicy):
        self._policy = policy

    @staticmethod
    def _order_details(order: OrderStatusEvent, order_ref: str) -> dict:
        return {
            "order": order,
            "broker_order_id": order.broker_order_id,
            "perm_id": order.perm_id,
            "order_ref": order_ref,
            "account": getattr(order, "account", "") or "",
            "client_id": getattr(order, "client_id", None),
        }

    @staticmethod
    def _int_or_none(value: object) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def reconcile_orders(
        self,
        broker_orders: list[OrderStatusEvent],
        oms_working_ids: set[int],
        known_order_refs: dict[str, str] | None = None,
        our_client_id_pattern: str = "",
        managed_account_id: str = "",
        managed_client_id: int | None = None,
        owned_order_ref_prefixes: tuple[str, ...] = (),
    ) -> list[Discrepancy]:
        """Compare broker open orders vs OMS working orders."""
        discrepancies = []
        known_order_refs = known_order_refs or {}
        oms_working_ids = {
            broker_order_id
            for raw_id in oms_working_ids
            if (broker_order_id := self._int_or_none(raw_id)) is not None
        }
        broker_ids = {
            broker_order_id
            for order in broker_orders
            if (broker_order_id := self._int_or_none(order.broker_order_id)) is not None
        }

        # Orders on broker but not in OMS
        for bo in broker_orders:
            broker_order_id = self._int_or_none(bo.broker_order_id)
            if broker_order_id not in oms_working_ids:
                order_ref = (getattr(bo, "order_ref", "") or "").strip()
                account = (getattr(bo, "account", "") or "").strip()
                client_id = self._int_or_none(getattr(bo, "client_id", None))
                details = self._order_details(bo, order_ref)
                details["broker_order_id"] = broker_order_id
                details["client_id"] = client_id

                if managed_account_id and account and account != managed_account_id:
                    details["managed_account_id"] = managed_account_id
                    details["scope_mismatch"] = "account"
                    discrepancies.append(
                        Discrepancy("foreign_order", DiscrepancyAction.HALT_AND_ALERT, details)
                    )
                    continue

                if managed_client_id is not None and client_id is not None and client_id != managed_client_id:
                    details["managed_client_id"] = managed_client_id
                    details["scope_mismatch"] = "client_id"
                    discrepancies.append(
                        Discrepancy("foreign_order", DiscrepancyAction.HALT_AND_ALERT, details)
                    )
                    continue

                if order_ref and order_ref in known_order_refs:
                    discrepancies.append(
                        Discrepancy(
                            "repair_order_mapping",
                            DiscrepancyAction.REPAIR_MAPPING,
                            {
                                "order": bo,
                                "order_ref": order_ref,
                                "oms_order_id": known_order_refs[order_ref],
                            },
                        )
                    )
                    continue
                if owned_order_ref_prefixes and order_ref and not order_ref.startswith(owned_order_ref_prefixes):
                    details["owned_order_ref_prefixes"] = list(owned_order_ref_prefixes)
                    details["scope_mismatch"] = "order_ref_prefix"
                    discrepancies.append(
                        Discrepancy("foreign_order", DiscrepancyAction.HALT_AND_ALERT, details)
                    )
                    continue
                if not order_ref:
                    action = self._policy.unknown_order_orphan
                else:
                    action = self._policy.unknown_order_with_our_tag
                discrepancies.append(
                    Discrepancy("unknown_order", action, details)
                )

        # Orders in OMS but not on broker
        for oms_id in oms_working_ids:
            if oms_id not in broker_ids:
                discrepancies.append(
                    Discrepancy(
                        "missing_order",
                        self._policy.oms_working_broker_missing,
                        {"oms_broker_order_id": oms_id},
                    )
                )

        return discrepancies

    def reconcile_positions(
        self,
        broker_positions: list[PositionSnapshot],
        oms_positions: dict[int, float],  # con_id -> expected qty
    ) -> list[Discrepancy]:
        """Compare broker positions vs OMS positions."""
        discrepancies = []
        seen_con_ids = set()

        for bp in broker_positions:
            seen_con_ids.add(bp.con_id)
            expected = oms_positions.get(bp.con_id, 0.0)
            if bp.qty != expected:
                if bp.con_id not in oms_positions and bp.qty != 0:
                    action = self._policy.unexpected_position
                else:
                    action = self._policy.position_qty_mismatch
                discrepancies.append(
                    Discrepancy(
                        "position_mismatch",
                        action,
                        {
                            "con_id": bp.con_id,
                            "symbol": bp.symbol,
                            "broker_qty": bp.qty,
                            "oms_qty": expected,
                        },
                    )
                )

        # OMS positions not on broker
        for con_id, expected_qty in oms_positions.items():
            if con_id not in seen_con_ids and expected_qty != 0:
                discrepancies.append(
                    Discrepancy(
                        "position_mismatch",
                        self._policy.position_qty_mismatch,
                        {
                            "con_id": con_id,
                            "broker_qty": 0.0,
                            "oms_qty": expected_qty,
                        },
                    )
                )

        return discrepancies
