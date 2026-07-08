"""Reconciliation discrepancy policy configuration."""
import os
from dataclasses import dataclass
from enum import Enum


class DiscrepancyAction(Enum):
    IMPORT = "import"
    CANCEL = "cancel"
    MARK_CANCELLED = "mark_cancelled"
    REPAIR_MAPPING = "repair_mapping"
    ADJUST_POSITION = "adjust_position"
    HALT_AND_ALERT = "halt_and_alert"


@dataclass
class DiscrepancyPolicy:
    """Config-driven policy for reconciliation mismatches."""

    unknown_order_with_our_tag: DiscrepancyAction = DiscrepancyAction.IMPORT
    unknown_order_orphan: DiscrepancyAction = DiscrepancyAction.HALT_AND_ALERT
    oms_working_broker_missing: DiscrepancyAction = DiscrepancyAction.MARK_CANCELLED
    unexpected_position: DiscrepancyAction = DiscrepancyAction.HALT_AND_ALERT
    position_qty_mismatch: DiscrepancyAction = DiscrepancyAction.ADJUST_POSITION

    @classmethod
    def from_environment(cls) -> "DiscrepancyPolicy":
        """Build policy with explicit opt-in for whole-account orphan cleanup."""
        own_all = os.environ.get("OMS_OWN_ALL_BROKER_ORDERS", "").strip().lower()
        policy = cls()
        if own_all in {"1", "true", "yes", "y"}:
            policy.unknown_order_orphan = DiscrepancyAction.CANCEL
        return policy
