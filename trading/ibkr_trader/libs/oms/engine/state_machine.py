"""Order state machine with valid transitions."""
import logging
from ..models.order import OMSOrder, OrderStatus, TERMINAL_STATUSES

logger = logging.getLogger(__name__)

# Valid state transitions
TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.CREATED: {OrderStatus.RISK_APPROVED, OrderStatus.REJECTED},
    OrderStatus.RISK_APPROVED: {
        OrderStatus.QUEUED,
        OrderStatus.ROUTED,
        OrderStatus.REJECTED,
        OrderStatus.CANCEL_REQUESTED,
        OrderStatus.EXPIRED,
    },
    OrderStatus.QUEUED: {
        OrderStatus.RISK_APPROVED,
        OrderStatus.ROUTED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    },
    # H3: CANCELLED added — broker can cancel before ACK (race condition)
    # RISK_APPROVED added — retryable broker rejects re-route through risk gateway
    OrderStatus.ROUTED: {
        OrderStatus.ACKED,
        OrderStatus.REJECTED,
        OrderStatus.CANCEL_REQUESTED,
        OrderStatus.CANCELLED,
        OrderStatus.RISK_APPROVED,
    },
    OrderStatus.ACKED: {
        OrderStatus.WORKING,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
    },
    OrderStatus.WORKING: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCEL_REQUESTED,
        OrderStatus.REPLACE_REQUESTED,
        OrderStatus.CANCELLED,
        OrderStatus.EXPIRED,
        OrderStatus.REJECTED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCEL_REQUESTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.CANCEL_REQUESTED: {
        OrderStatus.CANCELLED,
        OrderStatus.FILLED,
        OrderStatus.PARTIALLY_FILLED,
    },
    OrderStatus.REPLACE_REQUESTED: {
        OrderStatus.REPLACED,
        OrderStatus.CANCELLED,
        OrderStatus.FILLED,
        OrderStatus.REJECTED,
    },
    OrderStatus.REPLACED: {OrderStatus.DONE},
    OrderStatus.CANCELLED: {OrderStatus.DONE},
    OrderStatus.FILLED: {OrderStatus.DONE},
    OrderStatus.REJECTED: {OrderStatus.DONE},
    OrderStatus.EXPIRED: {OrderStatus.DONE},
}


def transition(order: OMSOrder, new_status: OrderStatus) -> bool:
    """Attempt state transition. Returns True if valid, False if rejected.
    Logs warning on invalid transition but does NOT raise — operational safety.
    """
    allowed = TRANSITIONS.get(order.status, set())
    if new_status not in allowed:
        logger.warning(
            f"Invalid transition: {order.oms_order_id} "
            f"{order.status.value} -> {new_status.value}"
        )
        return False
    order.status = new_status
    return True


def is_terminal(order: OMSOrder) -> bool:
    return order.status in TERMINAL_STATUSES


def is_done(order: OMSOrder) -> bool:
    return order.status == OrderStatus.DONE
