"""Trace ID generation for request tracking."""
import uuid


def generate_trace_id(oms_order_id: str = "") -> str:
    """Deterministic trace ID per intent. If oms_order_id provided, derive from it."""
    if oms_order_id:
        return f"tr-{oms_order_id[:8]}-{uuid.uuid4().hex[:6]}"
    return f"tr-{uuid.uuid4().hex[:14]}"
