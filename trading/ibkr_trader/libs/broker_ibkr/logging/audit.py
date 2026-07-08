"""Broker interaction audit logging."""
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("broker_ibkr.audit")


def log_broker_command(
    trace_id: str,
    oms_order_id: str,
    command: str,  # "submit", "cancel", "replace"
    payload: dict,
) -> None:
    logger.info(
        json.dumps(
            {
                "type": "broker_command",
                "trace_id": trace_id,
                "oms_order_id": oms_order_id,
                "command": command,
                "payload": payload,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
    )


def log_broker_response(
    trace_id: str,
    oms_order_id: str,
    event_type: str,  # "ack", "reject", "fill", "status"
    payload: dict,
) -> None:
    logger.info(
        json.dumps(
            {
                "type": "broker_response",
                "trace_id": trace_id,
                "oms_order_id": oms_order_id,
                "event_type": event_type,
                "payload": payload,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
    )
