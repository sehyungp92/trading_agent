"""Position snapshot payload builder."""
from __future__ import annotations

from typing import Any, Mapping

from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import lineage_to_payload

from ._shared import (
    account_alias_for,
    as_float,
    direction_from_qty,
    event_time,
    plain,
    redact_account_payload,
    stable_payload_hash,
)


def build_position_snapshot(
    position: Mapping[str, Any],
    *,
    lineage: Any = None,
    fill: Mapping[str, Any] | None = None,
    order: Mapping[str, Any] | None = None,
    source: str = "oms_db",
    source_snapshot_id: str = "",
    timestamp: Any = None,
) -> dict[str, Any]:
    raw_pos = plain(dict(position or {}))
    raw_fill = plain(dict(fill or {}))
    raw_order = plain(dict(order or {}))
    lineage_alias = str(lineage_to_payload(lineage).get("account_alias", "") or "")
    account_alias = account_alias_for(
        raw_pos.get("account_id") or raw_order.get("account_id") or raw_fill.get("account_id", ""),
        lineage_alias
        or raw_pos.get("account_alias")
        or raw_order.get("account_alias")
        or raw_fill.get("account_alias", ""),
    )
    pos = redact_account_payload(raw_pos, account_alias)
    fill_data = redact_account_payload(raw_fill, account_alias)
    order_data = redact_account_payload(raw_order, account_alias)
    symbol = str(
        pos.get("symbol")
        or pos.get("instrument_symbol")
        or fill_data.get("symbol")
        or order_data.get("symbol")
        or ""
    )
    strategy_id = str(pos.get("strategy_id") or order_data.get("strategy_id") or fill_data.get("strategy_id") or "")
    qty = as_float(pos.get("qty", pos.get("net_qty", 0.0)))
    avg_price = as_float(pos.get("avg_price"))
    mark_price = as_float(pos.get("mark_price", fill_data.get("price", avg_price)))

    payload = {
        "timestamp": event_time(timestamp or pos.get("last_update_at") or fill_data.get("timestamp")),
        "snapshot_id": source_snapshot_id or stable_payload_hash(
            "pos_snap_",
            {"strategy_id": strategy_id, "symbol": symbol, "qty": qty, "timestamp": event_time(timestamp)},
        ),
        "position_id": stable_payload_hash(
            "pos_",
            {
                "portfolio_id": pos.get("portfolio_id", ""),
                "account_alias": pos.get("account_alias", ""),
                "strategy_id": strategy_id,
                "symbol": symbol,
            },
        ),
        "portfolio_id": pos.get("portfolio_id", order_data.get("portfolio_id", "")),
        "account_alias": account_alias,
        "family_id": pos.get("family_id", order_data.get("family_id", "")),
        "strategy_id": strategy_id,
        "symbol": symbol,
        "asset_class": pos.get("asset_class", ""),
        "sector": pos.get("sector", ""),
        "industry": pos.get("industry", ""),
        "direction": pos.get("direction") or direction_from_qty(qty),
        "qty": qty,
        "avg_price": avg_price,
        "mark_price": mark_price,
        "notional": abs(qty) * mark_price,
        "unrealized_pnl": as_float(pos.get("unrealized_pnl")),
        "realized_pnl": as_float(pos.get("realized_pnl")),
        "open_risk_R": as_float(pos.get("open_risk_R")),
        "open_risk_dollars": as_float(pos.get("open_risk_dollars")),
        "stop_price": pos.get("stop_price"),
        "target_price": pos.get("target_price"),
        "entry_time": pos.get("entry_time", ""),
        "last_update_time": event_time(pos.get("last_update_at") or fill_data.get("timestamp")),
        "source": source,
        "source_snapshot_id": source_snapshot_id,
        "fill": fill_data,
        "order": order_data,
    }
    return enrich_payload(payload, lineage=lineage, event_type="position_snapshot", scope="portfolio")
