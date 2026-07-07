"""Family daily reconciliation snapshot builder."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from libs.instrumentation.event_contract import enrich_payload

from ._shared import artifact_hashes, as_float, as_int, event_time, plain, stable_payload_hash


def build_family_daily_snapshot(
    *,
    data_dir: str,
    lineage: Any = None,
    date_str: str,
    timezone_name: str = "UTC",
    family_id: str = "",
    daily_snapshot: Mapping[str, Any] | None = None,
    portfolio_snapshot: Mapping[str, Any] | None = None,
    allocation_snapshot: Mapping[str, Any] | None = None,
    portfolio_rules: Sequence[Mapping[str, Any]] | None = None,
    source: str = "daily_closeout",
) -> dict[str, Any]:
    daily = plain(dict(daily_snapshot or {}))
    portfolio = plain(dict(portfolio_snapshot or {}))
    allocation = plain(dict(allocation_snapshot or {}))
    per_strategy = dict(daily.get("per_strategy_summary") or {})
    rules = [plain(rule) for rule in list(portfolio_rules or [])]
    blocks = sum(1 for rule in rules if str(rule.get("result", "")).lower() == "block")
    scales = sum(1 for rule in rules if str(rule.get("result", "")).lower() == "scale")
    trade_count = as_int(daily.get("total_trades", daily.get("trade_count", 0)))
    win_count = as_int(daily.get("win_count", 0))
    loss_count = as_int(daily.get("loss_count", 0))
    gross_pnl = as_float(daily.get("gross_pnl", daily.get("gross_profit", 0.0)))
    net_pnl = as_float(daily.get("net_pnl", daily.get("pnl", 0.0)))
    fees = as_float(daily.get("fees_paid", daily.get("fees", 0.0)))

    payload = {
        "timestamp": event_time(),
        "snapshot_id": stable_payload_hash("family_snap_", {"date": date_str, "family_id": family_id, "daily": daily}),
        "date": date_str,
        "timezone": timezone_name,
        "source": source,
        "family_id": family_id,
        "active_strategy_ids": sorted(per_strategy.keys()),
        "active_strategy_count": len(per_strategy),
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "trade_count": trade_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": (win_count / trade_count) if trade_count else 0.0,
        "profit_factor": as_float(daily.get("profit_factor")),
        "max_drawdown": as_float(daily.get("max_drawdown", daily.get("max_drawdown_pct", 0.0))),
        "average_exposure": as_float(daily.get("average_exposure", portfolio.get("gross_exposure", 0.0))),
        "family_allocation_target": allocation.get("family_target_weights", {}).get(family_id),
        "realized_allocation": allocation.get("observed_family_weights", {}).get(family_id),
        "allocation_drift": allocation.get("drift_by_family", {}).get(family_id),
        "blocks_by_portfolio_rule": blocks,
        "scales_by_portfolio_rule": scales,
        "concurrent_position_counts": portfolio.get("open_position_counts", {}),
        "same_symbol_overlaps": daily.get("same_symbol_overlaps", 0),
        "sector_concentration": portfolio.get("exposure_by_sector", {}),
        "lineage_summary": {
            "lineage_gap": bool(daily.get("lineage_gaps") or portfolio.get("lineage_gaps") or allocation.get("lineage_gaps")),
            "artifact_hashes": artifact_hashes(data_dir, date_str),
        },
        "daily_snapshot": daily,
        "portfolio_snapshot_id": portfolio.get("snapshot_id", ""),
        "allocation_snapshot_id": allocation.get("snapshot_id", ""),
    }
    return enrich_payload(payload, lineage=lineage, event_type="family_daily_snapshot", scope="family")
