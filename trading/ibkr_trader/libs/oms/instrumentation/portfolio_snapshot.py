"""Portfolio snapshot payload builder."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import lineage_to_payload

from ._shared import as_float, direction_from_qty, event_time, plain, redact_account_payload, stable_payload_hash


def _exposures(positions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_strategy: dict[str, float] = defaultdict(float)
    by_family: dict[str, float] = defaultdict(float)
    by_symbol: dict[str, float] = defaultdict(float)
    by_sector: dict[str, float] = defaultdict(float)
    by_direction: dict[str, float] = defaultdict(float)
    gross = 0.0
    net = 0.0
    for raw in positions:
        pos = dict(raw)
        qty = as_float(pos.get("qty", pos.get("net_qty", 0.0)))
        price = as_float(pos.get("mark_price", pos.get("avg_price", 0.0)))
        signed = qty * price
        notional = abs(signed)
        gross += notional
        net += signed
        by_strategy[str(pos.get("strategy_id") or "_UNKNOWN_")] += notional
        by_family[str(pos.get("family_id") or "_UNKNOWN_")] += notional
        by_symbol[str(pos.get("symbol") or pos.get("instrument_symbol") or "_UNKNOWN_")] += notional
        by_sector[str(pos.get("sector") or "_UNKNOWN_")] += notional
        by_direction[direction_from_qty(qty)] += notional
    return {
        "gross_exposure": gross,
        "net_exposure": net,
        "exposure_by_strategy": dict(by_strategy),
        "exposure_by_family": dict(by_family),
        "exposure_by_symbol": dict(by_symbol),
        "exposure_by_sector": dict(by_sector),
        "exposure_by_direction": dict(by_direction),
        "open_position_count": sum(1 for p in positions if as_float(p.get("qty", p.get("net_qty", 0))) != 0),
    }


def build_portfolio_snapshot(
    *,
    lineage: Any = None,
    positions: Sequence[Mapping[str, Any]] | None = None,
    portfolio_risk: Mapping[str, Any] | None = None,
    account_state: Mapping[str, Any] | None = None,
    source: str = "oms_db",
    reconciliation_status: str = "unverified",
    timestamp: Any = None,
) -> dict[str, Any]:
    account_alias = str(lineage_to_payload(lineage).get("account_alias", "") or "")
    pos_list = [redact_account_payload(p, account_alias) for p in list(positions or [])]
    risk = plain(dict(portfolio_risk or {}))
    account = redact_account_payload(dict(account_state or {}), account_alias)
    exposure = _exposures(pos_list)

    payload = {
        "timestamp": event_time(timestamp),
        "snapshot_id": stable_payload_hash("portfolio_snap_", {"positions": pos_list, "risk": risk, "source": source}),
        "source": source,
        "equity": as_float(account.get("equity", account.get("net_liquidation", 0.0))),
        "cash": as_float(account.get("cash")),
        "buying_power": as_float(account.get("buying_power")),
        "margin_used": as_float(account.get("margin_used")),
        "currency": account.get("currency", "USD"),
        "account_alias": account.get("account_alias", account_alias),
        **exposure,
        "portfolio_heat_R": as_float(risk.get("open_risk_R")),
        "portfolio_heat_cap_R": as_float(risk.get("heat_cap_R", risk.get("portfolio_heat_cap_R", 0.0))),
        "daily_realized_R": as_float(risk.get("daily_realized_R")),
        "weekly_realized_R": as_float(risk.get("weekly_realized_R")),
        "daily_stop_R": as_float(risk.get("daily_stop_R", risk.get("portfolio_daily_stop_R", 0.0))),
        "weekly_stop_R": as_float(risk.get("weekly_stop_R", risk.get("portfolio_weekly_stop_R", 0.0))),
        "drawdown_pct": as_float(risk.get("drawdown_pct")),
        "drawdown_tier": risk.get("drawdown_tier", ""),
        "drawdown_size_multiplier": risk.get("drawdown_size_multiplier"),
        "open_position_counts": {"total": exposure["open_position_count"]},
        "max_position_caps": risk.get("max_position_caps", {}),
        "stale_data_flags": risk.get("stale_data_flags", []),
        "reconciliation_status": reconciliation_status,
        "positions": pos_list,
    }
    return enrich_payload(payload, lineage=lineage, event_type="portfolio_snapshot", scope="portfolio")
