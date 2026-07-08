"""Allocation snapshot payload builder."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import lineage_to_payload

from ._shared import as_float, event_time, plain, redact_account_payload, stable_payload_hash


def _observed_weights(positions: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, float], float]:
    strategy_notional: dict[str, float] = defaultdict(float)
    family_notional: dict[str, float] = defaultdict(float)
    for raw in positions:
        pos = dict(raw)
        qty = as_float(pos.get("qty", pos.get("net_qty", 0.0)))
        price = as_float(pos.get("mark_price", pos.get("avg_price", 0.0)))
        notional = abs(qty * price)
        strategy_notional[str(pos.get("strategy_id") or "_UNKNOWN_")] += notional
        family_notional[str(pos.get("family_id") or "_UNKNOWN_")] += notional
    gross = sum(strategy_notional.values())
    if gross <= 0:
        return {}, {}, 0.0
    return (
        {key: value / gross for key, value in sorted(strategy_notional.items())},
        {key: value / gross for key, value in sorted(family_notional.items())},
        gross,
    )


def _drift(target: Mapping[str, Any], observed: Mapping[str, float]) -> dict[str, float]:
    keys = set(target) | set(observed)
    return {
        key: observed.get(key, 0.0) - as_float(target.get(key, 0.0))
        for key in sorted(keys)
    }


def build_allocation_snapshot(
    *,
    lineage: Any = None,
    positions: Sequence[Mapping[str, Any]] | None = None,
    targets: Mapping[str, Any] | None = None,
    raw_nav: float = 0.0,
    allocated_nav: float = 0.0,
    source: str = "fill",
    timestamp: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    account_alias = str(lineage_to_payload(lineage).get("account_alias", "") or "")
    pos_list = [redact_account_payload(p, account_alias) for p in list(positions or [])]
    target_map = plain(dict(targets or {}))
    metadata_payload = redact_account_payload(dict(metadata or {}), account_alias)
    family_targets = dict(target_map.get("families", target_map.get("family_target_weights", {})) or {})
    strategy_targets = dict(target_map.get("strategies", target_map.get("strategy_target_weights", {})) or {})
    observed_strategy, observed_family, gross = _observed_weights(pos_list)
    raw_nav_value = as_float(raw_nav, 0.0)
    if raw_nav_value <= 0:
        raw_nav_value = gross
    allocated_nav_value = as_float(allocated_nav, 0.0)
    if allocated_nav_value <= 0:
        allocated_nav_value = raw_nav_value or gross

    payload = {
        "timestamp": event_time(timestamp),
        "snapshot_id": stable_payload_hash("alloc_snap_", {"positions": pos_list, "source": source}),
        "source": source,
        "family_target_weights": family_targets,
        "strategy_target_weights": strategy_targets,
        "observed_family_weights": observed_family,
        "observed_strategy_weights": observed_strategy,
        "drift_by_family": _drift(family_targets, observed_family),
        "drift_by_strategy": _drift(strategy_targets, observed_strategy),
        "raw_nav": raw_nav_value,
        "allocated_nav": allocated_nav_value,
        "metadata": metadata_payload,
    }
    return enrich_payload(payload, lineage=lineage, event_type="allocation_snapshot", scope="portfolio")
