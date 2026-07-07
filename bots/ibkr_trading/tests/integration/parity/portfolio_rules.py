from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from libs.oms.risk.portfolio_rules import PortfolioRulesConfig


def portfolio_rules_config_from_fixture(fixture: Mapping[str, Any]) -> PortfolioRulesConfig | None:
    family_cfg = fixture.get("family_config", {}) or {}
    payload = dict(family_cfg.get("portfolio_rules", {}) or {})
    family = str(fixture.get("family", "") or family_cfg.get("family", ""))
    strategy_ids = tuple(
        str(item["id"])
        for item in family_cfg.get("strategies", []) or []
        if item.get("id")
    )
    if not payload and family not in {"momentum", "stock", "swing"}:
        return None

    if family == "momentum":
        return _momentum_rules(payload, fixture, strategy_ids)
    if family == "stock":
        return _stock_rules(payload, fixture, strategy_ids)
    if family == "swing":
        return _swing_rules(payload, fixture, strategy_ids)
    return None


def _momentum_rules(
    payload: Mapping[str, Any],
    fixture: Mapping[str, Any],
    strategy_ids: tuple[str, ...],
) -> PortfolioRulesConfig:
    equity = float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0))
    reference_unit_risk = float(payload.get("reference_unit_risk_dollars", equity * 0.005))
    return PortfolioRulesConfig(
        initial_equity=equity,
        directional_cap_R=float(payload.get("directional_cap_R", 4.25)),
        directional_cap_long_R=float(payload.get("directional_cap_long_R", 10.0)),
        directional_cap_short_R=float(payload.get("directional_cap_short_R", 10.5)),
        max_total_active_positions=int(payload.get("max_total_active_positions", 8)),
        max_strategy_active_positions=_tuple_pairs(
            payload.get(
                "max_strategy_active_positions",
                (
                    ("NQ_REGIME", 3),
                    ("VdubusNQ_v4", 2),
                    ("NQDTC_v2.1", 2),
                    ("DownturnDominator_v1", 2),
                ),
            ),
            value_type=int,
            allowed=strategy_ids,
        ),
        max_family_contracts_mnq_eq=int(payload.get("max_family_contracts_mnq_eq", 40)),
        family_strategy_ids=strategy_ids,
        symbol_collision_action=str(payload.get("symbol_collision_action", "none")),
        cooldown_session_only=bool(payload.get("cooldown_session_only", True)),
        nqdtc_direction_filter_enabled=bool(payload.get("nqdtc_direction_filter_enabled", False)),
        nqdtc_agree_size_mult=float(payload.get("nqdtc_agree_size_mult", 1.25)),
        nqdtc_oppose_size_mult=float(payload.get("nqdtc_oppose_size_mult", 0.50)),
        strategy_priorities=_tuple_pairs(
            payload.get(
                "strategy_priorities",
                (
                    ("VdubusNQ_v4", 0),
                    ("NQ_REGIME", 0),
                    ("NQDTC_v2.1", 1),
                    ("DownturnDominator_v1", 1),
                ),
            ),
            value_type=int,
            allowed=strategy_ids,
        ),
        strategy_size_multipliers=_tuple_pairs(
            payload.get(
                "strategy_size_multipliers",
                (
                    ("NQ_REGIME", 0.75),
                    ("VdubusNQ_v4", 0.95),
                    ("NQDTC_v2.1", 1.0),
                    ("DownturnDominator_v1", 1.0),
                ),
            ),
            value_type=float,
            allowed=strategy_ids,
        ),
        priority_headroom_R=float(payload.get("priority_headroom_R", 1.0)),
        priority_reserve_threshold=int(payload.get("priority_reserve_threshold", 1)),
        reference_unit_risk_dollars=reference_unit_risk,
        portfolio_heat_cap_R=float(payload.get("portfolio_heat_cap_R", 10.0)),
        existing_position_mult=float(payload.get("existing_position_mult", 0.85)),
        heat_pressure_threshold=float(payload.get("heat_pressure_threshold", 0.65)),
        heat_pressure_mult=float(payload.get("heat_pressure_mult", 0.65)),
        same_direction_pressure_threshold=float(payload.get("same_direction_pressure_threshold", 0.65)),
        same_direction_pressure_mult=float(payload.get("same_direction_pressure_mult", 0.70)),
        max_trade_risk_R=float(payload.get("max_trade_risk_R", 2.0)),
        min_qty=int(payload.get("min_qty", 1)),
        fit_to_remaining_heat=bool(payload.get("fit_to_remaining_heat", True)),
        fit_to_remaining_directional_cap=bool(payload.get("fit_to_remaining_directional_cap", True)),
        fit_to_remaining_family_cap=bool(payload.get("fit_to_remaining_family_cap", True)),
        dd_tiers=_tuple_tuple_float(payload.get("dd_tiers", ((0.10, 1.00), (0.15, 0.60), (0.20, 0.30), (1.00, 0.00)))),
    )


def _stock_rules(
    payload: Mapping[str, Any],
    fixture: Mapping[str, Any],
    strategy_ids: tuple[str, ...],
) -> PortfolioRulesConfig:
    equity = float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0))
    reference_pct = float(payload.get("reference_unit_risk_pct", 0.00648))
    reference_unit_risk = float(payload.get("reference_unit_risk_dollars", max(equity * reference_pct, 1.0)))
    return PortfolioRulesConfig(
        initial_equity=equity,
        family_strategy_ids=strategy_ids,
        directional_cap_R=float(payload.get("directional_cap_R", 6.5)),
        directional_cap_long_R=float(payload.get("directional_cap_long_R", 6.25)),
        directional_cap_short_R=float(payload.get("directional_cap_short_R", 0.0)),
        symbol_collision_action=str(payload.get("symbol_collision_action", "half_size")),
        symbol_collision_pairs=_tuple_triples(
            payload.get("symbol_collision_pairs", ()),
            allowed=strategy_ids,
        ),
        strategy_priorities=_tuple_pairs(
            payload.get("strategy_priorities", (("IARIC_v1", 0), ("ALCB_v1", 1))),
            value_type=int,
            allowed=strategy_ids,
        ),
        priority_headroom_R=float(payload.get("priority_headroom_R", 1.15)),
        priority_reserve_threshold=int(payload.get("priority_reserve_threshold", 1)),
        reference_unit_risk_dollars=reference_unit_risk,
        reference_unit_risk_pct=reference_pct,
        max_total_active_positions=int(payload.get("max_total_active_positions", 12)),
        max_symbol_heat_R=float(payload.get("max_symbol_heat_R", 2.2)),
        same_sector_heat_cap_R=float(payload.get("same_sector_heat_cap_R", 3.8)),
        symbol_sector_map=_symbol_sector_map(fixture),
        max_single_strategy_trade_share=float(payload.get("max_single_strategy_trade_share", 0.85)),
        dynamic_allocation_enabled=bool(payload.get("dynamic_allocation_enabled", True)),
        dynamic_lookback_trades=int(payload.get("dynamic_lookback_trades", 60)),
        dynamic_min_mult=float(payload.get("dynamic_min_mult", 0.65)),
        dynamic_max_mult=float(payload.get("dynamic_max_mult", 1.22)),
        dynamic_positive_expectancy_boost=float(payload.get("dynamic_positive_expectancy_boost", 0.10)),
        dynamic_negative_expectancy_cut=float(payload.get("dynamic_negative_expectancy_cut", 0.18)),
        portfolio_heat_cap_R=float(payload.get("portfolio_heat_cap_R", payload.get("directional_cap_R", 6.5))),
        max_strategy_active_positions=_tuple_pairs(
            payload.get("max_strategy_active_positions", (("IARIC_v1", 9), ("ALCB_v1", 6))),
            value_type=int,
            allowed=strategy_ids,
        ),
        max_strategy_heat_R=_tuple_pairs(
            payload.get("max_strategy_heat_R", (("IARIC_v1", 4.6), ("ALCB_v1", 3.25))),
            value_type=float,
            allowed=strategy_ids,
        ),
        dd_tiers=_tuple_tuple_float(payload.get("dd_tiers", ((0.04, 1.00), (0.07, 0.75), (0.10, 0.40), (0.13, 0.00)))),
    )


def _swing_rules(
    payload: Mapping[str, Any],
    fixture: Mapping[str, Any],
    strategy_ids: tuple[str, ...],
) -> PortfolioRulesConfig:
    equity = float((fixture.get("account_state", {}) or {}).get("equity", 100_000.0))
    return PortfolioRulesConfig(
        initial_equity=float(payload.get("initial_equity", equity)),
        family_strategy_ids=strategy_ids,
        directional_cap_R=float(payload.get("directional_cap_R", 5.5)),
        directional_cap_long_R=float(payload.get("directional_cap_long_R", 4.0)),
        directional_cap_short_R=float(payload.get("directional_cap_short_R", 4.0)),
        symbol_collision_action=str(payload.get("symbol_collision_action", "half_size")),
        strategy_priorities=_tuple_pairs(
            payload.get("strategy_priorities", (("ATRSS", 0), ("TPC", 1), ("AKC_HELIX", 2), ("OVERLAY", 99))),
            value_type=int,
            allowed=strategy_ids,
        ),
        priority_headroom_R=float(payload.get("priority_headroom_R", 0.75)),
        priority_reserve_threshold=int(payload.get("priority_reserve_threshold", 1)),
        reference_unit_risk_dollars=float(payload.get("reference_unit_risk_dollars", 550.0)),
        dd_tiers=_tuple_tuple_float(payload.get("dd_tiers", ((0.04, 0.90), (0.07, 0.70), (0.10, 0.50), (0.14, 0.25), (0.18, 0.00)))),
        nqdtc_direction_filter_enabled=False,
    )


def _tuple_pairs(value: Any, *, value_type, allowed: tuple[str, ...]) -> tuple[tuple[str, Any], ...]:
    allowed_set = set(allowed)
    rows = value.items() if isinstance(value, Mapping) else value or ()
    pairs: list[tuple[str, Any]] = []
    for row in rows:
        if len(row) < 2:
            continue
        key = str(row[0])
        if allowed_set and key not in allowed_set:
            continue
        pairs.append((key, value_type(row[1])))
    return tuple(pairs)


def _tuple_triples(value: Any, *, allowed: tuple[str, ...]) -> tuple[tuple[str, str, str], ...]:
    allowed_set = set(allowed)
    triples: list[tuple[str, str, str]] = []
    for row in value or ():
        if len(row) < 3:
            continue
        holder, requester, action = str(row[0]), str(row[1]), str(row[2])
        if allowed_set and (holder not in allowed_set or requester not in allowed_set):
            continue
        triples.append((holder, requester, action))
    return tuple(triples)


def _tuple_tuple_float(value: Any) -> tuple[tuple[float, float], ...]:
    return tuple((float(row[0]), float(row[1])) for row in value or ())


def _symbol_sector_map(fixture: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    mapping: dict[str, str] = {}
    artifacts = fixture.get("artifacts", {}) or {}
    for family_key in ("iaric", "alcb"):
        for item in ((artifacts.get(family_key, {}) or {}).get("watchlist", []) or []):
            if item.get("symbol"):
                mapping[str(item["symbol"]).upper()] = str(item.get("sector", "Technology"))
        for item in ((artifacts.get(family_key, {}) or {}).get("candidates", []) or []):
            if item.get("symbol"):
                mapping[str(item["symbol"]).upper()] = str(item.get("sector", "Technology"))
    return tuple(sorted(mapping.items()))
