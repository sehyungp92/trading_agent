"""Crisis overlay for PortfolioRulesConfig.

Applies crisis-level risk reduction on top of regime-adjusted rules.
Uses the same _scale_dd_tiers() helper and dataclasses.replace() pattern
as regime/integration.py.

Key design: crisis can only TIGHTEN risk, never loosen.
"""
from __future__ import annotations

import dataclasses
import logging

from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
from regime.crisis.actions import resolve_crisis_action
from regime.crisis.context import CrisisContext
from regime.integration import _scale_dd_tiers

logger = logging.getLogger(__name__)


def apply_crisis_overlay(
    rules: PortfolioRulesConfig,
    crisis_ctx: CrisisContext,
    family_id: str | None = None,
    regime: str | None = None,
) -> PortfolioRulesConfig:
    """Apply crisis risk reduction to regime-adjusted rules.

    Multiplicative on regime_unit_risk_mult and family-specific caps, with
    _scale_dd_tiers on dd_tiers. Only tightens: all multipliers are clamped
    at <= 1.0.

    Args:
        rules: Regime-adjusted PortfolioRulesConfig (from build_*_rules)
        crisis_ctx: Current CrisisContext
        family_id: Optional downstream family name for action-layer policy.
        regime: Optional HMM regime code (G/R/S/D) for incremental action sizing.

    Returns:
        New PortfolioRulesConfig with crisis overlay applied.
    """
    action = resolve_crisis_action(crisis_ctx, family_id, regime=regime)
    if action.is_no_action():
        return rules

    new_rules = dataclasses.replace(
        rules,
        directional_cap_R=_scale_cap(
            rules.directional_cap_R,
            action.directional_cap_multiplier,
        ),
        directional_cap_long_R=_scale_optional_cap(
            rules.directional_cap_long_R,
            action.long_cap_multiplier,
        ),
        directional_cap_short_R=_scale_optional_cap(
            rules.directional_cap_short_R,
            action.short_cap_multiplier,
        ),
        priority_headroom_R=_scale_optional_cap(
            rules.priority_headroom_R,
            action.priority_headroom_multiplier,
        ),
        max_family_contracts_mnq_eq=_scale_contract_cap(
            rules.max_family_contracts_mnq_eq,
            action.max_family_contracts_multiplier,
        ),
        regime_unit_risk_mult=round(
            rules.regime_unit_risk_mult * action.risk_multiplier,
            4,
        ),
        regime_unit_risk_long_mult=round(
            rules.regime_unit_risk_long_mult * action.long_risk_multiplier,
            4,
        ),
        regime_unit_risk_short_mult=round(
            rules.regime_unit_risk_short_mult * action.short_risk_multiplier,
            4,
        ),
        dd_tiers=_scale_dd_tiers(rules.dd_tiers, action.dd_tier_multiplier),
        disabled_strategies=frozenset(
            set(rules.disabled_strategies) | set(action.disabled_strategies)
        ),
    )

    logger.info(
        "Crisis overlay applied: family=%s regime=%s level=%s provenance=%s, "
        "risk_mult=%.2f->%.4f, long_risk=%.2f, short_risk=%.2f, "
        "dd_mult=%.2f, cap_mult=%.2f, long_mult=%.2f, short_mult=%.2f",
        action.family_id,
        action.regime,
        crisis_ctx.alert_level,
        action.action_provenance,
        rules.regime_unit_risk_mult,
        new_rules.regime_unit_risk_mult,
        action.long_risk_multiplier,
        action.short_risk_multiplier,
        action.dd_tier_multiplier,
        action.directional_cap_multiplier,
        action.long_cap_multiplier,
        action.short_cap_multiplier,
    )

    return new_rules


def _scale_cap(value: float, multiplier: float) -> float:
    return round(max(0.0, float(value) * min(float(multiplier), 1.0)), 4)


def _scale_optional_cap(value: float, multiplier: float) -> float:
    if value <= 0:
        return 0.0
    return _scale_cap(value, multiplier)


def _scale_contract_cap(value: int, multiplier: float) -> int:
    if value <= 0:
        return 0
    return max(1, int(value * min(float(multiplier), 1.0)))
