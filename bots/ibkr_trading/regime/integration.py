"""Regime-to-config mapping tables and builder functions.

All policy for how macro regimes translate into portfolio rules, strategy
profiles, and overlay weights is centralized here. Optimized live/backtest
portfolio configs remain the baseline; regime logic applies bounded relative
tilts so it cannot silently replace family-specific optimized rules.
"""
from __future__ import annotations

import dataclasses
import logging
import math

from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
from regime.context import RegimeContext

logger = logging.getLogger(__name__)

_VALID_REGIMES = frozenset({"G", "R", "S", "D"})
_DEFAULT_DD_TIERS = PortfolioRulesConfig().dd_tiers


def _validated_regime(regime: str) -> str:
    """Return regime if valid, else fall back to 'G' with a warning."""
    if regime in _VALID_REGIMES:
        return regime
    logger.warning("Unknown regime %r, falling back to Recovery (G)", regime)
    return "G"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _finite_float(value: float | int | None, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _context_sizing_mult(ctx: RegimeContext) -> float:
    """Bound model leverage so regime can tilt, not replace, optimized risk.

    ``suggested_leverage_mult`` is already the regime engine's final leverage
    output, including its deliberately tiny stress-HMM dampener. Do not apply
    ``stress_level`` again here; the daily crisis detector is the actionable
    tail-risk layer.
    """
    leverage = _finite_float(ctx.suggested_leverage_mult, 1.0)
    leverage = _clamp(leverage, 0.50, 1.05)

    return round(_clamp(leverage, 0.45, 1.05), 4)


def _scale(value: float, multiplier: float) -> float:
    return round(float(value) * float(multiplier), 4)


def _scale_optional_cap(value: float, multiplier: float) -> float:
    if value <= 0:
        return 0.0
    return _scale(value, multiplier)


def _scale_dd_tiers(
    base_tiers: tuple[tuple[float, float], ...],
    threshold_mult: float,
) -> tuple[tuple[float, float], ...]:
    """Scale thresholds while preserving optimized size multipliers."""
    scaled = []
    for threshold, multiplier in base_tiers:
        if threshold >= 0.999:
            scaled.append((threshold, multiplier))
        else:
            scaled.append((round(max(0.001, threshold * threshold_mult), 4), multiplier))
    return tuple(scaled)


# Tier 1: portfolio rule overlays per regime and family.
#
# The directional_cap_R fields are reference values for the latest optimized
# live baselines. Builders use the multiplier fields against the supplied
# base_cfg so future optimized ports remain the source of truth.

STOCK_RULES: dict[str, dict] = {
    "G": {"directional_cap_R": 6.25, "cap_mult": 1.00, "regime_unit_risk_mult": 1.00,
          "risk_mult": 1.00, "priority_headroom_R": 1.15, "headroom_mult": 1.00,
          "symbol_collision_action": "half_size", "dd_threshold_mult": 1.00},
    "R": {"directional_cap_R": 5.625, "cap_mult": 0.90, "regime_unit_risk_mult": 0.95,
          "risk_mult": 0.95, "priority_headroom_R": 1.035, "headroom_mult": 0.90,
          "symbol_collision_action": "half_size", "dd_threshold_mult": 0.95},
    "S": {"directional_cap_R": 4.5, "cap_mult": 0.72, "regime_unit_risk_mult": 0.75,
          "risk_mult": 0.75, "priority_headroom_R": 0.805, "headroom_mult": 0.70,
          "symbol_collision_action": "block", "dd_threshold_mult": 0.85},
    "D": {"directional_cap_R": 3.4375, "cap_mult": 0.55, "regime_unit_risk_mult": 0.55,
          "risk_mult": 0.55, "priority_headroom_R": 0.6325, "headroom_mult": 0.55,
          "symbol_collision_action": "block", "dd_threshold_mult": 0.70},
}

MOMENTUM_RULES: dict[str, dict] = {
    "G": {"directional_cap_R": 4.25, "directional_cap_long_R": 10.0, "directional_cap_short_R": 7.35,
          "cap_mult": 1.00, "long_cap_mult": 1.00, "short_cap_mult": 0.70,
          "regime_unit_risk_mult": 1.00, "risk_mult": 1.00, "nqdtc_oppose_size_mult": 0.50,
          "nqdtc_direction_filter_enabled": True, "max_contracts_scale": 1.00,
          "dd_threshold_mult": 1.00},
    "R": {"directional_cap_R": 4.0375, "directional_cap_long_R": 10.0, "directional_cap_short_R": 8.925,
          "cap_mult": 0.95, "long_cap_mult": 1.00, "short_cap_mult": 0.85,
          "regime_unit_risk_mult": 0.95, "risk_mult": 0.95, "nqdtc_oppose_size_mult": 0.50,
          "nqdtc_direction_filter_enabled": True, "max_contracts_scale": 0.95,
          "dd_threshold_mult": 0.95},
    "S": {"directional_cap_R": 3.1875, "directional_cap_long_R": 7.0, "directional_cap_short_R": 10.5,
          "cap_mult": 0.75, "long_cap_mult": 0.70, "short_cap_mult": 1.00,
          "regime_unit_risk_mult": 0.75, "risk_mult": 0.75, "nqdtc_oppose_size_mult": 0.50,
          "nqdtc_direction_filter_enabled": True, "max_contracts_scale": 0.75,
          "dd_threshold_mult": 0.85},
    "D": {"directional_cap_R": 2.55, "directional_cap_long_R": 4.5, "directional_cap_short_R": 10.5,
          "cap_mult": 0.60, "long_cap_mult": 0.45, "short_cap_mult": 1.00,
          "regime_unit_risk_mult": 0.60, "risk_mult": 0.60, "nqdtc_oppose_size_mult": 0.50,
          "nqdtc_direction_filter_enabled": True, "max_contracts_scale": 0.50,
          "dd_threshold_mult": 0.70},
}

SWING_RULES: dict[str, dict] = {
    "G": {"directional_cap_R": 3.75, "directional_cap_long_R": 3.0, "directional_cap_short_R": 3.25,
          "cap_mult": 1.00, "long_cap_mult": 1.00, "short_cap_mult": 1.00,
          "regime_unit_risk_mult": 1.00, "risk_mult": 1.00, "dd_threshold_mult": 1.00},
    "R": {"directional_cap_R": 3.375, "directional_cap_long_R": 2.85, "directional_cap_short_R": 2.925,
          "cap_mult": 0.90, "long_cap_mult": 0.95, "short_cap_mult": 0.90,
          "regime_unit_risk_mult": 0.95, "risk_mult": 0.95, "dd_threshold_mult": 0.95},
    "S": {"directional_cap_R": 2.8125, "directional_cap_long_R": 1.95, "directional_cap_short_R": 3.25,
          "cap_mult": 0.75, "long_cap_mult": 0.65, "short_cap_mult": 1.00,
          "regime_unit_risk_mult": 0.80, "risk_mult": 0.80, "dd_threshold_mult": 0.85},
    "D": {"directional_cap_R": 2.25, "directional_cap_long_R": 1.35, "directional_cap_short_R": 3.25,
          "cap_mult": 0.60, "long_cap_mult": 0.45, "short_cap_mult": 1.00,
          "regime_unit_risk_mult": 0.65, "risk_mult": 0.65, "dd_threshold_mult": 0.70},
}

# Compatibility/default tiers. Builders scale the supplied base_cfg.dd_tiers so
# family-specific optimized drawdown multipliers are preserved.
DD_TIERS: dict[str, tuple[tuple[float, float], ...]] = {
    regime: _scale_dd_tiers(_DEFAULT_DD_TIERS, rules["dd_threshold_mult"])
    for regime, rules in STOCK_RULES.items()
}

# Overlay weights (swing only).
OVERLAY_WEIGHTS: dict[str, dict[str, float]] = {
    "G": {"QQQ": 0.60, "GLD": 0.40},
    "R": {"QQQ": 0.50, "GLD": 0.50},
    "S": {"QQQ": 0.20, "GLD": 0.80},
    "D": {"QQQ": 0.00, "GLD": 1.00},
}

# Tier 2: Per-regime strategy profiles. Growth values match the optimized
# live baseline, while stress/defensive regimes reduce concentration.
STOCK_PROFILES: dict[str, dict] = {
    "G": {"alcb_max_positions": 6, "iaric_pb_max_positions": 10, "disabled": frozenset()},
    "R": {"alcb_max_positions": 5, "iaric_pb_max_positions": 8, "disabled": frozenset()},
    "S": {"alcb_max_positions": 3, "iaric_pb_max_positions": 5, "disabled": frozenset()},
    "D": {"alcb_max_positions": 2, "iaric_pb_max_positions": 3, "disabled": frozenset()},
}

MOMENTUM_PROFILES: dict[str, dict] = {
    "G": {"disabled": frozenset({"DownturnDominator_v1"})},
    "R": {"disabled": frozenset({"DownturnDominator_v1"})},
    "S": {"disabled": frozenset()},
    "D": {"disabled": frozenset()},
}


def build_stock_rules(ctx: RegimeContext, base_cfg: PortfolioRulesConfig) -> PortfolioRulesConfig:
    """Return new PortfolioRulesConfig with regime-adjusted values for stock family."""
    regime = _validated_regime(ctx.regime)
    r = STOCK_RULES[regime]
    profile = STOCK_PROFILES[regime]
    context_mult = _context_sizing_mult(ctx)
    return dataclasses.replace(
        base_cfg,
        directional_cap_R=_scale(base_cfg.directional_cap_R, r["cap_mult"]),
        directional_cap_long_R=_scale_optional_cap(base_cfg.directional_cap_long_R, r["cap_mult"]),
        directional_cap_short_R=_scale_optional_cap(base_cfg.directional_cap_short_R, r["cap_mult"]),
        regime_unit_risk_mult=_scale(base_cfg.regime_unit_risk_mult, r["risk_mult"] * context_mult),
        priority_headroom_R=_scale(base_cfg.priority_headroom_R, r["headroom_mult"]),
        symbol_collision_action=r["symbol_collision_action"],
        dd_tiers=_scale_dd_tiers(base_cfg.dd_tiers, r["dd_threshold_mult"]),
        disabled_strategies=profile["disabled"],
    )


def build_momentum_rules(
    ctx: RegimeContext,
    base_cfg: PortfolioRulesConfig,
    base_max_contracts: int,
) -> PortfolioRulesConfig:
    """Return new PortfolioRulesConfig with regime-adjusted values for momentum family."""
    regime = _validated_regime(ctx.regime)
    r = MOMENTUM_RULES[regime]
    profile = MOMENTUM_PROFILES[regime]
    context_mult = _context_sizing_mult(ctx)
    contract_cap = int(base_max_contracts * r["max_contracts_scale"] * context_mult)
    contract_cap = min(base_max_contracts, max(1, contract_cap))
    return dataclasses.replace(
        base_cfg,
        directional_cap_R=_scale(base_cfg.directional_cap_R, r["cap_mult"]),
        directional_cap_long_R=_scale_optional_cap(base_cfg.directional_cap_long_R, r["long_cap_mult"]),
        directional_cap_short_R=_scale_optional_cap(base_cfg.directional_cap_short_R, r["short_cap_mult"]),
        nqdtc_direction_filter_enabled=(
            base_cfg.nqdtc_direction_filter_enabled and r["nqdtc_direction_filter_enabled"]
        ),
        nqdtc_oppose_size_mult=min(base_cfg.nqdtc_oppose_size_mult, r["nqdtc_oppose_size_mult"]),
        max_family_contracts_mnq_eq=contract_cap,
        regime_unit_risk_mult=_scale(base_cfg.regime_unit_risk_mult, r["risk_mult"] * context_mult),
        dd_tiers=_scale_dd_tiers(base_cfg.dd_tiers, r["dd_threshold_mult"]),
        disabled_strategies=profile["disabled"],
    )


def build_swing_rules(ctx: RegimeContext, base_cfg: PortfolioRulesConfig) -> PortfolioRulesConfig:
    """Return new PortfolioRulesConfig with regime-adjusted values for swing family."""
    regime = _validated_regime(ctx.regime)
    r = SWING_RULES[regime]
    context_mult = _context_sizing_mult(ctx)
    return dataclasses.replace(
        base_cfg,
        directional_cap_R=_scale(base_cfg.directional_cap_R, r["cap_mult"]),
        directional_cap_long_R=_scale_optional_cap(base_cfg.directional_cap_long_R, r["long_cap_mult"]),
        directional_cap_short_R=_scale_optional_cap(base_cfg.directional_cap_short_R, r["short_cap_mult"]),
        regime_unit_risk_mult=_scale(base_cfg.regime_unit_risk_mult, r["risk_mult"] * context_mult),
        dd_tiers=_scale_dd_tiers(base_cfg.dd_tiers, r["dd_threshold_mult"]),
    )
