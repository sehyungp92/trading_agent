"""Family-specific crisis action policy.

The detector answers "how stressed is the market?"  This module answers
"how should each downstream family tighten when WARNING/CRISIS is active?"
Keeping that policy centralized makes action-layer optimization testable
without moving the already-validated detector thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass

from regime.crisis import config as C
from regime.crisis.context import CrisisContext


@dataclass(frozen=True)
class CrisisActionPolicy:
    """Tightening policy applied on top of regime-adjusted portfolio rules."""

    family_id: str = "generic"
    regime: str = ""
    alert_level: str = C.ALERT_NORMAL
    alert_level_int: int = 0
    action_provenance: str = "none"
    risk_multiplier: float = 1.0
    long_risk_multiplier: float = 1.0
    short_risk_multiplier: float = 1.0
    dd_tier_multiplier: float = 1.0
    directional_cap_multiplier: float = 1.0
    long_cap_multiplier: float = 1.0
    short_cap_multiplier: float = 1.0
    overlay_exposure_multiplier: float = 1.0
    priority_headroom_multiplier: float = 1.0
    max_family_contracts_multiplier: float = 1.0
    position_limit_multiplier: float = 1.0
    disabled_strategies: frozenset[str] = frozenset()

    def is_no_action(self) -> bool:
        """Return True when the policy leaves regime-adjusted rules unchanged."""
        return (
            self.alert_level_int <= 0
            and self.risk_multiplier >= 1.0
            and self.long_risk_multiplier >= 1.0
            and self.short_risk_multiplier >= 1.0
            and self.dd_tier_multiplier >= 1.0
            and self.directional_cap_multiplier >= 1.0
            and self.long_cap_multiplier >= 1.0
            and self.short_cap_multiplier >= 1.0
            and self.overlay_exposure_multiplier >= 1.0
            and self.priority_headroom_multiplier >= 1.0
            and self.max_family_contracts_multiplier >= 1.0
            and self.position_limit_multiplier >= 1.0
            and not self.disabled_strategies
        )

    def to_dict(self) -> dict:
        """Serialize for instrumentation and event logs."""
        return {
            "family_id": self.family_id,
            "regime": self.regime,
            "alert_level": self.alert_level,
            "alert_level_int": self.alert_level_int,
            "action_provenance": self.action_provenance,
            "risk_multiplier": self.risk_multiplier,
            "long_risk_multiplier": self.long_risk_multiplier,
            "short_risk_multiplier": self.short_risk_multiplier,
            "dd_tier_multiplier": self.dd_tier_multiplier,
            "directional_cap_multiplier": self.directional_cap_multiplier,
            "long_cap_multiplier": self.long_cap_multiplier,
            "short_cap_multiplier": self.short_cap_multiplier,
            "overlay_exposure_multiplier": self.overlay_exposure_multiplier,
            "priority_headroom_multiplier": self.priority_headroom_multiplier,
            "max_family_contracts_multiplier": self.max_family_contracts_multiplier,
            "position_limit_multiplier": self.position_limit_multiplier,
            "disabled_strategies": sorted(self.disabled_strategies),
        }


_NO_ACTION = CrisisActionPolicy()


def stress_formation_risk_multiplier(mode: str, score: int) -> float:
    """Return optimized pre-action risk multiplier for shock/grind formation."""
    if score < C.STRESS_FORMATION_MIN_SCORE:
        return 1.0
    mode_l = (mode or "").lower()
    multiplier = 1.0
    if "shock" in mode_l:
        multiplier = min(multiplier, C.STRESS_FORMATION_RISK_MULT_SHOCK)
    if "grind" in mode_l:
        multiplier = min(multiplier, C.STRESS_FORMATION_RISK_MULT_GRIND)
    if "credit_impulse" in mode_l:
        multiplier = min(multiplier, C.STRESS_FORMATION_RISK_MULT_CREDIT_IMPULSE)
    return multiplier


_POLICIES: dict[str, dict[int, CrisisActionPolicy]] = {
    "generic": {
        2: CrisisActionPolicy(
            family_id="generic",
            alert_level=C.ALERT_WARNING,
            alert_level_int=2,
            risk_multiplier=C.RISK_MULT_WARNING,
            dd_tier_multiplier=C.DD_TIER_MULT_WARNING,
        ),
        3: CrisisActionPolicy(
            family_id="generic",
            alert_level=C.ALERT_CRISIS,
            alert_level_int=3,
            risk_multiplier=C.RISK_MULT_CRISIS,
            dd_tier_multiplier=C.DD_TIER_MULT_CRISIS,
        ),
    },
    "stock": {
        2: CrisisActionPolicy(
            family_id="stock",
            alert_level=C.ALERT_WARNING,
            alert_level_int=2,
            risk_multiplier=C.RISK_MULT_WARNING,
            dd_tier_multiplier=C.DD_TIER_MULT_WARNING,
            directional_cap_multiplier=0.85,
            long_cap_multiplier=0.75,
            short_cap_multiplier=0.85,
            priority_headroom_multiplier=0.75,
            position_limit_multiplier=0.75,
        ),
        3: CrisisActionPolicy(
            family_id="stock",
            alert_level=C.ALERT_CRISIS,
            alert_level_int=3,
            risk_multiplier=C.RISK_MULT_CRISIS,
            dd_tier_multiplier=C.DD_TIER_MULT_CRISIS,
            directional_cap_multiplier=0.60,
            long_cap_multiplier=0.40,
            short_cap_multiplier=0.60,
            priority_headroom_multiplier=0.50,
            position_limit_multiplier=0.50,
            disabled_strategies=frozenset({"ALCB_v1", "IARIC_v1"}),
        ),
    },
    "momentum": {
        2: CrisisActionPolicy(
            family_id="momentum",
            alert_level=C.ALERT_WARNING,
            alert_level_int=2,
            risk_multiplier=1.0,
            long_risk_multiplier=C.RISK_MULT_WARNING,
            short_risk_multiplier=1.0,
            dd_tier_multiplier=C.DD_TIER_MULT_WARNING,
            directional_cap_multiplier=0.80,
            long_cap_multiplier=0.70,
            short_cap_multiplier=1.00,
            max_family_contracts_multiplier=0.75,
        ),
        3: CrisisActionPolicy(
            family_id="momentum",
            alert_level=C.ALERT_CRISIS,
            alert_level_int=3,
            risk_multiplier=1.0,
            long_risk_multiplier=C.RISK_MULT_CRISIS,
            short_risk_multiplier=1.0,
            dd_tier_multiplier=C.DD_TIER_MULT_CRISIS,
            directional_cap_multiplier=0.55,
            long_cap_multiplier=0.35,
            short_cap_multiplier=1.00,
            max_family_contracts_multiplier=0.50,
        ),
    },
    "swing": {
        2: CrisisActionPolicy(
            family_id="swing",
            alert_level=C.ALERT_WARNING,
            alert_level_int=2,
            risk_multiplier=1.0,
            long_risk_multiplier=C.RISK_MULT_WARNING,
            short_risk_multiplier=1.0,
            dd_tier_multiplier=C.DD_TIER_MULT_WARNING,
            directional_cap_multiplier=0.85,
            long_cap_multiplier=0.75,
            short_cap_multiplier=1.00,
            overlay_exposure_multiplier=C.RISK_MULT_WARNING,
        ),
        3: CrisisActionPolicy(
            family_id="swing",
            alert_level=C.ALERT_CRISIS,
            alert_level_int=3,
            risk_multiplier=1.0,
            long_risk_multiplier=C.RISK_MULT_CRISIS,
            short_risk_multiplier=1.0,
            dd_tier_multiplier=C.DD_TIER_MULT_CRISIS,
            directional_cap_multiplier=0.60,
            long_cap_multiplier=0.40,
            short_cap_multiplier=1.00,
            overlay_exposure_multiplier=C.RISK_MULT_CRISIS,
        ),
    },
}


def _normalized_regime(regime: str | None) -> str:
    regime_u = (regime or "").upper()
    return regime_u if regime_u in {"G", "R", "S", "D"} else ""


def _action_level_int(crisis_ctx: CrisisContext) -> int:
    """Return the portfolio-action level, falling back for older contexts."""
    portfolio_level = int(max(0, min(3, crisis_ctx.portfolio_action_level_int)))
    if portfolio_level > 0:
        return portfolio_level
    alert_level = int(max(0, min(3, crisis_ctx.alert_level_int)))
    return alert_level if alert_level >= 2 else 0


def _is_limited_credit_bridge_warning(crisis_ctx: CrisisContext, level_int: int) -> bool:
    if level_int != 2:
        return False
    mode = (crisis_ctx.stress_formation_mode or "").lower()
    if "credit_impulse" not in mode:
        return False
    return crisis_ctx.primary_warning_count < C.WARNING_MIN_PRIMARY and crisis_ctx.primary_crisis_count <= 0


def _copy_policy(
    policy: CrisisActionPolicy,
    *,
    family_id: str | None = None,
    regime: str = "",
    action_provenance: str | None = None,
    risk_multiplier: float | None = None,
    long_risk_multiplier: float | None = None,
    short_risk_multiplier: float | None = None,
    dd_tier_multiplier: float | None = None,
    directional_cap_multiplier: float | None = None,
    long_cap_multiplier: float | None = None,
    short_cap_multiplier: float | None = None,
    overlay_exposure_multiplier: float | None = None,
    priority_headroom_multiplier: float | None = None,
    max_family_contracts_multiplier: float | None = None,
    position_limit_multiplier: float | None = None,
) -> CrisisActionPolicy:
    return CrisisActionPolicy(
        family_id=(family_id or policy.family_id).lower(),
        regime=regime,
        alert_level=policy.alert_level,
        alert_level_int=policy.alert_level_int,
        action_provenance=action_provenance or policy.action_provenance,
        risk_multiplier=min(float(risk_multiplier if risk_multiplier is not None else policy.risk_multiplier), 1.0),
        long_risk_multiplier=min(float(long_risk_multiplier if long_risk_multiplier is not None else policy.long_risk_multiplier), 1.0),
        short_risk_multiplier=min(float(short_risk_multiplier if short_risk_multiplier is not None else policy.short_risk_multiplier), 1.0),
        dd_tier_multiplier=min(float(dd_tier_multiplier if dd_tier_multiplier is not None else policy.dd_tier_multiplier), 1.0),
        directional_cap_multiplier=min(float(directional_cap_multiplier if directional_cap_multiplier is not None else policy.directional_cap_multiplier), 1.0),
        long_cap_multiplier=min(float(long_cap_multiplier if long_cap_multiplier is not None else policy.long_cap_multiplier), 1.0),
        short_cap_multiplier=min(float(short_cap_multiplier if short_cap_multiplier is not None else policy.short_cap_multiplier), 1.0),
        overlay_exposure_multiplier=min(float(overlay_exposure_multiplier if overlay_exposure_multiplier is not None else policy.overlay_exposure_multiplier), 1.0),
        priority_headroom_multiplier=min(float(priority_headroom_multiplier if priority_headroom_multiplier is not None else policy.priority_headroom_multiplier), 1.0),
        max_family_contracts_multiplier=min(float(max_family_contracts_multiplier if max_family_contracts_multiplier is not None else policy.max_family_contracts_multiplier), 1.0),
        position_limit_multiplier=min(float(position_limit_multiplier if position_limit_multiplier is not None else policy.position_limit_multiplier), 1.0),
        disabled_strategies=policy.disabled_strategies,
    )


def _apply_provenance_adjustment(
    policy: CrisisActionPolicy,
    crisis_ctx: CrisisContext,
    regime: str,
) -> CrisisActionPolicy:
    if not _is_limited_credit_bridge_warning(crisis_ctx, policy.alert_level_int):
        return policy

    return _copy_policy(
        policy,
        regime=regime,
        action_provenance="credit_impulse_bridge",
        risk_multiplier=max(policy.risk_multiplier, C.ACTION_CREDIT_BRIDGE_WARNING_RISK_MULT),
        long_risk_multiplier=max(policy.long_risk_multiplier, C.ACTION_CREDIT_BRIDGE_WARNING_RISK_MULT),
        dd_tier_multiplier=max(policy.dd_tier_multiplier, C.ACTION_CREDIT_BRIDGE_WARNING_DD_MULT),
        directional_cap_multiplier=max(
            policy.directional_cap_multiplier,
            C.ACTION_CREDIT_BRIDGE_WARNING_CAP_MULT,
        ),
        long_cap_multiplier=max(policy.long_cap_multiplier, C.ACTION_CREDIT_BRIDGE_WARNING_LONG_MULT),
        priority_headroom_multiplier=max(
            policy.priority_headroom_multiplier,
            C.ACTION_CREDIT_BRIDGE_WARNING_CAP_MULT,
        ),
        max_family_contracts_multiplier=max(
            policy.max_family_contracts_multiplier,
            C.ACTION_CREDIT_BRIDGE_WARNING_CONTRACTS_MULT,
        ),
        overlay_exposure_multiplier=max(
            policy.overlay_exposure_multiplier,
            C.ACTION_CREDIT_BRIDGE_WARNING_RISK_MULT,
        ),
        position_limit_multiplier=max(
            policy.position_limit_multiplier,
            C.ACTION_CREDIT_BRIDGE_WARNING_LONG_MULT,
        ),
    )


def _apply_regime_adjustment(policy: CrisisActionPolicy, regime: str) -> CrisisActionPolicy:
    if policy.alert_level_int != 2:
        return policy

    if regime == "S":
        return _copy_policy(
            policy,
            regime=regime,
            action_provenance=(
                f"{policy.action_provenance}+stress_regime"
                if policy.action_provenance != "confirmed"
                else "stress_regime"
            ),
            risk_multiplier=max(policy.risk_multiplier, C.ACTION_WARNING_RISK_MULT_STRESS_REGIME),
            long_risk_multiplier=max(policy.long_risk_multiplier, C.ACTION_WARNING_RISK_MULT_STRESS_REGIME),
            dd_tier_multiplier=max(policy.dd_tier_multiplier, C.ACTION_WARNING_DD_MULT_STRESS_REGIME),
            directional_cap_multiplier=max(
                policy.directional_cap_multiplier,
                C.ACTION_WARNING_CAP_MULT_STRESS_REGIME,
            ),
            long_cap_multiplier=max(policy.long_cap_multiplier, C.ACTION_WARNING_CAP_MULT_STRESS_REGIME),
            priority_headroom_multiplier=max(
                policy.priority_headroom_multiplier,
                C.ACTION_WARNING_CAP_MULT_STRESS_REGIME,
            ),
            max_family_contracts_multiplier=max(
                policy.max_family_contracts_multiplier,
                C.ACTION_WARNING_CAP_MULT_STRESS_REGIME,
            ),
            overlay_exposure_multiplier=max(
                policy.overlay_exposure_multiplier,
                C.ACTION_WARNING_RISK_MULT_STRESS_REGIME,
            ),
            position_limit_multiplier=max(
                policy.position_limit_multiplier,
                C.ACTION_WARNING_CAP_MULT_STRESS_REGIME,
            ),
        )

    if regime == "D":
        return _copy_policy(
            policy,
            regime=regime,
            action_provenance=(
                f"{policy.action_provenance}+defensive_regime"
                if policy.action_provenance != "confirmed"
                else "defensive_regime"
            ),
            risk_multiplier=max(policy.risk_multiplier, C.ACTION_WARNING_RISK_MULT_DEFENSIVE_REGIME),
            long_risk_multiplier=max(policy.long_risk_multiplier, C.ACTION_WARNING_RISK_MULT_DEFENSIVE_REGIME),
            dd_tier_multiplier=max(policy.dd_tier_multiplier, C.ACTION_WARNING_DD_MULT_DEFENSIVE_REGIME),
            directional_cap_multiplier=max(
                policy.directional_cap_multiplier,
                C.ACTION_WARNING_CAP_MULT_DEFENSIVE_REGIME,
            ),
            long_cap_multiplier=max(policy.long_cap_multiplier, C.ACTION_WARNING_CAP_MULT_DEFENSIVE_REGIME),
            priority_headroom_multiplier=max(
                policy.priority_headroom_multiplier,
                C.ACTION_WARNING_CAP_MULT_DEFENSIVE_REGIME,
            ),
            max_family_contracts_multiplier=max(
                policy.max_family_contracts_multiplier,
                C.ACTION_WARNING_CAP_MULT_DEFENSIVE_REGIME,
            ),
            overlay_exposure_multiplier=max(
                policy.overlay_exposure_multiplier,
                C.ACTION_WARNING_RISK_MULT_DEFENSIVE_REGIME,
            ),
            position_limit_multiplier=max(
                policy.position_limit_multiplier,
                C.ACTION_WARNING_CAP_MULT_DEFENSIVE_REGIME,
            ),
        )

    return policy


def resolve_crisis_action(
    crisis_ctx: CrisisContext,
    family_id: str | None = None,
    regime: str | None = None,
) -> CrisisActionPolicy:
    """Return the action policy for a family and crisis level.

    HMM regime remains the baseline risk envelope.  This resolver only chooses
    how much the daily crisis layer tightens that envelope for each family.
    """
    family = (family_id or "generic").lower()
    regime_u = _normalized_regime(regime)
    level_int = _action_level_int(crisis_ctx)
    if level_int <= 1:
        pre_action_mult = stress_formation_risk_multiplier(
            crisis_ctx.stress_formation_mode,
            crisis_ctx.stress_formation_score,
        )
        if pre_action_mult >= 1.0:
            return _NO_ACTION
        return CrisisActionPolicy(
            family_id=family,
            regime=regime_u,
            alert_level=C.ALERT_WATCH,
            alert_level_int=1,
            action_provenance="stress_formation",
            risk_multiplier=pre_action_mult,
            dd_tier_multiplier=1.0,
        )

    by_level = _POLICIES.get(family, _POLICIES["generic"])
    policy = by_level.get(level_int) or _POLICIES["generic"].get(level_int)
    if policy is None:
        return _NO_ACTION

    policy = _copy_policy(policy, family_id=policy.family_id, regime=regime_u, action_provenance="confirmed")
    policy = _apply_provenance_adjustment(policy, crisis_ctx, regime_u)
    policy = _apply_regime_adjustment(policy, regime_u)
    return policy
