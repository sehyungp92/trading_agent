"""CrisisContext: downstream-facing dataclass for strategy coordinators.

Mirrors RegimeContext pattern but at daily frequency with absolute thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass

from regime.crisis.config import (
    ALERT_LEVEL_INT,
    ALERT_NORMAL,
    DD_TIER_MULT_CRISIS,
    DD_TIER_MULT_NORMAL,
    DD_TIER_MULT_WARNING,
    DD_TIER_MULT_WATCH,
    RISK_MULT_CRISIS,
    RISK_MULT_NORMAL,
    RISK_MULT_WARNING,
    RISK_MULT_WATCH,
)


@dataclass(frozen=True)
class CrisisContext:
    """Daily crisis detection context for strategy coordinators.

    Additive-only: crisis can only tighten risk, never loosen beyond
    the regime-adjusted baseline.
    """

    alert_level: str = ALERT_NORMAL           # NORMAL/WATCH/WARNING/CRISIS
    alert_level_int: int = 0                   # 0-3

    # Split outputs:
    # - alert_level keeps the internal hysteresis/action state for compatibility.
    # - advisory_level is stricter, user-facing early warning.
    # - portfolio_action_level is NORMAL unless WARNING/CRISIS is actionable.
    advisory_level: str = ALERT_NORMAL
    advisory_level_int: int = 0
    advisory_reason: str = ""
    portfolio_action_level: str = ALERT_NORMAL
    portfolio_action_level_int: int = 0

    # Portfolio impact multipliers
    risk_multiplier: float = RISK_MULT_NORMAL  # normal/watch/action risk multiplier
    dd_tier_multiplier: float = DD_TIER_MULT_NORMAL  # 1.0 / 1.0 / 0.90 / 0.75

    # Indicator snapshot
    vix_level: float = 0.0
    credit_spread_bps: float = 0.0
    yield_curve_slope: float = 0.0
    yield_curve_20d_change: float = 0.0
    spy_3d_return: float = 0.0
    spy_5d_return: float = 0.0
    spy_tlt_correlation: float = 0.0
    spy_10d_return: float = 0.0
    spy_20d_return: float = 0.0
    vix_3d_change: float = 0.0
    credit_spread_20d_change_bps: float = 0.0
    stress_formation_score: int = 0
    stress_formation_mode: str = ""
    stress_formation_reason: str = ""

    # Confirming indicators (VIX term structure; SPY DD is now primary)
    vix_term_structure_ratio: float = 0.0      # VIX/VIX3M (0.0 if unavailable)
    spy_10d_drawdown: float = 0.0

    # Conjunction counts
    primary_watch_count: int = 0
    primary_warning_count: int = 0
    primary_crisis_count: int = 0

    # Recovery ramp (R3)
    recovery_ramp_mult: float = 1.0            # 0.75..1.0 during recovery, 1.0 otherwise

    # Metadata
    computed_at: str = ""                       # ISO timestamp
    data_as_of: str = ""                         # latest market data date used
    data_status: str = ""                        # fresh/stale/degraded-live note
    days_at_current_level: int = 0
    dominant_channel: str = ""                  # which indicator is most elevated

    def to_snapshot_dict(self) -> dict:
        """Serialize for DailySnapshot instrumentation."""
        return {
            "crisis_alert_level": self.alert_level,
            "crisis_alert_level_int": self.alert_level_int,
            "crisis_advisory_level": self.advisory_level,
            "crisis_advisory_level_int": self.advisory_level_int,
            "crisis_advisory_reason": self.advisory_reason,
            "crisis_portfolio_action_level": self.portfolio_action_level,
            "crisis_portfolio_action_level_int": self.portfolio_action_level_int,
            "crisis_risk_multiplier": self.risk_multiplier,
            "crisis_dd_tier_multiplier": self.dd_tier_multiplier,
            "vix_level": self.vix_level,
            "credit_spread_bps": self.credit_spread_bps,
            "yield_curve_slope": self.yield_curve_slope,
            "yield_curve_20d_change": self.yield_curve_20d_change,
            "spy_3d_return": self.spy_3d_return,
            "spy_5d_return": self.spy_5d_return,
            "spy_tlt_correlation": self.spy_tlt_correlation,
            "spy_10d_return": self.spy_10d_return,
            "spy_20d_return": self.spy_20d_return,
            "vix_3d_change": self.vix_3d_change,
            "credit_spread_20d_change_bps": self.credit_spread_20d_change_bps,
            "stress_formation_score": self.stress_formation_score,
            "stress_formation_mode": self.stress_formation_mode,
            "stress_formation_reason": self.stress_formation_reason,
            "vix_term_structure_ratio": self.vix_term_structure_ratio,
            "spy_10d_drawdown": self.spy_10d_drawdown,
            "primary_watch_count": self.primary_watch_count,
            "primary_warning_count": self.primary_warning_count,
            "primary_crisis_count": self.primary_crisis_count,
            "crisis_recovery_ramp_mult": self.recovery_ramp_mult,
            "crisis_data_as_of": self.data_as_of,
            "crisis_data_status": self.data_status,
            "days_at_current_level": self.days_at_current_level,
            "dominant_channel": self.dominant_channel,
            "computed_at": self.computed_at,
        }


def _risk_mult_for_level(level_int: int) -> float:
    return (RISK_MULT_NORMAL, RISK_MULT_WATCH, RISK_MULT_WARNING, RISK_MULT_CRISIS)[
        min(level_int, 3)
    ]


def _dd_mult_for_level(level_int: int) -> float:
    return (DD_TIER_MULT_NORMAL, DD_TIER_MULT_WATCH, DD_TIER_MULT_WARNING, DD_TIER_MULT_CRISIS)[
        min(level_int, 3)
    ]
