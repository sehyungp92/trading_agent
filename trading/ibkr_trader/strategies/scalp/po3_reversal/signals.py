from __future__ import annotations

from dataclasses import dataclass

from . import config
from .config import SetupTier


@dataclass(frozen=True, slots=True)
class Po3SignalScore:
    total: float
    tier: SetupTier


def score_components(
    *,
    h4_location: bool,
    daily_h4_alignment: bool,
    liquidity_sweep: bool,
    smt_present: bool,
    smt_strength: float,
    ifvg_close_through: bool,
    ifvg_retest_rejection: bool,
    displacement_body_pct: float,
    spread_clean: bool,
    atr_normal: bool,
    h1_target_clean: bool,
    tier_hint: SetupTier = SetupTier.NONE,
) -> Po3SignalScore:
    del tier_hint
    total = 0.0
    for flag in (
        h4_location,
        daily_h4_alignment,
        liquidity_sweep,
        smt_present,
        ifvg_close_through,
        ifvg_retest_rejection,
        spread_clean,
        atr_normal,
        h1_target_clean,
    ):
        total += 1.0 if flag else 0.0
    total += min(1.0, max(0.0, smt_strength))
    total += min(1.0, max(0.0, displacement_body_pct))
    if total >= config.SCORE_THRESHOLD_A:
        tier = SetupTier.A
    elif total >= config.SCORE_THRESHOLD_B:
        tier = SetupTier.B
    else:
        tier = SetupTier.NONE
    return Po3SignalScore(total=total, tier=tier)


def threshold_for_tier(tier: SetupTier) -> float:
    if tier is SetupTier.A:
        return config.SCORE_THRESHOLD_A
    if tier is SetupTier.B:
        return config.SCORE_THRESHOLD_B
    return float("inf")
