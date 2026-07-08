from __future__ import annotations

from strategies.swing.tpc.config import TPCSymbolConfig
from strategies.swing.tpc.models import PullbackType, RegimeGrade


def score_setup(
    regime_grade: RegimeGrade,
    pullback_type: PullbackType,
    confirmations: list[str],
    rr_ratio: float,
    has_news_risk: bool,
    asset_context_score: float,
    daily_has_room: bool,
    *,
    orderly_pullback: bool = True,
    atr_healthy: bool = True,
    score_model: str = "legacy",
) -> int:
    if score_model == "alpha7":
        return _score_setup_alpha7(
            regime_grade,
            pullback_type,
            confirmations,
            rr_ratio,
            has_news_risk,
            asset_context_score,
            daily_has_room,
            orderly_pullback=orderly_pullback,
            atr_healthy=atr_healthy,
        )
    score = 0
    score += 2 if regime_grade in {RegimeGrade.A_PLUS, RegimeGrade.VALID} else 0
    score += 2 if regime_grade == RegimeGrade.A_PLUS else 1
    score += 2
    score += 2 if orderly_pullback or pullback_type == PullbackType.TYPE_B else 0
    score += 2 if any("micro" in c or "higher_low" in c or "lower_high" in c for c in confirmations) else 0
    score += 2 if any("vwap" in c for c in confirmations) else 0
    score += 1 if any("volume" in c for c in confirmations) else 0
    score += 1 if atr_healthy else 0
    score += 2 if rr_ratio >= 2.0 else 0
    score += 1 if not has_news_risk else 0
    score += 1 if asset_context_score > 0 else 0
    score += 1 if daily_has_room else 0
    return score


def _score_setup_alpha7(
    regime_grade: RegimeGrade,
    pullback_type: PullbackType,
    confirmations: list[str],
    rr_ratio: float,
    has_news_risk: bool,
    asset_context_score: float,
    daily_has_room: bool,
    *,
    orderly_pullback: bool,
    atr_healthy: bool,
) -> int:
    """Seven-component setup score for structural alpha experiments."""

    confirmations_set = {str(item) for item in confirmations}
    has_structure = any("micro" in item or "higher_low" in item or "lower_high" in item for item in confirmations_set)
    has_vwap = any("vwap" in item for item in confirmations_set)
    has_volume = any("volume" in item for item in confirmations_set)

    score = 0
    score += 3 if regime_grade == RegimeGrade.A_PLUS else 2 if regime_grade == RegimeGrade.VALID else 0
    score += 2 if orderly_pullback or pullback_type == PullbackType.TYPE_B else 1
    score += 2 if has_structure else 0
    score += 2 if has_vwap else 0
    score += 1 if has_volume else 0
    score += (2 if rr_ratio >= 2.0 else 0) + (1 if daily_has_room else 0)
    score += (1 if atr_healthy else 0) + (1 if asset_context_score > 0 and not has_news_risk else 0)
    return score


def compute_risk_pct(score: int, pullback_type: PullbackType, cfg: TPCSymbolConfig) -> float | None:
    if score >= cfg.score_a_plus_min:
        base = cfg.risk_a_plus_pct
    elif score >= cfg.score_a_min:
        base = cfg.risk_a_pct
    elif score >= cfg.score_b_min:
        base = cfg.risk_b_pct
    else:
        return None
    if pullback_type in {PullbackType.TYPE_B, PullbackType.TYPE_C}:
        base *= 0.6
    if cfg.dynamic_risk_enabled:
        span = max(cfg.dynamic_risk_score_ceiling - cfg.dynamic_risk_score_floor, 1e-9)
        quality = (float(score) - cfg.dynamic_risk_score_floor) / span
        quality = min(max(quality, 0.0), 1.0)
        curve = max(float(cfg.dynamic_risk_curve), 1e-6)
        shaped = quality ** curve
        risk_mult = cfg.dynamic_risk_min_mult + shaped * (cfg.dynamic_risk_max_mult - cfg.dynamic_risk_min_mult)
        base *= max(risk_mult, 0.0)
    return min(base, cfg.max_risk_pct)


def compute_position_size(
    equity: float,
    risk_pct: float,
    entry: float,
    stop: float,
    cfg: TPCSymbolConfig,
) -> int:
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0 or equity <= 0:
        return 0
    raw_qty = int((equity * risk_pct) // risk_per_share)
    notional_cap_qty = int((equity * cfg.max_position_notional_pct) // max(entry, 1e-9))
    return max(0, min(raw_qty, notional_cap_qty))
