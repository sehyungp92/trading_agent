from __future__ import annotations

from strategies.momentum.nq_regime import config
from strategies.momentum.nq_regime.config import Grade, ModuleId


def grade_for(module: ModuleId, score: int, *, vetoes: tuple[str, ...] = (), target_room_r: float = 0.0) -> Grade:
    if vetoes or target_room_r < config.TARGET_ROOM_MIN_R:
        return Grade.INVALID
    if module is ModuleId.STRUCTURAL_EXPANSION:
        if score >= config.STRUCTURAL_A_PLUS_SCORE:
            return Grade.A_PLUS
        if score >= config.STRUCTURAL_MIN_SCORE:
            return Grade.A
        if score >= 5:
            return Grade.B
        return Grade.INVALID
    if module is ModuleId.LIQUIDITY_REVERSION:
        if score >= config.REVERSION_A_PLUS_SCORE:
            return Grade.A_PLUS
        if score >= config.REVERSION_A_SCORE:
            return Grade.A
        if score >= config.REVERSION_MIN_SCORE:
            return Grade.B
        return Grade.INVALID
    if module is ModuleId.SECOND_WIND:
        if score >= config.SECOND_WIND_A_PLUS_SCORE:
            return Grade.A_PLUS
        if score >= config.SECOND_WIND_A_SCORE:
            return Grade.A
        if score >= config.SECOND_WIND_MIN_SCORE:
            return Grade.B
        return Grade.INVALID
    return Grade.INVALID


def risk_pct_for_grade(grade: Grade, *, post_news: bool = False, after_loss: bool = False) -> float:
    if post_news:
        risk = config.RISK_PCT_POST_NEWS
    elif grade is Grade.A_PLUS:
        risk = config.RISK_PCT_A_PLUS
    elif grade is Grade.A:
        risk = config.RISK_PCT_A
    elif grade is Grade.B:
        risk = config.RISK_PCT_B
    else:
        risk = 0.0
    if after_loss and risk > 0:
        risk *= config.RISK_REDUCTION_AFTER_LOSS
    return risk

