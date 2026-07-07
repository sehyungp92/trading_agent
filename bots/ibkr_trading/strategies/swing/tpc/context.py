from __future__ import annotations

from typing import Any

import numpy as np

from strategies.swing._shared.etf_core import ETFBarInput
from strategies.swing._shared.models import Direction
from strategies.swing.tpc.config import TPCSymbolConfig


def score_asset_context(
    bar_input: ETFBarInput,
    direction: Direction,
    cfg: TPCSymbolConfig,
) -> tuple[float, dict[str, Any]]:
    if not cfg.asset_context_enabled:
        return 1.0, {"enabled": False}
    ind = bar_input.indicators
    votes: list[tuple[str, float]] = []
    _add_trend_vote(
        votes,
        "context_1h",
        _trend_alignment(
            direction,
            close=ind.get("context_close_1h", np.nan),
            fast=ind.get("context_sma20_1h", np.nan),
            slow=ind.get("context_sma50_1h", np.nan),
            ret=ind.get("context_ret12_1h", np.nan),
        ),
        0.40,
    )
    _add_trend_vote(
        votes,
        "context_daily",
        _trend_alignment(
            direction,
            close=ind.get("context_close_daily", np.nan),
            fast=ind.get("context_sma20_daily", np.nan),
            slow=ind.get("context_sma50_daily", np.nan),
            ret=ind.get("context_ret20_daily", np.nan),
        ),
        0.35,
    )
    _add_trend_vote(
        votes,
        "self_4h_di",
        _di_alignment(direction, ind.get("plus_di_4h", np.nan), ind.get("minus_di_4h", np.nan)),
        0.15,
    )
    _add_trend_vote(
        votes,
        "self_4h_ma",
        _self_ma_alignment(bar_input, direction, ind),
        0.10,
    )
    if not votes:
        return 0.0, {"enabled": True, "reason": "no_context"}
    score = float(np.clip(sum(value for _name, value in votes), -1.0, 1.0))
    return score, {
        "enabled": True,
        "score": score,
        "votes": {name: value for name, value in votes},
        "context_symbol": cfg.asset_context_symbol,
    }


def _add_trend_vote(votes: list[tuple[str, float]], name: str, alignment: float | None, weight: float) -> None:
    if alignment is None:
        return
    votes.append((name, float(weight) * float(alignment)))


def _trend_alignment(
    direction: Direction,
    *,
    close: float,
    fast: float,
    slow: float,
    ret: float,
) -> float | None:
    values = [close, fast, slow, ret]
    if any(not np.isfinite(value) for value in values):
        return None
    if direction == Direction.LONG:
        aligned = close >= fast and fast >= slow and ret >= 0.0
        opposed = close < fast and fast < slow and ret < 0.0
    else:
        aligned = close <= fast and fast <= slow and ret <= 0.0
        opposed = close > fast and fast > slow and ret > 0.0
    if aligned:
        return 1.0
    if opposed:
        return -1.0
    return 0.0


def _di_alignment(direction: Direction, plus_di: float, minus_di: float) -> float | None:
    if not np.isfinite(plus_di) or not np.isfinite(minus_di):
        return None
    if direction == Direction.LONG:
        return 1.0 if plus_di > minus_di else -1.0
    return 1.0 if minus_di > plus_di else -1.0


def _self_ma_alignment(bar_input: ETFBarInput, direction: Direction, indicators: dict[str, float]) -> float | None:
    bars_4h = bar_input.bars_4h
    ma50 = indicators.get("ma50_4h", np.nan)
    ma100 = indicators.get("ma100_4h", np.nan)
    if bars_4h is None or len(bars_4h) == 0 or not np.isfinite(ma50) or not np.isfinite(ma100):
        return None
    close = float(bars_4h.closes[-1])
    if direction == Direction.LONG:
        return 1.0 if close > ma50 > ma100 else -1.0 if close < ma50 < ma100 else 0.0
    return 1.0 if close < ma50 < ma100 else -1.0 if close > ma50 > ma100 else 0.0
