from __future__ import annotations

from datetime import datetime, timedelta, timezone

from strategies.scalp.po3_reversal.config import TradeDirection
from strategies.scalp.po3_reversal.models import PriceBar
from strategies.scalp.po3_reversal.smt import detect_smt_divergence


def _bar(i: int, high: float, low: float) -> PriceBar:
    ts = datetime(2026, 4, 29, 14, 0, tzinfo=timezone.utc) + timedelta(minutes=i)
    return PriceBar(ts, open=(high + low) / 2, high=high, low=low, close=(high + low) / 2)


def test_bullish_smt_when_es_sweeps_low_and_nq_holds() -> None:
    nq = [_bar(i, 100 + i * 0.1, 95.0) for i in range(8)]
    es = [_bar(i, 100 + i * 0.1, 95.0) for i in range(8)]
    nq.append(_bar(8, 101.0, 95.25))
    es.append(_bar(8, 101.0, 94.0))

    result = detect_smt_divergence(nq, es, TradeDirection.LONG, min_strength=0.01)

    assert result.present
    assert result.direction is TradeDirection.LONG

