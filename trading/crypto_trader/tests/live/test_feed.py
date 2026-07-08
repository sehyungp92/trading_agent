"""Tests for live feed — BarAssembler and LiveFeed."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from crypto_trader.core.models import Bar, TimeFrame
from crypto_trader.live.feed import BarAssembler, LiveFeed


def _make_candle(ts_ms: int, o=100.0, h=105.0, l=95.0, c=102.0, v=1000.0):
    return {"T": ts_ms, "o": str(o), "h": str(h), "l": str(l), "c": str(c), "v": str(v)}


class TestBarAssembler:
    def test_first_poll_emits_bar(self):
        info = MagicMock()
        now_ms = int(datetime(2026, 4, 20, 10, 15, tzinfo=timezone.utc).timestamp() * 1000)
        # 3 candles: second-to-last is completed
        candles = [
            _make_candle(now_ms - 1800_000),  # oldest
            _make_candle(now_ms - 900_000),    # completed (most recent closed)
            _make_candle(now_ms),               # currently forming
        ]
        info.candles_snapshot.return_value = candles

        asm = BarAssembler(info, ["BTC"], [TimeFrame.M15])
        bars = asm.poll_all()
        assert len(bars) == 1
        assert bars[0].symbol == "BTC"
        assert bars[0].timeframe == TimeFrame.M15

    def test_duplicate_suppression(self):
        info = MagicMock()
        now_ms = int(datetime(2026, 4, 20, 10, 15, tzinfo=timezone.utc).timestamp() * 1000)
        candles = [
            _make_candle(now_ms - 1800_000),
            _make_candle(now_ms - 900_000),
            _make_candle(now_ms),
        ]
        info.candles_snapshot.return_value = candles

        asm = BarAssembler(info, ["BTC"], [TimeFrame.M15])
        bars1 = asm.poll_all()
        assert len(bars1) == 1

        # Same candles → no new bar
        bars2 = asm.poll_all()
        assert len(bars2) == 0

    def test_new_bar_after_advance(self):
        info = MagicMock()
        t1 = int(datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
        t2 = int(datetime(2026, 4, 20, 10, 15, tzinfo=timezone.utc).timestamp() * 1000)
        t3 = int(datetime(2026, 4, 20, 10, 30, tzinfo=timezone.utc).timestamp() * 1000)

        asm = BarAssembler(info, ["BTC"], [TimeFrame.M15])

        # First poll
        info.candles_snapshot.return_value = [
            _make_candle(t1), _make_candle(t2), _make_candle(t3),
        ]
        bars1 = asm.poll_all()
        assert len(bars1) == 1  # t2 is completed

        # Time advances
        t4 = t3 + 900_000
        info.candles_snapshot.return_value = [
            _make_candle(t2), _make_candle(t3), _make_candle(t4),
        ]
        bars2 = asm.poll_all()
        assert len(bars2) == 1  # t3 is newly completed

    def test_emission_order_multiple_tfs(self):
        info = MagicMock()

        asm = BarAssembler(info, ["BTC"], [TimeFrame.D1, TimeFrame.H1, TimeFrame.M15])

        # D1 poll
        d1_ts = int(datetime(2026, 4, 19, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        d1_candles = [_make_candle(d1_ts - 86400_000), _make_candle(d1_ts), _make_candle(d1_ts + 86400_000)]

        # H1 poll
        h1_ts = int(datetime(2026, 4, 20, 9, 0, tzinfo=timezone.utc).timestamp() * 1000)
        h1_candles = [_make_candle(h1_ts - 3600_000), _make_candle(h1_ts), _make_candle(h1_ts + 3600_000)]

        # M15 poll
        m15_ts = int(datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
        m15_candles = [_make_candle(m15_ts - 900_000), _make_candle(m15_ts), _make_candle(m15_ts + 900_000)]

        info.candles_snapshot.side_effect = [d1_candles, h1_candles, m15_candles]

        bars = asm.poll_all()
        assert len(bars) == 3
        # Order should be D1, H1, M15
        assert bars[0].timeframe == TimeFrame.D1
        assert bars[1].timeframe == TimeFrame.H1
        assert bars[2].timeframe == TimeFrame.M15

    def test_insufficient_candles(self):
        info = MagicMock()
        info.candles_snapshot.return_value = [_make_candle(1000)]
        asm = BarAssembler(info, ["BTC"], [TimeFrame.M15])
        bars = asm.poll_all()
        assert len(bars) == 0

    def test_set_last_emitted(self):
        info = MagicMock()
        asm = BarAssembler(info, ["BTC"], [TimeFrame.M15])
        ts = datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc)
        asm.set_last_emitted("BTC", TimeFrame.M15, ts)

        # Poll with same timestamp → no new bar
        ts_ms = int(ts.timestamp() * 1000)
        info.candles_snapshot.return_value = [
            _make_candle(ts_ms - 900_000),
            _make_candle(ts_ms),
            _make_candle(ts_ms + 900_000),
        ]
        bars = asm.poll_all()
        assert len(bars) == 0


class TestLiveFeed:
    def test_timeframe_union(self):
        info = MagicMock()
        strategy_tfs = {
            "momentum": [TimeFrame.M15, TimeFrame.H1, TimeFrame.H4],
            "trend": [TimeFrame.H1, TimeFrame.D1],
            "breakout": [TimeFrame.M30, TimeFrame.H4],
        }
        feed = LiveFeed(info, ["BTC"], strategy_tfs)
        assert TimeFrame.M15 in feed._all_tfs
        assert TimeFrame.M30 in feed._all_tfs
        assert TimeFrame.H1 in feed._all_tfs
        assert TimeFrame.H4 in feed._all_tfs
        assert TimeFrame.D1 in feed._all_tfs

    def test_warmup_loading(self):
        info = MagicMock()
        strategy_tfs = {"momentum": [TimeFrame.M15]}
        feed = LiveFeed(info, ["BTC"], strategy_tfs)

        # Mock candles_snapshot for warmup
        ts = int(datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
        candles = [_make_candle(ts + i * 900_000) for i in range(10)]
        info.candles_snapshot.return_value = candles

        warmup_bars = feed.load_warmup_bars(info, {TimeFrame.M15: 5})
        assert len(warmup_bars) > 0
        assert all(b.timeframe == TimeFrame.M15 for b in warmup_bars)
