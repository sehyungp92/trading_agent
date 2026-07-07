import json
from datetime import datetime, timezone

from trading_assistant.schemas.events import EventMetadata, TradeEvent, DailySnapshot


class TestEventMetadata:
    def test_event_id_is_deterministic(self):
        """Same inputs produce the same event_id."""
        em1 = EventMetadata(
            bot_id="bot1",
            exchange_timestamp=datetime(2026, 3, 1, 14, 0, tzinfo=timezone.utc),
            local_timestamp=datetime(2026, 3, 1, 14, 0, 0, 50000, tzinfo=timezone.utc),
            data_source_id="binance_spot_ws",
            event_type="trade",
            payload_key="bot1|BTCUSDT|2026-03-01T14:00:00",
        )
        em2 = EventMetadata(
            bot_id="bot1",
            exchange_timestamp=datetime(2026, 3, 1, 14, 0, tzinfo=timezone.utc),
            local_timestamp=datetime(2026, 3, 1, 14, 0, 0, 50000, tzinfo=timezone.utc),
            data_source_id="binance_spot_ws",
            event_type="trade",
            payload_key="bot1|BTCUSDT|2026-03-01T14:00:00",
        )
        assert em1.event_id == em2.event_id
        assert len(em1.event_id) == 16

    def test_clock_skew_computed(self):
        em = EventMetadata(
            bot_id="bot1",
            exchange_timestamp=datetime(2026, 3, 1, 14, 0, 0, tzinfo=timezone.utc),
            local_timestamp=datetime(2026, 3, 1, 14, 0, 0, 50000, tzinfo=timezone.utc),
            data_source_id="binance_spot_ws",
            event_type="trade",
            payload_key="key1",
        )
        assert em.clock_skew_ms == -50  # exchange is 50ms behind local


class TestTradeEvent:
    def test_roundtrip_json(self):
        """TradeEvent serializes to JSON and back."""
        trade = TradeEvent(
            trade_id="t001",
            bot_id="bot1",
            pair="BTCUSDT",
            side="LONG",
            entry_time=datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc),
            exit_time=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
            entry_price=50000.0,
            exit_price=50500.0,
            position_size=0.1,
            pnl=50.0,
            pnl_pct=1.0,
            entry_signal="EMA cross + RSI < 30",
            exit_reason="TAKE_PROFIT",
            market_regime="trending_up",
        )
        data = json.loads(trade.model_dump_json())
        restored = TradeEvent.model_validate(data)
        assert restored.trade_id == "t001"
        assert restored.pnl == 50.0


class TestDailySnapshot:
    def test_basic_construction(self):
        snap = DailySnapshot(
            date="2026-03-01",
            bot_id="bot1",
            total_trades=10,
            win_count=6,
            loss_count=4,
            gross_pnl=500.0,
            net_pnl=480.0,
            win_rate=0.6,
        )
        assert snap.win_rate == 0.6
