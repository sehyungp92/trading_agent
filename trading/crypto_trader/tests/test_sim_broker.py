"""Tests for SimBroker: fill mechanics, funding, liquidation, and integration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from crypto_trader.backtest.metrics import compute_metrics
from crypto_trader.broker.sim_broker import SimBroker
from crypto_trader.core.clock import SimClock
from crypto_trader.core.engine import StrategyEngine, StrategyContext, MultiTimeFrameBars
from crypto_trader.core.events import EventBus
from crypto_trader.core.models import (
    Bar,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    TimeFrame,
)
from crypto_trader.exchange.funding import FundingHelper
from tests.conftest import make_bar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _broker(**kwargs) -> SimBroker:
    """Create a SimBroker with sensible defaults for testing."""
    defaults = dict(
        initial_equity=100_000.0,
        taker_fee_bps=3.5,
        maker_fee_bps=1.0,
        slippage_bps=2.0,
        spread_bps=2.0,
    )
    defaults.update(kwargs)
    return SimBroker(**defaults)


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1. Market buy fill
# ---------------------------------------------------------------------------

class TestMarketBuyFill:
    def test_fills_at_open_plus_spread_and_slippage(self):
        broker = _broker()
        order = Order(
            order_id="", symbol="BTC", side=Side.LONG,
            order_type=OrderType.MARKET, qty=0.1,
        )
        broker.submit_order(order)

        bar = make_bar(_dt("2025-01-01T00:15:00"), o=50000, h=51000, l=49000, c=50500)
        fills = broker.process_bar(bar)

        assert len(fills) == 1
        fill = fills[0]
        # Expected: open * (1 + (spread_bps/2 + slippage_bps) / 10000)
        # = 50000 * (1 + (1.0 + 2.0) / 10000) = 50000 * 1.0003 = 50015.0
        expected_price = 50000 * (1 + (2.0 / 2 + 2.0) / 10_000)
        assert fill.fill_price == pytest.approx(expected_price, rel=1e-9)
        assert fill.side == Side.LONG
        assert fill.qty == 0.1

        # Taker fee
        expected_commission = 0.1 * expected_price * 3.5 / 10_000
        assert fill.commission == pytest.approx(expected_commission, rel=1e-9)

        # Equity = initial - commission + unrealized PnL (bar close vs entry)
        unrealized = 0.1 * (50500 - expected_price)  # long: close - entry
        assert broker.get_equity() == pytest.approx(
            100_000.0 - expected_commission + unrealized, rel=1e-6
        )


# ---------------------------------------------------------------------------
# 2. Market sell fill
# ---------------------------------------------------------------------------

class TestMarketSellFill:
    def test_fills_at_open_minus_spread_and_slippage(self):
        broker = _broker()
        order = Order(
            order_id="", symbol="BTC", side=Side.SHORT,
            order_type=OrderType.MARKET, qty=0.1,
        )
        broker.submit_order(order)

        bar = make_bar(_dt("2025-01-01T00:15:00"), o=50000, h=51000, l=49000, c=50500)
        fills = broker.process_bar(bar)

        assert len(fills) == 1
        fill = fills[0]
        expected_price = 50000 * (1 - (2.0 / 2 + 2.0) / 10_000)
        assert fill.fill_price == pytest.approx(expected_price, rel=1e-9)
        assert fill.side == Side.SHORT


# ---------------------------------------------------------------------------
# 3. Stop trigger
# ---------------------------------------------------------------------------

class TestStopTrigger:
    def test_stop_hit_within_bar_range(self):
        """Long stop (sell at stop_price) when bar.low <= stop_price."""
        broker = _broker()

        # Create a long position first
        buy = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=0.1)
        broker.submit_order(buy)
        bar0 = make_bar(_dt("2025-01-01T00:00:00"), o=50000, h=51000, l=49500, c=50500)
        broker.process_bar(bar0)

        # Submit stop
        stop = Order(
            order_id="", symbol="BTC", side=Side.SHORT,
            order_type=OrderType.STOP, qty=0.1, stop_price=49000,
        )
        broker.submit_order(stop)

        bar1 = make_bar(_dt("2025-01-01T00:15:00"), o=50000, h=50200, l=48500, c=49000)
        fills = broker.process_bar(bar1)

        assert len(fills) == 1
        fill = fills[0]
        # Fill at stop_price * (1 - slippage_bps / 10000)
        expected = 49000 * (1 - 2.0 / 10_000)
        assert fill.fill_price == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. Stop gap detection
# ---------------------------------------------------------------------------

class TestStopGapDetection:
    def test_bar_opens_past_stop_fills_at_open(self):
        """If bar opens already past stop, fill at open (worse fill)."""
        broker = _broker()

        buy = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=0.1)
        broker.submit_order(buy)
        bar0 = make_bar(_dt("2025-01-01T00:00:00"), o=50000, h=51000, l=49500, c=50500)
        broker.process_bar(bar0)

        stop = Order(
            order_id="", symbol="BTC", side=Side.SHORT,
            order_type=OrderType.STOP, qty=0.1, stop_price=49000,
        )
        broker.submit_order(stop)

        # Bar opens at 48000 — gap below stop at 49000
        bar1 = make_bar(_dt("2025-01-01T00:15:00"), o=48000, h=48500, l=47500, c=48000)
        fills = broker.process_bar(bar1)

        assert len(fills) == 1
        # Gap open → fill at open * (1 - slippage_bps / 10000)
        expected = 48000 * (1 - 2.0 / 10_000)
        assert fills[0].fill_price == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. Stop-limit trigger and reject
# ---------------------------------------------------------------------------

class TestStopLimitTriggerAndReject:
    def test_stop_limit_fills_when_within_limit(self):
        broker = _broker()

        order = Order(
            order_id="", symbol="BTC", side=Side.LONG,
            order_type=OrderType.STOP_LIMIT, qty=0.1,
            stop_price=51000, limit_price=51500,
        )
        broker.submit_order(order)

        bar = make_bar(_dt("2025-01-01T00:15:00"), o=50500, h=52000, l=50000, c=51500)
        fills = broker.process_bar(bar)

        assert len(fills) == 1
        # Stop price + slippage: 51000 * (1 + 2.0/10000) = 51010.2, within limit of 51500
        expected_price = 51000 * (1 + 2.0 / 10_000)
        assert fills[0].fill_price == pytest.approx(expected_price, rel=1e-9)

    def test_stop_limit_rejects_when_gap_exceeds_limit(self):
        broker = _broker()

        order = Order(
            order_id="", symbol="BTC", side=Side.LONG,
            order_type=OrderType.STOP_LIMIT, qty=0.1,
            stop_price=51000, limit_price=51200,
        )
        broker.submit_order(order)

        # Opens at 51500 — gap above stop and above limit
        bar = make_bar(_dt("2025-01-01T00:15:00"), o=51500, h=52000, l=51000, c=51800)
        fills = broker.process_bar(bar)

        assert len(fills) == 0
        # Order should be rejected
        assert order.status == OrderStatus.REJECTED


# ---------------------------------------------------------------------------
# 6. Limit trade-through (conservative model)
# ---------------------------------------------------------------------------

class TestLimitTradeThrough:
    def test_buy_limit_fills_on_strict_trade_through(self):
        """Buy limit fills only when bar.low < limit_price (strict <)."""
        broker = _broker()

        order = Order(
            order_id="", symbol="BTC", side=Side.LONG,
            order_type=OrderType.LIMIT, qty=0.1, limit_price=49500,
        )
        broker.submit_order(order)

        # Bar low touches limit exactly — should NOT fill (need strict <)
        bar_no_fill = make_bar(_dt("2025-01-01T00:15:00"), o=50000, h=50500, l=49500, c=50000)
        fills = broker.process_bar(bar_no_fill)
        assert len(fills) == 0

        # Bar low goes below limit — should fill
        bar_fill = make_bar(_dt("2025-01-01T00:30:00"), o=50000, h=50500, l=49400, c=49800)
        fills = broker.process_bar(bar_fill)
        assert len(fills) == 1
        assert fills[0].fill_price == 49500  # Fills at limit, no slippage

        # Maker fee (not taker)
        expected_commission = 0.1 * 49500 * 1.0 / 10_000
        assert fills[0].commission == pytest.approx(expected_commission, rel=1e-9)

    def test_sell_limit_fills_on_strict_trade_through(self):
        """Sell limit fills only when bar.high > limit_price (strict >)."""
        broker = _broker()

        # Open a long position first so we have something to sell
        buy = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=0.1)
        broker.submit_order(buy)
        bar0 = make_bar(_dt("2025-01-01T00:00:00"), o=50000, h=51000, l=49500, c=50500)
        broker.process_bar(bar0)

        sell = Order(
            order_id="", symbol="BTC", side=Side.SHORT,
            order_type=OrderType.LIMIT, qty=0.1, limit_price=51000,
        )
        broker.submit_order(sell)

        # High exactly at limit — should NOT fill
        bar_no = make_bar(_dt("2025-01-01T00:15:00"), o=50500, h=51000, l=50000, c=50800)
        fills = broker.process_bar(bar_no)
        assert len(fills) == 0

        # High above limit — fills
        bar_yes = make_bar(_dt("2025-01-01T00:30:00"), o=50500, h=51100, l=50000, c=50800)
        fills = broker.process_bar(bar_yes)
        assert len(fills) == 1
        assert fills[0].fill_price == 51000


# ---------------------------------------------------------------------------
# 7. TTL expiry
# ---------------------------------------------------------------------------

class TestTTLExpiry:
    def test_order_cancelled_after_ttl_bars(self):
        broker = _broker()

        order = Order(
            order_id="", symbol="BTC", side=Side.LONG,
            order_type=OrderType.LIMIT, qty=0.1, limit_price=40000,
            ttl_bars=3,
        )
        broker.submit_order(order)

        # Process 3 bars — order should survive
        for i in range(3):
            bar = make_bar(
                _dt("2025-01-01T00:00:00") + timedelta(minutes=15 * (i + 1)),
                o=50000, h=51000, l=49000, c=50500,
            )
            broker.process_bar(bar)

        assert order.status in (OrderStatus.PENDING, OrderStatus.WORKING)

        # 4th bar — TTL exceeded, order expires
        bar4 = make_bar(_dt("2025-01-01T01:00:00"), o=50000, h=51000, l=49000, c=50500)
        broker.process_bar(bar4)

        assert order.status == OrderStatus.EXPIRED


# ---------------------------------------------------------------------------
# 8. OCA cancellation
# ---------------------------------------------------------------------------

class TestOCACancellation:
    def test_sibling_orders_cancelled_on_fill(self):
        broker = _broker()

        # Submit two orders in the same OCA group
        stop = Order(
            order_id="", symbol="BTC", side=Side.SHORT,
            order_type=OrderType.STOP, qty=0.1, stop_price=49000,
            oca_group="exit1",
        )
        take_profit = Order(
            order_id="", symbol="BTC", side=Side.SHORT,
            order_type=OrderType.LIMIT, qty=0.1, limit_price=52000,
            oca_group="exit1",
        )
        broker.submit_order(stop)
        broker.submit_order(take_profit)

        # Bar that triggers the stop
        bar = make_bar(_dt("2025-01-01T00:15:00"), o=50000, h=50200, l=48500, c=49000)
        fills = broker.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].side == Side.SHORT

        # The take-profit should be cancelled
        assert take_profit.status == OrderStatus.CANCELLED


# ---------------------------------------------------------------------------
# 9. Round-trip PnL
# ---------------------------------------------------------------------------

class TestRoundTripPnL:
    def test_buy_sell_roundtrip(self):
        broker = _broker()

        # Buy
        buy = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=0.1)
        broker.submit_order(buy)
        bar0 = make_bar(_dt("2025-01-01T00:00:00"), o=50000, h=51000, l=49500, c=50500)
        fills0 = broker.process_bar(bar0)
        assert len(fills0) == 1
        entry_price = fills0[0].fill_price
        entry_commission = fills0[0].commission

        # Sell
        sell = Order(order_id="", symbol="BTC", side=Side.SHORT, order_type=OrderType.MARKET, qty=0.1)
        broker.submit_order(sell)
        bar1 = make_bar(_dt("2025-01-01T00:15:00"), o=51000, h=51500, l=50500, c=51200)
        fills1 = broker.process_bar(bar1)
        assert len(fills1) == 1
        exit_price = fills1[0].fill_price
        exit_commission = fills1[0].commission

        # Position should be closed
        assert broker.get_position("BTC") is None

        # Trade should be created
        trades = broker.closed_trades
        assert len(trades) == 1
        trade = trades[0]
        assert trade.symbol == "BTC"
        assert trade.direction == Side.LONG

        # PnL: (exit_price - entry_price) * qty
        raw_pnl = (exit_price - entry_price) * 0.1
        assert trade.pnl == pytest.approx(raw_pnl, rel=1e-6)

        # Equity should reflect PnL minus both commissions
        expected_equity = 100_000 + raw_pnl - entry_commission - exit_commission
        assert broker.get_equity() == pytest.approx(expected_equity, rel=1e-6)


# ---------------------------------------------------------------------------
# 10. Funding accrual
# ---------------------------------------------------------------------------

class TestFundingAccrual:
    def test_funding_applied_at_hourly_boundaries(self):
        # Create funding data: 0.01% per hour
        funding_df = pd.DataFrame({
            "ts": [
                int(datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000),
                int(datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc).timestamp() * 1000),
                int(datetime(2025, 1, 1, 2, 0, tzinfo=timezone.utc).timestamp() * 1000),
            ],
            "rate": [0.0001, 0.0001, 0.0001],  # 0.01%
        })
        fh = FundingHelper(funding_df)
        broker = _broker(funding_helper=fh)

        # Open a long position
        buy = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=1.0)
        broker.submit_order(buy)
        bar0 = make_bar(_dt("2025-01-01T00:00:00"), o=50000, h=51000, l=49500, c=50000)
        broker.process_bar(bar0)

        equity_before = broker.get_equity()

        # Process bars through an hourly boundary
        bar1 = make_bar(_dt("2025-01-01T00:15:00"), o=50000, h=50100, l=49900, c=50000)
        broker.process_bar(bar1)
        bar2 = make_bar(_dt("2025-01-01T00:30:00"), o=50000, h=50100, l=49900, c=50000)
        broker.process_bar(bar2)
        bar3 = make_bar(_dt("2025-01-01T00:45:00"), o=50000, h=50100, l=49900, c=50000)
        broker.process_bar(bar3)
        bar4 = make_bar(_dt("2025-01-01T01:00:00"), o=50000, h=50100, l=49900, c=50000)
        broker.process_bar(bar4)

        # Funding should have been accrued (long pays positive rate)
        assert len(broker._funding_log) > 0
        total_funding = sum(f["cost"] for f in broker._funding_log)
        assert total_funding > 0  # Long pays positive funding

        # Equity should decrease due to funding
        equity_after = broker.get_equity()
        assert equity_after < equity_before


# ---------------------------------------------------------------------------
# 11. Liquidation
# ---------------------------------------------------------------------------

class TestLiquidation:
    def test_position_force_closed_on_margin_breach(self):
        broker = _broker(initial_equity=1000, default_leverage=100)

        # Open a leveraged long
        buy = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=1.0)
        broker.submit_order(buy)
        # low must stay above liquidation price (~49515 for 100x leverage)
        bar0 = make_bar(_dt("2025-01-01T00:00:00"), o=50000, h=50500, l=49800, c=50000)
        broker.process_bar(bar0)

        assert broker.get_position("BTC") is not None

        # Massive price drop — should trigger liquidation
        bar1 = make_bar(_dt("2025-01-01T00:15:00"), o=49000, h=49000, l=40000, c=40000)
        broker.process_bar(bar1)

        # Position should be liquidated (closed)
        assert broker.get_position("BTC") is None


# ---------------------------------------------------------------------------
# 12. Integration: trivial strategy through full StrategyEngine loop
# ---------------------------------------------------------------------------

class BuyEveryNBarsStrategy:
    """Trivial strategy for integration testing: buy every N bars, close after M bars."""

    def __init__(self, buy_every: int = 10, hold_bars: int = 5):
        self._buy_every = buy_every
        self._hold_bars = hold_bars
        self._bar_count = 0
        self._entry_bar: int | None = None

    @property
    def name(self) -> str:
        return "BuyEveryNBars"

    @property
    def symbols(self) -> list[str]:
        return ["BTC"]

    @property
    def timeframes(self) -> list[TimeFrame]:
        return [TimeFrame.M15]

    def on_init(self, ctx: StrategyContext) -> None:
        pass

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        if bar.timeframe != TimeFrame.M15:
            return

        self._bar_count += 1

        # Close position after hold_bars
        if self._entry_bar is not None and (self._bar_count - self._entry_bar) >= self._hold_bars:
            pos = ctx.broker.get_position("BTC")
            if pos is not None:
                close = Order(
                    order_id="", symbol="BTC", side=Side.SHORT,
                    order_type=OrderType.MARKET, qty=pos.qty, tag="close",
                )
                ctx.broker.submit_order(close)
                self._entry_bar = None

        # Buy every N bars if flat
        elif self._bar_count % self._buy_every == 0:
            pos = ctx.broker.get_position("BTC")
            if pos is None:
                buy = Order(
                    order_id="", symbol="BTC", side=Side.LONG,
                    order_type=OrderType.MARKET, qty=0.01, tag="entry",
                )
                ctx.broker.submit_order(buy)
                self._entry_bar = self._bar_count

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        pass

    def on_shutdown(self, ctx: StrategyContext) -> None:
        pass


class ListFeed:
    """Minimal feed that yields a list of bars (for testing without Parquet)."""

    def __init__(self, bars: list[Bar]):
        self._bars = bars
        self._history: dict[tuple[str, TimeFrame], list[Bar]] = {}

    def subscribe(self, symbol: str, timeframes: list[TimeFrame]) -> None:
        pass

    def __iter__(self):
        for bar in self._bars:
            key = (bar.symbol, bar.timeframe)
            if key not in self._history:
                self._history[key] = []
            self._history[key].append(bar)
            yield bar

    def get_history(self, symbol: str, timeframe: TimeFrame, count: int) -> list[Bar]:
        bars = self._history.get((symbol, timeframe), [])
        return bars[-count:]


class TestIntegrationTrivialStrategy:
    def test_full_engine_loop(self):
        """Run BuyEveryNBarsStrategy through StrategyEngine with SimBroker."""
        # Generate 50 M15 bars with a simple price pattern
        base_time = _dt("2025-01-01T00:00:00")
        bars = []
        price = 50000.0
        for i in range(50):
            ts = base_time + timedelta(minutes=15 * i)
            # Oscillating price
            price_delta = 100 * (1 if i % 4 < 2 else -1)
            o = price + price_delta
            h = o + 200
            l = o - 200
            c = o + 50
            bars.append(make_bar(ts, o, h, l, c))
            price = c

        feed = ListFeed(bars)
        broker = SimBroker(initial_equity=100_000.0)
        clock = SimClock()
        events = EventBus()
        strategy = BuyEveryNBarsStrategy(buy_every=10, hold_bars=5)

        engine = StrategyEngine(
            strategy=strategy,
            broker=broker,
            feed=feed,
            clock=clock,
            events=events,
        )

        # Track events
        bar_events = []
        fill_events = []
        from crypto_trader.core.events import BarEvent, FillEvent
        events.subscribe(BarEvent, lambda e: bar_events.append(e))
        events.subscribe(FillEvent, lambda e: fill_events.append(e))

        engine.run()

        # Verify basic invariants
        assert len(bar_events) == 50
        assert len(fill_events) > 0

        # Verify fills have reasonable prices (near open)
        for fe in fill_events:
            assert fe.fill.fill_price > 0
            assert fe.fill.qty > 0

        # Verify trades were created
        trades = broker.closed_trades
        assert len(trades) > 0

        # Verify equity tracking
        eq_hist = broker.equity_history
        assert len(eq_hist) == 50
        # First fill happens after bar 10 (buy_every=10), so early entries are at initial equity
        # but later entries should differ due to fills and commissions
        assert eq_hist[0][1] == 100_000.0  # No fills yet on bar 0
        # After all fills, equity should differ from initial
        final_equity = eq_hist[-1][1]
        assert final_equity != 100_000.0

        # Verify commissions are deducted
        total_commissions = sum(f.fill.commission for f in fill_events)
        assert total_commissions > 0

    def test_multi_tf_bar_emission_order(self):
        """Verify higher-TF bars are emitted before primary TF bars at boundaries."""
        base_time = _dt("2025-01-01T00:00:00")

        # Create M15 bars spanning a full hour + H1 bars
        m15_bars = []
        for i in range(8):  # 2 hours of M15 bars
            ts = base_time + timedelta(minutes=15 * i)
            m15_bars.append(make_bar(ts, 50000, 51000, 49000, 50500, tf=TimeFrame.M15))

        h1_bars = [
            make_bar(base_time, 50000, 51000, 49000, 50500, tf=TimeFrame.H1),
            make_bar(base_time + timedelta(hours=1), 50000, 51000, 49000, 50500, tf=TimeFrame.H1),
        ]

        # Feed that interleaves bars correctly
        all_bars = []
        for i, m15 in enumerate(m15_bars):
            # At :45 boundary, emit H1 first
            if m15.timestamp.minute == 45:
                h1_open = m15.timestamp.replace(minute=0)
                for h1 in h1_bars:
                    if h1.timestamp == h1_open:
                        all_bars.append(h1)
            all_bars.append(m15)

        feed = ListFeed(all_bars)
        broker = SimBroker()
        clock = SimClock()
        events = EventBus()

        # Dummy strategy that records bar order
        received_bars: list[Bar] = []

        class RecorderStrategy:
            name = "Recorder"
            symbols = ["BTC"]
            timeframes = [TimeFrame.M15, TimeFrame.H1]

            def on_init(self, ctx): pass
            def on_bar(self, bar, ctx): received_bars.append(bar)
            def on_fill(self, fill, ctx): pass
            def on_shutdown(self, ctx): pass

        engine = StrategyEngine(
            strategy=RecorderStrategy(),
            broker=broker,
            feed=feed,
            clock=clock,
            events=events,
        )
        engine.run()

        # Find the H1 bar in the output and verify it comes before its M15 bar
        for i, bar in enumerate(received_bars):
            if bar.timeframe == TimeFrame.H1 and i + 1 < len(received_bars):
                # Next bar should be M15 (the :45 bar that triggered emission)
                next_bar = received_bars[i + 1]
                assert next_bar.timeframe == TimeFrame.M15
                assert next_bar.timestamp.minute == 45


# ---------------------------------------------------------------------------
# 13. Order validation
# ---------------------------------------------------------------------------

class TestOrderValidation:
    def test_rejects_zero_qty(self):
        broker = _broker()
        order = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=0)
        oid = broker.submit_order(order)
        assert oid == ""
        assert order.status == OrderStatus.REJECTED

    def test_rejects_negative_qty(self):
        broker = _broker()
        order = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=-1.0)
        oid = broker.submit_order(order)
        assert oid == ""
        assert order.status == OrderStatus.REJECTED

    def test_rejects_limit_without_price(self):
        broker = _broker()
        order = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.LIMIT, qty=0.1)
        oid = broker.submit_order(order)
        assert oid == ""
        assert order.status == OrderStatus.REJECTED

    def test_rejects_stop_without_price(self):
        broker = _broker()
        order = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.STOP, qty=0.1)
        oid = broker.submit_order(order)
        assert oid == ""
        assert order.status == OrderStatus.REJECTED

    def test_rejects_stop_limit_without_prices(self):
        broker = _broker()
        order = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.STOP_LIMIT, qty=0.1, stop_price=50000)
        oid = broker.submit_order(order)
        assert oid == ""
        assert order.status == OrderStatus.REJECTED


# ---------------------------------------------------------------------------
# 14. Trade records: total commission and bars_held
# ---------------------------------------------------------------------------

class TestTradeRecords:
    def test_trade_records_total_commission(self):
        """Trade.commission should include both entry and exit commissions."""
        broker = _broker()

        buy = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=0.1)
        broker.submit_order(buy)
        bar0 = make_bar(_dt("2025-01-01T00:00:00"), o=50000, h=51000, l=49500, c=50500)
        fills0 = broker.process_bar(bar0)
        entry_comm = fills0[0].commission

        sell = Order(order_id="", symbol="BTC", side=Side.SHORT, order_type=OrderType.MARKET, qty=0.1)
        broker.submit_order(sell)
        bar1 = make_bar(_dt("2025-01-01T00:15:00"), o=51000, h=51500, l=50500, c=51200)
        fills1 = broker.process_bar(bar1)
        exit_comm = fills1[0].commission

        trades = broker.closed_trades
        assert len(trades) == 1
        assert trades[0].commission == pytest.approx(entry_comm + exit_comm, rel=1e-6)

    def test_trade_records_bars_held(self):
        """Trade.bars_held should count bars the position was open."""
        broker = _broker()

        buy = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=0.1)
        broker.submit_order(buy)
        bar0 = make_bar(_dt("2025-01-01T00:00:00"), o=50000, h=51000, l=49500, c=50500)
        broker.process_bar(bar0)

        # 3 more bars while position is open
        for i in range(1, 4):
            bar = make_bar(
                _dt("2025-01-01T00:00:00") + timedelta(minutes=15 * i),
                o=50500, h=51000, l=50000, c=50500,
            )
            broker.process_bar(bar)

        # Close on bar 4
        sell = Order(order_id="", symbol="BTC", side=Side.SHORT, order_type=OrderType.MARKET, qty=0.1)
        broker.submit_order(sell)
        bar4 = make_bar(_dt("2025-01-01T01:15:00"), o=51000, h=51500, l=50500, c=51200)
        broker.process_bar(bar4)

        trades = broker.closed_trades
        assert len(trades) == 1
        # Position opened during bar0, bars 1-4 increment counter = 4 bars held
        assert trades[0].bars_held == 4


# ---------------------------------------------------------------------------
# 15. Margin-based cash accounting
# ---------------------------------------------------------------------------

class TestMarginAccounting:
    def test_cash_deducts_margin_not_notional(self):
        """Opening a leveraged position should only deduct margin from cash."""
        broker = _broker(default_leverage=10)

        buy = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=1.0)
        broker.submit_order(buy)
        bar0 = make_bar(_dt("2025-01-01T00:00:00"), o=50000, h=51000, l=49500, c=50000)
        broker.process_bar(bar0)

        entry_price = 50000 * (1 + (2.0 / 2 + 2.0) / 10_000)  # ~50015
        commission = 1.0 * entry_price * 3.5 / 10_000
        expected_margin = entry_price / 10  # 10x leverage

        # Cash should be: initial - margin - commission (not initial - notional - commission)
        # Equity = cash + unrealized. With close=50000 and entry ~50015, unrealized ≈ -15
        assert broker._cash == pytest.approx(100_000 - expected_margin - commission, rel=1e-6)


# ---------------------------------------------------------------------------
# 16. PositionClosedEvent emission
# ---------------------------------------------------------------------------

class TestPositionClosedEvent:
    def test_engine_emits_position_closed_event(self):
        """StrategyEngine should emit PositionClosedEvent when a trade is created."""
        from crypto_trader.core.events import PositionClosedEvent

        base_time = _dt("2025-01-01T00:00:00")
        bars = []
        for i in range(20):
            ts = base_time + timedelta(minutes=15 * i)
            bars.append(make_bar(ts, 50000, 51000, 49000, 50500))

        feed = ListFeed(bars)
        broker = SimBroker(initial_equity=100_000.0)
        clock = SimClock()
        events = EventBus()
        strategy = BuyEveryNBarsStrategy(buy_every=5, hold_bars=3)

        engine = StrategyEngine(
            strategy=strategy, broker=broker, feed=feed,
            clock=clock, events=events,
        )

        closed_events = []
        events.subscribe(PositionClosedEvent, lambda e: closed_events.append(e))
        engine.run()

        # There should be at least one closed trade
        assert len(broker.closed_trades) > 0
        # And a matching PositionClosedEvent for each
        assert len(closed_events) == len(broker.closed_trades)
        for evt in closed_events:
            assert evt.trade.symbol == "BTC"
            assert evt.trade.pnl is not None


# ---------------------------------------------------------------------------
# 17. Engine close_open_positions dispatches fills through strategy
# ---------------------------------------------------------------------------

class _FillTrackingStrategy:
    """Strategy that tracks on_fill calls and records PositionClosedEvents."""

    name = "fill_tracker"
    symbols = ["BTC"]
    timeframes = [TimeFrame.M15]

    def __init__(self):
        self.fills: list[Fill] = []
        self.closed_events = []
        self._bought = False

    def on_init(self, ctx: StrategyContext) -> None:
        from crypto_trader.core.events import PositionClosedEvent
        ctx.events.subscribe(PositionClosedEvent, lambda e: self.closed_events.append(e))

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        if not self._bought:
            buy = Order(
                order_id="buy1", symbol="BTC", side=Side.LONG,
                order_type=OrderType.MARKET, qty=0.01, tag="entry",
            )
            ctx.broker.submit_order(buy)
            self._bought = True

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        self.fills.append(fill)

    def on_shutdown(self, ctx: StrategyContext) -> None:
        pass


class TestEngineCloseOpenPositions:
    def test_dispatches_fills_to_strategy(self):
        """engine.close_open_positions() should call strategy.on_fill for each fill."""
        base = _dt("2025-01-01T00:00:00")
        bars = [make_bar(base + timedelta(minutes=15 * i), 50000, 51000, 49000, 50500)
                for i in range(5)]
        feed = ListFeed(bars)
        broker = SimBroker(initial_equity=100_000.0)
        clock = SimClock()
        events = EventBus()
        strategy = _FillTrackingStrategy()
        engine = StrategyEngine(
            strategy=strategy, broker=broker, feed=feed,
            clock=clock, events=events,
        )
        engine.run()

        # Position should be open after engine.run()
        assert broker.get_position("BTC") is not None
        fills_before = len(strategy.fills)
        events_before = len(strategy.closed_events)

        # close_open_positions should dispatch fills and events
        fills = engine.close_open_positions()
        assert len(fills) == 1
        assert fills[0].tag == "backtest_end"
        assert len(strategy.fills) == fills_before + 1
        assert strategy.fills[-1].tag == "backtest_end"
        assert len(strategy.closed_events) == events_before + 1
        assert strategy.closed_events[-1].trade.exit_reason == "backtest_end"
        # Position should be gone
        assert broker.get_position("BTC") is None

    def test_emits_position_closed_event(self):
        """engine.close_open_positions() should emit PositionClosedEvent."""
        from crypto_trader.core.events import PositionClosedEvent

        base = _dt("2025-01-01T00:00:00")
        bars = [make_bar(base + timedelta(minutes=15 * i), 50000, 51000, 49000, 50500)
                for i in range(5)]
        feed = ListFeed(bars)
        broker = SimBroker(initial_equity=100_000.0)
        clock = SimClock()
        events = EventBus()
        strategy = _FillTrackingStrategy()

        external_events = []
        events.subscribe(PositionClosedEvent, lambda e: external_events.append(e))

        engine = StrategyEngine(
            strategy=strategy, broker=broker, feed=feed,
            clock=clock, events=events,
        )
        engine.run()
        engine.close_open_positions()

        # Should have both the entry trade's event AND the backtest_end event
        backtest_end_events = [e for e in external_events if e.trade.exit_reason == "backtest_end"]
        assert len(backtest_end_events) == 1


# ---------------------------------------------------------------------------
# 18. Broker cancels orphaned orders on position close
# ---------------------------------------------------------------------------

class TestOrphanOrderCancellation:
    def test_remaining_orders_cancelled_on_position_close(self):
        """When a position is fully closed, remaining orders for that symbol
        should be cancelled to prevent orphaned positions."""
        broker = _broker()
        bar1 = make_bar(_dt("2025-01-01T00:00:00"), 50000, 51000, 49000, 50500)

        # Open a LONG position
        entry = Order(order_id="e1", symbol="BTC", side=Side.LONG,
                      order_type=OrderType.MARKET, qty=0.1)
        broker.submit_order(entry)
        broker.process_bar(bar1)
        assert broker.get_position("BTC") is not None

        # Submit a TP MARKET and a full-size STOP (simulates TP + trail on same bar)
        tp = Order(order_id="tp1", symbol="BTC", side=Side.SHORT,
                   order_type=OrderType.MARKET, qty=0.025, tag="tp1")
        stop = Order(order_id="stop1", symbol="BTC", side=Side.SHORT,
                     order_type=OrderType.STOP, qty=0.1, stop_price=49000.0,
                     tag="protective_stop")
        broker.submit_order(tp)
        broker.submit_order(stop)

        # Bar that triggers the stop (low goes below stop price)
        bar2 = make_bar(_dt("2025-01-01T00:15:00"), 50500, 50600, 48500, 48800)
        fills = broker.process_bar(bar2)

        # Stop fills first (closes full position), TP should be cancelled
        stop_fills = [f for f in fills if f.tag == "protective_stop"]
        tp_fills = [f for f in fills if f.tag == "tp1"]
        assert len(stop_fills) == 1
        assert len(tp_fills) == 0  # TP was cancelled before it could fill

        # No orphaned position
        assert broker.get_position("BTC") is None

        # Only 1 trade created (the stop exit), not 2
        close_trades = [t for t in broker.closed_trades if t.symbol == "BTC"]
        assert len(close_trades) == 1

    def test_partial_exit_does_not_cancel_orders(self):
        """Partial exits (TP1) should NOT cancel remaining orders."""
        broker = _broker()
        bar1 = make_bar(_dt("2025-01-01T00:00:00"), 50000, 51000, 49000, 50500)

        # Open a LONG
        entry = Order(order_id="e1", symbol="BTC", side=Side.LONG,
                      order_type=OrderType.MARKET, qty=0.1)
        broker.submit_order(entry)
        broker.process_bar(bar1)

        # Submit TP1 (partial) and a protective stop
        tp = Order(order_id="tp1", symbol="BTC", side=Side.SHORT,
                   order_type=OrderType.MARKET, qty=0.025, tag="tp1")
        stop = Order(order_id="stop1", symbol="BTC", side=Side.SHORT,
                     order_type=OrderType.STOP, qty=0.075, stop_price=49000.0,
                     tag="protective_stop")
        broker.submit_order(tp)
        broker.submit_order(stop)

        # Bar where stop doesn't trigger (price stays above)
        bar2 = make_bar(_dt("2025-01-01T00:15:00"), 50500, 52000, 50000, 51500)
        fills = broker.process_bar(bar2)

        # TP1 should fill (partial exit)
        tp_fills = [f for f in fills if f.tag == "tp1"]
        assert len(tp_fills) == 1

        # Position should still exist with reduced qty
        pos = broker.get_position("BTC")
        assert pos is not None
        assert pos.qty == pytest.approx(0.075, abs=1e-10)

        # Stop should still be pending
        pending = [o for o in broker._pending_orders
                   if o.status in (OrderStatus.PENDING, OrderStatus.WORKING)]
        assert len(pending) == 1
        assert pending[0].tag == "protective_stop"

    def test_different_symbol_orders_not_cancelled(self):
        """Closing BTC position should not cancel ETH orders."""
        broker = _broker()
        bar_btc = make_bar(_dt("2025-01-01T00:00:00"), 50000, 51000, 49000, 50500, sym="BTC")
        bar_eth = make_bar(_dt("2025-01-01T00:00:00"), 3000, 3100, 2900, 3050, sym="ETH")

        # Open BTC and ETH positions
        broker.submit_order(Order(order_id="e1", symbol="BTC", side=Side.LONG,
                                  order_type=OrderType.MARKET, qty=0.1))
        broker.submit_order(Order(order_id="e2", symbol="ETH", side=Side.LONG,
                                  order_type=OrderType.MARKET, qty=1.0))
        broker.process_bar(bar_btc)
        broker.process_bar(bar_eth)

        # Submit stops for both
        broker.submit_order(Order(order_id="s1", symbol="BTC", side=Side.SHORT,
                                  order_type=OrderType.STOP, qty=0.1, stop_price=49000.0))
        broker.submit_order(Order(order_id="s2", symbol="ETH", side=Side.SHORT,
                                  order_type=OrderType.STOP, qty=1.0, stop_price=2900.0))

        # Close BTC via stop trigger
        bar2 = make_bar(_dt("2025-01-01T00:15:00"), 50000, 50100, 48500, 48800, sym="BTC")
        broker.process_bar(bar2)

        # BTC closed, ETH stop still pending
        assert broker.get_position("BTC") is None
        assert broker.get_position("ETH") is not None
        eth_pending = [o for o in broker._pending_orders
                       if o.symbol == "ETH" and o.status in (OrderStatus.PENDING, OrderStatus.WORKING)]
        assert len(eth_pending) == 1


class _SameBarStopStrategy:
    """Submit one market entry and attach a same-bar protective stop."""

    name = "same_bar_stop"
    symbols = ["BTC"]
    timeframes = [TimeFrame.M15]

    def __init__(self) -> None:
        self._submitted = False

    def on_init(self, ctx: StrategyContext) -> None:
        pass

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        if not self._submitted:
            ctx.broker.submit_order(Order(
                order_id="entry1",
                symbol="BTC",
                side=Side.LONG,
                order_type=OrderType.MARKET,
                qty=1.0,
                tag="entry",
            ))
            self._submitted = True

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        if fill.tag == "entry":
            ctx.broker.submit_order(Order(
                order_id="stop1",
                symbol="BTC",
                side=Side.SHORT,
                order_type=OrderType.STOP,
                qty=fill.qty,
                stop_price=95.0,
                tag="protective_stop",
            ))

    def on_shutdown(self, ctx: StrategyContext) -> None:
        pass


class _HoldOpenStrategy:
    """Submit one market entry and leave the position open for terminal marking."""

    name = "hold_open"
    symbols = ["BTC"]
    timeframes = [TimeFrame.M15]

    def __init__(self) -> None:
        self._submitted = False

    def on_init(self, ctx: StrategyContext) -> None:
        pass

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        if not self._submitted:
            ctx.broker.submit_order(Order(
                order_id="entry1",
                symbol="BTC",
                side=Side.LONG,
                order_type=OrderType.MARKET,
                qty=1.0,
                tag="entry",
            ))
            self._submitted = True

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        pass

    def on_shutdown(self, ctx: StrategyContext) -> None:
        pass


class TestEquityFinalization:
    def test_same_bar_stop_rewrites_current_bar_equity_snapshot(self):
        base = _dt("2026-01-01T00:00:00")
        bars = [
            make_bar(base, 100.0, 100.0, 100.0, 100.0),
            make_bar(base + timedelta(minutes=15), 100.0, 101.0, 95.0, 101.0),
        ]
        feed = ListFeed(bars)
        broker = SimBroker(
            initial_equity=10_000.0,
            taker_fee_bps=0.0,
            maker_fee_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
        )
        engine = StrategyEngine(
            strategy=_SameBarStopStrategy(),
            broker=broker,
            feed=feed,
            clock=SimClock(),
            events=EventBus(),
        )

        engine.run()

        assert len(broker.closed_trades) == 1
        assert broker.closed_trades[0].net_pnl == pytest.approx(-5.0)
        assert broker.equity_history[-1][1] == pytest.approx(9_995.0)

        metrics = compute_metrics(broker)
        assert metrics.net_profit == pytest.approx(-5.0)
        assert metrics.max_drawdown_pct > 0.0

    def test_liquidation_rewrites_current_bar_equity_snapshot(self):
        broker = SimBroker(
            initial_equity=10.0,
            taker_fee_bps=0.0,
            maker_fee_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            default_leverage=100.0,
        )
        broker.submit_order(Order(
            order_id="entry1",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=10.0,
        ))
        broker.process_bar(make_bar(_dt("2026-01-01T00:00:00"), 100.0, 100.0, 100.0, 100.0))
        broker.process_bar(make_bar(_dt("2026-01-01T00:15:00"), 100.0, 100.0, 50.0, 90.0))

        assert broker.get_position("BTC") is None
        assert broker.closed_trades[0].exit_reason == "liquidation"
        assert broker.equity_history[-1][1] == pytest.approx(-490.0)
        assert broker.get_equity() == pytest.approx(-490.0)

    def test_same_bar_liquidation_does_not_leave_orphan_protective_stop(self):
        base = _dt("2026-01-01T00:00:00")
        bars = [
            make_bar(base, 100.0, 100.0, 100.0, 100.0),
            make_bar(base + timedelta(minutes=15), 100.0, 100.0, 50.0, 90.0),
        ]
        feed = ListFeed(bars)
        broker = SimBroker(
            initial_equity=10.0,
            taker_fee_bps=0.0,
            maker_fee_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            default_leverage=100.0,
        )
        engine = StrategyEngine(
            strategy=_SameBarStopStrategy(),
            broker=broker,
            feed=feed,
            clock=SimClock(),
            events=EventBus(),
        )

        engine.run()

        assert broker.get_position("BTC") is None
        assert len(broker.closed_trades) == 1
        assert broker.closed_trades[0].exit_reason == "liquidation"
        assert broker.get_open_orders("BTC") == []


class TestTerminalMarks:
    def test_engine_marks_open_positions_instead_of_creating_backtest_end_trade(self):
        base = _dt("2026-01-01T00:00:00")
        bars = [
            make_bar(base, 100.0, 100.0, 100.0, 100.0),
            make_bar(base + timedelta(minutes=15), 100.0, 106.0, 99.0, 105.0),
        ]
        feed = ListFeed(bars)
        broker = SimBroker(
            initial_equity=10_000.0,
            taker_fee_bps=0.0,
            maker_fee_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
        )
        engine = StrategyEngine(
            strategy=_HoldOpenStrategy(),
            broker=broker,
            feed=feed,
            clock=SimClock(),
            events=EventBus(),
        )

        engine.run()
        marks = engine.mark_open_positions()

        assert broker.closed_trades == []
        assert len(marks) == 1
        assert marks[0].symbol == "BTC"
        assert marks[0].unrealized_pnl_net == pytest.approx(5.0)
        assert broker.equity_history[-1][1] == pytest.approx(10_005.0)

        metrics = compute_metrics(broker)
        assert metrics.total_trades == 0
        assert metrics.terminal_mark_count == 1
        assert metrics.terminal_mark_pnl_net == pytest.approx(5.0)
        assert metrics.net_profit == pytest.approx(5.0)
        assert broker.liquidation_equity_history[-1][1] == pytest.approx(10_005.0)

    def test_mark_open_positions_replaces_final_snapshot_and_is_idempotent(self):
        base = _dt("2026-01-01T00:00:00")
        bars = [
            make_bar(base, 100.0, 100.0, 100.0, 100.0),
            make_bar(base + timedelta(minutes=15), 100.0, 106.0, 99.0, 105.0),
        ]
        feed = ListFeed(bars)
        broker = SimBroker(
            initial_equity=10_000.0,
            taker_fee_bps=3.5,
            maker_fee_bps=1.0,
            slippage_bps=2.0,
            spread_bps=2.0,
        )
        engine = StrategyEngine(
            strategy=_HoldOpenStrategy(),
            broker=broker,
            feed=feed,
            clock=SimClock(),
            events=EventBus(),
        )

        engine.run()
        pre_mark_len = len(broker.equity_history)
        pre_liq_len = len(broker.liquidation_equity_history)

        first_marks = engine.mark_open_positions()
        assert len(first_marks) == 1
        assert len(broker.equity_history) == pre_mark_len
        assert len(broker.liquidation_equity_history) == pre_liq_len

        second_marks = engine.mark_open_positions()
        assert len(second_marks) == 1
        assert len(broker.equity_history) == pre_mark_len
        assert len(broker.liquidation_equity_history) == pre_liq_len
        assert broker.equity_history[-1][1] == pytest.approx(broker.get_equity())
        assert broker.liquidation_equity_history[-1][1] < broker.equity_history[-1][1]

    def test_liquidation_equity_history_tracks_exit_costs_for_open_positions(self):
        base = _dt("2026-01-01T00:00:00")
        bars = [
            make_bar(base, 100.0, 100.0, 100.0, 100.0),
            make_bar(base + timedelta(minutes=15), 100.0, 106.0, 99.0, 105.0),
        ]
        feed = ListFeed(bars)
        broker = SimBroker(
            initial_equity=10_000.0,
            taker_fee_bps=3.5,
            maker_fee_bps=1.0,
            slippage_bps=2.0,
            spread_bps=2.0,
        )
        engine = StrategyEngine(
            strategy=_HoldOpenStrategy(),
            broker=broker,
            feed=feed,
            clock=SimClock(),
            events=EventBus(),
        )

        engine.run()

        assert len(broker.liquidation_equity_history) == len(broker.equity_history)
        assert broker.liquidation_equity_history[-1][1] < broker.equity_history[-1][1]

        metrics = compute_metrics(broker)
        assert metrics.net_profit == pytest.approx(
            broker.liquidation_equity_history[-1][1] - broker.initial_equity
        )
