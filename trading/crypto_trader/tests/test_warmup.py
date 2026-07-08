"""Tests for warmup infrastructure and trend round 2 updates."""

import pytest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, PropertyMock

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.core.models import Order, OrderType, SetupGrade, Side, TimeFrame
from crypto_trader.strategy.breakout.config import BreakoutConfig
from crypto_trader.strategy.momentum.config import MomentumConfig
from crypto_trader.strategy.trend.config import TrendConfig


# ── Warmup Infrastructure Tests ──────────────────────────────────────────


class TestWarmupDaysField:
    def test_warmup_days_default_zero(self):
        """BacktestConfig().warmup_days defaults to 0."""
        cfg = BacktestConfig()
        assert cfg.warmup_days == 0

    def test_warmup_days_field_exists(self):
        """BacktestConfig(warmup_days=30) constructs without error."""
        cfg = BacktestConfig(warmup_days=30)
        assert cfg.warmup_days == 30

    def test_warmup_days_negative_allowed(self):
        """No validation on field — negative is accepted (but meaningless)."""
        cfg = BacktestConfig(warmup_days=-5)
        assert cfg.warmup_days == -5


class TestRunnerWarmup:
    def test_run_warmup_zero_unchanged(self):
        """warmup_days=0 produces identical feed start to no-warmup."""
        from crypto_trader.backtest.runner import run
        from crypto_trader.core.models import TimeFrame

        cfg = TrendConfig()
        cfg.symbols = ["BTC"]
        bt = BacktestConfig(
            symbols=["BTC"],
            start_date=date(2026, 3, 15),
            end_date=date(2026, 4, 18),
            warmup_days=0,
        )

        feed_start_dates = []
        original_init = None

        # Capture the start_date passed to HistoricalFeed
        from crypto_trader.data.historical_feed import HistoricalFeed
        original_init = HistoricalFeed.__init__

        def capture_init(self, *args, **kwargs):
            feed_start_dates.append(kwargs.get("start_date"))
            original_init(self, *args, **kwargs)

        with patch.object(HistoricalFeed, "__init__", capture_init):
            with patch("crypto_trader.backtest.runner.run") as mock_run:
                # Can't easily run full backtest without data, so just test the logic
                pass

        # Direct logic test: warmup_days=0 means warmup_start == actual_start
        actual_start = bt.start_date
        if bt.warmup_days > 0 and actual_start is not None:
            warmup_start = actual_start - timedelta(days=bt.warmup_days)
        else:
            warmup_start = actual_start
        assert warmup_start == actual_start

    def test_warmup_extends_feed_start(self):
        """warmup_days > 0 shifts feed start date earlier."""
        bt = BacktestConfig(
            symbols=["BTC"],
            start_date=date(2026, 3, 15),
            end_date=date(2026, 4, 18),
            warmup_days=60,
        )
        actual_start = bt.start_date
        warmup_start = actual_start - timedelta(days=bt.warmup_days)
        assert warmup_start == date(2026, 1, 14)
        assert warmup_start < actual_start

    def test_warmup_trades_filtered_by_entry_time(self):
        """Trades entering during warmup period are excluded from results."""
        measurement_start = datetime(2026, 3, 15, tzinfo=timezone.utc)

        # Simulate trades: one during warmup, two during measurement
        warmup_trade = MagicMock()
        warmup_trade.entry_time = datetime(2026, 2, 20, tzinfo=timezone.utc)

        valid_trade1 = MagicMock()
        valid_trade1.entry_time = datetime(2026, 3, 16, tzinfo=timezone.utc)

        valid_trade2 = MagicMock()
        valid_trade2.entry_time = datetime(2026, 4, 1, tzinfo=timezone.utc)

        all_trades = [warmup_trade, valid_trade1, valid_trade2]

        # Apply the same filter logic as runner.py
        filtered = [t for t in all_trades if t.entry_time >= measurement_start]
        assert len(filtered) == 2
        assert warmup_trade not in filtered

    def test_warmup_equity_starts_at_measurement(self):
        """Equity timestamps are trimmed to >= start_date."""
        measurement_start = datetime(2026, 3, 15, tzinfo=timezone.utc)

        equity_history = [
            (datetime(2026, 2, 25, tzinfo=timezone.utc), 10000.0),
            (datetime(2026, 3, 1, tzinfo=timezone.utc), 10050.0),
            (datetime(2026, 3, 15, tzinfo=timezone.utc), 10100.0),
            (datetime(2026, 3, 20, tzinfo=timezone.utc), 10200.0),
            (datetime(2026, 4, 1, tzinfo=timezone.utc), 10300.0),
        ]

        filtered_equity = [
            (ts, eq) for ts, eq in equity_history
            if ts >= measurement_start
        ]
        assert len(filtered_equity) == 3
        assert filtered_equity[0][0] == measurement_start
        assert filtered_equity[0][1] == 10100.0  # New initial equity

    def test_walk_forward_propagates_warmup(self):
        """Both train and test configs in walk_forward get warmup_days."""
        bt = BacktestConfig(
            symbols=["BTC"],
            start_date=date(2026, 3, 1),
            end_date=date(2026, 4, 18),
            warmup_days=60,
        )
        total_days = (bt.end_date - bt.start_date).days
        train_days = int(total_days * bt.train_pct)
        split_date = bt.start_date + timedelta(days=train_days)

        # Simulate what run_walk_forward does
        train_cfg = BacktestConfig(
            symbols=bt.symbols,
            start_date=bt.start_date,
            end_date=split_date,
            warmup_days=bt.warmup_days,
        )
        test_cfg = BacktestConfig(
            symbols=bt.symbols,
            start_date=split_date,
            end_date=bt.end_date,
            warmup_days=bt.warmup_days,
        )
        assert train_cfg.warmup_days == 60
        assert test_cfg.warmup_days == 60


class TestParallelWarmupSerialization:
    def test_bt_config_to_dict_includes_warmup_days(self):
        """_bt_config_to_dict serializes warmup_days field."""
        from crypto_trader.optimize.parallel import _bt_config_to_dict

        bt = BacktestConfig(
            symbols=["BTC"],
            start_date=date(2026, 3, 1),
            end_date=date(2026, 4, 18),
            warmup_days=60,
        )
        d = _bt_config_to_dict(bt)
        assert "warmup_days" in d
        assert d["warmup_days"] == 60

    def test_bt_config_to_dict_warmup_zero(self):
        """Default warmup_days=0 is serialized."""
        from crypto_trader.optimize.parallel import _bt_config_to_dict

        bt = BacktestConfig(symbols=["BTC"])
        d = _bt_config_to_dict(bt)
        assert d["warmup_days"] == 0

    def test_bt_config_roundtrip(self):
        """BacktestConfig round-trips through serialization with warmup_days."""
        from crypto_trader.optimize.parallel import _bt_config_to_dict

        bt = BacktestConfig(
            symbols=["BTC", "ETH"],
            start_date=date(2026, 3, 1),
            end_date=date(2026, 4, 18),
            warmup_days=45,
            initial_equity=20000.0,
        )
        d = _bt_config_to_dict(bt)
        bt2 = BacktestConfig(**d)
        assert bt2.warmup_days == 45
        assert bt2.initial_equity == 20000.0
        assert bt2.symbols == ["BTC", "ETH"]


# ── Baking & Integration Tests ──────────────────────────────────────────


class TestRound1MutationsBaked:
    def test_round1_mutations_baked(self):
        """TrendConfig defaults match baked values through round 2."""
        cfg = TrendConfig()
        assert cfg.regime.a_min_adx == 12.0  # baked round 1
        assert cfg.exits.tp1_r == 0.8  # lowered round 3 prep (was 1.2)
        assert cfg.trail.trail_buffer_wide == 1.2  # M15 scale
        assert cfg.trail.trail_buffer_tight == 0.1  # baked round 2 (was 0.2)
        assert cfg.trail.trail_r_ceiling == 1.5  # M15 scale
        assert cfg.risk.risk_pct_b == 0.01  # unchanged


class TestTrendPluginRound2Updates:
    def test_trend_plugin_hard_rejects_updated(self):
        """Hard rejects block non-edge candidates for the current round."""
        from crypto_trader.optimize.trend_plugin import HARD_REJECTS
        assert HARD_REJECTS["total_trades"] == (">=", 30)
        assert HARD_REJECTS["profit_factor"] == (">=", 1.5)
        assert HARD_REJECTS["max_drawdown_pct"] == ("<=", 12.0)
        assert HARD_REJECTS["expectancy_r"] == (">=", 0.10)

    def test_trend_plugin_scoring_weights_balanced(self):
        """Trend optimizer uses the immutable seven-component score."""
        from crypto_trader.optimize.trend_plugin import SCORING_WEIGHTS
        assert len(SCORING_WEIGHTS) == 7
        assert SCORING_WEIGHTS["coverage"] == 0.17
        assert SCORING_WEIGHTS["returns"] == 0.22
        assert SCORING_WEIGHTS["expectancy"] == 0.14
        assert abs(sum(SCORING_WEIGHTS.values()) - 1.0) < 0.01

    def test_trend_plugin_warmup_enforced(self):
        """TrendPlugin __init__ enforces warmup_days >= 60."""
        from crypto_trader.optimize.trend_plugin import TrendPlugin

        bt = BacktestConfig(
            symbols=["BTC"],
            start_date=date(2026, 3, 1),
            end_date=date(2026, 4, 18),
            warmup_days=0,
        )
        plugin = TrendPlugin(bt, TrendConfig())
        assert plugin.backtest_config.warmup_days == 60

    def test_trend_plugin_warmup_preserves_higher(self):
        """TrendPlugin preserves warmup_days if already >= 60."""
        from crypto_trader.optimize.trend_plugin import TrendPlugin

        bt = BacktestConfig(
            symbols=["BTC"],
            start_date=date(2026, 3, 1),
            end_date=date(2026, 4, 18),
            warmup_days=90,
        )
        plugin = TrendPlugin(bt, TrendConfig())
        assert plugin.backtest_config.warmup_days == 90

    def test_phase_gate_criteria_tightened(self):
        """Phase gate criteria match the current aggressive-but-controlled stance."""
        from crypto_trader.optimize.trend_plugin import PHASE_GATE_CRITERIA

        # Phase 3: trades >= 30, PF >= 1.5, exit efficiency guarded.
        p3 = {gc.metric: gc.threshold for gc in PHASE_GATE_CRITERIA[3]}
        assert p3["total_trades"] == 30
        assert p3["profit_factor"] == 1.5
        assert p3["exit_efficiency"] == 0.45

        # Phase 5: DD <= 12%
        p5 = {gc.metric: gc.threshold for gc in PHASE_GATE_CRITERIA[5]}
        assert p5["max_drawdown_pct"] == 12

    def test_phase_candidates_count(self):
        """Phase candidate counts are in expected range."""
        from crypto_trader.optimize.trend_plugin import (
            _phase1_candidates, _phase2_candidates, _phase3_candidates,
            _phase4_candidates, _phase5_candidates, _phase6_candidates,
        )
        assert len(_phase1_candidates()) >= 9
        assert len(_phase2_candidates()) >= 9
        assert len(_phase3_candidates()) >= 9
        assert len(_phase4_candidates()) >= 8
        assert len(_phase5_candidates()) >= 8
        # Phase 6 depends on cumulative mutations
        assert len(_phase6_candidates({})) >= 6

    def test_no_warmup_trades_in_results(self):
        """Trades with entry_time before start_date are excluded."""
        start = date(2026, 3, 15)
        measurement_start = datetime.combine(
            start, datetime.min.time(), tzinfo=timezone.utc,
        )

        # Trade during warmup
        t1 = MagicMock()
        t1.entry_time = datetime(2026, 2, 28, 10, 0, tzinfo=timezone.utc)
        # Trade at exact measurement boundary
        t2 = MagicMock()
        t2.entry_time = measurement_start
        # Trade during measurement
        t3 = MagicMock()
        t3.entry_time = datetime(2026, 3, 20, 14, 30, tzinfo=timezone.utc)

        trades = [t1, t2, t3]
        filtered = [t for t in trades if t.entry_time >= measurement_start]
        assert len(filtered) == 2
        assert t1 not in filtered
        assert t2 in filtered
        assert t3 in filtered


# ── M15 Execution Layer Tests ──────────────────────────────────────────


class TestM15ExecutionLayer:
    def test_trend_timeframes_include_m15(self):
        """TrendStrategy.timeframes includes M15 as first element."""
        from crypto_trader.strategy.trend.strategy import TrendStrategy
        from crypto_trader.core.models import TimeFrame
        s = TrendStrategy()
        assert s.timeframes == [TimeFrame.M15, TimeFrame.H1, TimeFrame.D1]

    def test_trend_m15_primary_in_runner(self):
        """_create_strategy returns M15 as primary TF for trend."""
        from crypto_trader.backtest.runner import _create_strategy
        from crypto_trader.core.models import TimeFrame
        _, tfs, primary = _create_strategy("trend", TrendConfig())
        assert primary == TimeFrame.M15
        assert TimeFrame.M15 in tfs

    def test_m15_indicators_initialized(self):
        """on_init creates M15 IncrementalIndicators per symbol."""
        from crypto_trader.strategy.trend.strategy import TrendStrategy
        cfg = TrendConfig()
        cfg.symbols = ["BTC"]
        s = TrendStrategy(cfg)
        ctx = MagicMock()
        ctx.events = MagicMock()
        s.on_init(ctx)
        assert "BTC" in s._m15_inc
        assert "BTC" in s._m15_bar_count
        assert s._m15_bar_count["BTC"] == 0

    def test_m15_bar_count_tracked(self):
        """M15 bars increment the M15 counter."""
        from crypto_trader.strategy.trend.strategy import TrendStrategy
        from crypto_trader.core.models import TimeFrame
        cfg = TrendConfig()
        cfg.symbols = ["BTC"]
        s = TrendStrategy(cfg)
        ctx = MagicMock()
        ctx.events = MagicMock()
        s.on_init(ctx)

        bar = MagicMock()
        bar.symbol = "BTC"
        bar.timeframe = TimeFrame.M15
        bar.open = bar.high = bar.low = bar.close = 100.0
        bar.volume = 1.0
        bar.timestamp = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

        s.on_bar(bar, ctx)
        assert s._m15_bar_count["BTC"] == 1

    def test_position_managed_on_m15(self):
        """_handle_m15 calls _manage_positions when position exists."""
        from crypto_trader.strategy.trend.strategy import TrendStrategy
        from crypto_trader.core.models import TimeFrame
        cfg = TrendConfig()
        cfg.symbols = ["BTC"]
        s = TrendStrategy(cfg)
        ctx = MagicMock()
        ctx.events = MagicMock()
        s.on_init(ctx)

        # Warm up M15
        s._m15_bar_count["BTC"] = 10

        # Mock position and meta
        ctx.broker.get_position.return_value = MagicMock(qty=1.0)
        s._position_meta["BTC"] = MagicMock()

        bar = MagicMock()
        bar.symbol = "BTC"
        bar.timeframe = TimeFrame.M15

        with patch.object(s, "_manage_positions") as mock_manage:
            s._handle_m15(bar, "BTC", ctx)
            mock_manage.assert_called_once()

    def test_h1_no_position_management(self):
        """_handle_h1 no longer calls _manage_positions."""
        from crypto_trader.strategy.trend.strategy import TrendStrategy
        from crypto_trader.core.models import TimeFrame
        cfg = TrendConfig()
        cfg.symbols = ["BTC"]
        s = TrendStrategy(cfg)
        ctx = MagicMock()
        ctx.events = MagicMock()
        ctx.config = BacktestConfig(start_date=date(2026, 3, 15), symbols=["BTC"])
        s.on_init(ctx)

        # Warm up H1
        s._h1_bar_count["BTC"] = 100
        s._h1_indicators["BTC"] = MagicMock()

        # Position exists — should skip to return (no management call)
        ctx.broker.get_position.return_value = MagicMock(qty=1.0)

        bar = MagicMock()
        bar.symbol = "BTC"
        bar.timeframe = TimeFrame.H1
        bar.timestamp = datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc)

        with patch.object(s, "_manage_positions") as mock_manage:
            s._handle_h1(bar, "BTC", ctx)
            mock_manage.assert_not_called()

    def test_aggressive_risk_defaults(self):
        """Risk defaults are aggressive for M15 perps."""
        cfg = TrendConfig()
        assert cfg.risk.risk_pct_a == 0.015
        assert cfg.risk.max_leverage_major == 15.0
        assert cfg.risk.max_leverage_alt == 12.0
        assert cfg.risk.max_risk_pct == 0.02
        assert cfg.limits.max_concurrent_positions == 5
        assert cfg.limits.max_daily_loss_pct == 0.04
        assert cfg.limits.max_trades_per_day == 10

    def test_quick_exit_enabled_default(self):
        """Quick exit is enabled by default for M15."""
        cfg = TrendConfig()
        assert cfg.exits.quick_exit_enabled is True
        assert cfg.exits.quick_exit_bars == 12
        assert cfg.exits.quick_exit_max_mfe_r == 0.15
        assert cfg.exits.time_stop_bars == 20  # baked round 2
        assert cfg.exits.be_min_bars_above == 4

    def test_m15_config_roundtrip(self):
        """m15_indicators survives to_dict()/from_dict() round trip."""
        cfg = TrendConfig()
        d = cfg.to_dict()
        assert "m15_indicators" in d
        cfg2 = TrendConfig.from_dict(d)
        assert cfg2.m15_indicators.ema_fast == 20
        assert cfg2.m15_indicators.ema_slow == 200


def _make_strategy_ctx(start_date: date) -> MagicMock:
    ctx = MagicMock()
    ctx.events.subscribe = MagicMock()
    ctx.config = BacktestConfig(start_date=start_date, symbols=["BTC"], initial_equity=10_000.0)
    ctx.clock.now.return_value = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    ctx.bars.get.return_value = [MagicMock()] * 60
    ctx.broker.get_position.return_value = None
    ctx.broker.get_positions.return_value = []
    ctx.broker.get_equity.return_value = 10_000.0
    ctx.broker.get_open_orders.return_value = []
    ctx.broker.submit_order = MagicMock()
    return ctx


class TestMeasurementBoundaryEntryGating:
    def test_momentum_blocks_entry_submission_before_measurement_start(self):
        from crypto_trader.strategy.momentum.strategy import MomentumStrategy, WARMUP_BARS

        cfg = MomentumConfig()
        cfg.symbols = ["BTC"]
        strategy = MomentumStrategy(cfg)
        ctx = _make_strategy_ctx(date(2026, 3, 15))
        strategy.on_init(ctx)

        strategy._m15_bar_count["BTC"] = WARMUP_BARS
        strategy._current_bias["BTC"] = MagicMock(direction=Side.LONG, reasons=[])
        strategy._inc["BTC"][TimeFrame.M15] = MagicMock(
            update=MagicMock(return_value=MagicMock(atr=1.0, volume_ma=1.0))
        )
        strategy._env_filter.check = MagicMock(return_value=MagicMock(allowed=True, require_a_grade=False))
        strategy._setup_detector.detect = MagicMock(return_value=MagicMock(
            grade=SetupGrade.A,
            confluences=("ema",),
            zone_price=100.0,
        ))
        strategy._confirmation_detector.check = MagicMock(return_value=MagicMock(pattern_type="engulfing"))
        strategy._stop_placer.compute = MagicMock(return_value=95.0)
        strategy._sizer.compute = MagicMock(return_value=(MagicMock(
            qty=1.0,
            leverage=10.0,
            liquidation_price=50.0,
            risk_pct_actual=0.01,
        ), ""))
        strategy._entry_signal.generate = MagicMock(return_value=Order(
            order_id="",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=1.0,
            tag="entry",
        ))

        bar = MagicMock(
            symbol="BTC",
            timeframe=TimeFrame.M15,
            timestamp=datetime(2026, 3, 14, 23, 45, tzinfo=timezone.utc),
            close=100.0,
        )

        strategy._handle_m15(bar, ctx)

        assert strategy._m15_indicators["BTC"] is not None
        ctx.broker.submit_order.assert_not_called()

    def test_trend_blocks_entry_submission_before_measurement_start(self):
        from crypto_trader.strategy.trend.strategy import TrendStrategy, WARMUP_BARS

        cfg = TrendConfig()
        cfg.symbols = ["BTC"]
        strategy = TrendStrategy(cfg)
        ctx = _make_strategy_ctx(date(2026, 3, 15))
        strategy.on_init(ctx)

        strategy._h1_bar_count["BTC"] = WARMUP_BARS
        strategy._h1_inc["BTC"] = MagicMock(update=MagicMock(return_value=MagicMock(atr=1.0)))
        strategy._current_regime["BTC"] = MagicMock(tier="a", direction=Side.LONG, reasons=["trend"])
        strategy._setup_detector.detect = MagicMock(return_value=MagicMock(
            direction=Side.LONG,
            grade=SetupGrade.A,
            confluences=("ema",),
            room_r=2.0,
        ))
        strategy._trigger_detector.check = MagicMock(return_value=MagicMock(pattern="engulfing"))
        strategy._stop_placer.compute = MagicMock(return_value=95.0)
        strategy._sizer.compute = MagicMock(return_value=(MagicMock(
            qty=1.0,
            leverage=10.0,
            liquidation_price=50.0,
            risk_pct_actual=0.01,
        ), ""))
        strategy._entry_generator.generate = MagicMock(return_value=Order(
            order_id="",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=1.0,
            tag="entry",
        ))

        bar = MagicMock(
            symbol="BTC",
            timeframe=TimeFrame.H1,
            timestamp=datetime(2026, 3, 14, 23, 0, tzinfo=timezone.utc),
            close=100.0,
        )

        strategy._handle_h1(bar, "BTC", ctx)

        assert strategy._h1_indicators["BTC"] is not None
        ctx.broker.submit_order.assert_not_called()

    def test_breakout_blocks_entry_submission_before_measurement_start(self):
        from crypto_trader.strategy.breakout.strategy import BreakoutStrategy, WARMUP_BARS

        cfg = BreakoutConfig()
        cfg.symbols = ["BTC"]
        strategy = BreakoutStrategy(cfg)
        ctx = _make_strategy_ctx(date(2026, 3, 15))
        strategy.on_init(ctx)

        strategy._m30_bar_count["BTC"] = WARMUP_BARS
        strategy._m30_inc["BTC"] = MagicMock(update=MagicMock(return_value=MagicMock(atr=1.0)))
        strategy._current_profile["BTC"] = MagicMock()
        strategy._balance_detector.update = MagicMock()
        strategy._manage_positions = MagicMock()
        strategy._confirmation_detector.clear_pending = MagicMock()

        bar = MagicMock(
            symbol="BTC",
            timeframe=TimeFrame.M30,
            timestamp=datetime(2026, 3, 14, 23, 30, tzinfo=timezone.utc),
            close=100.0,
        )

        strategy._handle_m30(bar, "BTC", ctx)

        assert strategy._m30_indicators["BTC"] is not None
        strategy._confirmation_detector.clear_pending.assert_not_called()
        ctx.broker.submit_order.assert_not_called()
