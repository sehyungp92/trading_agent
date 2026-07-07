"""Tests for DrawdownAnalyzer — drawdown episode segmentation + attribution."""
from trading_assistant.schemas.drawdown_analysis import DrawdownAttribution, DrawdownEpisode
from trading_assistant.skills.drawdown_analyzer import DrawdownAnalyzer
from tests.factories import make_trade


def _make_trade(trade_id: str, date_str: str, pnl: float,
                root_causes: list[str] | None = None,
                regime: str = "trending_up"):
    return make_trade(
        trade_id=trade_id, pnl=pnl, pair="BTCUSDT",
        entry_price=50000, market_regime=regime,
        root_causes=root_causes or [],
        entry_time=f"{date_str}T10:00:00",
        exit_time=f"{date_str}T11:00:00",
    )


class TestDrawdownAnalyzer:
    def test_identifies_single_drawdown(self):
        # Equity curve: 100, 150, 70, -50, -100, 200
        # Peak is 150 after t2. t3-t5 drop below peak.
        # t6 (+300) brings cumulative to 200 which exceeds peak of 150 => recovered.
        # Drawdown trades: t3, t4, t5 (3 trades in the episode).
        trades = [
            _make_trade("t1", "2026-02-20", 100),
            _make_trade("t2", "2026-02-21", 50),
            _make_trade("t3", "2026-02-22", -80, ["regime_mismatch"]),
            _make_trade("t4", "2026-02-23", -120, ["weak_signal"]),
            _make_trade("t5", "2026-02-24", -50, ["regime_mismatch"]),
            _make_trade("t6", "2026-02-25", 300),
        ]
        analyzer = DrawdownAnalyzer(bot_id="bot1", date="2026-03-01")
        attr = analyzer.compute(trades)

        assert isinstance(attr, DrawdownAttribution)
        assert len(attr.episodes) == 1
        assert attr.episodes[0].trade_count == 3
        assert attr.episodes[0].recovered is True
        assert attr.episodes[0].drawdown_pct > 0

    def test_no_drawdown(self):
        trades = [
            _make_trade("t1", "2026-02-20", 100),
            _make_trade("t2", "2026-02-21", 50),
            _make_trade("t3", "2026-02-22", 200),
        ]
        analyzer = DrawdownAnalyzer(bot_id="bot1", date="2026-03-01")
        attr = analyzer.compute(trades)
        assert len(attr.episodes) == 0

    def test_root_cause_attribution(self):
        trades = [
            _make_trade("t1", "2026-02-20", 100),
            _make_trade("t2", "2026-02-21", -80, ["regime_mismatch"]),
            _make_trade("t3", "2026-02-22", -60, ["regime_mismatch"]),
            _make_trade("t4", "2026-02-23", -40, ["weak_signal"]),
            _make_trade("t5", "2026-02-24", 200),
        ]
        analyzer = DrawdownAnalyzer(bot_id="bot1", date="2026-03-01")
        attr = analyzer.compute(trades)

        assert "regime_mismatch" in attr.top_contributing_root_causes
        assert attr.top_contributing_root_causes["regime_mismatch"] == 2

    def test_largest_single_loss(self):
        trades = [
            _make_trade("t1", "2026-02-20", 1000),
            _make_trade("t2", "2026-02-21", -200),
            _make_trade("t3", "2026-02-22", -50),
            _make_trade("t4", "2026-02-23", 500),
        ]
        analyzer = DrawdownAnalyzer(bot_id="bot1", date="2026-03-01")
        attr = analyzer.compute(trades)
        assert attr.largest_single_loss_pct > 0

    def test_empty_trades(self):
        analyzer = DrawdownAnalyzer(bot_id="bot1", date="2026-03-01")
        attr = analyzer.compute([])
        assert len(attr.episodes) == 0
        assert attr.largest_single_loss_pct == 0.0


class TestDrawdownEpisode:
    def test_creates_episode(self):
        ep = DrawdownEpisode(
            bot_id="bot1",
            start_date="2026-02-20",
            end_date="2026-02-25",
            peak_pnl=5000.0,
            trough_pnl=4200.0,
            drawdown_pct=16.0,
            trade_count=12,
            duration_days=5,
        )
        assert ep.drawdown_pct == 16.0
        assert ep.duration_days == 5

    def test_recovery_flag(self):
        ep = DrawdownEpisode(
            bot_id="bot1",
            start_date="2026-02-20",
            end_date="2026-02-25",
            peak_pnl=5000.0,
            trough_pnl=4200.0,
            recovered=True,
            recovery_date="2026-02-28",
        )
        assert ep.recovered is True


class TestDrawdownAttribution:
    def test_creates_attribution(self):
        attr = DrawdownAttribution(
            bot_id="bot1",
            date="2026-03-01",
            episodes=[
                DrawdownEpisode(
                    bot_id="bot1",
                    start_date="2026-02-20",
                    end_date="2026-02-25",
                    peak_pnl=5000.0,
                    trough_pnl=4200.0,
                    drawdown_pct=16.0,
                    trade_count=12,
                    duration_days=5,
                ),
            ],
            top_contributing_root_causes={"regime_mismatch": 5, "weak_signal": 3},
            largest_single_loss_pct=4.2,
        )
        assert len(attr.episodes) == 1
        assert attr.top_contributing_root_causes["regime_mismatch"] == 5

    def test_max_drawdown_property(self):
        attr = DrawdownAttribution(
            bot_id="bot1",
            date="2026-03-01",
            episodes=[
                DrawdownEpisode(bot_id="bot1", start_date="a", end_date="b",
                                peak_pnl=100, trough_pnl=90, drawdown_pct=10.0),
                DrawdownEpisode(bot_id="bot1", start_date="c", end_date="d",
                                peak_pnl=200, trough_pnl=150, drawdown_pct=25.0),
            ],
        )
        assert attr.max_drawdown_pct == 25.0

    def test_empty_episodes(self):
        attr = DrawdownAttribution(bot_id="bot1", date="2026-03-01")
        assert attr.max_drawdown_pct == 0.0
