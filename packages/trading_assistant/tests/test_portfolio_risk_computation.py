"""Tests for cross-bot portfolio risk computation."""
from trading_assistant.schemas.daily_metrics import BotDailySummary
from trading_assistant.skills.compute_portfolio_risk import PortfolioRiskComputer


def _make_summary(bot_id: str, exposure_pct: float = 20.0, **kwargs) -> BotDailySummary:
    defaults = dict(
        date="2026-03-01",
        bot_id=bot_id,
        total_trades=10,
        win_count=6,
        loss_count=4,
        gross_pnl=100.0,
        exposure_pct=exposure_pct,
    )
    defaults.update(kwargs)
    return BotDailySummary(**defaults)


class TestPortfolioRiskComputer:
    def test_total_exposure(self):
        summaries = [
            _make_summary("bot1", exposure_pct=30.0),
            _make_summary("bot2", exposure_pct=25.0),
        ]
        computer = PortfolioRiskComputer(
            date="2026-03-01",
            bot_summaries=summaries,
            position_details={
                "bot1": [{"symbol": "BTC", "direction": "LONG", "exposure_pct": 30.0}],
                "bot2": [{"symbol": "ETH", "direction": "LONG", "exposure_pct": 25.0}],
            },
        )
        card = computer.compute()
        assert card.total_exposure_pct == 55.0

    def test_exposure_by_symbol(self):
        computer = PortfolioRiskComputer(
            date="2026-03-01",
            bot_summaries=[_make_summary("bot1"), _make_summary("bot2")],
            position_details={
                "bot1": [{"symbol": "BTC", "direction": "LONG", "exposure_pct": 30.0}],
                "bot2": [{"symbol": "BTC", "direction": "LONG", "exposure_pct": 25.0}],
            },
        )
        card = computer.compute()
        assert card.exposure_by_symbol["BTC"] == 55.0

    def test_exposure_by_direction(self):
        computer = PortfolioRiskComputer(
            date="2026-03-01",
            bot_summaries=[_make_summary("bot1"), _make_summary("bot2")],
            position_details={
                "bot1": [{"symbol": "BTC", "direction": "LONG", "exposure_pct": 30.0}],
                "bot2": [{"symbol": "BTC", "direction": "SHORT", "exposure_pct": 10.0}],
            },
        )
        card = computer.compute()
        assert card.exposure_by_direction["LONG"] == 30.0
        assert card.exposure_by_direction["SHORT"] == 10.0

    def test_single_symbol_concentration_alert(self):
        computer = PortfolioRiskComputer(
            date="2026-03-01",
            bot_summaries=[_make_summary("bot1"), _make_summary("bot2")],
            position_details={
                "bot1": [{"symbol": "BTC", "direction": "LONG", "exposure_pct": 40.0}],
                "bot2": [{"symbol": "BTC", "direction": "LONG", "exposure_pct": 20.0}],
            },
            max_single_symbol_pct=50.0,
        )
        card = computer.compute()
        btc_alerts = [a for a in card.crowding_alerts if a.symbol == "BTC"]
        assert len(btc_alerts) == 1
        assert btc_alerts[0].alert_type == "single_symbol_concentration"

    def test_same_side_alert(self):
        computer = PortfolioRiskComputer(
            date="2026-03-01",
            bot_summaries=[_make_summary("b1"), _make_summary("b2"), _make_summary("b3")],
            position_details={
                "b1": [{"symbol": "BTC", "direction": "LONG", "exposure_pct": 10.0}],
                "b2": [{"symbol": "BTC", "direction": "LONG", "exposure_pct": 10.0}],
                "b3": [{"symbol": "BTC", "direction": "LONG", "exposure_pct": 10.0}],
            },
        )
        card = computer.compute()
        same_side = [a for a in card.crowding_alerts if a.alert_type == "same_side"]
        assert len(same_side) == 1

    def test_total_exposure_alert(self):
        computer = PortfolioRiskComputer(
            date="2026-03-01",
            bot_summaries=[_make_summary("bot1"), _make_summary("bot2")],
            position_details={
                "bot1": [{"symbol": "BTC", "direction": "LONG", "exposure_pct": 50.0}],
                "bot2": [{"symbol": "ETH", "direction": "LONG", "exposure_pct": 60.0}],
            },
            max_total_exposure_pct=80.0,
        )
        card = computer.compute()
        total_alerts = [a for a in card.crowding_alerts if a.alert_type == "total_exposure"]
        assert len(total_alerts) == 1

    def test_concentration_score(self):
        # All in one symbol = high concentration
        computer = PortfolioRiskComputer(
            date="2026-03-01",
            bot_summaries=[_make_summary("bot1")],
            position_details={
                "bot1": [{"symbol": "BTC", "direction": "LONG", "exposure_pct": 100.0}],
            },
        )
        card = computer.compute()
        assert card.concentration_score > 80  # single symbol = highly concentrated

    def test_no_alerts_when_diversified(self):
        computer = PortfolioRiskComputer(
            date="2026-03-01",
            bot_summaries=[_make_summary("bot1"), _make_summary("bot2")],
            position_details={
                "bot1": [{"symbol": "BTC", "direction": "LONG", "exposure_pct": 10.0}],
                "bot2": [{"symbol": "ETH", "direction": "SHORT", "exposure_pct": 10.0}],
            },
        )
        card = computer.compute()
        assert card.crowding_alerts == []
