# tests/test_portfolio_risk.py
"""Tests for portfolio risk schemas."""
from trading_assistant.schemas.portfolio_risk import PortfolioRiskCard, CrowdingAlert


class TestPortfolioRiskCard:
    def test_creates_minimal_card(self):
        card = PortfolioRiskCard(
            date="2026-03-01",
            total_exposure_pct=45.0,
            exposure_by_symbol={"BTC": 30.0, "ETH": 15.0},
            exposure_by_direction={"LONG": 35.0, "SHORT": 10.0},
        )
        assert card.total_exposure_pct == 45.0

    def test_crowding_alerts_empty_by_default(self):
        card = PortfolioRiskCard(date="2026-03-01")
        assert card.crowding_alerts == []

    def test_concentration_score_bounds(self):
        card = PortfolioRiskCard(
            date="2026-03-01",
            concentration_score=85.0,
        )
        assert 0 <= card.concentration_score <= 100

    def test_with_correlation_matrix(self):
        card = PortfolioRiskCard(
            date="2026-03-01",
            correlation_matrix={"bot1_bot2": 0.75, "bot1_bot3": 0.3},
        )
        assert card.correlation_matrix["bot1_bot2"] == 0.75


class TestCrowdingAlert:
    def test_creates_alert(self):
        alert = CrowdingAlert(
            alert_type="high_correlation",
            description="bot1 and bot2 correlation > 0.7",
            severity="high",
            bots_involved=["bot1", "bot2"],
        )
        assert alert.alert_type == "high_correlation"
        assert len(alert.bots_involved) == 2

    def test_single_symbol_concentration(self):
        alert = CrowdingAlert(
            alert_type="single_symbol_concentration",
            description="BTC > 50% of total exposure",
            severity="medium",
            symbol="BTC",
            exposure_pct=65.0,
        )
        assert alert.symbol == "BTC"
