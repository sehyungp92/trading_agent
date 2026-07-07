"""Risk calculation utilities."""


class RiskCalculator:
    """Computes risk metrics for orders and positions."""

    @staticmethod
    def compute_order_risk_dollars(
        planned_entry: float,
        stop_price: float,
        qty: int,
        point_value: float,
    ) -> float:
        return abs(planned_entry - stop_price) * point_value * qty

    @staticmethod
    def compute_risk_R(risk_dollars: float, unit_risk_dollars: float) -> float:
        if unit_risk_dollars <= 0:
            return float("inf")
        return risk_dollars / unit_risk_dollars

    @staticmethod
    def compute_unit_risk_dollars(
        nav: float, unit_risk_pct: float, vol_factor: float = 1.0
    ) -> float:
        """unit_risk = NAV * unit_risk_pct * vol_factor"""
        return nav * unit_risk_pct * vol_factor
