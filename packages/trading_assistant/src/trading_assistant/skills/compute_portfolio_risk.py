"""Cross-bot portfolio risk computation — daily, no LLM calls.

Computes exposure, concentration, and crowding alerts.
If crowding alerts trigger, the daily report leads with them.
"""
from __future__ import annotations

import math
from collections import defaultdict

from trading_assistant.schemas.daily_metrics import BotDailySummary
from trading_assistant.schemas.portfolio_risk import PortfolioRiskCard, CrowdingAlert


class PortfolioRiskComputer:
    """Computes a PortfolioRiskCard from bot summaries and position details."""

    def __init__(
        self,
        date: str,
        bot_summaries: list[BotDailySummary],
        position_details: dict[str, list[dict]],
        max_single_symbol_pct: float = 50.0,
        max_total_exposure_pct: float = 80.0,
        correlation_threshold: float = 0.7,
        historical_pnl: dict[str, list[float]] | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        self.date = date
        self.bot_summaries = bot_summaries
        self.position_details = position_details  # bot_id → [{symbol, direction, exposure_pct}]
        self.max_single_symbol_pct = max_single_symbol_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.correlation_threshold = correlation_threshold
        self.historical_pnl = historical_pnl  # bot_id → list of daily PnL values
        self.sector_map = sector_map  # symbol → sector

    def compute(self) -> PortfolioRiskCard:
        """Build the full risk card."""
        exposure_by_symbol = self._compute_exposure_by_symbol()
        exposure_by_direction = self._compute_exposure_by_direction()
        total_exposure = sum(exposure_by_symbol.values())
        concentration_score = self._compute_concentration_score(exposure_by_symbol, total_exposure)
        alerts = self._detect_alerts(exposure_by_symbol, exposure_by_direction, total_exposure)

        # Correlation matrix from historical PnL
        correlation_matrix = self._compute_correlation_matrix()

        # High correlation alerts
        for pair_key, corr_val in correlation_matrix.items():
            if corr_val > self.correlation_threshold:
                bots = pair_key.split("_", 1)
                alerts.append(CrowdingAlert(
                    alert_type="high_correlation",
                    description=(
                        f"Bots {bots[0]} and {bots[1]} have PnL correlation "
                        f"{corr_val:.2f} > {self.correlation_threshold}"
                    ),
                    severity="medium",
                    bots_involved=bots,
                ))

        # Sector exposure and alerts
        sector_exposure = self._compute_sector_exposure()
        for sector, pct in sector_exposure.items():
            if pct > self.max_single_symbol_pct:
                alerts.append(CrowdingAlert(
                    alert_type="sector_concentration",
                    description=(
                        f"Sector '{sector}' exposure {pct:.1f}% exceeds "
                        f"{self.max_single_symbol_pct}%"
                    ),
                    severity="medium",
                ))

        return PortfolioRiskCard(
            date=self.date,
            total_exposure_pct=total_exposure,
            exposure_by_symbol=exposure_by_symbol,
            exposure_by_direction=exposure_by_direction,
            correlation_matrix=correlation_matrix,
            concentration_score=concentration_score,
            crowding_alerts=alerts,
        )

    def _compute_exposure_by_symbol(self) -> dict[str, float]:
        by_symbol: dict[str, float] = defaultdict(float)
        for positions in self.position_details.values():
            for pos in positions:
                by_symbol[pos["symbol"]] += pos["exposure_pct"]
        return dict(by_symbol)

    def _compute_exposure_by_direction(self) -> dict[str, float]:
        by_dir: dict[str, float] = defaultdict(float)
        for positions in self.position_details.values():
            for pos in positions:
                by_dir[pos["direction"]] += pos["exposure_pct"]
        return dict(by_dir)

    def _compute_concentration_score(
        self, exposure_by_symbol: dict[str, float], total_exposure: float
    ) -> float:
        """Herfindahl-Hirschman Index normalized to 0–100. Higher = more concentrated."""
        if total_exposure == 0:
            return 0.0
        shares = [exp / total_exposure for exp in exposure_by_symbol.values()]
        hhi = sum(s * s for s in shares)
        # HHI ranges from 1/N (perfectly diversified) to 1.0 (single symbol)
        # Normalize: 0 = perfectly diversified, 100 = single symbol
        n = len(shares)
        if n <= 1:
            return 100.0
        min_hhi = 1.0 / n
        return ((hhi - min_hhi) / (1.0 - min_hhi)) * 100.0

    def _detect_alerts(
        self,
        exposure_by_symbol: dict[str, float],
        exposure_by_direction: dict[str, float],
        total_exposure: float,
    ) -> list[CrowdingAlert]:
        alerts: list[CrowdingAlert] = []

        # Single symbol concentration
        for symbol, exp in exposure_by_symbol.items():
            if exp > self.max_single_symbol_pct:
                alerts.append(CrowdingAlert(
                    alert_type="single_symbol_concentration",
                    description=f"{symbol} > {self.max_single_symbol_pct}% of total exposure ({exp:.1f}%)",
                    severity="medium",
                    symbol=symbol,
                    exposure_pct=exp,
                ))

        # Total exposure too high
        if total_exposure > self.max_total_exposure_pct:
            alerts.append(CrowdingAlert(
                alert_type="total_exposure",
                description=f"Total exposure {total_exposure:.1f}% exceeds {self.max_total_exposure_pct}%",
                severity="high",
            ))

        # All bots on same side of same symbol
        symbol_direction_bots: dict[tuple[str, str], list[str]] = defaultdict(list)
        for bot_id, positions in self.position_details.items():
            for pos in positions:
                key = (pos["symbol"], pos["direction"])
                symbol_direction_bots[key].append(bot_id)

        for (symbol, direction), bots in symbol_direction_bots.items():
            if len(bots) >= 3:  # 3+ bots on same side of same asset
                alerts.append(CrowdingAlert(
                    alert_type="same_side",
                    description=f"{len(bots)} bots all {direction} on {symbol}",
                    severity="high",
                    bots_involved=bots,
                    symbol=symbol,
                ))

        return alerts

    # ── Correlation matrix (0A) ──────────────────────────────────────

    def _compute_correlation_matrix(self) -> dict[str, float]:
        """Compute pairwise Pearson correlation from daily PnL series between bots.

        Returns a dict with keys like ``"botA_botB"`` (sorted pair) mapping to
        the Pearson correlation coefficient.  Requires ``historical_pnl`` to be
        set; returns empty dict otherwise.
        """
        if not self.historical_pnl:
            return {}

        bot_ids = sorted(self.historical_pnl.keys())
        if len(bot_ids) < 2:
            return {}

        matrix: dict[str, float] = {}
        for i in range(len(bot_ids)):
            for j in range(i + 1, len(bot_ids)):
                a_id, b_id = bot_ids[i], bot_ids[j]
                corr = self._pearson(
                    self.historical_pnl[a_id],
                    self.historical_pnl[b_id],
                )
                if corr is not None:
                    matrix[f"{a_id}_{b_id}"] = round(corr, 4)
        return matrix

    @staticmethod
    def _pearson(xs: list[float], ys: list[float]) -> float | None:
        """Pearson correlation for two equal-length numeric lists.

        Returns None when insufficient data or zero variance.
        """
        n = min(len(xs), len(ys))
        if n < 2:
            return None
        xs, ys = xs[:n], ys[:n]

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)

        denom = math.sqrt(var_x * var_y)
        if denom == 0:
            return None
        return cov / denom

    # ── Sector exposure (0E) ─────────────────────────────────────────

    def _compute_sector_exposure(self) -> dict[str, float]:
        """Aggregate exposure by sector using the optional sector_map.

        Returns a dict of sector → total exposure percentage.
        If no sector_map is provided, returns an empty dict.
        """
        if not self.sector_map:
            return {}

        by_sector: dict[str, float] = defaultdict(float)
        for positions in self.position_details.values():
            for pos in positions:
                symbol = pos["symbol"]
                sector = self.sector_map.get(symbol, "unknown")
                by_sector[sector] += pos["exposure_pct"]
        return dict(by_sector)

    # ── Drawdown correlation (0G) ────────────────────────────────────

    @classmethod
    def compute_drawdown_correlation(
        cls,
        family_equity_curves: dict[str, list[float]],
    ) -> dict:
        """Compute drawdown correlation across strategy families.

        Args:
            family_equity_curves: family name → list of equity values
                (e.g. cumulative PnL per day).

        Returns:
            Dict with:
            - ``simultaneous_drawdown_days``: days where ALL families were in drawdown
            - ``family_drawdown_overlap``: pairwise overlap analysis
            - ``worst_portfolio_drawdown_pct``: worst combined drawdown
            - ``systemic_risk_score``: 0-100 composite score
        """
        if not family_equity_curves or len(family_equity_curves) < 2:
            return {
                "simultaneous_drawdown_days": 0,
                "family_drawdown_overlap": {},
                "worst_portfolio_drawdown_pct": 0.0,
                "systemic_risk_score": 0.0,
            }

        # Compute drawdown series for each family
        dd_series: dict[str, list[float]] = {}
        for family, curve in family_equity_curves.items():
            dd_series[family] = cls._drawdown_series(curve)

        families = sorted(dd_series.keys())
        min_len = min(len(dd_series[f]) for f in families)
        if min_len == 0:
            return {
                "simultaneous_drawdown_days": 0,
                "family_drawdown_overlap": {},
                "worst_portfolio_drawdown_pct": 0.0,
                "systemic_risk_score": 0.0,
            }

        # Simultaneous drawdown: days where ALL families are in drawdown (< 0)
        simul_days = 0
        for day_idx in range(min_len):
            if all(dd_series[f][day_idx] < 0 for f in families):
                simul_days += 1

        # Portfolio combined equity (sum of curves) → worst drawdown
        combined_curve = []
        for day_idx in range(min_len):
            combined_curve.append(sum(
                family_equity_curves[f][day_idx]
                for f in families
                if day_idx < len(family_equity_curves[f])
            ))
        combined_dd = cls._drawdown_series(combined_curve)
        worst_dd = min(combined_dd) if combined_dd else 0.0

        # Pairwise overlap: days both families are in drawdown simultaneously
        overlap: dict[str, dict] = {}
        for i in range(len(families)):
            for j in range(i + 1, len(families)):
                fa, fb = families[i], families[j]
                pair_len = min(len(dd_series[fa]), len(dd_series[fb]))
                overlap_days = sum(
                    1 for d in range(pair_len)
                    if dd_series[fa][d] < 0 and dd_series[fb][d] < 0
                )
                # Correlation of drawdown series
                corr = cls._pearson(
                    dd_series[fa][:pair_len],
                    dd_series[fb][:pair_len],
                )
                overlap[f"{fa}_{fb}"] = {
                    "overlap_days": overlap_days,
                    "correlation": round(corr, 4) if corr is not None else 0.0,
                }

        # Systemic risk score: 0–100 composite
        # Components: simultaneous ratio + avg pairwise DD correlation + worst DD magnitude
        simul_ratio = simul_days / min_len if min_len > 0 else 0.0
        avg_corr = 0.0
        if overlap:
            corr_vals = [v["correlation"] for v in overlap.values()]
            avg_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
        dd_severity = min(abs(worst_dd) / 10.0, 1.0)  # cap at 10% for max score component

        systemic_score = round(
            (simul_ratio * 40 + max(avg_corr, 0) * 30 + dd_severity * 30), 1
        )
        systemic_score = max(0.0, min(100.0, systemic_score))

        return {
            "simultaneous_drawdown_days": simul_days,
            "family_drawdown_overlap": overlap,
            "worst_portfolio_drawdown_pct": round(worst_dd, 4),
            "systemic_risk_score": systemic_score,
        }

    @staticmethod
    def _drawdown_series(equity_curve: list[float]) -> list[float]:
        """Convert an equity curve into a drawdown series (percentages, negative).

        Each element is the current drawdown from peak as a percentage
        (0.0 means at peak, negative means below peak).
        """
        if not equity_curve:
            return []

        dd = []
        peak = equity_curve[0]
        for val in equity_curve:
            if val > peak:
                peak = val
            if peak != 0:
                dd.append(((val - peak) / abs(peak)) * 100.0)
            else:
                dd.append(0.0)
        return dd
