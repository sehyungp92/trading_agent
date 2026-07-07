# skills/ground_truth_computer.py
"""Ground truth computer — deterministic evaluation function.

Computes a legacy helper composite score from daily curated summaries using the
objective_weights_v1 6-component z-score-normalized formula:
  expected_total_r (30%), calmar (20%), profit_factor (15%),
  expectancy (15%), inverse_drawdown (10%), process_quality (10%).

Replay-backed monthly/phased-auto candidate ranking is governed by immutable
score profiles, not this helper composite. soul.md uses broader prompt bands
that are only directionally aligned with this historical helper formula.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

from trading_assistant.schemas.learning_ledger import GroundTruthSnapshot
from trading_assistant.schemas.objective_weights import (
    W_CALMAR,
    W_EXPECTANCY,
    W_EXPECTED_R,
    W_INV_DRAWDOWN,
    W_PROCESS,
    W_PROFIT_FACTOR,
)


class GroundTruthComputer:
    """Computes legacy helper ground truth performance snapshots."""

    # Helper composite weights imported from schemas.objective_weights.
    # Monthly replay candidate ranking uses immutable_score_profiles_v1 instead.
    _W_EXPECTED_R = W_EXPECTED_R
    _W_CALMAR = W_CALMAR
    _W_PROFIT_FACTOR = W_PROFIT_FACTOR
    _W_EXPECTANCY = W_EXPECTANCY
    _W_INV_DRAWDOWN = W_INV_DRAWDOWN
    _W_PROCESS = W_PROCESS
    _Z_CLIP = 3.0
    _PF_CAP = 10.0  # cap extreme profit factors for z-score stability
    _MIN_TRADES = 10
    _NEUTRAL_SCORE = 0.5
    _HISTORY_DAYS = 90

    def __init__(self, curated_dir: Path) -> None:
        self._curated_dir = curated_dir

    def compute_snapshot(
        self,
        bot_id: str,
        as_of_date: str,
        period_days: int = 30,
    ) -> GroundTruthSnapshot:
        """Compute a ground truth snapshot for a bot over a rolling window."""
        # Load 90-day history once — 30-day window is a subset
        history = self._load_dailies(bot_id, as_of_date, self._HISTORY_DAYS)
        # Filter to entries within the period_days calendar window
        period_cutoff = (
            datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=period_days)
        ).strftime("%Y-%m-%d")
        dailies = [d for d in history if d.get("date", "") > period_cutoff]

        if not dailies:
            return GroundTruthSnapshot(
                snapshot_date=as_of_date,
                bot_id=bot_id,
                period_days=period_days,
                composite_score=self._NEUTRAL_SCORE,
            )

        # Core metrics from daily summaries
        pnl_total = sum(d.get("net_pnl", 0.0) for d in dailies)
        trade_count = sum(d.get("total_trades", 0) for d in dailies)

        if trade_count < self._MIN_TRADES:
            return GroundTruthSnapshot(
                snapshot_date=as_of_date,
                bot_id=bot_id,
                period_days=period_days,
                pnl_total=pnl_total,
                trade_count=trade_count,
                composite_score=self._NEUTRAL_SCORE,
            )

        # Compute individual metrics
        win_rate = self._compute_win_rate(dailies)
        sharpe = self._compute_sharpe(dailies)
        calmar = self._compute_calmar(dailies, period_days)
        profit_factor = self._compute_profit_factor(dailies)
        max_dd = self._compute_max_drawdown(dailies)
        avg_pq = self._compute_avg_process_quality(dailies)
        expected_r = self._compute_expected_total_r(dailies, period_days)
        expectancy = self._compute_expectancy(dailies)

        composite = self._compute_composite(
            expected_r, calmar, profit_factor, expectancy, max_dd, avg_pq,
            history,
        )

        return GroundTruthSnapshot(
            snapshot_date=as_of_date,
            bot_id=bot_id,
            period_days=period_days,
            pnl_total=round(pnl_total, 2),
            sharpe_ratio_30d=round(sharpe, 4),
            win_rate=round(win_rate, 4),
            calmar_ratio_30d=round(calmar, 4),
            profit_factor=round(profit_factor, 4),
            max_drawdown_pct=round(max_dd, 4),
            avg_process_quality=round(avg_pq, 2),
            expected_total_r=round(expected_r, 4),
            expectancy=round(expectancy, 4),
            composite_score=round(composite, 4),
            trade_count=trade_count,
        )

    def compute_all_bots(
        self,
        bots: list[str],
        as_of_date: str,
    ) -> dict[str, GroundTruthSnapshot]:
        """Compute snapshots for all bots."""
        return {bot: self.compute_snapshot(bot, as_of_date) for bot in bots}

    def _load_dailies(
        self, bot_id: str, as_of_date: str, days: int,
    ) -> list[dict]:
        """Load daily summaries for the rolling window."""
        end = datetime.strptime(as_of_date, "%Y-%m-%d")
        dailies: list[dict] = []
        for d in range(days):
            date_str = (end - timedelta(days=d)).strftime("%Y-%m-%d")
            path = self._curated_dir / date_str / bot_id / "summary.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        data.setdefault("date", date_str)
                        dailies.append(data)
                except (json.JSONDecodeError, OSError):
                    pass
        return dailies

    @staticmethod
    def _compute_win_rate(dailies: list[dict]) -> float:
        total_wins = sum(d.get("winning_trades", 0) for d in dailies)
        total_trades = sum(d.get("total_trades", 0) for d in dailies)
        if total_trades == 0:
            return 0.0
        return total_wins / total_trades

    @staticmethod
    def _compute_sharpe(dailies: list[dict]) -> float:
        """Annualized Sharpe from daily returns."""
        returns = [d.get("net_pnl", 0.0) for d in dailies if d.get("total_trades", 0) > 0]
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = math.sqrt(var) if var > 0 else 0.0
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(252)

    @staticmethod
    def _compute_max_drawdown(dailies: list[dict]) -> float:
        """Max drawdown percentage from daily PnL series."""
        if not dailies:
            return 0.0
        # Sort by date ascending
        sorted_dailies = sorted(dailies, key=lambda d: d.get("date", ""))
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for d in sorted_dailies:
            equity += d.get("net_pnl", 0.0)
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def _compute_calmar(self, dailies: list[dict], period_days: int) -> float:
        """Annualized Calmar = (total_pnl * 365/period) / max_drawdown_pct."""
        total_pnl = sum(d.get("net_pnl", 0.0) for d in dailies)
        max_dd = self._compute_max_drawdown(dailies)
        if max_dd <= 0:
            return 0.0  # no drawdown = no Calmar (not inf)
        return (total_pnl * 365 / max(period_days, 1)) / (max_dd * 100)

    @staticmethod
    def _aggregate_win_loss(dailies: list[dict]) -> tuple[int, int, float, float]:
        """Aggregate win/loss counts and gross amounts from daily summaries.

        Returns (total_wins, total_losses, gross_wins, gross_losses).
        """
        total_wins = 0
        total_losses = 0
        gross_wins = 0.0
        gross_losses = 0.0
        for d in dailies:
            avg_win = d.get("avg_win", 0.0)
            avg_loss = abs(d.get("avg_loss", 0.0))
            wins = d.get("win_count", 0) or d.get("winning_trades", 0)
            losses = d.get("loss_count", 0) or d.get("losing_trades", 0)
            total_wins += wins
            total_losses += losses
            gross_wins += avg_win * wins
            gross_losses += avg_loss * losses
        return total_wins, total_losses, gross_wins, gross_losses

    def _compute_profit_factor(self, dailies: list[dict]) -> float:
        """Aggregate profit factor = gross_wins / gross_losses."""
        _, _, gross_wins, gross_losses = self._aggregate_win_loss(dailies)
        if gross_losses <= 0:
            return self._PF_CAP if gross_wins > 0 else 1.0
        return min(self._PF_CAP, gross_wins / gross_losses)

    @staticmethod
    def _compute_avg_process_quality(dailies: list[dict]) -> float:
        """Average process quality score across daily summaries."""
        scores = []
        for d in dailies:
            pq = d.get("avg_process_quality")
            if pq is None:
                pq = d.get("process_quality_avg")
            if isinstance(pq, (int, float)):
                scores.append(float(pq))
        if not scores:
            return 50.0  # neutral default
        return sum(scores) / len(scores)

    @staticmethod
    def _compute_expected_total_r(dailies: list[dict], period_days: int) -> float:
        """Annualized net PnL: pnl_total * 365 / period_days."""
        pnl_total = sum(d.get("net_pnl", 0.0) for d in dailies)
        return pnl_total * 365 / max(period_days, 1)

    def _compute_expectancy(self, dailies: list[dict]) -> float:
        """Aggregated win_rate × (avg_win / avg_loss) from curated daily data."""
        total_wins, total_losses, gross_wins, gross_losses = self._aggregate_win_loss(dailies)
        total_trades = total_wins + total_losses
        if total_trades == 0 or total_losses == 0 or gross_losses == 0:
            return 0.0
        win_rate = total_wins / total_trades
        agg_avg_win = gross_wins / total_wins if total_wins > 0 else 0.0
        agg_avg_loss = gross_losses / total_losses
        return win_rate * (agg_avg_win / agg_avg_loss)

    def _compute_composite(
        self,
        expected_r: float,
        calmar: float,
        profit_factor: float,
        expectancy: float,
        max_dd: float,
        avg_pq: float,
        history: list[dict],
    ) -> float:
        """Absolute helper composite score via z-score normalization, output [0, 1].

        Formula: 0.30*z_expected_r + 0.20*z_calmar + 0.15*z_pf + 0.15*z_expectancy
                 + 0.10*z_inv_dd + 0.10*z_process_quality
        Z-scores normalized against bot's own 90-day history, clipped to [-3, 3].
        See schemas/objective_weights.py for weight rationale and the intentional
        divergence between this z-score scale, ParameterSearcher's ratio scale,
        and the monthly immutable score profiles.
        """
        # Compute historical baselines for z-scoring
        hist_expected_rs = self._rolling_expected_rs(history)
        hist_calmars = self._rolling_calmars(history)
        hist_profit_factors = self._rolling_profit_factors(history)
        hist_expectancies = self._rolling_expectancies(history)
        hist_drawdowns = self._rolling_drawdowns(history)
        hist_pq = self._rolling_process_qualities(history)

        z_expected_r = self._z_score(expected_r, hist_expected_rs)
        z_calmar = self._z_score(calmar, hist_calmars)
        z_pf = self._z_score(profit_factor, hist_profit_factors)
        z_expectancy = self._z_score(expectancy, hist_expectancies)
        # Invert drawdown (lower is better)
        z_inv_dd = self._z_score(-max_dd, [-dd for dd in hist_drawdowns])
        z_pq = self._z_score(avg_pq, hist_pq)

        composite = (
            self._W_EXPECTED_R * z_expected_r
            + self._W_CALMAR * z_calmar
            + self._W_PROFIT_FACTOR * z_pf
            + self._W_EXPECTANCY * z_expectancy
            + self._W_INV_DRAWDOWN * z_inv_dd
            + self._W_PROCESS * z_pq
        )

        # Normalize to 0-1 range (z-scores clipped to [-3, 3] → raw in [-3, 3])
        # Map [-3, 3] → [0, 1]
        return max(0.0, min(1.0, (composite + self._Z_CLIP) / (2 * self._Z_CLIP)))

    def _z_score(self, value: float, history: list[float]) -> float:
        """Compute z-score clipped to [-3, 3]."""
        if len(history) < 2:
            return 0.0
        mean = sum(history) / len(history)
        var = sum((v - mean) ** 2 for v in history) / len(history)
        std = math.sqrt(var) if var > 0 else 0.0
        if std == 0:
            return 0.0
        z = (value - mean) / std
        return max(-self._Z_CLIP, min(self._Z_CLIP, z))

    def _rolling_calmars(self, dailies: list[dict]) -> list[float]:
        """Rolling 7-day Calmar ratios for z-score history."""
        sorted_dailies = sorted(dailies, key=lambda d: d.get("date", ""))
        if len(sorted_dailies) < 5:
            return [0.0]
        window = 7
        calmars: list[float] = []
        for i in range(len(sorted_dailies) - window + 1):
            w = sorted_dailies[i:i + window]
            calmar = self._compute_calmar(w, window)
            calmars.append(calmar)
        return calmars if calmars else [0.0]

    def _rolling_profit_factors(self, dailies: list[dict]) -> list[float]:
        """Per-day profit factors from individual daily summaries."""
        pfs: list[float] = []
        for d in dailies:
            total = d.get("total_trades", 0)
            if total > 0:
                pf = self._compute_profit_factor([d])
                pfs.append(pf)
        return pfs if pfs else [1.0]

    def _rolling_expected_rs(self, dailies: list[dict]) -> list[float]:
        """Rolling 7-day annualized net PnL for z-score baseline."""
        sorted_dailies = sorted(dailies, key=lambda d: d.get("date", ""))
        if len(sorted_dailies) < 5:
            return [0.0]
        window = 7
        ers: list[float] = []
        for i in range(len(sorted_dailies) - window + 1):
            w = sorted_dailies[i:i + window]
            er = self._compute_expected_total_r(w, window)
            ers.append(er)
        return ers if ers else [0.0]

    def _rolling_expectancies(self, dailies: list[dict]) -> list[float]:
        """Per-day expectancy values for z-score baseline."""
        exps: list[float] = []
        for d in dailies:
            total = d.get("total_trades", 0)
            if total > 0:
                exp = self._compute_expectancy([d])
                exps.append(exp)
        return exps if exps else [0.0]

    @staticmethod
    def _rolling_drawdowns(dailies: list[dict]) -> list[float]:
        """Daily drawdown values (max_drawdown_pct from each daily summary)."""
        dds: list[float] = []
        for d in dailies:
            dd = d.get("max_drawdown_pct", 0.0)
            if isinstance(dd, (int, float)):
                dds.append(float(dd))
        return dds if dds else [0.0]

    @staticmethod
    def _rolling_process_qualities(dailies: list[dict]) -> list[float]:
        """Process quality scores from individual daily summaries."""
        pqs: list[float] = []
        for d in dailies:
            pq = d.get("avg_process_quality") or d.get("process_quality_avg")
            if isinstance(pq, (int, float)):
                pqs.append(float(pq))
        return pqs if pqs else [50.0]

    def compute_portfolio_snapshot(
        self,
        bots: list[str],
        as_of_date: str,
        allocations: dict[str, float] | None = None,
        period_days: int = 30,
    ) -> dict:
        """Compute an allocation-weighted portfolio composite score.

        Formula: sum(bot_composite * bot_allocation_pct) for all bots.
        When allocations is None, uses equal weighting.

        Returns dict with portfolio_composite, per_bot_composites, and metadata.
        """
        if not bots:
            return {"portfolio_composite": 0.5, "per_bot": {}, "as_of_date": as_of_date}

        if allocations is None:
            weight = 1.0 / len(bots)
            allocations = {bot: weight for bot in bots}

        per_bot: dict[str, dict] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for bot in bots:
            snapshot = self.compute_snapshot(bot, as_of_date, period_days=period_days)
            alloc = allocations.get(bot, 0.0)
            per_bot[bot] = {
                "composite": snapshot.composite_score,
                "allocation": alloc,
                "pnl_total": snapshot.pnl_total,
                "trade_count": snapshot.trade_count,
            }
            weighted_sum += snapshot.composite_score * alloc
            total_weight += alloc

        portfolio_composite = weighted_sum / total_weight if total_weight > 0 else 0.5

        return {
            "portfolio_composite": round(portfolio_composite, 4),
            "per_bot": per_bot,
            "as_of_date": as_of_date,
            "period_days": period_days,
        }
