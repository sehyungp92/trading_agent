# skills/suggestion_validator.py
"""Suggestion validator — backtests parameter suggestions against historical data.

For parameter changes (target_param + proposed_value): replays 30 days of trades
with the proposed parameter value and compares outcomes to baseline.
For structural changes: records as not_testable with evidence summary.
"""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from trading_assistant.schemas.suggestion_validation import (
    SuggestionValidationResult,
    ValidationEvidence,
)

logger = logging.getLogger(__name__)


class SuggestionValidator:
    """Validates suggestions against historical trade data before recording."""

    def __init__(
        self,
        curated_dir: Path,
        lookback_days: int = 30,
    ) -> None:
        self._curated_dir = curated_dir
        self._lookback_days = lookback_days

    def validate(
        self,
        suggestion_id: str,
        bot_id: str,
        category: str,
        target_param: str | None = None,
        proposed_value: float | None = None,
        title: str = "",
        end_date: str = "",
    ) -> SuggestionValidationResult:
        """Validate a suggestion against historical data.

        For parameter suggestions (target_param + proposed_value): replays trades.
        For structural suggestions: returns not_testable.
        """
        result = SuggestionValidationResult(
            suggestion_id=suggestion_id,
            bot_id=bot_id,
            target_param=target_param or "",
            proposed_value=proposed_value,
        )

        if not target_param or proposed_value is None:
            result.evidence = ValidationEvidence(
                validated=False,
                method="not_testable",
                notes=f"Structural suggestion: '{title}' — no parameter to backtest",
            )
            return result

        # Load historical trades for replay
        trades = self._load_trades(bot_id, end_date)
        if not trades:
            result.evidence = ValidationEvidence(
                validated=False,
                method="not_testable",
                notes="No historical trade data available for backtesting",
            )
            return result

        # Compute baseline metrics
        baseline = self._compute_metrics(trades)

        # Replay with proposed parameter
        filtered_trades = self._replay_with_param(
            trades, target_param, proposed_value,
        )
        proposed = self._compute_metrics(filtered_trades)

        # Compute improvement
        baseline_pnl = baseline.get("pnl", 0)
        proposed_pnl = proposed.get("pnl", 0)
        improvement = 0.0
        if baseline_pnl != 0:
            improvement = ((proposed_pnl - baseline_pnl) / abs(baseline_pnl)) * 100

        # Regime breakdown
        regime_breakdown = self._compute_regime_breakdown(
            trades, filtered_trades, target_param, proposed_value,
        )

        # Degradation: > 5% worse (handle negative baselines correctly)
        if baseline_pnl >= 0:
            degradation = proposed_pnl < baseline_pnl * 0.95
        else:
            degradation = proposed_pnl < baseline_pnl * 1.05

        result.evidence = ValidationEvidence(
            validated=True,
            method="backtest_replay",
            baseline_metrics=baseline,
            proposed_metrics=proposed,
            improvement_pct=round(improvement, 2),
            sample_size=len(trades),
            regime_breakdown=regime_breakdown,
            notes=f"Baseline: {len(trades)} trades, Proposed: {len(filtered_trades)} trades after replay",
        )
        result.degradation_detected = degradation
        result.requires_review = degradation

        return result

    def _load_trades(self, bot_id: str, end_date: str) -> list[dict]:
        """Load trades from curated trades.jsonl over the lookback window."""
        from datetime import datetime, timedelta

        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = datetime.now()

        trades: list[dict] = []
        for d in range(self._lookback_days):
            date_str = (end - timedelta(days=d)).strftime("%Y-%m-%d")
            trades_path = self._curated_dir / date_str / bot_id / "trades.jsonl"
            if not trades_path.exists():
                continue
            try:
                for line in trades_path.read_text(encoding="utf-8").strip().splitlines():
                    if line.strip():
                        trades.append(json.loads(line))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load trades from %s: %s", trades_path, exc)
        return trades

    def _replay_with_param(
        self,
        trades: list[dict],
        target_param: str,
        proposed_value: float,
    ) -> list[dict]:
        """Replay trades filtering by the proposed parameter change.

        Simulates parameter effects by filtering trades based on common
        parameter types:
        - stop_loss / take_profit: filters out trades where the param would
          have changed the outcome
        - signal threshold: filters out trades below the proposed threshold
        - position_size_pct: scales PnL proportionally
        """
        result = []
        for trade in trades:
            modified = dict(trade)

            # Signal threshold parameters
            if "threshold" in target_param or "signal" in target_param:
                signal_strength = trade.get("signal_strength", 0)
                if signal_strength < proposed_value:
                    continue  # Would have been filtered
                result.append(modified)

            # Stop loss parameter
            elif "stop" in target_param:
                mae = abs(trade.get("mae_pct", trade.get("mae", 0)))
                # If MAE exceeded proposed stop, trade would have stopped out at a loss
                if mae > proposed_value:
                    pnl_key = "pnl" if "pnl" in trade else "net_pnl"
                    modified[pnl_key] = -proposed_value
                result.append(modified)

            # Position sizing
            elif "size" in target_param or "position" in target_param:
                current_size = trade.get("position_size_pct", 1.0)
                if current_size > 0:
                    scale = proposed_value / current_size
                    pnl_key = "pnl" if "pnl" in trade else "net_pnl"
                    modified[pnl_key] = trade.get(pnl_key, 0) * scale
                result.append(modified)

            # Filter threshold
            elif "filter" in target_param:
                filter_value = trade.get(target_param, trade.get("filter_score", 0))
                if filter_value < proposed_value:
                    continue
                result.append(modified)

            else:
                # Unknown parameter — can't simulate, include as-is
                result.append(modified)

        return result

    @staticmethod
    def _compute_metrics(trades: list[dict]) -> dict:
        """Compute aggregate metrics from trade list."""
        if not trades:
            return {"pnl": 0, "win_rate": 0, "sharpe": 0, "max_dd": 0, "trade_count": 0}

        pnls = []
        for t in trades:
            pnl = t.get("pnl") or t.get("net_pnl", 0)
            try:
                pnls.append(float(pnl))
            except (TypeError, ValueError):
                pnls.append(0)

        total_pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0

        # Sharpe approximation
        sharpe = 0.0
        if len(pnls) >= 2:
            mean_pnl = statistics.mean(pnls)
            stdev = statistics.stdev(pnls)
            if stdev > 0:
                sharpe = mean_pnl / stdev

        # Max drawdown
        peak = 0.0
        max_dd = 0.0
        cumulative = 0.0
        for p in pnls:
            cumulative += p
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        return {
            "pnl": round(total_pnl, 4),
            "win_rate": round(win_rate, 4),
            "sharpe": round(sharpe, 4),
            "max_dd": round(max_dd, 4),
            "trade_count": len(pnls),
        }

    def _compute_regime_breakdown(
        self,
        original_trades: list[dict],
        filtered_trades: list[dict],
        target_param: str = "",
        proposed_value: float = 0.0,
    ) -> dict:
        """Compute per-regime baseline vs proposed metrics."""
        # Group original trades by regime
        regime_groups: dict[str, list[dict]] = {}
        for t in original_trades:
            regime = t.get("regime", t.get("market_regime", "unknown"))
            regime_groups.setdefault(regime, []).append(t)

        # Group filtered trades by regime
        filtered_groups: dict[str, list[dict]] = {}
        for t in filtered_trades:
            regime = t.get("regime", t.get("market_regime", "unknown"))
            filtered_groups.setdefault(regime, []).append(t)

        breakdown: dict = {}
        for regime, orig_trades in regime_groups.items():
            filt_trades = filtered_groups.get(regime, [])
            breakdown[regime] = {
                "baseline": self._compute_metrics(orig_trades),
                "proposed": self._compute_metrics(filt_trades),
            }

        return breakdown
