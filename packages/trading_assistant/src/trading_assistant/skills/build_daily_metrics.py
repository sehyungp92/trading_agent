"""Data reduction pipeline — events -> per-bot curated daily files.

Claude sees reduced, classified, pre-scored data. This deterministic pipeline
does the heavy lifting of classification. Claude does interpretation and synthesis.

Output directory: data/curated/<date>/<bot_id>/
"""
from __future__ import annotations

import json
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from filelock import FileLock, Timeout
from pydantic import BaseModel

from trading_assistant.schemas.daily_metrics import (
    BotDailySummary,
    FilterAnalysis,
    NotableMissedRecord,
    PerStrategySummary,
    ProcessFailureRecord,
    RegimeAnalysis,
    RootCauseSummary,
    WinnerLoserRecord,
)
from trading_assistant.schemas.events import (
    HealthReportSnapshot,
    MissedOpportunityEvent,
    PipelineFunnelSnapshot,
    TradeEvent,
    normalize_strategy_id,
)
from trading_assistant.schemas.exit_efficiency import ExitEfficiencyStats
from trading_assistant.schemas.factor_attribution import FactorAttribution, FactorStats
from trading_assistant.skills.lineage_utils import build_lineage_summary

logger = logging.getLogger(__name__)


class DailyMetricsBuilder:
    """Builds curated daily metrics for a single bot on a single day."""

    def __init__(self, date: str, bot_id: str, bot_timezone: str = "UTC") -> None:
        self.date = date
        self.bot_id = bot_id
        self.bot_timezone = bot_timezone

    @staticmethod
    def _unwrap_event_payload(event: dict) -> dict | None:
        payload = event.get("payload", event)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _safe_float(value: object) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _first_float(self, payload: dict, *keys: str) -> float | None:
        for key in keys:
            value = self._safe_float(payload.get(key))
            if value is not None:
                return value
        return None

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, (str, int, float)):
            return [str(value)]
        if not isinstance(value, (list, tuple, set)):
            return []
        return [str(item) for item in value if isinstance(item, (str, int, float))]

    @staticmethod
    def _event_timestamp(event: dict, payload: dict) -> str:
        metadata = payload.get("event_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        for candidate in (
            payload.get("timestamp"),
            payload.get("exchange_timestamp"),
            metadata.get("exchange_timestamp"),
            metadata.get("local_timestamp"),
            event.get("timestamp"),
            event.get("exchange_timestamp"),
        ):
            if candidate not in (None, ""):
                return str(candidate)
        return ""

    @staticmethod
    def _normalize_order_status(value: object) -> str:
        raw = str(value or "UNKNOWN").strip()
        if not raw:
            return "UNKNOWN"
        normalized = raw.replace("-", "_").replace(" ", "_").upper()
        status_map = {
            "FILL": "FILLED",
            "FILLED": "FILLED",
            "EXECUTED": "FILLED",
            "DONE": "FILLED",
            "PARTIAL": "PARTIAL_FILL",
            "PARTIAL_FILL": "PARTIAL_FILL",
            "PARTIAL_FILLED": "PARTIAL_FILL",
            "PARTIALLY_FILLED": "PARTIAL_FILL",
            "CANCELED": "CANCELLED",
            "CANCELLED": "CANCELLED",
            "API_CANCELLED": "CANCELLED",
            "APICANCELLED": "CANCELLED",
        }
        return status_map.get(normalized, normalized)

    def build_summary(
        self,
        trades: list[TradeEvent],
        daily_snapshot: dict | None = None,
        missed: list[MissedOpportunityEvent] | None = None,
    ) -> BotDailySummary:
        """Aggregate trade list into daily summary stats."""
        if not trades:
            summary = BotDailySummary(date=self.date, bot_id=self.bot_id)
            self._merge_snapshot_into_summary(summary, daily_snapshot, trades=[], missed=missed or [])
            return summary

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_pnl = sum(t.pnl for t in trades)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
        avg_pq = sum(t.process_quality_score for t in trades) / len(trades)

        summary = BotDailySummary(
            date=self.date,
            bot_id=self.bot_id,
            total_trades=len(trades),
            win_count=len(wins),
            loss_count=len(losses),
            gross_pnl=gross_pnl,
            net_pnl=gross_pnl,  # fees not separately tracked yet
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_process_quality=avg_pq,
        )
        self._merge_snapshot_into_summary(summary, daily_snapshot, trades=trades, missed=missed or [])
        return summary

    @staticmethod
    def _count_missed_winners(missed: list[MissedOpportunityEvent]) -> int:
        winners = 0
        for event in missed:
            if event.would_have_hit_tp is True:
                winners += 1
            elif event.would_have_hit_tp is None and (event.outcome_24h or 0.0) > 0:
                winners += 1
        return winners

    def _merge_snapshot_into_summary(
        self,
        summary: BotDailySummary,
        daily_snapshot: dict | None,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
    ) -> None:
        lineage_events = [*trades, *missed]
        if lineage_events:
            summary.lineage_summary = build_lineage_summary(lineage_events)
            summary.lineage_gap = bool(summary.lineage_summary.get("lineage_gap"))
        if not daily_snapshot:
            if missed:
                summary.missed_count = len(missed)
                summary.missed_would_have_won = self._count_missed_winners(missed)
            return

        # Preserve trade-derived metrics when real trade events exist.
        if not trades:
            summary.total_trades = int(daily_snapshot.get("total_trades", summary.total_trades) or 0)
            summary.win_count = int(daily_snapshot.get("win_count", summary.win_count) or 0)
            summary.loss_count = int(daily_snapshot.get("loss_count", summary.loss_count) or 0)
            summary.gross_pnl = float(daily_snapshot.get("gross_pnl", summary.gross_pnl) or 0.0)
            summary.net_pnl = float(daily_snapshot.get("net_pnl", summary.net_pnl) or 0.0)
            summary.avg_win = float(daily_snapshot.get("avg_win", summary.avg_win) or 0.0)
            summary.avg_loss = float(daily_snapshot.get("avg_loss", summary.avg_loss) or 0.0)
            summary.avg_process_quality = float(
                daily_snapshot.get("avg_process_quality", summary.avg_process_quality) or 0.0
            )

        summary.max_drawdown_pct = float(
            daily_snapshot.get("max_drawdown_pct", summary.max_drawdown_pct) or 0.0
        )
        summary.sharpe_rolling_30d = float(
            daily_snapshot.get("sharpe_rolling_30d", summary.sharpe_rolling_30d) or 0.0
        )
        summary.sortino_rolling_30d = float(
            daily_snapshot.get("sortino_rolling_30d", summary.sortino_rolling_30d) or 0.0
        )
        summary.calmar_rolling_30d = float(
            daily_snapshot.get("calmar_rolling_30d", summary.calmar_rolling_30d) or 0.0
        )
        summary.exposure_pct = float(daily_snapshot.get("exposure_pct", summary.exposure_pct) or 0.0)
        summary.error_count = int(daily_snapshot.get("error_count", summary.error_count) or 0)
        summary.uptime_pct = float(daily_snapshot.get("uptime_pct", summary.uptime_pct) or 0.0)
        if not summary.lineage_summary and daily_snapshot.get("lineage_summary"):
            summary.lineage_summary = daily_snapshot.get("lineage_summary")
            summary.lineage_gap = bool(
                daily_snapshot.get("lineage_gap")
                or summary.lineage_summary.get("lineage_gap", False)
            )

        if missed:
            summary.missed_count = len(missed)
            summary.missed_would_have_won = self._count_missed_winners(missed)
        else:
            summary.missed_count = int(daily_snapshot.get("missed_count", summary.missed_count) or 0)
            summary.missed_would_have_won = int(
                daily_snapshot.get("missed_would_have_won", summary.missed_would_have_won) or 0
            )

    def top_winners(self, trades: list[TradeEvent], n: int = 5) -> list[WinnerLoserRecord]:
        """Top N winning trades by PnL, descending."""
        wins = sorted([t for t in trades if t.pnl > 0], key=lambda t: t.pnl, reverse=True)
        return [self._to_winner_loser(t) for t in wins[:n]]

    def top_losers(self, trades: list[TradeEvent], n: int = 5) -> list[WinnerLoserRecord]:
        """Top N losing trades by PnL, ascending (most negative first)."""
        losses = sorted([t for t in trades if t.pnl <= 0], key=lambda t: t.pnl)
        return [self._to_winner_loser(t) for t in losses[:n]]

    def process_failures(
        self, trades: list[TradeEvent], threshold: int = 60
    ) -> list[ProcessFailureRecord]:
        """Trades with process_quality_score below threshold."""
        return [
            ProcessFailureRecord(
                trade_id=t.trade_id,
                bot_id=t.bot_id,
                pair=t.pair,
                process_quality_score=t.process_quality_score,
                root_causes=t.root_causes,
                pnl=t.pnl,
                entry_signal=t.entry_signal,
                market_regime=t.market_regime,
            )
            for t in trades
            if t.process_quality_score < threshold
        ]

    def notable_missed(
        self,
        missed: list[MissedOpportunityEvent],
        trades: list[TradeEvent],
    ) -> list[NotableMissedRecord]:
        """Missed opportunities where outcome > 2x average win."""
        wins = [t for t in trades if t.pnl > 0]
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
        threshold = 2.0 * avg_win if avg_win > 0 else float("inf")

        return [
            NotableMissedRecord(
                bot_id=m.bot_id,
                pair=m.pair,
                signal=m.signal,
                blocked_by=m.blocked_by,
                hypothetical_entry=m.hypothetical_entry,
                outcome_24h=m.outcome_24h or 0.0,
                confidence=m.confidence,
                assumption_tags=m.assumption_tags,
            )
            for m in missed
            if (m.outcome_24h or 0.0) > threshold
        ]

    def regime_analysis(self, trades: list[TradeEvent]) -> RegimeAnalysis:
        """PnL breakdown by market regime."""
        regime_pnl: dict[str, float] = defaultdict(float)
        regime_count: dict[str, int] = defaultdict(int)
        regime_wins: dict[str, int] = defaultdict(int)

        for t in trades:
            regime = t.market_regime or "unknown"
            regime_pnl[regime] += t.pnl
            regime_count[regime] += 1
            if t.pnl > 0:
                regime_wins[regime] += 1

        regime_win_rate = {
            r: regime_wins.get(r, 0) / c for r, c in regime_count.items()
        }

        return RegimeAnalysis(
            bot_id=self.bot_id,
            date=self.date,
            regime_pnl=dict(regime_pnl),
            regime_trade_count=dict(regime_count),
            regime_win_rate=regime_win_rate,
        )

    def filter_analysis(self, missed: list[MissedOpportunityEvent]) -> FilterAnalysis:
        """Impact analysis for each filter."""
        block_counts: dict[str, int] = defaultdict(int)
        saved_pnl: dict[str, float] = defaultdict(float)
        missed_pnl: dict[str, float] = defaultdict(float)

        for m in missed:
            filt = m.blocked_by or "unknown"
            block_counts[filt] += 1
            outcome = m.outcome_24h or 0.0
            if outcome <= 0:
                saved_pnl[filt] += abs(outcome)
            else:
                missed_pnl[filt] += outcome

        return FilterAnalysis(
            bot_id=self.bot_id,
            date=self.date,
            filter_block_counts=dict(block_counts),
            filter_saved_pnl=dict(saved_pnl),
            filter_missed_pnl=dict(missed_pnl),
        )

    def root_cause_summary(self, trades: list[TradeEvent]) -> RootCauseSummary:
        """Distribution of root causes across all trades."""
        dist: dict[str, int] = defaultdict(int)
        for t in trades:
            for cause in t.root_causes:
                dist[cause] += 1

        return RootCauseSummary(
            bot_id=self.bot_id,
            date=self.date,
            distribution=dict(dist),
            total_trades=len(trades),
        )

    def hourly_performance(self, trades: list[TradeEvent]) -> dict:
        """Compute time-of-day performance buckets via HourlyAnalyzer."""
        from trading_assistant.skills.hourly_analyzer import HourlyAnalyzer
        analyzer = HourlyAnalyzer(bot_id=self.bot_id, date=self.date, bot_timezone=self.bot_timezone)
        return analyzer.compute(trades).model_dump(mode="json")

    def slippage_stats(self, trades: list[TradeEvent]) -> dict:
        """Compute per-symbol, per-hour slippage distributions via SlippageAnalyzer."""
        from trading_assistant.skills.slippage_analyzer import SlippageAnalyzer
        analyzer = SlippageAnalyzer(bot_id=self.bot_id, date=self.date, bot_timezone=self.bot_timezone)
        return analyzer.compute(trades).model_dump(mode="json")

    def factor_attribution(self, trades: list[TradeEvent]) -> FactorAttribution:
        """Per-factor performance attribution across daily trades."""
        factor_data: dict[str, dict] = {}

        for t in trades:
            if not t.signal_factors:
                continue
            for fd in t.signal_factors:
                name = fd["factor_name"]
                if name not in factor_data:
                    factor_data[name] = {
                        "trade_count": 0,
                        "win_count": 0,
                        "total_pnl": 0.0,
                        "contributions": [],
                    }
                bucket = factor_data[name]
                bucket["trade_count"] += 1
                if t.pnl > 0:
                    bucket["win_count"] += 1
                bucket["total_pnl"] += t.pnl
                bucket["contributions"].append(fd.get("contribution", 0.0))

        factors = sorted(
            [
                FactorStats(
                    factor_name=name,
                    trade_count=d["trade_count"],
                    win_count=d["win_count"],
                    total_pnl=d["total_pnl"],
                    avg_contribution=(
                        sum(d["contributions"]) / len(d["contributions"])
                        if d["contributions"]
                        else 0.0
                    ),
                )
                for name, d in factor_data.items()
            ],
            key=lambda f: f.factor_name,
        )

        return FactorAttribution(
            bot_id=self.bot_id,
            date=self.date,
            factors=factors,
        )

    def exit_efficiency(self, trades: list[TradeEvent]) -> dict:
        """Measure how well exits capture available price moves.

        For each trade with post-exit price data, compute:
        - continuation: how much price moved after exit
        - exit_efficiency: actual_capture / max_available_move
        - premature exit detection (left money on the table)
        """
        efficiencies: list[float] = []
        premature_count = 0
        reason_effs: dict[str, list[float]] = defaultdict(list)
        regime_effs: dict[str, list[float]] = defaultdict(list)
        move_1h_pcts: list[float] = []
        move_4h_pcts: list[float] = []
        premature_details: list[tuple[float, dict]] = []  # (continuation_amt, info)

        for t in trades:
            if t.post_exit_1h_price is None and t.post_exit_4h_price is None:
                continue

            # Collect move percentages (prefer direct field, fall back to price computation)
            if t.post_exit_1h_move_pct is not None:
                move_1h_pcts.append(t.post_exit_1h_move_pct)
            elif t.post_exit_1h_price is not None and t.exit_price:
                move_1h_pcts.append((t.post_exit_1h_price - t.exit_price) / t.exit_price * 100)

            if t.post_exit_4h_move_pct is not None:
                move_4h_pcts.append(t.post_exit_4h_move_pct)
            elif t.post_exit_4h_price is not None and t.exit_price:
                move_4h_pcts.append((t.post_exit_4h_price - t.exit_price) / t.exit_price * 100)

            # Compute continuation amounts
            cont_1h = (t.post_exit_1h_price - t.exit_price) if t.post_exit_1h_price is not None else None
            cont_4h = (t.post_exit_4h_price - t.exit_price) if t.post_exit_4h_price is not None else None

            # For SHORT trades, flip sign: price going down after exit = left money
            if t.side == "SHORT":
                if cont_1h is not None:
                    cont_1h = -cont_1h
                if cont_4h is not None:
                    cont_4h = -cont_4h

            # Detect premature exit: positive continuation means price kept going
            # in the trade's favor after exit
            best_continuation = max(
                c for c in [cont_1h, cont_4h] if c is not None
            )
            if best_continuation > 0:
                premature_count += 1
                move_pct = (best_continuation / t.exit_price * 100) if t.exit_price else 0.0
                premature_details.append((best_continuation, {
                    "trade_id": t.trade_id,
                    "pair": t.pair,
                    "exit_reason": t.exit_reason or "unknown",
                    "move_pct": round(move_pct, 4),
                }))

            # Compute exit efficiency = actual_capture / max_available_move
            # actual_capture is the trade's PnL (unsigned magnitude)
            actual_capture = abs(t.pnl)
            # max_available = actual capture + best continuation (if positive)
            max_available = actual_capture + max(best_continuation, 0.0)

            if max_available > 0:
                eff = actual_capture / max_available
            else:
                eff = 1.0  # no further move available, exit was optimal

            efficiencies.append(eff)

            reason = t.exit_reason or "unknown"
            regime = t.market_regime or "unknown"
            reason_effs[reason].append(eff)
            regime_effs[regime].append(eff)

        total_with_data = len(efficiencies)
        avg_eff = sum(efficiencies) / total_with_data if total_with_data else 0.0
        premature_pct = premature_count / total_with_data if total_with_data else 0.0

        by_exit_reason = {
            r: sum(vals) / len(vals) for r, vals in reason_effs.items()
        }
        by_regime = {
            r: sum(vals) / len(vals) for r, vals in regime_effs.items()
        }

        stats = ExitEfficiencyStats(
            bot_id=self.bot_id,
            date=self.date,
            avg_efficiency=avg_eff,
            premature_exit_pct=premature_pct,
            by_exit_reason=by_exit_reason,
            by_regime=by_regime,
            total_trades_with_data=total_with_data,
        )
        result = stats.model_dump(mode="json")

        # Enhanced output: post-exit move percentages
        if move_1h_pcts:
            result["avg_1h_move_pct"] = round(sum(move_1h_pcts) / len(move_1h_pcts), 4)
        if move_4h_pcts:
            result["avg_4h_move_pct"] = round(sum(move_4h_pcts) / len(move_4h_pcts), 4)

        # Worst premature exits (top 5 by continuation amount)
        premature_details.sort(key=lambda x: x[0], reverse=True)
        if premature_details:
            result["worst_premature_exits"] = [d for _, d in premature_details[:5]]

        return result

    def build_per_strategy_from_snapshot(
        self, daily_snapshot: dict,
    ) -> dict[str, PerStrategySummary]:
        """Map DailySnapshot.per_strategy_summary (loose dict) into typed objects."""
        result = {}
        raw = daily_snapshot.get("per_strategy_summary", {})

        for strategy_id, data in raw.items():
            if not isinstance(data, dict):
                continue

            # Normalize win_rate: bots may emit 0-100 or 0-1
            win_rate = data.get("win_rate", 0.0)
            if win_rate > 1.0:
                win_rate = win_rate / 100.0

            result[strategy_id] = PerStrategySummary(
                strategy_id=strategy_id,
                trades=data.get("trades", 0),
                win_count=data.get("win_count", 0),
                loss_count=data.get("loss_count", 0),
                gross_pnl=data.get("gross_pnl", 0.0),
                net_pnl=data.get("net_pnl", 0.0),
                win_rate=win_rate,
                avg_win=data.get("avg_win", 0.0),
                avg_loss=data.get("avg_loss", 0.0),
                best_trade_pnl=data.get("best_trade_pnl", 0.0),
                worst_trade_pnl=data.get("worst_trade_pnl", 0.0),
                avg_entry_slippage_bps=data.get("avg_entry_slippage_bps"),
            )

        return result

    def build_excursion_stats(self, trades: list[TradeEvent]) -> dict:
        """Build aggregate excursion statistics from trades with MFE/MAE data."""
        excursion_trades = [t for t in trades if t.mfe_pct is not None and t.mae_pct is not None]

        if not excursion_trades:
            return {"coverage": 0, "total_trades": len(trades), "stats": {}}

        mfe_pcts = [t.mfe_pct for t in excursion_trades]
        mae_pcts = [t.mae_pct for t in excursion_trades]
        efficiencies = [t.exit_efficiency for t in excursion_trades if t.exit_efficiency is not None]
        mfe_rs = [t.mfe_r for t in excursion_trades if t.mfe_r is not None]
        mae_rs = [t.mae_r for t in excursion_trades if t.mae_r is not None]

        def _stats(values: list[float]) -> dict:
            if not values:
                return {}
            s = sorted(values)
            n = len(s)
            return {
                "mean": sum(s) / n,
                "median": s[n // 2],
                "p25": s[max(0, n // 4)],
                "p75": s[max(0, 3 * n // 4)],
                "min": s[0],
                "max": s[-1],
            }

        # Winners vs losers breakdown
        winners = [t for t in excursion_trades if t.pnl > 0]
        losers = [t for t in excursion_trades if t.pnl <= 0]

        return {
            "coverage": len(excursion_trades),
            "total_trades": len(trades),
            "coverage_pct": len(excursion_trades) / len(trades) * 100 if trades else 0,
            "stats": {
                "mfe_pct": _stats(mfe_pcts),
                "mae_pct": _stats(mae_pcts),
                "exit_efficiency": _stats(efficiencies),
                "mfe_r": _stats(mfe_rs),
                "mae_r": _stats(mae_rs),
            },
            "winners": {
                "count": len(winners),
                "avg_mfe_pct": sum(t.mfe_pct for t in winners) / len(winners) if winners else 0,
                "avg_mae_pct": sum(t.mae_pct for t in winners) / len(winners) if winners else 0,
                "avg_exit_efficiency": (
                    sum(t.exit_efficiency for t in winners if t.exit_efficiency is not None)
                    / sum(1 for t in winners if t.exit_efficiency is not None)
                    if any(t.exit_efficiency is not None for t in winners) else 0
                ),
            },
            "losers": {
                "count": len(losers),
                "avg_mfe_pct": sum(t.mfe_pct for t in losers) / len(losers) if losers else 0,
                "avg_mae_pct": sum(t.mae_pct for t in losers) / len(losers) if losers else 0,
            },
        }

    def build_experiment_breakdown(self, snapshot_data: dict) -> dict:
        """Extract experiment breakdown from DailySnapshot data.

        Args:
            snapshot_data: Dict containing DailySnapshot fields including
                optional 'experiment_breakdown' dict.

        Returns:
            Dict with per-experiment, per-variant daily metrics.
        """
        breakdown = snapshot_data.get("experiment_breakdown", {})
        if not breakdown:
            return {}
        return breakdown

    def coordinator_impact(self, coordination_events: list[dict]) -> dict:
        """Aggregate coordinator events into a daily summary.

        Args:
            coordination_events: list of raw coordinator event dicts
                (parsed from coordination_YYYY-MM-DD.jsonl)

        Returns:
            dict with event counts, action breakdown, and rule summary.
        """
        if not coordination_events:
            return {
                "bot_id": self.bot_id,
                "date": self.date,
                "total_events": 0,
                "by_action": {},
                "by_rule": {},
                "symbols_affected": [],
                "events": [],
            }

        by_action: dict[str, int] = defaultdict(int)
        by_rule: dict[str, int] = defaultdict(int)
        symbols: set[str] = set()

        for evt in coordination_events:
            action = evt.get("action", "unknown")
            rule = evt.get("rule", "unknown")
            symbol = evt.get("symbol", "")
            by_action[action] += 1
            by_rule[rule] += 1
            if symbol:
                symbols.add(symbol)

        return {
            "bot_id": self.bot_id,
            "date": self.date,
            "total_events": len(coordination_events),
            "by_action": dict(by_action),
            "by_rule": dict(by_rule),
            "symbols_affected": sorted(symbols),
            "events": coordination_events,
        }

    def build_filter_decision_summary(self, filter_events: list[dict]) -> dict:
        """Summarize per-filter pass/block decisions from FilterDecisionEvent data."""
        from collections import defaultdict

        per_filter: dict[str, dict] = defaultdict(lambda: {
            "pass_count": 0,
            "block_count": 0,
            "margins": [],
            "near_misses": 0,
        })

        for evt in filter_events:
            payload = evt.get("payload", evt)
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    continue

            name = payload.get("filter_name", "unknown")
            passed = payload.get("passed", True)
            threshold = payload.get("threshold", 0)
            actual = payload.get("actual_value", 0)

            bucket = per_filter[name]
            if passed:
                bucket["pass_count"] += 1
            else:
                bucket["block_count"] += 1

            margin_pct = 0.0
            if threshold != 0:
                margin_pct = ((actual - threshold) / abs(threshold)) * 100
            bucket["margins"].append(margin_pct)

            if abs(margin_pct) < 10:
                bucket["near_misses"] += 1

        summary = {}
        for name, data in per_filter.items():
            margins = data["margins"]
            total = data["pass_count"] + data["block_count"]
            summary[name] = {
                "pass_count": data["pass_count"],
                "block_count": data["block_count"],
                "total": total,
                "block_rate": data["block_count"] / total if total > 0 else 0,
                "avg_margin_pct": sum(margins) / len(margins) if margins else 0,
                "near_miss_count": data["near_misses"],
                "near_miss_rate": data["near_misses"] / len(margins) if margins else 0,
            }

        return {
            "bot_id": self.bot_id,
            "date": self.date,
            "filters": summary,
            "total_evaluations": sum(d["pass_count"] + d["block_count"] for d in per_filter.values()),
        }

    def build_indicator_snapshot_summary(self, indicator_events: list[dict]) -> dict:
        """Summarize indicator values at signal evaluation points."""
        from collections import defaultdict

        indicator_values: dict[str, list[float]] = defaultdict(list)
        decision_counts: dict[str, int] = defaultdict(int)
        signal_strengths: list[float] = []

        for evt in indicator_events:
            payload = evt.get("payload", evt)
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    continue

            indicators = payload.get("indicators", {})
            for name, value in indicators.items():
                if isinstance(value, (int, float)):
                    indicator_values[name].append(value)

            decision = payload.get("decision", "skip")
            decision_counts[decision] += 1

            strength = payload.get("signal_strength", 0)
            if isinstance(strength, (int, float)):
                signal_strengths.append(strength)

        indicator_stats = {}
        for name, values in indicator_values.items():
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            indicator_stats[name] = {
                "count": n,
                "mean": sum(values) / n if n else 0,
                "min": sorted_vals[0] if n else 0,
                "max": sorted_vals[-1] if n else 0,
                "median": sorted_vals[n // 2] if n else 0,
            }

        return {
            "bot_id": self.bot_id,
            "date": self.date,
            "indicator_stats": indicator_stats,
            "decision_counts": dict(decision_counts),
            "total_snapshots": sum(decision_counts.values()),
            "avg_signal_strength": sum(signal_strengths) / len(signal_strengths) if signal_strengths else 0,
        }

    def build_orderbook_summary(self, orderbook_events: list[dict]) -> dict:
        """Summarize order book state at trading decision points."""
        from collections import defaultdict

        spreads: list[float] = []
        imbalances: list[float] = []
        by_context: dict[str, dict] = defaultdict(lambda: {
            "count": 0, "spreads": [], "imbalances": [],
        })

        for evt in orderbook_events:
            payload = evt.get("payload", evt)
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    continue

            spread = payload.get("spread_bps", 0)
            spreads.append(spread)

            bid_depth = payload.get("bid_depth_10bps", 0)
            ask_depth = payload.get("ask_depth_10bps", 0)
            imbalance = bid_depth / ask_depth if ask_depth > 0 else 0
            imbalances.append(imbalance)

            context = payload.get("trade_context", "signal")
            by_context[context]["count"] += 1
            by_context[context]["spreads"].append(spread)
            by_context[context]["imbalances"].append(imbalance)

        def _stats(values: list[float]) -> dict:
            if not values:
                return {"count": 0, "mean": 0, "min": 0, "max": 0, "median": 0}
            s = sorted(values)
            n = len(s)
            return {
                "count": n,
                "mean": sum(s) / n,
                "min": s[0],
                "max": s[-1],
                "median": s[n // 2],
            }

        context_summary = {}
        for ctx, data in by_context.items():
            context_summary[ctx] = {
                "count": data["count"],
                "spread_stats": _stats(data["spreads"]),
                "imbalance_stats": _stats(data["imbalances"]),
            }

        return {
            "bot_id": self.bot_id,
            "date": self.date,
            "spread_stats": _stats(spreads),
            "imbalance_stats": _stats(imbalances),
            "by_context": context_summary,
            "total_snapshots": len(spreads),
        }

    def build_parameter_change_log(self, param_events: list[dict]) -> dict:
        """Summarize parameter changes for the day."""
        changes: list[dict] = []
        by_strategy: dict[str, list[dict]] = {}

        for evt in param_events:
            payload = self._unwrap_event_payload(evt)
            if payload is None:
                continue

            change = {
                "strategy_id": payload.get("strategy_id", ""),
                "param_name": payload.get("param_name", ""),
                "old_value": payload.get("old_value"),
                "new_value": payload.get("new_value"),
                "reason": payload.get("reason", ""),
                "timestamp": self._event_timestamp(evt, payload),
            }
            changes.append(change)
            sid = change["strategy_id"] or "unknown"
            by_strategy.setdefault(sid, []).append(change)

        return {
            "bot_id": self.bot_id,
            "date": self.date,
            "total_changes": len(changes),
            "changes": changes,
            "by_strategy": {
                sid: {"count": len(items), "params": [c["param_name"] for c in items]}
                for sid, items in by_strategy.items()
            },
        }

    def build_order_lifecycle_summary(self, order_events: list[dict]) -> dict:
        """Summarize order lifecycle events for daily execution review."""
        status_counts: Counter[str] = Counter()
        reject_reasons: Counter[str] = Counter()
        rejects: list[dict] = []
        cancels: list[dict] = []
        partial_fills: list[dict] = []
        slippage_samples: list[float] = []
        latency_samples: list[float] = []
        order_ids: set[str] = set()
        processed_events = 0
        by_strategy: dict[str, dict] = {}

        for event in order_events:
            payload = self._unwrap_event_payload(event)
            if payload is None:
                continue

            processed_events += 1
            event_type = str(payload.get("event_type") or event.get("event_type") or "").lower()
            status = self._normalize_order_status(
                payload.get("status")
                or payload.get("order_status")
                or payload.get("execution_status")
            )
            if status == "UNKNOWN" and event_type in {"fill", "fills", "inferred_fill"}:
                status = "FILLED"
            status_counts[status] += 1

            order_id = str(
                payload.get("order_id")
                or payload.get("oms_order_id")
                or payload.get("kis_order_id")
                or payload.get("client_order_id")
                or payload.get("idempotency_key")
                or payload.get("fill_id")
                or ""
            )
            if order_id:
                order_ids.add(order_id)

            strategy = str(
                payload.get("assistant_strategy_id")
                or payload.get("strategy_type")
                or payload.get("strategy_id")
                or payload.get("strategy")
                or "unknown"
            )
            bucket = by_strategy.setdefault(strategy, {
                "event_count": 0,
                "status_counts": {},
                "reject_count": 0,
                "cancel_count": 0,
                "partial_fill_count": 0,
                "fill_count": 0,
            })
            bucket["event_count"] += 1
            bucket["status_counts"][status] = bucket["status_counts"].get(status, 0) + 1

            requested_qty = self._first_float(payload, "requested_qty", "qty", "quantity")
            filled_qty = self._first_float(payload, "filled_qty", "fill_qty", "qty_filled", "executed_qty")
            if filled_qty is None and status in {"FILLED", "PARTIAL_FILL"}:
                filled_qty = self._first_float(payload, "qty", "quantity")

            requested_price = self._first_float(payload, "requested_price", "limit_price")
            fill_price = self._first_float(payload, "fill_price", "avg_fill_price")
            if fill_price is None and status in {"FILLED", "PARTIAL_FILL"}:
                fill_price = self._first_float(payload, "price")

            slippage_bps = self._first_float(payload, "slippage_bps")
            if slippage_bps is None and requested_price not in (None, 0.0) and fill_price is not None:
                slippage_bps = abs(fill_price - requested_price) / requested_price * 10_000
            if status in {"FILLED", "PARTIAL_FILL"} and slippage_bps is not None:
                slippage_samples.append(slippage_bps)

            latency_ms = self._first_float(payload, "latency_ms")
            if latency_ms is not None:
                latency_samples.append(latency_ms)

            event_timestamp = self._event_timestamp(event, payload)
            evidence = {
                "event_type": event_type,
                "order_id": order_id,
                "pair": str(payload.get("pair") or payload.get("symbol") or payload.get("contract") or ""),
                "side": str(payload.get("side") or ""),
                "order_type": str(payload.get("order_type") or ""),
                "status": status,
                "requested_qty": requested_qty or 0.0,
                "filled_qty": filled_qty or 0.0,
                "requested_price": requested_price,
                "fill_price": fill_price,
                "slippage_bps": round(slippage_bps, 3) if slippage_bps is not None else None,
                "latency_ms": round(latency_ms, 3) if latency_ms is not None else None,
                "timestamp": event_timestamp,
                "related_trade_id": str(payload.get("related_trade_id") or payload.get("trade_id") or ""),
                "strategy_type": strategy,
            }

            if status == "REJECTED":
                reason = str(payload.get("reject_reason") or payload.get("rejection_reason") or "").strip()
                if reason:
                    reject_reasons[reason] += 1
                rejects.append({**evidence, "reject_reason": reason})
                bucket["reject_count"] += 1
            elif status == "CANCELLED":
                cancels.append(evidence)
                bucket["cancel_count"] += 1
            elif status == "PARTIAL_FILL":
                partial_fills.append(evidence)
                bucket["partial_fill_count"] += 1
            elif status == "FILLED":
                bucket["fill_count"] += 1

        avg_fill_slippage_bps = statistics.mean(slippage_samples) if slippage_samples else 0.0
        avg_latency_ms = statistics.mean(latency_samples) if latency_samples else 0.0
        normalized_by_strategy = {
            strategy: {
                **bucket,
                "status_counts": dict(sorted(bucket["status_counts"].items())),
            }
            for strategy, bucket in sorted(by_strategy.items())
        }

        return {
            "bot_id": self.bot_id,
            "date": self.date,
            "total_events": processed_events,
            "distinct_orders": len(order_ids),
            "status_counts": dict(sorted(status_counts.items())),
            "reject_count": len(rejects),
            "cancel_count": len(cancels),
            "partial_fill_count": len(partial_fills),
            "fill_count": status_counts.get("FILLED", 0),
            "avg_fill_slippage_bps": round(avg_fill_slippage_bps, 3),
            "max_fill_slippage_bps": round(max(slippage_samples), 3) if slippage_samples else 0.0,
            "avg_latency_ms": round(avg_latency_ms, 3),
            "top_reject_reasons": [
                {"reason": reason, "count": count}
                for reason, count in sorted(
                    reject_reasons.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:5]
            ],
            "by_strategy": normalized_by_strategy,
            "rejects": rejects,
            "cancels": cancels,
            "partial_fills": partial_fills,
        }

    def build_process_quality_summary(self, process_quality_events: list[dict]) -> dict:
        """Summarize standalone process-quality score records."""
        scores: list[float] = []
        classification_counts: Counter[str] = Counter()
        root_cause_counts: Counter[str] = Counter()
        worst_scores: list[dict] = []

        for event in process_quality_events:
            payload = self._unwrap_event_payload(event)
            if payload is None:
                continue

            score = self._safe_float(payload.get("process_quality_score"))
            if score is None:
                score = self._safe_float(payload.get("score"))
            if score is None:
                continue

            scores.append(score)
            classification = str(payload.get("classification") or "unknown").strip().lower() or "unknown"
            classification_counts[classification] += 1
            root_causes = list(dict.fromkeys(self._string_list(payload.get("root_causes"))))
            root_cause_counts.update(root_causes)
            worst_scores.append({
                "trade_id": str(payload.get("trade_id", "")),
                "process_quality_score": round(score, 3),
                "classification": classification,
                "root_causes": root_causes,
                "negative_factors": list(dict.fromkeys(self._string_list(payload.get("negative_factors")))),
                "evidence_refs": list(dict.fromkeys(self._string_list(payload.get("evidence_refs")))),
            })

        worst_scores.sort(key=lambda item: (item["process_quality_score"], item["trade_id"]))
        return {
            "bot_id": self.bot_id,
            "date": self.date,
            "total_scores": len(scores),
            "avg_score": round(statistics.mean(scores), 3) if scores else 0.0,
            "min_score": round(min(scores), 3) if scores else 0.0,
            "max_score": round(max(scores), 3) if scores else 0.0,
            "low_score_count": sum(1 for score in scores if score < 60),
            "classification_counts": dict(sorted(classification_counts.items())),
            "root_cause_counts": dict(sorted(root_cause_counts.items())),
            "worst_scores": worst_scores[:5],
        }

    def build_stop_adjustment_analysis(self, events: list[dict]) -> dict:
        """Aggregate stop adjustment events by type and trigger.

        Returns a summary dict with counts and avg/max tightening per
        adjustment_type, plus trigger frequency counts.
        """
        by_type: dict[str, list[float]] = defaultdict(list)
        trigger_counts: dict[str, int] = Counter()
        all_distances: list[float] = []

        for evt in events:
            payload = self._unwrap_event_payload(evt)
            if payload is None:
                continue
            old_stop = self._safe_float(payload.get("old_stop"))
            new_stop = self._safe_float(payload.get("new_stop"))
            adj_type = payload.get("adjustment_type", "unknown")
            trigger = payload.get("trigger", "unknown")

            if old_stop is not None and new_stop is not None and old_stop != 0:
                distance = abs(new_stop - old_stop) / abs(old_stop)
            elif old_stop is not None and new_stop is not None:
                distance = abs(new_stop - old_stop)
            else:
                distance = 0.0

            by_type[adj_type].append(distance)
            trigger_counts[trigger] += 1
            all_distances.append(distance)

        by_adjustment_type = {}
        for adj_type, distances in sorted(by_type.items()):
            by_adjustment_type[adj_type] = {
                "count": len(distances),
                "avg_tightening": round(statistics.mean(distances), 6) if distances else 0.0,
                "max_tightening": round(max(distances), 6) if distances else 0.0,
            }

        return {
            "bot_id": self.bot_id,
            "date": self.date,
            "total_adjustments": len(all_distances),
            "avg_tightening_distance": round(statistics.mean(all_distances), 6) if all_distances else 0.0,
            "by_adjustment_type": by_adjustment_type,
            "by_trigger": dict(sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)),
        }

    def build_execution_latency_analysis(self, trades: list[TradeEvent]) -> dict:
        """Analyze execution pipeline timing from enriched trade events.

        Computes per-stage latency stats and identifies bottlenecks.
        """
        # Normalize swing bot key names to canonical names
        _KEY_MAP = {
            "signal_generated_at": "signal_detected_at",
            "oms_received_at": "intent_created_at",
            "fill_confirmed_at": "fill_received_at",
        }
        _STAGES = [
            ("signal_detected_at", "intent_created_at", "signal_to_intent"),
            ("intent_created_at", "risk_checked_at", "intent_to_risk_check"),
            ("risk_checked_at", "order_submitted_at", "risk_check_to_order"),
            ("order_submitted_at", "fill_received_at", "order_to_fill"),
        ]

        stage_durations: dict[str, list[float]] = defaultdict(list)
        total_durations: list[float] = []
        regime_totals: dict[str, list[float]] = defaultdict(list)
        latency_slippage_pairs: list[tuple[float, float]] = []

        trades_with_data = 0
        for t in trades:
            if not t.execution_timestamps:
                continue
            ts = {_KEY_MAP.get(k, k): v for k, v in t.execution_timestamps.items()}
            trades_with_data += 1

            total_ms = 0.0
            for start_key, end_key, stage_name in _STAGES:
                start_val = ts.get(start_key)
                end_val = ts.get(end_key)
                if start_val is not None and end_val is not None:
                    try:
                        dur = float(end_val) - float(start_val)
                        stage_durations[stage_name].append(dur)
                        total_ms += dur
                    except (TypeError, ValueError):
                        pass

            if total_ms > 0:
                total_durations.append(total_ms)
                regime = t.market_regime or "unknown"
                regime_totals[regime].append(total_ms)
                if t.entry_slippage_bps:
                    latency_slippage_pairs.append((total_ms, t.entry_slippage_bps))

        if trades_with_data == 0:
            return {"coverage": 0, "total_with_data": 0}

        stages_summary: dict[str, dict] = {}
        bottleneck_stage = ""
        bottleneck_p95 = 0.0
        for stage_name in ["signal_to_intent", "intent_to_risk_check", "risk_check_to_order", "order_to_fill"]:
            vals = stage_durations.get(stage_name, [])
            if not vals:
                continue
            sorted_vals = sorted(vals)
            p95_idx = max(0, int(len(sorted_vals) * 0.95) - 1)
            p95 = sorted_vals[p95_idx]
            stages_summary[stage_name] = {
                "mean_ms": round(statistics.mean(vals), 2),
                "median_ms": round(statistics.median(vals), 2),
                "p95_ms": round(p95, 2),
            }
            if p95 > bottleneck_p95:
                bottleneck_p95 = p95
                bottleneck_stage = stage_name

        by_regime = {}
        for regime, vals in regime_totals.items():
            by_regime[regime] = {
                "trade_count": len(vals),
                "mean_total_ms": round(statistics.mean(vals), 2),
            }

        # Pearson correlation between latency and slippage
        latency_slippage_correlation: float | None = None
        if len(latency_slippage_pairs) >= 5:
            lats = [p[0] for p in latency_slippage_pairs]
            slips = [p[1] for p in latency_slippage_pairs]
            try:
                latency_slippage_correlation = round(self._pearson(lats, slips), 4)
            except Exception:
                pass

        return {
            "coverage": round(trades_with_data / max(len(trades), 1), 4),
            "total_with_data": trades_with_data,
            "stages": stages_summary,
            "bottleneck_stage": bottleneck_stage,
            "by_regime": by_regime,
            "latency_slippage_correlation": latency_slippage_correlation,
        }

    def build_sizing_analysis(self, trades: list[TradeEvent]) -> dict:
        """Analyze position sizing methodology effectiveness."""
        model_stats: dict[str, dict] = defaultdict(lambda: {
            "trades": [], "pnl_vals": [], "risk_pcts": [], "risk_effs": [],
        })

        trades_with_data = 0
        for t in trades:
            if not t.sizing_inputs:
                continue
            trades_with_data += 1
            model = t.sizing_inputs.get("sizing_model", "unknown")
            model_stats[model]["trades"].append(t)
            model_stats[model]["pnl_vals"].append(t.pnl)
            risk_pct = t.sizing_inputs.get("target_risk_pct")
            if risk_pct is not None:
                model_stats[model]["risk_pcts"].append(float(risk_pct))
            unit_risk = t.sizing_inputs.get("unit_risk_usd", 0)
            if unit_risk and float(unit_risk) != 0:
                model_stats[model]["risk_effs"].append(t.pnl / float(unit_risk))

        if trades_with_data == 0:
            return {"coverage": 0, "total_with_data": 0}

        by_model: dict[str, dict] = {}
        all_risk_effs: list[float] = []
        for model, data in sorted(model_stats.items()):
            tc = len(data["trades"])
            wins = sum(1 for t in data["trades"] if t.pnl > 0)
            by_model[model] = {
                "trade_count": tc,
                "win_rate": round(wins / tc, 4) if tc else 0.0,
                "avg_pnl": round(statistics.mean(data["pnl_vals"]), 4) if data["pnl_vals"] else 0.0,
                "avg_target_risk_pct": round(statistics.mean(data["risk_pcts"]), 4) if data["risk_pcts"] else None,
                "avg_risk_efficiency": round(statistics.mean(data["risk_effs"]), 4) if data["risk_effs"] else None,
            }
            all_risk_effs.extend(data["risk_effs"])

        return {
            "coverage": round(trades_with_data / max(len(trades), 1), 4),
            "total_with_data": trades_with_data,
            "by_sizing_model": by_model,
            "overall_risk_efficiency": round(statistics.mean(all_risk_effs), 4) if all_risk_effs else None,
        }

    def build_param_outcome_correlation(self, trades: list[TradeEvent]) -> dict:
        """Identify strategy parameters most correlated with trade outcomes."""
        trades_with_data = 0
        all_params: dict[str, list[tuple[float, float]]] = defaultdict(list)  # param_name -> [(value, pnl)]
        boolean_params: dict[str, dict[bool, list[float]]] = defaultdict(
            lambda: {True: [], False: []},
        )  # param_name -> {True: [pnl, ...], False: [pnl, ...]}

        for t in trades:
            if not t.strategy_params_at_entry:
                continue
            trades_with_data += 1
            for key, val in t.strategy_params_at_entry.items():
                # Check boolean BEFORE float since float(True)==1.0
                if isinstance(val, bool):
                    boolean_params[key][val].append(t.pnl)
                    continue
                try:
                    all_params[key].append((float(val), t.pnl))
                except (TypeError, ValueError):
                    pass  # skip non-numeric params

        if trades_with_data == 0:
            return {"coverage": 0, "total_with_data": 0, "param_count": 0}

        # Discard params with single unique value (no variation)
        varied_params = {
            k: v for k, v in all_params.items()
            if len(set(p[0] for p in v)) > 1 and len(v) >= 3
        }

        correlations: list[dict] = []

        # Boolean params: 2-group comparison instead of 3-bucket numeric split
        for param_name, state_pnls in sorted(boolean_params.items()):
            enabled_pnls = state_pnls[True]
            disabled_pnls = state_pnls[False]
            if len(enabled_pnls) < 2 or len(disabled_pnls) < 2:
                continue
            en_wins = sum(1 for p in enabled_pnls if p > 0)
            dis_wins = sum(1 for p in disabled_pnls if p > 0)
            en_wr = round(en_wins / len(enabled_pnls), 4)
            dis_wr = round(dis_wins / len(disabled_pnls), 4)
            spread = abs(en_wr - dis_wr)
            correlations.append({
                "param_name": param_name,
                "type": "boolean",
                "enabled_win_rate": en_wr,
                "disabled_win_rate": dis_wr,
                "enabled_avg_pnl": round(statistics.mean(enabled_pnls), 4),
                "disabled_avg_pnl": round(statistics.mean(disabled_pnls), 4),
                "pnl_delta": round(
                    statistics.mean(disabled_pnls) - statistics.mean(enabled_pnls), 4,
                ),
                "enabled_count": len(enabled_pnls),
                "disabled_count": len(disabled_pnls),
                "spread": round(spread, 4),
            })

        # Numeric params: 3-bucket split
        for param_name, pairs in varied_params.items():
            sorted_pairs = sorted(pairs, key=lambda x: x[0])
            n = len(sorted_pairs)
            third = max(1, n // 3)
            buckets = [
                sorted_pairs[:third],
                sorted_pairs[third:2 * third],
                sorted_pairs[2 * third:],
            ]
            bucket_stats = []
            for _i, bucket in enumerate(buckets):
                if not bucket:
                    continue
                vals = [p[0] for p in bucket]
                pnls = [p[1] for p in bucket]
                wins = sum(1 for p in pnls if p > 0)
                bucket_stats.append({
                    "range": f"{min(vals):.4g}-{max(vals):.4g}",
                    "trade_count": len(bucket),
                    "win_rate": round(wins / len(bucket), 4),
                    "avg_pnl": round(statistics.mean(pnls), 4),
                })

            if len(bucket_stats) >= 2:
                win_rates = [b["win_rate"] for b in bucket_stats]
                spread = max(win_rates) - min(win_rates)
                correlations.append({
                    "param_name": param_name,
                    "buckets": bucket_stats,
                    "spread": round(spread, 4),
                })

        correlations.sort(key=lambda x: x["spread"], reverse=True)

        return {
            "coverage": round(trades_with_data / max(len(trades), 1), 4),
            "total_with_data": trades_with_data,
            "param_count": len(varied_params) + len(boolean_params),
            "top_correlations": correlations[:10],
        }

    def build_portfolio_context_analysis(self, trades: list[TradeEvent]) -> dict:
        """Analyze portfolio state at entry — exposure levels and crowding effects."""
        trades_with_data = 0
        exposure_pnl: list[tuple[float, float, bool]] = []  # (exposure, pnl, is_crowded)

        for t in trades:
            if not t.portfolio_state_at_entry:
                continue
            trades_with_data += 1
            exposure = float(t.portfolio_state_at_entry.get("exposure", 0))
            correlated = t.portfolio_state_at_entry.get("correlated_positions", [])
            is_crowded = len(correlated) > 2 if isinstance(correlated, list) else False
            exposure_pnl.append((exposure, t.pnl, is_crowded))

        if trades_with_data == 0:
            return {"coverage": 0, "total_with_data": 0}

        # Bucket by exposure tertile
        sorted_by_exp = sorted(exposure_pnl, key=lambda x: x[0])
        n = len(sorted_by_exp)
        third = max(1, n // 3)
        level_names = ["low", "medium", "high"]
        bucket_slices = [sorted_by_exp[:third], sorted_by_exp[third:2*third], sorted_by_exp[2*third:]]

        by_exposure_level: dict[str, dict] = {}
        for name, bucket in zip(level_names, bucket_slices, strict=False):
            if not bucket:
                continue
            pnls = [b[1] for b in bucket]
            wins = sum(1 for p in pnls if p > 0)
            by_exposure_level[name] = {
                "trade_count": len(bucket),
                "win_rate": round(wins / len(bucket), 4),
                "avg_pnl": round(statistics.mean(pnls), 4),
            }

        crowded = [e for e in exposure_pnl if e[2]]
        uncrowded = [e for e in exposure_pnl if not e[2]]
        crowded_wr = round(sum(1 for c in crowded if c[1] > 0) / len(crowded), 4) if crowded else None
        uncrowded_wr = round(sum(1 for c in uncrowded if c[1] > 0) / len(uncrowded), 4) if uncrowded else None

        return {
            "coverage": round(trades_with_data / max(len(trades), 1), 4),
            "total_with_data": trades_with_data,
            "by_exposure_level": by_exposure_level,
            "crowding_count": len(crowded),
            "crowded_win_rate": crowded_wr,
            "uncrowded_win_rate": uncrowded_wr,
        }

    def build_market_condition_summary(self, trades: list[TradeEvent]) -> dict:
        """Summarize market conditions at entry, grouped by regime."""
        trades_with_data = 0
        regime_conditions: dict[str, list[dict]] = defaultdict(list)
        all_condition_pnl: dict[str, list[tuple[float, float]]] = defaultdict(list)

        for t in trades:
            conditions = dict(t.market_conditions_at_entry) if t.market_conditions_at_entry else {}
            # Supplement with inline fields if not already in the dict
            for field, key in [
                (t.atr_at_entry, "atr_at_entry"),
                (t.volume_24h, "volume_24h"),
                (t.spread_at_entry, "spread_at_entry"),
                (t.funding_rate, "funding_rate"),
            ]:
                if key not in conditions and field:
                    conditions[key] = field

            if not conditions:
                continue
            trades_with_data += 1
            regime = t.market_regime or "unknown"
            regime_conditions[regime].append(conditions)

            for key, val in conditions.items():
                try:
                    all_condition_pnl[key].append((float(val), t.pnl))
                except (TypeError, ValueError):
                    pass

        if trades_with_data == 0:
            return {"coverage": 0, "total_with_data": 0}

        by_regime: dict[str, dict] = {}
        for regime, cond_list in sorted(regime_conditions.items()):
            # Average all numeric keys across the regime
            all_keys: set[str] = set()
            for c in cond_list:
                all_keys.update(c.keys())
            avg_conditions: dict[str, float] = {}
            for key in sorted(all_keys):
                vals = []
                for c in cond_list:
                    try:
                        vals.append(float(c[key]))
                    except (KeyError, TypeError, ValueError):
                        pass
                if vals:
                    avg_conditions[key] = round(statistics.mean(vals), 6)
            by_regime[regime] = {
                "trade_count": len(cond_list),
                "avg_conditions": avg_conditions,
            }

        # Find conditions most correlated with PnL
        condition_pnl_correlations: list[dict] = []
        for key, pairs in all_condition_pnl.items():
            if len(pairs) < 5:
                continue
            vals = [p[0] for p in pairs]
            pnls = [p[1] for p in pairs]
            try:
                corr = self._pearson(vals, pnls)
                condition_pnl_correlations.append({
                    "key": key,
                    "correlation": round(corr, 4),
                    "sample_size": len(pairs),
                })
            except Exception:
                pass

        condition_pnl_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "coverage": round(trades_with_data / max(len(trades), 1), 4),
            "total_with_data": trades_with_data,
            "by_regime": by_regime,
            "condition_pnl_correlations": condition_pnl_correlations[:5],
        }

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False))
        std_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5
        if std_x == 0 or std_y == 0:
            return 0.0
        return cov / (std_x * std_y)

    def write_curated(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        base_dir: Path,
        coordination_events: list[dict] | None = None,
        daily_snapshot: dict | None = None,
        findings_dir: Path | None = None,
        filter_decision_events: list[dict] | None = None,
        indicator_snapshot_events: list[dict] | None = None,
        orderbook_context_events: list[dict] | None = None,
        parameter_change_events: list[dict] | None = None,
        order_events: list[dict] | None = None,
        process_quality_events: list[dict] | None = None,
        stop_adjustment_events: list[dict] | None = None,
        post_exit_events: list[dict] | None = None,
        funnel_snapshots: list[PipelineFunnelSnapshot] | None = None,
        health_snapshots: list[HealthReportSnapshot] | None = None,
        strategy_registry=None,
    ) -> Path:
        """Write all curated files to base_dir/<date>/<bot_id>/. Returns output dir.

        Acquires a per-(date, bot) filelock so concurrent runs serialize cleanly
        instead of racing each other's file writes.
        """
        output_dir = base_dir / self.date / self.bot_id
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        lock_path = output_dir.parent / f".{self.bot_id}.write.lock"
        try:
            with FileLock(str(lock_path), timeout=300):
                return self._write_curated_locked(
                    trades,
                    missed,
                    base_dir,
                    coordination_events=coordination_events,
                    daily_snapshot=daily_snapshot,
                    findings_dir=findings_dir,
                    filter_decision_events=filter_decision_events,
                    indicator_snapshot_events=indicator_snapshot_events,
                    orderbook_context_events=orderbook_context_events,
                    parameter_change_events=parameter_change_events,
                    order_events=order_events,
                    process_quality_events=process_quality_events,
                    stop_adjustment_events=stop_adjustment_events,
                    post_exit_events=post_exit_events,
                    funnel_snapshots=funnel_snapshots,
                    health_snapshots=health_snapshots,
                    strategy_registry=strategy_registry,
                )
        except Timeout:
            logger.error(
                "Timed out acquiring write lock for %s/%s after 300s",
                self.date, self.bot_id,
            )
            raise

    def _write_curated_locked(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        base_dir: Path,
        coordination_events: list[dict] | None = None,
        daily_snapshot: dict | None = None,
        findings_dir: Path | None = None,
        filter_decision_events: list[dict] | None = None,
        indicator_snapshot_events: list[dict] | None = None,
        orderbook_context_events: list[dict] | None = None,
        parameter_change_events: list[dict] | None = None,
        order_events: list[dict] | None = None,
        process_quality_events: list[dict] | None = None,
        stop_adjustment_events: list[dict] | None = None,
        post_exit_events: list[dict] | None = None,
        funnel_snapshots: list[PipelineFunnelSnapshot] | None = None,
        health_snapshots: list[HealthReportSnapshot] | None = None,
        strategy_registry=None,
    ) -> Path:
        """Inner write — caller must hold the per-(date, bot) write lock."""
        self._strategy_registry = strategy_registry
        output_dir = base_dir / self.date / self.bot_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Merge standalone post_exit events into trades BEFORE any computation
        # so exit_efficiency, trades.jsonl, etc. all benefit from backfilled data.
        if post_exit_events:
            trade_map = {t.trade_id: t for t in trades}
            for evt in post_exit_events:
                payload = self._unwrap_event_payload(evt)
                if not payload:
                    continue
                tid = payload.get("trade_id", "")
                trade = trade_map.get(tid)
                if not trade:
                    continue
                if trade.post_exit_1h_price is None:
                    val = self._safe_float(payload.get("post_exit_1h_price"))
                    if val is not None:
                        trade.post_exit_1h_price = val
                if trade.post_exit_4h_price is None:
                    val = self._safe_float(payload.get("post_exit_4h_price"))
                    if val is not None:
                        trade.post_exit_4h_price = val
                if trade.post_exit_1h_move_pct is None:
                    val = self._safe_float(payload.get("post_exit_1h_move_pct"))
                    if val is not None:
                        trade.post_exit_1h_move_pct = val
                if trade.post_exit_4h_move_pct is None:
                    val = self._safe_float(payload.get("post_exit_4h_move_pct"))
                    if val is not None:
                        trade.post_exit_4h_move_pct = val
                if not trade.post_exit_backfill_status:
                    trade.post_exit_backfill_status = payload.get("backfill_status", "complete")

        summary = self.build_summary(trades, daily_snapshot=daily_snapshot, missed=missed)

        # Enrich summary with per-strategy data from snapshot
        if daily_snapshot:
            per_strategy = self.build_per_strategy_from_snapshot(daily_snapshot)
            if per_strategy:
                summary.per_strategy_summary = per_strategy

        winners = self.top_winners(trades)
        losers = self.top_losers(trades)
        failures = self.process_failures(trades)
        notable = self.notable_missed(missed, trades)
        regime = self.regime_analysis(trades)
        filters = self.filter_analysis(missed)
        root_causes = self.root_cause_summary(trades)

        self._write_jsonl(output_dir / "trades.jsonl", trades)
        self._write_jsonl(output_dir / "missed.jsonl", missed)
        self._write_json(output_dir / "summary.json", summary.model_dump(mode="json"))
        self._write_json(output_dir / "winners.json", [w.model_dump(mode="json") for w in winners])
        self._write_json(output_dir / "losers.json", [loser.model_dump(mode="json") for loser in losers])
        self._write_json(output_dir / "process_failures.json", [f.model_dump(mode="json") for f in failures])
        self._write_json(output_dir / "notable_missed.json", [n.model_dump(mode="json") for n in notable])
        self._write_json(output_dir / "regime_analysis.json", regime.model_dump(mode="json"))
        self._write_json(output_dir / "filter_analysis.json", filters.model_dump(mode="json"))
        self._write_json(output_dir / "root_cause_summary.json", root_causes.model_dump(mode="json"))
        self._write_json(output_dir / "hourly_performance.json", self.hourly_performance(trades))
        self._write_json(output_dir / "slippage_stats.json", self.slippage_stats(trades))
        factor_attr = self.factor_attribution(trades)
        self._write_json(output_dir / "factor_attribution.json", factor_attr.model_dump(mode="json"))
        self._write_json(output_dir / "exit_efficiency.json", self.exit_efficiency(trades))

        # Record factor history for rolling analysis
        if findings_dir is not None:
            try:
                from trading_assistant.schemas.signal_factor_history import DailyFactorSnapshot, FactorDayStats
                from trading_assistant.skills.signal_factor_tracker import SignalFactorTracker

                snapshot = DailyFactorSnapshot(
                    date=self.date,
                    bot_id=self.bot_id,
                    factors=[
                        FactorDayStats(
                            factor_name=f.factor_name,
                            trade_count=f.trade_count,
                            win_rate=f.win_rate,
                            avg_pnl=f.avg_pnl,
                            total_pnl=f.total_pnl,
                            avg_contribution=f.avg_contribution,
                        )
                        for f in factor_attr.factors
                    ],
                )
                SignalFactorTracker(findings_dir).record_daily(snapshot)
            except Exception:
                pass  # Don't block pipeline on history recording failure

        # Write excursion stats if any trades carry MFE/MAE data
        excursion = self.build_excursion_stats(trades)
        if excursion["coverage"] > 0:
            self._write_json(output_dir / "excursion_stats.json", excursion)

        # Write regime bps mapping for empirical cost modeling.
        from trading_assistant.skills.slippage_analyzer import SlippageAnalyzer as _SlipAnalyzer
        _sa = _SlipAnalyzer(bot_id=self.bot_id, date=self.date)
        self._write_json(output_dir / "regime_bps.json", _sa.export_regime_bps(trades))

        # Write coordinator impact if events provided
        if coordination_events:
            self._write_json(
                output_dir / "coordinator_impact.json",
                self.coordinator_impact(coordination_events),
            )

        # Write overlay state summary if present in snapshot
        if daily_snapshot:
            overlay_summary = daily_snapshot.get("overlay_state_summary")
            if overlay_summary:
                self._write_json(output_dir / "overlay_state_summary.json", overlay_summary)

        # 1.4: Write experiment data if present in snapshot
        if daily_snapshot:
            experiment_data = self.build_experiment_breakdown(daily_snapshot)
            if experiment_data:
                self._write_json(output_dir / "experiment_data.json", experiment_data)

        # 1.5: Write signal health if any trades carry signal_evolution data
        if any(t.signal_evolution for t in trades):
            from trading_assistant.skills.signal_health_analyzer import SignalHealthAnalyzer
            analyzer = SignalHealthAnalyzer(bot_id=self.bot_id, date=self.date)
            report = analyzer.compute(trades)
            self._write_json(output_dir / "signal_health.json", report.model_dump(mode="json"))

        # 2.6: Write fill quality if any trades carry fill detail data
        if any(t.entry_fill_details or t.exit_fill_details for t in trades):
            from trading_assistant.skills.fill_quality_analyzer import FillQualityAnalyzer
            analyzer = FillQualityAnalyzer(bot_id=self.bot_id, date=self.date)
            report = analyzer.compute(trades)
            self._write_json(output_dir / "fill_quality.json", report.model_dump(mode="json"))

        # 2B: Write enriched event summaries if events provided
        if filter_decision_events:
            self._write_json(
                output_dir / "filter_decisions.json",
                self.build_filter_decision_summary(filter_decision_events),
            )

        if indicator_snapshot_events:
            self._write_json(
                output_dir / "indicator_snapshots.json",
                self.build_indicator_snapshot_summary(indicator_snapshot_events),
            )

        if orderbook_context_events:
            self._write_json(
                output_dir / "orderbook_stats.json",
                self.build_orderbook_summary(orderbook_context_events),
            )

        if parameter_change_events:
            self._write_json(
                output_dir / "parameter_changes.json",
                self.build_parameter_change_log(parameter_change_events),
            )

        if order_events:
            self._write_json(
                output_dir / "order_lifecycle.json",
                self.build_order_lifecycle_summary(order_events),
            )

        if process_quality_events:
            self._write_json(
                output_dir / "process_quality.json",
                self.build_process_quality_summary(process_quality_events),
            )

        # Write stop adjustment analysis if events provided
        if stop_adjustment_events:
            self._write_json(
                output_dir / "stop_adjustment_analysis.json",
                self.build_stop_adjustment_analysis(stop_adjustment_events),
            )

        # Write enriched instrumentation analyses
        exec_latency = self.build_execution_latency_analysis(trades)
        if exec_latency.get("coverage", 0) > 0:
            self._write_json(output_dir / "execution_latency.json", exec_latency)

        sizing = self.build_sizing_analysis(trades)
        if sizing.get("coverage", 0) > 0:
            self._write_json(output_dir / "sizing_analysis.json", sizing)

        param_corr = self.build_param_outcome_correlation(trades)
        if param_corr.get("coverage", 0) > 0:
            self._write_json(output_dir / "param_outcome_correlation.json", param_corr)

        portfolio_ctx = self.build_portfolio_context_analysis(trades)
        if portfolio_ctx.get("coverage", 0) > 0:
            self._write_json(output_dir / "portfolio_context.json", portfolio_ctx)

        market_cond = self.build_market_condition_summary(trades)
        if market_cond.get("coverage", 0) > 0:
            self._write_json(output_dir / "market_conditions.json", market_cond)

        # Engine-level decomposition (requires strategy_registry on self)
        if getattr(self, "_strategy_registry", None) is not None:
            try:
                from trading_assistant.skills.engine_decomposer import EngineDecomposer

                decomposer = EngineDecomposer(self._strategy_registry)
                decomposition = decomposer.decompose(trades, self.bot_id, period=self.date)
                if decomposition.engines:
                    self._write_json(
                        output_dir / "engine_decomposition.json",
                        decomposition.model_dump(mode="json"),
                    )
            except Exception:
                logger.warning(
                    "Engine decomposition failed for %s/%s — skipping",
                    self.bot_id, self.date, exc_info=True,
                )

        # Ablation flag analysis (works from strategy_params_at_entry, no registry needed)
        try:
            from trading_assistant.skills.ablation_analyzer import AblationAnalyzer

            ablation = AblationAnalyzer()
            analysis = ablation.analyze(trades, self.bot_id, period=self.date)
            if analysis.flags:
                self._write_json(
                    output_dir / "ablation_analysis.json",
                    analysis.model_dump(mode="json"),
                )
        except Exception:
            logger.warning(
                "Ablation analysis failed for %s/%s — skipping",
                self.bot_id, self.date, exc_info=True,
            )

        # Exit tier analysis (requires strategy_registry on self)
        if getattr(self, "_strategy_registry", None) is not None:
            try:
                from trading_assistant.skills.exit_tier_analyzer import ExitTierAnalyzer

                exit_analyzer = ExitTierAnalyzer(self._strategy_registry)
                exit_analysis = exit_analyzer.analyze(trades, self.bot_id, period=self.date)
                if exit_analysis and exit_analysis.get("tiers"):
                    self._write_json(
                        output_dir / "exit_tier_analysis.json",
                        exit_analysis,
                    )
            except Exception:
                logger.warning(
                    "Exit tier analysis failed for %s/%s — skipping",
                    self.bot_id, self.date, exc_info=True,
                )

        # Crypto perpetual curated files (conditional on crypto-specific fields)
        funding = self._build_funding_analysis(trades)
        if funding:
            self._write_json(output_dir / "funding_analysis.json", funding)

        grade = self._build_grade_analysis(trades)
        if grade:
            self._write_json(output_dir / "grade_analysis.json", grade)

        confluence = self._build_confluence_analysis(trades)
        if confluence:
            self._write_json(output_dir / "confluence_analysis.json", confluence)

        leverage = self._build_leverage_analysis(trades)
        if leverage:
            self._write_json(output_dir / "leverage_analysis.json", leverage)

        if funnel_snapshots:
            self._write_jsonl(output_dir / "funnel_snapshots.jsonl", funnel_snapshots)
            funnel_analysis = self._build_funnel_analysis(funnel_snapshots)
            if funnel_analysis:
                self._write_json(output_dir / "funnel_analysis.json", funnel_analysis)

        if health_snapshots:
            self._write_jsonl(output_dir / "health_snapshots.jsonl", health_snapshots)
            health_summary = self._build_health_summary(health_snapshots)
            if health_summary:
                self._write_json(output_dir / "health_summary.json", health_summary)

        # Write applied regime config from snapshot (per-bot, varies by family)
        if daily_snapshot:
            applied_config = daily_snapshot.get("applied_regime_config")
            if applied_config:
                self._write_json(output_dir / "applied_regime_config.json", applied_config)

        return output_dir

    # --- Crypto perpetual analysis builders ---

    def _build_funding_analysis(self, trades: list[TradeEvent]) -> dict | None:
        """Build funding cost analysis for crypto perpetual trades."""
        if not any(t.funding_paid != 0 for t in trades):
            return None
        total_funding = sum(t.funding_paid for t in trades)
        gross_pnl = sum(t.pnl for t in trades)
        per_direction: dict[str, dict] = {}
        for direction in ("LONG", "SHORT"):
            dir_trades = [t for t in trades if t.side == direction]
            if dir_trades:
                per_direction[direction] = {
                    "count": len(dir_trades),
                    "total_funding": sum(t.funding_paid for t in dir_trades),
                    "avg_funding": sum(t.funding_paid for t in dir_trades) / len(dir_trades),
                }
        per_symbol: dict[str, float] = {}
        for t in trades:
            sym = t.pair or "UNKNOWN"
            per_symbol[sym] = per_symbol.get(sym, 0.0) + t.funding_paid
        # Trades where funding cost exceeded trade pnl
        funding_losers = []
        for t in trades:
            if t.funding_paid > 0 and t.funding_paid > t.pnl:
                hold_hours = (t.exit_time - t.entry_time).total_seconds() / 3600
                funding_losers.append({
                    "trade_id": t.trade_id,
                    "pnl": t.pnl,
                    "funding_paid": t.funding_paid,
                    "hold_hours": round(hold_hours, 2),
                })
        total_hold_hours = sum(
            (t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades
        )
        funded_trades = [t for t in trades if t.funding_paid != 0]
        return {
            "coverage": len(funded_trades),
            "total_funding_paid": total_funding,
            "gross_pnl": gross_pnl,
            "funding_pct_of_gross": abs(total_funding / gross_pnl) if gross_pnl else 0.0,
            "per_direction": per_direction,
            "per_symbol": per_symbol,
            "funding_losers": funding_losers,
            "avg_funding_per_hour": total_funding / total_hold_hours if total_hold_hours > 0 else 0.0,
        }

    def _build_grade_analysis(self, trades: list[TradeEvent]) -> dict | None:
        """Build setup grade performance analysis for crypto trades."""
        if not any(t.setup_grade for t in trades):
            return None
        from collections import defaultdict
        grade_buckets: dict[str, list[TradeEvent]] = defaultdict(list)
        for t in trades:
            if t.setup_grade:
                grade_buckets[t.setup_grade].append(t)
        per_grade: dict[str, dict] = {}
        for grade, bucket in sorted(grade_buckets.items()):
            wins = [t for t in bucket if t.pnl > 0]
            avg_confluences = (
                statistics.mean(len(t.confluences) for t in bucket)
                if any(t.confluences for t in bucket) else 0.0
            )
            per_grade[grade] = {
                "count": len(bucket),
                "win_rate": len(wins) / len(bucket) if bucket else 0.0,
                "avg_pnl": statistics.mean(t.pnl for t in bucket),
                "avg_r": statistics.mean(
                    t.pnl_pct / (abs(t.mae_pct) if t.mae_pct is not None and t.mae_pct != 0 else abs(t.pnl_pct) or 1.0)
                    for t in bucket
                ),
                "avg_confluences": round(avg_confluences, 2),
            }
        # Grade expectancy gap (A - B if both exist)
        a_pnl = per_grade.get("A", {}).get("avg_pnl", 0.0)
        b_pnl = per_grade.get("B", {}).get("avg_pnl", 0.0)
        graded_trades = [t for t in trades if t.setup_grade]
        return {
            "coverage": len(graded_trades),
            "per_grade": per_grade,
            "grade_expectancy_gap": a_pnl - b_pnl,
        }

    def _build_confluence_analysis(self, trades: list[TradeEvent]) -> dict | None:
        """Build confluence factor analysis for crypto trades."""
        if not any(t.confluences for t in trades):
            return None
        from collections import defaultdict
        # By count
        count_buckets: dict[int, list[TradeEvent]] = defaultdict(list)
        for t in trades:
            count_buckets[len(t.confluences)].append(t)
        by_count: dict[str, dict] = {}
        for n, bucket in sorted(count_buckets.items()):
            wins = [t for t in bucket if t.pnl > 0]
            by_count[str(n)] = {
                "count": len(bucket),
                "win_rate": len(wins) / len(bucket) if bucket else 0.0,
                "avg_pnl": statistics.mean(t.pnl for t in bucket),
            }
        # By factor (present vs absent)
        all_factors: set[str] = set()
        for t in trades:
            all_factors.update(t.confluences)
        by_factor: dict[str, dict] = {}
        for factor in sorted(all_factors):
            present = [t for t in trades if factor in t.confluences]
            absent = [t for t in trades if factor not in t.confluences]
            present_wr = len([t for t in present if t.pnl > 0]) / len(present) if present else 0.0
            absent_wr = len([t for t in absent if t.pnl > 0]) / len(absent) if absent else 0.0
            by_factor[factor] = {
                "present_win_rate": round(present_wr, 4),
                "absent_win_rate": round(absent_wr, 4),
                "lift": round(present_wr - absent_wr, 4),
            }
        confluence_trades = [t for t in trades if t.confluences]
        return {"coverage": len(confluence_trades), "by_count": by_count, "by_factor": by_factor}

    def _build_leverage_analysis(self, trades: list[TradeEvent]) -> dict | None:
        """Build leverage utilization analysis for crypto trades."""
        leverages: list[float] = []
        for t in trades:
            lev = 1.0
            if t.sizing_inputs and isinstance(t.sizing_inputs, dict):
                lev = float(t.sizing_inputs.get("leverage", 1.0))
            leverages.append(lev)
        if not any(lev > 1.0 for lev in leverages):
            return None
        avg_leverage = statistics.mean(leverages)
        max_leverage = max(leverages)
        # Max configured leverage (estimate from data or use max seen * 1.2)
        max_configured = max_leverage  # conservative estimate
        # Per-grade leverage
        per_grade: dict[str, float] = {}
        from collections import defaultdict
        grade_levs: dict[str, list[float]] = defaultdict(list)
        for t, lev in zip(trades, leverages, strict=False):
            if t.setup_grade:
                grade_levs[t.setup_grade].append(lev)
        for grade, levs in grade_levs.items():
            per_grade[grade] = round(statistics.mean(levs), 2)
        # Near-liquidation count (mae_r > 0.8 * (1/leverage))
        near_liq = 0
        worst_mae_r = 0.0
        for t, lev in zip(trades, leverages, strict=False):
            if t.mae_r is not None and lev > 1.0:
                threshold = 0.8 * (1.0 / lev)
                if t.mae_r > threshold:
                    near_liq += 1
                if t.mae_r > worst_mae_r:
                    worst_mae_r = t.mae_r
        leveraged_trades = [lev for lev in leverages if lev > 1.0]
        return {
            "coverage": len(leveraged_trades),
            "avg_leverage": round(avg_leverage, 2),
            "max_leverage": round(max_leverage, 2),
            "leverage_utilization_pct": round(avg_leverage / max_configured, 4) if max_configured > 0 else 0.0,
            "per_grade": per_grade,
            "worst_mae_r": round(worst_mae_r, 4),
            "near_liquidation_count": near_liq,
        }

    def _build_funnel_analysis(
        self, snapshots: list[PipelineFunnelSnapshot],
    ) -> dict | None:
        """Summarize crypto strategy funnel conversion and drop-off stages."""
        if not snapshots:
            return None

        stages = [
            "bars_received",
            "indicators_ready",
            "setups_detected",
            "confirmations",
            "entries_attempted",
            "fills",
            "trades_closed",
        ]
        totals = {stage: 0 for stage in stages}
        per_strategy: dict[str, dict] = {}
        per_symbol: dict[str, dict] = {}
        assessments: Counter[str] = Counter()

        for snapshot in snapshots:
            strategy_id = normalize_strategy_id(self.bot_id, snapshot.strategy_id) or "unknown"
            assessment = snapshot.assessment or "unknown"
            assessments[assessment] += 1
            funnel = snapshot.funnel or self._flat_funnel_from_snapshot(snapshot)
            strategy_totals = per_strategy.setdefault(
                strategy_id, {stage: 0 for stage in stages},
            )
            for stage in stages:
                stage_values = funnel.get(stage, {})
                if isinstance(stage_values, dict):
                    stage_total = sum(
                        int(value or 0) for value in stage_values.values()
                        if isinstance(value, (int, float))
                    )
                    for symbol, value in stage_values.items():
                        if isinstance(value, (int, float)):
                            symbol_totals = per_symbol.setdefault(
                                str(symbol), {s: 0 for s in stages},
                            )
                            symbol_totals[stage] += int(value)
                elif isinstance(stage_values, (int, float)):
                    stage_total = int(stage_values)
                else:
                    stage_total = 0
                totals[stage] += stage_total
                strategy_totals[stage] += stage_total

        conversion_rates = self._funnel_conversion_rates(totals)
        top_dropoff = self._top_funnel_dropoff(totals, conversion_rates)

        return {
            "coverage": len(snapshots),
            "stage_totals": totals,
            "conversion_rates": conversion_rates,
            "top_dropoff": top_dropoff,
            "assessment_counts": dict(assessments),
            "per_strategy_breakdown": {
                sid: {
                    "stage_totals": values,
                    "conversion_rates": self._funnel_conversion_rates(values),
                }
                for sid, values in per_strategy.items()
            },
            "per_symbol_breakdown": {
                symbol: {
                    "stage_totals": values,
                    "conversion_rates": self._funnel_conversion_rates(values),
                }
                for symbol, values in per_symbol.items()
            },
        }

    @staticmethod
    def _flat_funnel_from_snapshot(snapshot: PipelineFunnelSnapshot) -> dict:
        signals = snapshot.signals_generated
        qualified = snapshot.setups_qualified or signals
        return {
            "bars_received": signals,
            "indicators_ready": signals,
            "setups_detected": qualified,
            "confirmations": snapshot.confirmations_passed,
            "entries_attempted": snapshot.entries_taken,
            "fills": snapshot.entries_taken,
            "trades_closed": snapshot.wins + snapshot.losses,
        }

    @staticmethod
    def _funnel_conversion_rates(totals: dict[str, int]) -> dict[str, float]:
        pairs = [
            ("bars_to_indicators", "bars_received", "indicators_ready"),
            ("indicators_to_setups", "indicators_ready", "setups_detected"),
            ("setups_to_confirmations", "setups_detected", "confirmations"),
            ("confirmations_to_entries", "confirmations", "entries_attempted"),
            ("entries_to_fills", "entries_attempted", "fills"),
            ("fills_to_closed", "fills", "trades_closed"),
        ]
        rates: dict[str, float] = {}
        for label, before, after in pairs:
            denominator = totals.get(before, 0)
            rates[label] = round(totals.get(after, 0) / denominator, 4) if denominator else 0.0
        return rates

    @staticmethod
    def _top_funnel_dropoff(
        totals: dict[str, int], conversion_rates: dict[str, float],
    ) -> dict:
        labels = {
            "bars_to_indicators": ("bars_received", "indicators_ready"),
            "indicators_to_setups": ("indicators_ready", "setups_detected"),
            "setups_to_confirmations": ("setups_detected", "confirmations"),
            "confirmations_to_entries": ("confirmations", "entries_attempted"),
            "entries_to_fills": ("entries_attempted", "fills"),
            "fills_to_closed": ("fills", "trades_closed"),
        }
        candidates: list[dict] = []
        for label, (before, after) in labels.items():
            before_count = totals.get(before, 0)
            after_count = totals.get(after, 0)
            if before_count <= 0:
                continue
            candidates.append({
                "stage": label,
                "from": before,
                "to": after,
                "lost": max(before_count - after_count, 0),
                "conversion_rate": conversion_rates.get(label, 0.0),
            })
        if not candidates:
            return {}
        return min(candidates, key=lambda item: (item["conversion_rate"], -item["lost"]))

    def _build_health_summary(
        self, snapshots: list[HealthReportSnapshot],
    ) -> dict | None:
        """Summarize process-health telemetry for prompt gating."""
        if not snapshots:
            return None

        latest = snapshots[-1]
        reports = [snapshot.report or self._flat_health_report(snapshot) for snapshot in snapshots]
        alerts: list[dict] = []
        max_last_bar_age = 0.0
        max_queue_depth = 0
        max_funding_drift = 0.0
        websocket_disconnects = 0
        error_count = 0

        for snapshot, report in zip(snapshots, reports, strict=False):
            for alert in report.get("alerts", []) if isinstance(report, dict) else []:
                if isinstance(alert, dict):
                    alerts.append(alert)
            data_flow = report.get("data_flow", {}) if isinstance(report, dict) else {}
            if isinstance(data_flow, dict):
                for item in data_flow.values():
                    if isinstance(item, dict):
                        age = item.get("last_bar_age_sec", 0)
                        if isinstance(age, (int, float)):
                            max_last_bar_age = max(max_last_bar_age, float(age))
            system = report.get("system", {}) if isinstance(report, dict) else {}
            if isinstance(system, dict):
                error_count = max(error_count, int(system.get("total_errors", 0) or 0))
            max_queue_depth = max(max_queue_depth, int(snapshot.queue_depth or 0))
            websocket_disconnects += int(snapshot.websocket_disconnects_24h or 0)
            for value in snapshot.funding_drift_per_symbol.values():
                if isinstance(value, (int, float)):
                    max_funding_drift = max(max_funding_drift, abs(float(value)))

        severity_rank = {"healthy": 0, "warning": 1, "degraded": 2, "error": 2, "critical": 3}
        worst_alert_rank = 0
        for alert in alerts:
            sev = str(alert.get("severity", "")).lower()
            worst_alert_rank = max(worst_alert_rank, severity_rank.get(sev, 0))
        latest_report = latest.report or {}
        latest_assessment = (
            latest.severity
            or latest_report.get("assessment", "")
            or "unknown"
        )

        return {
            "coverage": len(snapshots),
            "latest_timestamp": latest.timestamp,
            "latest_assessment": latest_assessment,
            "high_severity_alert_count": sum(
                1 for alert in alerts
                if str(alert.get("severity", "")).lower() in {"error", "critical", "high"}
            ),
            "worst_alert_rank": worst_alert_rank,
            "max_last_bar_age_sec": round(max_last_bar_age, 2),
            "max_queue_depth": max_queue_depth,
            "websocket_disconnects_24h": websocket_disconnects,
            "error_count_24h": max(error_count, max(int(s.error_count_24h or 0) for s in snapshots)),
            "max_funding_drift": round(max_funding_drift, 6),
            "alerts": alerts[:20],
        }

    @staticmethod
    def _flat_health_report(snapshot: HealthReportSnapshot) -> dict:
        return {
            "assessment": snapshot.severity,
            "system": {"total_errors": snapshot.error_count_24h},
            "alerts": [{
                "severity": snapshot.severity,
                "name": "health_report",
                "message": snapshot.notes,
            }] if snapshot.severity else [],
        }

    def _atomic_write_text(self, path: Path, content: str) -> None:
        """Write content to path atomically via temp file + os.replace.

        Prevents readers from seeing a half-written file if the process is
        interrupted, and prevents one of two overlapping runs for the same
        bot/date from leaving a partially-written file. ``os.replace`` is
        atomic within a directory on both POSIX and Windows.
        """
        import os as _os
        import tempfile
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with _os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            _os.replace(tmp_name, path)
        except Exception:
            try:
                _os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _write_json(self, path: Path, data: dict | list) -> None:
        self._atomic_write_text(path, json.dumps(data, indent=2, default=str))

    def _write_jsonl(self, path: Path, records: list[BaseModel]) -> None:
        if not records:
            self._atomic_write_text(path, "")
            return
        self._atomic_write_text(
            path,
            "\n".join(record.model_dump_json() for record in records) + "\n",
        )

    def _to_winner_loser(self, t: TradeEvent) -> WinnerLoserRecord:
        return WinnerLoserRecord(
            trade_id=t.trade_id,
            bot_id=t.bot_id,
            pair=t.pair,
            side=t.side,
            pnl=t.pnl,
            pnl_pct=t.pnl_pct,
            entry_signal=t.entry_signal,
            exit_reason=t.exit_reason,
            market_regime=t.market_regime,
            process_quality_score=t.process_quality_score,
            root_causes=t.root_causes,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            atr_at_entry=t.atr_at_entry,
        )


# ── Standalone portfolio-level helpers (Phase 0) ────────────────────


_PORTFOLIO_RULE_EVENT_TYPES = {"portfolio_rule_check", "portfolio_rule", "risk_decision", "risk_decisions"}


def _unwrap_raw_event_payload(event: dict) -> dict | None:
    payload = event.get("payload", event)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return None
    return payload if isinstance(payload, dict) else None


def _portfolio_rule_event_type(event: dict, payload: dict) -> str:
    for source in (event, payload):
        value = source.get("event_type") or source.get("type")
        if value:
            return str(value)
    return ""


def _portfolio_rule_name(payload: dict) -> str:
    for key in ("rule_name", "rule", "blocking_rule"):
        value = payload.get(key)
        if value:
            return str(value)

    for item in payload.get("rule_evaluations") or payload.get("evaluations") or []:
        if not isinstance(item, dict):
            continue
        if item.get("passed") is False or item.get("approved") is False:
            return str(item.get("rule") or item.get("rule_name") or "unknown")
    return "unknown"


def _portfolio_rule_result(payload: dict) -> str:
    result = str(payload.get("result") or "").lower()
    if result:
        return result

    action = str(payload.get("action") or "").lower()
    if action:
        return action

    decision = str(payload.get("decision") or "").lower()
    if decision:
        return decision

    if payload.get("allowed") is False or payload.get("approved") is False:
        return "blocked"
    return "pass"


def build_portfolio_rules_summary(raw_events: list[dict]) -> dict:
    """Scan raw events for portfolio rule entries and summarize (0B).

    Args:
        raw_events: list of raw event dicts (from JSONL ingest).  Each
            ``portfolio_rule_check``/``portfolio_rule``/``risk_decision`` event is expected to have
            a rule or risk decision ID, a pass/deny result, and optional denial details.

    Returns:
        Dict with rule evaluation counts, block counts, and per-rule breakdown.
    """
    rule_checks: list[tuple[dict, dict]] = []
    for event in raw_events:
        payload = _unwrap_raw_event_payload(event)
        if payload is None:
            continue
        if _portfolio_rule_event_type(event, payload) in _PORTFOLIO_RULE_EVENT_TYPES:
            rule_checks.append((event, payload))

    if not rule_checks:
        return {
            "total_evaluations": 0,
            "total_blocks": 0,
            "by_rule": {},
            "blocked_symbols": [],
            "records": [],
        }

    by_rule: dict[str, dict] = defaultdict(lambda: {
        "evaluations": 0,
        "blocks": 0,
        "block_reasons": [],
    })
    blocked_symbols: set[str] = set()
    records: list[dict] = []

    for event, payload in rule_checks:
        rule_name = _portfolio_rule_name(payload)
        result = _portfolio_rule_result(payload)
        details = payload.get("details", {})
        if not isinstance(details, dict):
            details = {}
        reason = (
            details.get("reason")
            or payload.get("reason")
            or payload.get("denial_reason")
            or payload.get("deny_reason")
            or payload.get("block_reason")
            or payload.get("blocking_rule")
            or ""
        )
        symbol = details.get("blocked_symbol") or payload.get("blocked_symbol") or payload.get("symbol") or ""
        records.append({
            "event_type": _portfolio_rule_event_type(event, payload),
            "strategy_id": payload.get("strategy_id") or event.get("strategy_id") or "",
            "rule_id": payload.get("rule_id") or payload.get("portfolio_rule_id") or "",
            "portfolio_rule_id": payload.get("portfolio_rule_id") or payload.get("rule_id") or "",
            "risk_decision_id": payload.get("risk_decision_id") or event.get("risk_decision_id") or "",
            "result": result,
            "deny_reason": reason,
            "symbol": symbol,
        })

        bucket = by_rule[rule_name]
        bucket["evaluations"] += 1

        if result in ("fail", "block", "blocked", "reject", "rejected", "deny", "denied"):
            bucket["blocks"] += 1
            if reason:
                bucket["block_reasons"].append(str(reason))
            if symbol:
                blocked_symbols.add(str(symbol))

    total_evals = sum(v["evaluations"] for v in by_rule.values())
    total_blocks = sum(v["blocks"] for v in by_rule.values())

    return {
        "total_evaluations": total_evals,
        "total_blocks": total_blocks,
        "by_rule": {
            name: {
                "evaluations": data["evaluations"],
                "blocks": data["blocks"],
                "block_rate": data["blocks"] / data["evaluations"] if data["evaluations"] > 0 else 0.0,
                "block_reasons": data["block_reasons"],
            }
            for name, data in by_rule.items()
        },
        "blocked_symbols": sorted(blocked_symbols),
        "records": records,
    }


def build_family_snapshots(
    bot_summaries: list[BotDailySummary],
    strategy_registry,
) -> list[dict]:
    """Aggregate bot summaries by strategy family (0C).

    Args:
        bot_summaries: Per-bot daily summaries for the day.
        strategy_registry: Object with a ``get_family(strategy_id) -> str``
            method (or a dict mapping strategy_id to family name).

    Returns:
        List of FamilyDailySnapshot dicts, one per family.
    """
    from trading_assistant.schemas.portfolio_metrics import FamilyDailySnapshot

    # Determine the family lookup function
    if hasattr(strategy_registry, "get_family"):
        get_family = strategy_registry.get_family
    elif isinstance(strategy_registry, dict):
        def get_family(sid):
            return strategy_registry.get(sid, "default")
    else:
        def get_family(sid):
            return "default"

    # Group per-strategy data by family
    family_data: dict[str, dict] = defaultdict(lambda: {
        "strategy_ids": set(),
        "total_net_pnl": 0.0,
        "total_fees": 0.0,
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "max_drawdown_pct": 0.0,
        "exposure_pcts": [],
        "date": "",
    })

    for summary in bot_summaries:
        date = summary.date

        # If per_strategy_summary is available, use it for family grouping
        if summary.per_strategy_summary:
            for sid, strat in summary.per_strategy_summary.items():
                family = get_family(sid)
                bucket = family_data[family]
                bucket["date"] = date
                bucket["strategy_ids"].add(sid)
                bucket["total_net_pnl"] += strat.net_pnl
                bucket["trade_count"] += strat.trades
                bucket["win_count"] += strat.win_count
                bucket["loss_count"] += strat.loss_count
        else:
            # Fallback: treat whole bot as a single strategy
            family = get_family(summary.bot_id)
            bucket = family_data[family]
            bucket["date"] = date
            bucket["strategy_ids"].add(summary.bot_id)
            bucket["total_net_pnl"] += summary.net_pnl
            bucket["trade_count"] += summary.total_trades
            bucket["win_count"] += summary.win_count
            bucket["loss_count"] += summary.loss_count
            bucket["max_drawdown_pct"] = max(
                bucket["max_drawdown_pct"], summary.max_drawdown_pct
            )
            if summary.exposure_pct > 0:
                bucket["exposure_pcts"].append(summary.exposure_pct)

    snapshots: list[dict] = []
    for family, data in sorted(family_data.items()):
        avg_exp = (
            sum(data["exposure_pcts"]) / len(data["exposure_pcts"])
            if data["exposure_pcts"]
            else 0.0
        )
        snap = FamilyDailySnapshot(
            family=family,
            date=data["date"],
            strategy_ids=sorted(data["strategy_ids"]),
            total_net_pnl=data["total_net_pnl"],
            total_fees=data["total_fees"],
            trade_count=data["trade_count"],
            win_count=data["win_count"],
            loss_count=data["loss_count"],
            max_drawdown_pct=data["max_drawdown_pct"],
            avg_exposure_pct=avg_exp,
            active_strategies=len(data["strategy_ids"]),
        )
        snapshots.append(snap.model_dump(mode="json"))

    return snapshots


def build_concurrent_position_analysis(trade_events: list[dict]) -> dict:
    """Extract correlated_pairs_detail from trade event payloads (0D).

    Scans trade events for ``correlated_pairs_detail`` fields (emitted by
    bots that track concurrent open positions) and aggregates them.

    Args:
        trade_events: list of raw trade event dicts.

    Returns:
        Dict with pair co-occurrence counts, total concurrent windows,
        and per-pair summary.
    """
    pair_occurrences: dict[str, int] = defaultdict(int)
    pair_pnl: dict[str, float] = defaultdict(float)
    total_windows = 0

    for evt in trade_events:
        payload = evt.get("payload", evt)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue

        details = payload.get("correlated_pairs_detail")
        if not details:
            continue

        if isinstance(details, list):
            for entry in details:
                pair_key = entry.get("pair", "")
                if not pair_key:
                    # Build key from symbols
                    symbols = sorted([
                        entry.get("symbol_a", ""),
                        entry.get("symbol_b", ""),
                    ])
                    pair_key = f"{symbols[0]}_{symbols[1]}"
                pair_occurrences[pair_key] += 1
                pair_pnl[pair_key] += entry.get("combined_pnl", 0.0)
                total_windows += 1
        elif isinstance(details, dict):
            for pair_key, info in details.items():
                count = info.get("count", 1) if isinstance(info, dict) else 1
                pnl = info.get("combined_pnl", 0.0) if isinstance(info, dict) else 0.0
                pair_occurrences[pair_key] += count
                pair_pnl[pair_key] += pnl
                total_windows += count

    per_pair = {
        pair: {
            "occurrences": pair_occurrences[pair],
            "total_combined_pnl": round(pair_pnl[pair], 4),
            "avg_combined_pnl": round(pair_pnl[pair] / pair_occurrences[pair], 4) if pair_occurrences[pair] > 0 else 0.0,
        }
        for pair in sorted(pair_occurrences.keys())
    }

    return {
        "total_concurrent_windows": total_windows,
        "unique_pairs": len(pair_occurrences),
        "per_pair": per_pair,
    }


def build_sector_exposure(trade_events: list[dict]) -> dict:
    """Aggregate sector/industry exposure from stock trade events (0E).

    Args:
        trade_events: list of raw trade event dicts.

    Returns:
        Dict with per-sector exposure breakdown and sector count.
    """
    sector_data: dict[str, dict] = {}

    for evt in trade_events:
        payload = evt.get("payload", evt)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue

        sector = payload.get("sector", "")
        if not sector:
            continue
        symbol = payload.get("pair", "")
        strategy = payload.get("strategy_id", "")
        direction = payload.get("side", "LONG")
        exposure = payload.get("position_size_quote", 0.0)

        if sector not in sector_data:
            sector_data[sector] = {
                "total_exposure": 0.0, "position_count": 0,
                "symbols": set(), "strategies": set(),
                "long_exposure": 0.0, "short_exposure": 0.0,
            }
        entry = sector_data[sector]
        entry["total_exposure"] += exposure
        entry["position_count"] += 1
        entry["symbols"].add(symbol)
        entry["strategies"].add(strategy)
        if direction == "LONG":
            entry["long_exposure"] += exposure
        else:
            entry["short_exposure"] += exposure

    result = {}
    for sector, data in sector_data.items():
        result[sector] = {
            "total_exposure": round(data["total_exposure"], 2),
            "position_count": data["position_count"],
            "symbols": sorted(data["symbols"]),
            "strategies": sorted(data["strategies"]),
            "long_exposure": round(data["long_exposure"], 2),
            "short_exposure": round(data["short_exposure"], 2),
        }
    return {"sectors": result, "sector_count": len(result)}


def build_macro_regime_analysis(daily_snapshots: list[dict], date: str) -> dict:
    """Extract portfolio-level macro regime analysis from DailySnapshot data.

    Picks regime_context from any snapshot (portfolio-wide, same for all bots),
    and aggregates applied_regime_config per bot.

    Args:
        daily_snapshots: list of raw DailySnapshot dicts (one per bot).
        date: YYYY-MM-DD date string.

    Returns:
        Dict with regime state + per-bot applied configs, or empty dict if
        no regime data present.
    """
    regime_context = None
    per_bot_configs: dict[str, dict] = {}

    for snap in daily_snapshots:
        # Pick regime_context from first bot that has it (same portfolio-wide)
        if regime_context is None and snap.get("regime_context"):
            regime_context = snap["regime_context"]

        bot_id = snap.get("bot_id", "")
        applied = snap.get("applied_regime_config")
        if bot_id and applied:
            per_bot_configs[bot_id] = applied

    if not regime_context:
        return {}

    return {
        "date": date,
        "macro_regime": regime_context.get("macro_regime", ""),
        "regime_confidence": regime_context.get("regime_confidence", 0.0),
        "stress_level": regime_context.get("stress_level", 0.0),
        "stress_onset": regime_context.get("stress_onset", False),
        "shift_velocity": regime_context.get("shift_velocity", 0.0),
        "suggested_leverage_mult": regime_context.get("suggested_leverage_mult", 1.0),
        "computed_at": regime_context.get("computed_at", ""),
        "per_bot_configs": per_bot_configs,
    }
