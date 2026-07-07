"""DailyAggregator — live mode only. Computes DailySnapshot at UTC midnight."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone

from crypto_trader.instrumentation.types import (
    DailySnapshot,
    ErrorEvent,
    EventMetadata,
    HealthReportSnapshot,
    InstrumentedTradeEvent,
    MissedOpportunityEvent,
    PipelineFunnelSnapshot,
)


def _has_explicit_economics(event: InstrumentedTradeEvent) -> bool:
    return any((
        event.price_pnl_gross != 0.0,
        event.total_fees != 0.0,
        event.realized_pnl_net != 0.0,
        event.funding_paid != 0.0,
    ))


def _event_net_pnl(event: InstrumentedTradeEvent) -> float:
    if _has_explicit_economics(event):
        return event.realized_pnl_net
    return event.pnl


def _event_gross_price_pnl(event: InstrumentedTradeEvent) -> float:
    if _has_explicit_economics(event):
        if event.price_pnl_gross != 0.0:
            return event.price_pnl_gross
        return event.realized_pnl_net + event.funding_paid + event.total_fees
    return event.pnl + event.commission


def _event_total_fees(event: InstrumentedTradeEvent) -> float:
    if _has_explicit_economics(event):
        return event.total_fees
    return event.commission


def _event_realized_r(event: InstrumentedTradeEvent) -> float:
    for value in (event.realized_r_net, event.r_multiple, event.geometric_r):
        if value is not None:
            return float(value)
    return 0.0


class DailyAggregator:
    """Accumulates trade/missed events during the day, computes DailySnapshot.

    Implements the Sink protocol so it can be added directly to EventEmitter.
    """

    def __init__(self, bot_id: str = "") -> None:
        self._bot_id = bot_id
        self._today_trades: list[InstrumentedTradeEvent] = []
        self._today_missed: list[MissedOpportunityEvent] = []
        self._trades_by_date: defaultdict[str, list[InstrumentedTradeEvent]] = defaultdict(list)
        self._missed_by_date: defaultdict[str, list[MissedOpportunityEvent]] = defaultdict(list)
        self._equity_history: list[tuple[datetime, float]] = []
        self._current_date: str = ""

    # --- Sink protocol methods ---

    def write_trade(self, event: InstrumentedTradeEvent) -> None:
        self._trades_by_date[self._event_utc_date(event)].append(event)
        self._today_trades.append(event)

    def write_missed(self, event: MissedOpportunityEvent) -> None:
        self._missed_by_date[self._event_utc_date(event)].append(event)
        self._today_missed.append(event)

    def write_daily(self, event: DailySnapshot) -> None:
        pass  # We produce these, not consume them

    def write_error(self, event: ErrorEvent) -> None:
        pass  # Not tracked by daily aggregator

    def write_funnel(self, event: PipelineFunnelSnapshot) -> None:
        pass  # Not tracked by daily aggregator

    def write_health_report(self, event: HealthReportSnapshot) -> None:
        pass  # Not tracked by daily aggregator

    def write_event(self, event_type: str, event) -> None:
        pass  # Generic assistant events are not part of daily trade aggregation

    # --- Legacy convenience aliases ---

    def record_trade(self, event: InstrumentedTradeEvent) -> None:
        self.write_trade(event)

    def record_missed(self, event: MissedOpportunityEvent) -> None:
        self.write_missed(event)

    def record_equity(self, timestamp: datetime, equity: float) -> None:
        self._equity_history.append((timestamp, equity))

    def compute_snapshot(self, date_str: str) -> DailySnapshot:
        """Build a DailySnapshot from events whose exchange timestamp is on date_str."""
        trades = list(self._trades_by_date.pop(date_str, []))
        missed_for_date = list(self._missed_by_date.pop(date_str, []))
        missed = list({
            (event.logical_event_id or event.opportunity_id or event.metadata.event_id): event
            for event in missed_for_date
        }.values())

        win_count = sum(1 for t in trades if _event_net_pnl(t) > 0)
        loss_count = sum(1 for t in trades if _event_net_pnl(t) <= 0)
        gross_pnl = sum(_event_gross_price_pnl(t) for t in trades)
        net_pnl = sum(_event_net_pnl(t) for t in trades)
        total_fees = sum(_event_total_fees(t) for t in trades)
        funding_paid = sum(t.funding_paid for t in trades)
        realized_R = sum(_event_realized_r(t) for t in trades)

        # Max drawdown from equity history
        max_dd = 0.0
        peak = 0.0
        for _, eq in self._equity_for_date(date_str):
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = (peak - eq) / peak * 100
                max_dd = max(max_dd, dd)

        # Process quality average
        scores = [t.process_quality_score for t in trades]
        avg_quality = sum(scores) / len(scores) if scores else 0.0

        # Root cause distribution
        rc_dist: dict[str, int] = defaultdict(int)
        for t in trades:
            for rc in t.root_causes:
                rc_dist[rc] += 1

        # Missed opportunities that would have won
        missed_won = sum(
            1 for m in missed
            if m.would_have_hit_tp is True
        )

        # Per-strategy summary
        per_strat: dict[str, dict] = defaultdict(
            lambda: {
                "trades": 0,
                "pnl": 0.0,
                "gross_pnl": 0.0,
                "net_pnl": 0.0,
                "fees": 0.0,
                "funding": 0.0,
                "realized_R": 0.0,
            }
        )
        for t in trades:
            sid = t.metadata.strategy_id
            per_strat[sid]["trades"] += 1
            trade_net_pnl = _event_net_pnl(t)
            per_strat[sid]["gross_pnl"] += _event_gross_price_pnl(t)
            per_strat[sid]["pnl"] += trade_net_pnl
            per_strat[sid]["net_pnl"] += trade_net_pnl
            per_strat[sid]["fees"] += _event_total_fees(t)
            per_strat[sid]["funding"] += t.funding_paid
            per_strat[sid]["realized_R"] += _event_realized_r(t)

        # Rolling metrics placeholder (need 30d equity history for proper calc)
        sharpe_30d = self._rolling_sharpe(30)
        sortino_30d = self._rolling_sortino(30)

        metadata = EventMetadata.create(
            bot_id=self._bot_id,
            strategy_id="portfolio",
            exchange_ts=datetime.now(timezone.utc),
            event_type="daily_snapshot",
            payload_key=date_str,
        )

        snapshot = DailySnapshot(
            metadata=metadata,
            date=date_str,
            total_trades=len(trades),
            win_count=win_count,
            loss_count=loss_count,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            max_drawdown_pct=max_dd,
            sharpe_rolling_30d=sharpe_30d,
            sortino_rolling_30d=sortino_30d,
            calmar_rolling_30d=0.0,
            exposure_pct=0.0,
            missed_count=len(missed),
            missed_would_have_won=missed_won,
            avg_process_quality=avg_quality,
            root_cause_distribution=dict(rc_dist),
            per_strategy_summary=dict(per_strat),
            family_summary={
                "family_level_trades": len(trades),
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "fees": total_fees,
                "funding": funding_paid,
                "realized_R": realized_R,
            },
        )

        # Keep newer-day events buffered if closeout runs late or races ingestion.
        self._today_trades = [
            event for event in self._today_trades
            if self._event_utc_date(event) != date_str
        ]
        self._today_missed = [
            event for event in self._today_missed
            if self._event_utc_date(event) != date_str
        ]

        return snapshot

    @staticmethod
    def _event_utc_date(event: InstrumentedTradeEvent | MissedOpportunityEvent) -> str:
        timestamp = event.metadata.exchange_timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).date().isoformat()

    def _equity_for_date(self, date_str: str) -> list[tuple[datetime, float]]:
        return [
            (timestamp, equity)
            for timestamp, equity in self._equity_history
            if (
                timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
            ).astimezone(timezone.utc).date().isoformat() == date_str
        ]

    def _rolling_sharpe(self, days: int) -> float:
        """Compute rolling Sharpe from equity history."""
        if len(self._equity_history) < 2:
            return 0.0

        # Get daily returns from equity snapshots
        daily: dict[str, float] = {}
        for ts, eq in self._equity_history:
            day = ts.strftime("%Y-%m-%d")
            daily[day] = eq

        sorted_days = sorted(daily.keys())[-days:]
        if len(sorted_days) < 2:
            return 0.0

        values = [daily[d] for d in sorted_days]
        returns = [(values[i] / values[i - 1]) - 1.0 for i in range(1, len(values))]

        if not returns:
            return 0.0

        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std = math.sqrt(var) if var > 0 else 0.0

        if std == 0:
            return 0.0

        return (mean_r / std) * math.sqrt(365)

    def _rolling_sortino(self, days: int) -> float:
        """Compute rolling Sortino from equity history."""
        if len(self._equity_history) < 2:
            return 0.0

        daily: dict[str, float] = {}
        for ts, eq in self._equity_history:
            day = ts.strftime("%Y-%m-%d")
            daily[day] = eq

        sorted_days = sorted(daily.keys())[-days:]
        if len(sorted_days) < 2:
            return 0.0

        values = [daily[d] for d in sorted_days]
        returns = [(values[i] / values[i - 1]) - 1.0 for i in range(1, len(values))]

        if not returns:
            return 0.0

        mean_r = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return 99.0 if mean_r > 0 else 0.0

        down_var = sum(r ** 2 for r in downside) / len(downside)
        down_std = math.sqrt(down_var) if down_var > 0 else 0.0

        if down_std == 0:
            return 0.0

        return (mean_r / down_std) * math.sqrt(365)
