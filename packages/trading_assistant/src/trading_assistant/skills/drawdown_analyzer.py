"""DrawdownAnalyzer — segments equity curve into drawdown episodes and attributes to root causes.

A drawdown episode starts when cumulative PnL drops below a running peak and
ends when cumulative PnL returns to or exceeds that peak.
"""
from __future__ import annotations

from collections import Counter

from trading_assistant.schemas.drawdown_analysis import DrawdownAttribution, DrawdownEpisode
from trading_assistant.schemas.events import TradeEvent


class DrawdownAnalyzer:
    def __init__(self, bot_id: str, date: str) -> None:
        self._bot_id = bot_id
        self._date = date

    def compute(self, trades: list[TradeEvent]) -> DrawdownAttribution:
        if not trades:
            return DrawdownAttribution(bot_id=self._bot_id, date=self._date)

        # Build equity curve
        cumulative = 0.0
        peak = 0.0
        episodes: list[DrawdownEpisode] = []
        current_dd_trades: list[TradeEvent] = []
        dd_peak = 0.0
        in_drawdown = False

        all_root_causes: Counter[str] = Counter()
        largest_single_loss_pct = 0.0

        for t in trades:
            cumulative += t.pnl

            if t.pnl < 0 and abs(t.pnl_pct) > largest_single_loss_pct:
                largest_single_loss_pct = abs(t.pnl_pct)

            if cumulative > peak:
                # New high — close any open drawdown
                if in_drawdown and current_dd_trades:
                    episodes.append(self._make_episode(
                        current_dd_trades, dd_peak, recovered=True,
                        recovery_date=t.exit_time.strftime("%Y-%m-%d"),
                    ))
                    current_dd_trades = []
                    in_drawdown = False
                peak = cumulative
                dd_peak = peak
            elif cumulative < peak:
                if not in_drawdown:
                    in_drawdown = True
                    dd_peak = peak
                current_dd_trades.append(t)

        # Close unclosed drawdown
        if in_drawdown and current_dd_trades:
            episodes.append(self._make_episode(current_dd_trades, dd_peak, recovered=False))

        # Aggregate root causes across all episodes
        for ep in episodes:
            all_root_causes.update(ep.root_cause_distribution)

        return DrawdownAttribution(
            bot_id=self._bot_id,
            date=self._date,
            episodes=episodes,
            top_contributing_root_causes=dict(all_root_causes),
            largest_single_loss_pct=largest_single_loss_pct,
        )

    def _make_episode(
        self,
        trades: list[TradeEvent],
        peak_pnl: float,
        recovered: bool = False,
        recovery_date: str | None = None,
    ) -> DrawdownEpisode:
        cumulative = peak_pnl
        trough = peak_pnl
        root_causes: Counter[str] = Counter()
        regimes: Counter[str] = Counter()

        for t in trades:
            cumulative += t.pnl
            if cumulative < trough:
                trough = cumulative
            for rc in t.root_causes:
                root_causes[rc] += 1
            if t.market_regime:
                regimes[t.market_regime] += 1

        dd_pct = ((peak_pnl - trough) / peak_pnl * 100) if peak_pnl > 0 else 0.0

        return DrawdownEpisode(
            bot_id=self._bot_id,
            start_date=trades[0].entry_time.strftime("%Y-%m-%d"),
            end_date=trades[-1].exit_time.strftime("%Y-%m-%d"),
            peak_pnl=peak_pnl,
            trough_pnl=trough,
            drawdown_pct=dd_pct,
            trade_count=len(trades),
            duration_days=(trades[-1].exit_time - trades[0].entry_time).days + 1,
            recovered=recovered,
            recovery_date=recovery_date,
            contributing_trades=[t.trade_id for t in trades],
            dominant_regime=regimes.most_common(1)[0][0] if regimes else "",
            root_cause_distribution=dict(root_causes),
        )
