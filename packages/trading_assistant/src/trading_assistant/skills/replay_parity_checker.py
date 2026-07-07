"""Replay parity threshold checks."""
from __future__ import annotations

from trading_assistant.schemas.replay_parity import ReplayParityReport, ReplayParityStatus


class ReplayParityChecker:
    def __init__(
        self,
        *,
        min_entry_match_rate: float = 0.95,
        min_exit_match_rate: float = 0.90,
        min_side_quantity_match_rate: float = 0.95,
        max_pnl_delta_pct: float = 5.0,
        max_drawdown_delta_pct: float = 5.0,
        max_fill_price_delta_bps: float = 25.0,
        max_fee_slippage_delta_bps: float = 25.0,
    ) -> None:
        self.min_entry_match_rate = min_entry_match_rate
        self.min_exit_match_rate = min_exit_match_rate
        self.min_side_quantity_match_rate = min_side_quantity_match_rate
        self.max_pnl_delta_pct = max_pnl_delta_pct
        self.max_drawdown_delta_pct = max_drawdown_delta_pct
        self.max_fill_price_delta_bps = max_fill_price_delta_bps
        self.max_fee_slippage_delta_bps = max_fee_slippage_delta_bps

    def classify(self, report: ReplayParityReport) -> ReplayParityStatus:
        if report.trade_count_live <= 0 and report.trade_count_replay <= 0:
            return ReplayParityStatus.INSUFFICIENT_DATA
        failures = [
            report.entry_match_rate < self.min_entry_match_rate,
            report.exit_match_rate < self.min_exit_match_rate,
            report.side_quantity_match_rate < self.min_side_quantity_match_rate,
            abs(report.pnl_delta_pct) > self.max_pnl_delta_pct,
            abs(report.drawdown_delta_pct) > self.max_drawdown_delta_pct,
            abs(report.fill_price_delta_bps) > self.max_fill_price_delta_bps,
            abs(report.fee_slippage_delta_bps) > self.max_fee_slippage_delta_bps,
        ]
        if any(failures):
            return ReplayParityStatus.FAIL
        if report.known_gaps or report.missing_trade_explanations or report.extra_simulated_trade_explanations:
            return ReplayParityStatus.PASS_WITH_KNOWN_GAPS
        return ReplayParityStatus.PASS

    def checked(self, report: ReplayParityReport) -> ReplayParityReport:
        report.status = self.classify(report)
        return report
