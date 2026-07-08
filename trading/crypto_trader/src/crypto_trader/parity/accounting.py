"""Economic reconciliation helpers for parity reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from crypto_trader.core.models import TerminalMark, Trade
from crypto_trader.core.runtime_types import TradeOutcome


@dataclass(frozen=True, slots=True)
class EquityReconciliation:
    expected_equity: float
    actual_equity: float
    realized_net: float
    terminal_unrealized_net: float
    mismatch: float

    @property
    def passed(self) -> bool:
        return abs(self.mismatch) < 1e-6

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "expected_equity": self.expected_equity,
            "actual_equity": self.actual_equity,
            "realized_net": self.realized_net,
            "terminal_unrealized_net": self.terminal_unrealized_net,
            "mismatch": self.mismatch,
        }


def reconcile_equity(
    *,
    initial_equity: float,
    final_equity: float,
    trades: Iterable[Trade],
    terminal_marks: Iterable[TerminalMark] = (),
) -> EquityReconciliation:
    realized_net = sum(TradeOutcome.from_trade(trade).realized_pnl_net for trade in trades)
    terminal_unrealized_net = sum(mark.unrealized_pnl_net for mark in terminal_marks)
    expected = initial_equity + realized_net + terminal_unrealized_net
    return EquityReconciliation(
        expected_equity=expected,
        actual_equity=final_equity,
        realized_net=realized_net,
        terminal_unrealized_net=terminal_unrealized_net,
        mismatch=final_equity - expected,
    )
