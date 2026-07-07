"""Position reconciler — compares portfolio state with exchange state."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from crypto_trader.core.models import Position

log = structlog.get_logger()
_POSITION_EPS = 1e-8


@dataclass
class Discrepancy:
    """A single discrepancy between expected and actual positions."""

    symbol: str
    kind: str  # "phantom" | "missing" | "qty_mismatch" | "direction_mismatch"
    expected: str  # human-readable description
    actual: str


class PositionReconciler:
    """Compares coordinator's position book with exchange state.

    Used on startup and periodically during runtime.
    """

    def reconcile(
        self,
        expected_positions: dict[str, Position | None],
        actual_positions: list[Position],
    ) -> list[Discrepancy]:
        """Compare expected vs actual positions.

        Args:
            expected_positions: {symbol: Position | None} from portfolio state
            actual_positions: list of Position from exchange

        Returns:
            List of discrepancies (empty = all reconciled)
        """
        discrepancies = []

        actual_by_symbol = {
            p.symbol: p
            for p in actual_positions
            if abs(p.qty) > _POSITION_EPS
        }

        # Check for expected positions missing on exchange
        for symbol, expected in expected_positions.items():
            actual = actual_by_symbol.get(symbol)
            if expected is None:
                if actual is not None:
                    discrepancies.append(Discrepancy(
                        symbol=symbol,
                        kind="phantom",
                        expected="flat",
                        actual=f"{actual.direction.value} {actual.qty}",
                    ))
                continue

            if actual is None:
                discrepancies.append(Discrepancy(
                    symbol=symbol,
                    kind="missing",
                    expected=f"{expected.direction.value} {expected.qty}",
                    actual="flat",
                ))
                continue

            if expected.metadata.get("direction_conflict"):
                discrepancies.append(Discrepancy(
                    symbol=symbol,
                    kind="direction_mismatch",
                    expected="mixed local directions",
                    actual=f"{actual.direction.value} {actual.qty}",
                ))
                continue

            if expected.direction != actual.direction:
                discrepancies.append(Discrepancy(
                    symbol=symbol,
                    kind="direction_mismatch",
                    expected=expected.direction.value,
                    actual=actual.direction.value,
                ))
                continue

            qty_known = bool(expected.metadata.get("qty_known", True))
            if qty_known and abs(expected.qty - actual.qty) > _POSITION_EPS:
                discrepancies.append(Discrepancy(
                    symbol=symbol,
                    kind="qty_mismatch",
                    expected=f"{expected.qty}",
                    actual=f"{actual.qty}",
                ))

        # Check for phantom positions (on exchange but not expected)
        expected_symbols = set(expected_positions.keys())
        for actual in actual_positions:
            if abs(actual.qty) <= _POSITION_EPS:
                continue
            if actual.symbol not in expected_symbols:
                discrepancies.append(Discrepancy(
                    symbol=actual.symbol,
                    kind="phantom",
                    expected="flat",
                    actual=f"{actual.direction.value} {actual.qty}",
                ))

        if discrepancies:
            for d in discrepancies:
                log.warning(
                    "reconciler.discrepancy",
                    symbol=d.symbol,
                    kind=d.kind,
                    expected=d.expected,
                    actual=d.actual,
                )
        else:
            log.info("reconciler.all_reconciled", n_positions=len(actual_positions))

        return discrepancies
