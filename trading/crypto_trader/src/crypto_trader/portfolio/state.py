"""Portfolio state — mutable shared state for multi-strategy position tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from crypto_trader.core.models import Side


@dataclass
class OpenRisk:
    """A single open risk entry tracked by the portfolio."""

    strategy_id: str
    symbol: str
    direction: Side
    risk_R: float
    entry_time: object = None  # datetime | None
    risk_id: str = ""
    position_instance_id: str = ""
    intent_id: str = ""
    client_order_id: str = ""
    order_id: str = ""
    exchange_order_id: str = ""
    order_qty: float = 0.0
    filled_qty: float = 0.0
    applied_fill_ids: list[str] = field(default_factory=list)


@dataclass
class PortfolioState:
    """Shared mutable state across all strategies in a portfolio.

    Tracks open risks, daily P&L, and equity for portfolio-level rule evaluation.
    """

    equity: float = 0.0
    peak_equity: float = 0.0
    open_risks: list[OpenRisk] = field(default_factory=list)
    daily_pnl_R: dict[str, float] = field(default_factory=dict)  # per-strategy
    portfolio_daily_pnl_R: float = 0.0
    current_day: date | None = None

    def total_heat_R(self) -> float:
        """Total open risk across all strategies."""
        return sum(r.risk_R for r in self.open_risks)

    def directional_risk_R(self, direction: Side) -> float:
        """Total open risk in one direction."""
        return sum(r.risk_R for r in self.open_risks if r.direction == direction)

    def symbol_risk_R(self, symbol: str, direction: Side | None = None) -> float:
        """Total open risk for a symbol, optionally filtered by direction."""
        total = 0.0
        for r in self.open_risks:
            if r.symbol == symbol:
                if direction is None or r.direction == direction:
                    total += r.risk_R
        return total

    def strategy_position_count(self, strategy_id: str) -> int:
        """Count open positions for a specific strategy."""
        return sum(1 for r in self.open_risks if r.strategy_id == strategy_id)

    def total_positions(self) -> int:
        """Count all open positions across all strategies."""
        return len(self.open_risks)

    def dd_pct(self) -> float:
        """Current drawdown as fraction of peak equity."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def strategy_daily_pnl_R(self, strategy_id: str) -> float:
        """Get daily P&L in R-units for a strategy."""
        return self.daily_pnl_R.get(strategy_id, 0.0)

    def reset_daily(self, new_day: date) -> None:
        """Reset daily P&L counters for a new trading day."""
        self.daily_pnl_R.clear()
        self.portfolio_daily_pnl_R = 0.0
        self.current_day = new_day

    def add_risk(self, risk: OpenRisk) -> None:
        """Register a new open risk."""
        self.open_risks.append(risk)

    def find_risk(self, risk_id: str) -> OpenRisk | None:
        """Return an open risk by its exact ledger id."""
        if not risk_id:
            return None
        for index, risk in enumerate(self.open_risks):
            if risk.risk_id == risk_id:
                return risk
        return None

    def remove_risk(
        self,
        strategy_id: str,
        symbol: str,
        *,
        risk_id: str = "",
        order_refs: set[str] | None = None,
    ) -> OpenRisk | None:
        """Remove and return one matching open risk."""
        removed = self.remove_risks(
            strategy_id,
            symbol,
            risk_id=risk_id,
            order_refs=order_refs,
            remove_all=False,
        )
        return removed[0] if removed else None

    def remove_risks(
        self,
        strategy_id: str,
        symbol: str,
        *,
        risk_id: str = "",
        order_refs: set[str] | None = None,
        remove_all: bool = False,
    ) -> list[OpenRisk]:
        """Remove and return matching risks.

        Exact ids/order refs are preferred. ``remove_all`` is the fallback for
        legacy close events that only carry strategy and symbol.
        """
        refs = {str(ref) for ref in (order_refs or set()) if ref}
        removed: list[OpenRisk] = []
        kept: list[OpenRisk] = []
        for index, risk in enumerate(self.open_risks):
            if risk.strategy_id != strategy_id or risk.symbol != symbol:
                kept.append(risk)
                continue

            matches_exact = bool(risk_id and risk.risk_id == risk_id)
            matches_ref = bool(refs and (self._risk_refs(risk) & refs))
            matches_fallback = not risk_id and not refs
            if matches_exact or matches_ref or matches_fallback:
                removed.append(risk)
                if not remove_all:
                    kept.extend(self.open_risks[index + 1:])
                    break
                continue
            kept.append(risk)

        self.open_risks = kept
        return removed

    @staticmethod
    def _risk_refs(risk: OpenRisk) -> set[str]:
        return {
            str(value)
            for value in (
                risk.risk_id,
                risk.position_instance_id,
                risk.intent_id,
                risk.client_order_id,
                risk.order_id,
                risk.exchange_order_id,
            )
            if value
        }

    def update_equity(self, equity: float) -> None:
        """Update equity and peak equity."""
        self.equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "open_risks": [
                {
                    "strategy_id": r.strategy_id,
                    "symbol": r.symbol,
                    "direction": r.direction.value,
                    "risk_R": r.risk_R,
                    "entry_time": str(r.entry_time) if r.entry_time else None,
                    "risk_id": r.risk_id,
                    "position_instance_id": r.position_instance_id,
                    "intent_id": r.intent_id,
                    "client_order_id": r.client_order_id,
                    "order_id": r.order_id,
                    "exchange_order_id": r.exchange_order_id,
                    "order_qty": r.order_qty,
                    "filled_qty": r.filled_qty,
                    "applied_fill_ids": list(r.applied_fill_ids),
                }
                for r in self.open_risks
            ],
            "daily_pnl_R": dict(self.daily_pnl_R),
            "portfolio_daily_pnl_R": self.portfolio_daily_pnl_R,
            "current_day": str(self.current_day) if self.current_day else None,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PortfolioState":
        """Restore portfolio state from a persisted JSON payload."""
        state = cls(
            equity=float(payload.get("equity", 0.0) or 0.0),
            peak_equity=float(payload.get("peak_equity", 0.0) or 0.0),
            daily_pnl_R={
                str(k): float(v)
                for k, v in dict(payload.get("daily_pnl_R") or {}).items()
            },
            portfolio_daily_pnl_R=float(payload.get("portfolio_daily_pnl_R", 0.0) or 0.0),
            current_day=_parse_date(payload.get("current_day")),
        )
        for raw in payload.get("open_risks") or []:
            if not isinstance(raw, dict):
                continue
            try:
                direction = raw.get("direction", Side.LONG)
                if not isinstance(direction, Side):
                    direction = Side(direction)
                state.add_risk(OpenRisk(
                    strategy_id=str(raw.get("strategy_id") or ""),
                    symbol=str(raw.get("symbol") or ""),
                    direction=direction,
                    risk_R=float(raw.get("risk_R", 0.0) or 0.0),
                    entry_time=_parse_datetime(raw.get("entry_time")),
                    risk_id=str(raw.get("risk_id") or ""),
                    position_instance_id=str(raw.get("position_instance_id") or ""),
                    intent_id=str(raw.get("intent_id") or ""),
                    client_order_id=str(raw.get("client_order_id") or ""),
                    order_id=str(raw.get("order_id") or ""),
                    exchange_order_id=str(raw.get("exchange_order_id") or ""),
                    order_qty=float(raw.get("order_qty", 0.0) or 0.0),
                    filled_qty=float(raw.get("filled_qty", 0.0) or 0.0),
                    applied_fill_ids=[
                        str(fill_id)
                        for fill_id in raw.get("applied_fill_ids") or []
                        if fill_id
                    ],
                ))
            except (TypeError, ValueError):
                continue
        return state


def _parse_date(value: object) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None
