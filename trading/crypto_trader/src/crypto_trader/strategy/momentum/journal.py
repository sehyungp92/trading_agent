"""Trade journaling — records context-rich trade entries for inspection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from crypto_trader.core.models import Trade


@dataclass
class JournalEntry:
    # Identification
    datetime_utc: str = ""
    exchange: str = "hyperliquid"
    instrument: str = ""
    direction: str = ""
    setup_grade: str = ""
    # Context
    h4_bias_notes: str = ""
    h1_trend_notes: str = ""
    pullback_confluences: str = ""
    confirmation_type: str = ""
    # Execution
    entry_price: float = 0.0
    stop_price: float = 0.0
    liquidation_price: float = 0.0
    leverage: float = 0.0
    risk_pct: float = 0.0
    position_size: float = 0.0
    # Costs
    fees_paid: float = 0.0
    slippage_estimated: float = 0.0
    funding_paid: float = 0.0
    # Exit
    exit_distribution: str = ""  # JSON string of partial exits
    exit_reason: str = ""
    # Results
    final_r: float = 0.0
    pnl_usd: float = 0.0
    rule_compliance: str = "yes"
    mae_r: float = 0.0
    mfe_r: float = 0.0
    bars_held: int = 0
    entry_method: str = ""
    signal_variant: str = ""


class TradeJournal:
    def __init__(self, strategy_name: str = "momentum_pullback", run_id: str = "") -> None:
        self._strategy_name = strategy_name
        self._run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._entries: list[JournalEntry] = []

    def record(self, trade: Trade, context: dict) -> JournalEntry:
        final_r = trade.economic_r_multiple
        entry = JournalEntry(
            datetime_utc=trade.entry_time.isoformat() if trade.entry_time else "",
            instrument=trade.symbol,
            direction=trade.direction.value if trade.direction else "",
            setup_grade=trade.setup_grade.value if trade.setup_grade else "",
            entry_price=trade.entry_price,
            stop_price=context.get("stop_price", 0.0),
            liquidation_price=context.get("liquidation_price", 0.0),
            leverage=context.get("leverage", 0.0),
            risk_pct=context.get("risk_pct", 0.0),
            position_size=trade.qty,
            fees_paid=trade.commission,
            funding_paid=trade.funding_paid,
            exit_distribution=json.dumps(context.get("exit_distribution", [])),
            exit_reason=trade.exit_reason,
            final_r=final_r if final_r is not None else 0.0,
            pnl_usd=trade.net_pnl,
            mae_r=trade.mae_r or 0.0,
            mfe_r=trade.mfe_r or 0.0,
            bars_held=trade.bars_held,
            confirmation_type=trade.confirmation_type or "",
            entry_method=trade.entry_method or "",
            signal_variant=context.get("signal_variant", ""),
            h4_bias_notes=context.get("h4_bias_notes", ""),
            h1_trend_notes=context.get("h1_trend_notes", ""),
            pullback_confluences=",".join(context.get("confluences", [])),
        )
        self._entries.append(entry)
        return entry

    def save(self, output_dir: Path | str = Path("output/journal")) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self._strategy_name}_{self._run_id}.jsonl"
        with open(path, "w") as f:
            for entry in self._entries:
                f.write(json.dumps(asdict(entry)) + "\n")
        return path

    def to_dataframe(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame()
        return pd.DataFrame([asdict(e) for e in self._entries])

    def export_csv(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return path

    @property
    def entries(self) -> list[JournalEntry]:
        return self._entries
