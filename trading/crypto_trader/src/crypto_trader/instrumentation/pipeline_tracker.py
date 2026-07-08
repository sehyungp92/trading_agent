"""Signal pipeline funnel tracker — per-strategy, per-period counters at each pipeline stage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone


# Gate-to-funnel-stage mapping
_GATE_STAGE_MAP = {
    "indicators": "indicators_ready",
    "warmup": "indicators_ready",
    "setup": "setups_detected",
    "confirmation": "confirmations",
    "model1_confirmation": "confirmations",
    "entry_order": "entries_attempted",
    "execute_entry_order": "entries_attempted",
}


@dataclass
class PipelineFunnel:
    """Snapshot of pipeline counters for one strategy over one period."""

    strategy_id: str
    period_start: datetime
    period_end: datetime
    bars_received: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    indicators_ready: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    setups_detected: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    confirmations: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    entries_attempted: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fills: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    trades_closed: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    gate_rejections: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "bars_received": dict(self.bars_received),
            "indicators_ready": dict(self.indicators_ready),
            "setups_detected": dict(self.setups_detected),
            "confirmations": dict(self.confirmations),
            "entries_attempted": dict(self.entries_attempted),
            "fills": dict(self.fills),
            "trades_closed": dict(self.trades_closed),
            "gate_rejections": {
                sym: dict(gates) for sym, gates in self.gate_rejections.items()
            },
        }

    def total(self, stage: str) -> int:
        """Sum across all symbols for a given stage."""
        counter = getattr(self, stage, None)
        if counter is None:
            return 0
        return sum(counter.values())


class PipelineTracker:
    """Tracks signal pipeline progression per strategy.

    Records bars → indicators → setups → confirmations → entries → fills → closed
    and surfaces which gate blocks the pipeline.
    """

    def __init__(self, strategy_id: str) -> None:
        self._strategy_id = strategy_id
        self._period_start = datetime.now(timezone.utc)

        self._bars_received: dict[str, int] = defaultdict(int)
        self._indicators_ready: dict[str, int] = defaultdict(int)
        self._setups_detected: dict[str, int] = defaultdict(int)
        self._confirmations: dict[str, int] = defaultdict(int)
        self._entries_attempted: dict[str, int] = defaultdict(int)
        self._fills: dict[str, int] = defaultdict(int)
        self._trades_closed: dict[str, int] = defaultdict(int)
        self._gate_rejections: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def record_bar(self, sym: str) -> None:
        """Record that a bar was received for this symbol."""
        self._bars_received[sym] += 1

    def record_gate(self, sym: str, gate_name: str, passed: bool) -> None:
        """Record a gate evaluation. Maps to funnel stage on pass, rejection on fail."""
        if passed:
            stage = _GATE_STAGE_MAP.get(gate_name)
            if stage is not None:
                counter = getattr(self, f"_{stage}", None)
                if counter is not None:
                    counter[sym] += 1
        else:
            self._gate_rejections[sym][gate_name] += 1

    def record_fill(self, sym: str) -> None:
        """Record a fill for this symbol."""
        self._fills[sym] += 1

    def record_trade_closed(self, sym: str) -> None:
        """Record a trade closure for this symbol."""
        self._trades_closed[sym] += 1

    def snapshot_and_reset(self) -> PipelineFunnel:
        """Take a snapshot of current counters and reset for next period."""
        now = datetime.now(timezone.utc)
        funnel = PipelineFunnel(
            strategy_id=self._strategy_id,
            period_start=self._period_start,
            period_end=now,
            bars_received=dict(self._bars_received),
            indicators_ready=dict(self._indicators_ready),
            setups_detected=dict(self._setups_detected),
            confirmations=dict(self._confirmations),
            entries_attempted=dict(self._entries_attempted),
            fills=dict(self._fills),
            trades_closed=dict(self._trades_closed),
            gate_rejections={
                sym: dict(gates) for sym, gates in self._gate_rejections.items()
            },
        )

        # Reset
        self._period_start = now
        self._bars_received = defaultdict(int)
        self._indicators_ready = defaultdict(int)
        self._setups_detected = defaultdict(int)
        self._confirmations = defaultdict(int)
        self._entries_attempted = defaultdict(int)
        self._fills = defaultdict(int)
        self._trades_closed = defaultdict(int)
        self._gate_rejections = defaultdict(lambda: defaultdict(int))

        return funnel

    @staticmethod
    def assess(funnel: PipelineFunnel) -> str:
        """Assess the funnel health.

        Returns one of:
        - "normal": entries attempted or legitimate no-signal period
        - "no_signals": indicators ready but no setups — legitimate
        - "pipeline_broken": no bars received for any subscribed symbol
        - "stalled": bars received but indicators never ready
        - "gate_blocked": setups found but a single gate rejects 100% of attempts
        """
        total_bars = funnel.total("bars_received")
        total_indicators = funnel.total("indicators_ready")
        total_setups = funnel.total("setups_detected")
        total_entries = funnel.total("entries_attempted")

        if total_bars == 0:
            return "pipeline_broken"

        if total_indicators == 0:
            return "stalled"

        if total_setups == 0:
            return "no_signals"

        if total_entries == 0 and total_setups > 0:
            # Check if a single gate blocks everything
            for sym, gates in funnel.gate_rejections.items():
                for gate_name, count in gates.items():
                    sym_setups = funnel.setups_detected.get(sym, 0)
                    if sym_setups > 0 and count >= sym_setups:
                        return "gate_blocked"

        return "normal"
