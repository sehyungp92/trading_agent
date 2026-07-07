# skills/process_quality_scorer.py
"""Deterministic process quality scorer — no LLM calls.

Evaluates each trade against configurable thresholds to detect execution
quality issues: regime mismatches, weak signals, slippage spikes, etc.
Score starts at 100 and deductions are applied per detected issue.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.process_quality import ProcessQualityResult, RootCause, ScoringDeduction


@dataclass
class ScorerConfig:
    """Configurable thresholds and deduction weights."""
    signal_strength_min: float = 0.5
    max_spread_bps: float = 20.0
    regime_allow_map: dict[str, list[str]] = field(default_factory=lambda: {
        "trending_up": ["LONG"],
        "trending_down": ["SHORT"],
        "ranging": ["LONG", "SHORT"],
        "volatile": ["LONG", "SHORT"],
    })
    # Per-cause deduction points
    deductions: dict[str, int] = field(default_factory=lambda: {
        "regime_mismatch": 25,
        "weak_signal": 20,
        "slippage_spike": 15,
        "early_exit": 15,
        "filter_blocked_good": 10,
    })


class ProcessQualityScorer:
    """Deterministic trade process quality scorer."""

    def __init__(self, config: ScorerConfig | None = None) -> None:
        self._config = config or ScorerConfig()

    def score(self, trade: TradeEvent) -> ProcessQualityResult:
        """Score a single trade's process quality. Returns ProcessQualityResult."""
        deductions: list[ScoringDeduction] = []

        # Check regime mismatch
        regime = trade.market_regime.lower() if trade.market_regime else ""
        side = trade.side.upper() if trade.side else ""
        if regime and regime in self._config.regime_allow_map:
            allowed_sides = self._config.regime_allow_map[regime]
            if side and side not in allowed_sides:
                pts = self._config.deductions.get("regime_mismatch", 25)
                deductions.append(ScoringDeduction(
                    root_cause=RootCause.REGIME_MISMATCH,
                    points=pts,
                    evidence=f"{side} trade in {regime} regime (allowed: {allowed_sides})",
                ))

        # Check weak signal
        if trade.entry_signal_strength < self._config.signal_strength_min:
            pts = self._config.deductions.get("weak_signal", 20)
            deductions.append(ScoringDeduction(
                root_cause=RootCause.WEAK_SIGNAL,
                points=pts,
                evidence=f"Signal strength {trade.entry_signal_strength} < min {self._config.signal_strength_min}",
            ))

        # Check slippage spike
        if trade.spread_at_entry > self._config.max_spread_bps:
            pts = self._config.deductions.get("slippage_spike", 15)
            deductions.append(ScoringDeduction(
                root_cause=RootCause.SLIPPAGE_SPIKE,
                points=pts,
                evidence=f"Spread {trade.spread_at_entry} bps > max {self._config.max_spread_bps} bps",
            ))

        # Check early exit (MANUAL exit reason)
        exit_reason = trade.exit_reason.upper() if trade.exit_reason else ""
        if exit_reason == "MANUAL":
            pts = self._config.deductions.get("early_exit", 15)
            deductions.append(ScoringDeduction(
                root_cause=RootCause.EARLY_EXIT,
                points=pts,
                evidence=f"Manual exit (exit_reason={trade.exit_reason})",
            ))

        # Check filter_blocked_good (blocked but would have been profitable)
        if trade.blocked_by and trade.pnl > 0:
            pts = self._config.deductions.get("filter_blocked_good", 10)
            deductions.append(ScoringDeduction(
                root_cause=RootCause.FILTER_BLOCKED_GOOD,
                points=pts,
                evidence=f"Blocked by '{trade.blocked_by}' but PnL was {trade.pnl:.2f}",
            ))

        # Compute final score
        total_deduction = sum(d.points for d in deductions)
        final_score = max(0, 100 - total_deduction)

        root_causes = [d.root_cause for d in deductions]
        evidence_refs = [d.evidence for d in deductions]

        return ProcessQualityResult(
            score=final_score,
            root_causes=root_causes,
            deductions=deductions,
            evidence_refs=evidence_refs,
        )
