"""ProcessQualityScorer — rules-based deterministic scoring engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from crypto_trader.instrumentation.types import (
    FilterDecision,
    MarketContext,
    ROOT_CAUSE_TAXONOMY,
)

if TYPE_CHECKING:
    from crypto_trader.core.models import Trade


@dataclass(frozen=True)
class ScoringRule:
    """A single quality scoring rule."""
    criterion: str
    condition_fn: Callable[..., bool]
    delta: int
    root_cause: str


def _regime_mismatch(
    trade: Trade,
    ctx: MarketContext | None,
    decisions: list[FilterDecision],
    sizing: dict,
) -> bool:
    """Trade direction != higher-TF bias at entry."""
    if ctx is None:
        return False
    direction = trade.direction.value  # "LONG" or "SHORT"

    # Check bias_direction (momentum), regime_direction (trend),
    # h4_context_direction (breakout)
    bias = ctx.bias_direction or ctx.regime_direction or ctx.h4_context_direction
    if bias is None:
        return False

    # Normalize: bias values are "LONG"/"SHORT" or "long"/"short"
    bias_upper = bias.upper()
    return bias_upper != direction


def _premature_stop(
    trade: Trade,
    ctx: MarketContext | None,
    decisions: list[FilterDecision],
    sizing: dict,
) -> bool:
    """Stopped within 2 bars of entry."""
    return (
        trade.bars_held <= 2
        and trade.exit_reason in ("protective_stop", "breakeven_stop")
        and trade.r_multiple is not None
        and trade.r_multiple < 0
    )


def _early_exit(
    trade: Trade,
    ctx: MarketContext | None,
    decisions: list[FilterDecision],
    sizing: dict,
) -> bool:
    """MFE > 2R but r_multiple < 0.5R (gave back >75% of MFE)."""
    if trade.mfe_r is None or trade.r_multiple is None:
        return False
    return trade.mfe_r > 2.0 and trade.r_multiple < 0.5


def _weak_signal(
    trade: Trade,
    ctx: MarketContext | None,
    decisions: list[FilterDecision],
    sizing: dict,
) -> bool:
    """B-grade setup AND loss."""
    if trade.setup_grade is None or trade.r_multiple is None:
        return False
    return trade.setup_grade.value == "B" and trade.r_multiple < 0


def _risk_cap_hit(
    trade: Trade,
    ctx: MarketContext | None,
    decisions: list[FilterDecision],
    sizing: dict,
) -> bool:
    """Risk was clamped by max_risk_pct."""
    return sizing.get("risk_clamped", False)


def _funding_adverse(
    trade: Trade,
    ctx: MarketContext | None,
    decisions: list[FilterDecision],
    sizing: dict,
) -> bool:
    """Funding rate at entry opposes trade direction."""
    if ctx is None:
        return False
    rate = ctx.funding_rate
    if rate == 0:
        return False
    direction = trade.direction.value
    # Positive funding hurts longs (they pay), negative hurts shorts
    if direction == "LONG" and rate > 0.0005:
        return True
    if direction == "SHORT" and rate < -0.0005:
        return True
    return False


def _funding_favorable(
    trade: Trade,
    ctx: MarketContext | None,
    decisions: list[FilterDecision],
    sizing: dict,
) -> bool:
    """Funding rate at entry supports trade direction."""
    if ctx is None:
        return False
    rate = ctx.funding_rate
    if rate == 0:
        return False
    direction = trade.direction.value
    if direction == "LONG" and rate < -0.0005:
        return True
    if direction == "SHORT" and rate > 0.0005:
        return True
    return False


def _strong_signal(
    trade: Trade,
    ctx: MarketContext | None,
    decisions: list[FilterDecision],
    sizing: dict,
) -> bool:
    """A-grade setup AND exit_efficiency > 0.6."""
    if trade.setup_grade is None:
        return False
    if trade.setup_grade.value != "A":
        return False
    if trade.mfe_r is None or trade.r_multiple is None:
        return False
    if trade.r_multiple <= 0 or trade.mfe_r <= 0:
        return False
    eff = trade.r_multiple / trade.mfe_r
    return eff > 0.6


def _good_execution(
    trade: Trade,
    ctx: MarketContext | None,
    decisions: list[FilterDecision],
    sizing: dict,
) -> bool:
    """Winner with exit_efficiency > 0.7."""
    if trade.mfe_r is None or trade.r_multiple is None:
        return False
    if trade.r_multiple <= 0 or trade.mfe_r <= 0:
        return False
    eff = trade.r_multiple / trade.mfe_r
    return eff > 0.7


def _exceptional_win(
    trade: Trade,
    ctx: MarketContext | None,
    decisions: list[FilterDecision],
    sizing: dict,
) -> bool:
    """R-multiple > 3.0."""
    return trade.r_multiple is not None and trade.r_multiple > 3.0


def _normal_win(
    trade: Trade,
    ctx: MarketContext | None,
    decisions: list[FilterDecision],
    sizing: dict,
) -> bool:
    """0 < R < 3.0."""
    return trade.r_multiple is not None and 0 < trade.r_multiple <= 3.0


def _normal_loss(
    trade: Trade,
    ctx: MarketContext | None,
    decisions: list[FilterDecision],
    sizing: dict,
) -> bool:
    """-1.5 < R < 0."""
    return trade.r_multiple is not None and -1.5 < trade.r_multiple < 0


def _regime_aligned(
    trade: Trade,
    ctx: MarketContext | None,
    decisions: list[FilterDecision],
    sizing: dict,
) -> bool:
    """Trade direction matches regime AND winner."""
    if ctx is None or trade.r_multiple is None:
        return False
    if trade.r_multiple <= 0:
        return False
    bias = ctx.bias_direction or ctx.regime_direction or ctx.h4_context_direction
    if bias is None:
        return False
    return bias.upper() == trade.direction.value


# ---------------------------------------------------------------------------
# Scoring rules table
# ---------------------------------------------------------------------------

SCORING_RULES: list[ScoringRule] = [
    ScoringRule("premature_stop", _premature_stop, -15, "premature_stop"),
    ScoringRule("early_exit", _early_exit, -20, "early_exit"),
    ScoringRule("regime_mismatch", _regime_mismatch, -10, "regime_mismatch"),
    ScoringRule("weak_signal", _weak_signal, -5, "weak_signal"),
    ScoringRule("risk_cap_hit", _risk_cap_hit, -5, "risk_cap_hit"),
    ScoringRule("funding_adverse", _funding_adverse, -3, "funding_adverse"),
    ScoringRule("funding_favorable", _funding_favorable, 3, "funding_favorable"),
    ScoringRule("strong_signal", _strong_signal, 5, "strong_signal"),
    ScoringRule("good_execution", _good_execution, 5, "good_execution"),
    # Tagging rules (delta=0)
    ScoringRule("exceptional_win", _exceptional_win, 0, "exceptional_win"),
    ScoringRule("normal_win", _normal_win, 0, "normal_win"),
    ScoringRule("normal_loss", _normal_loss, 0, "normal_loss"),
    ScoringRule("regime_aligned", _regime_aligned, 0, "regime_aligned"),
]

# Validate all root causes are in taxonomy
assert all(r.root_cause in ROOT_CAUSE_TAXONOMY for r in SCORING_RULES), (
    "ScoringRule root_cause must be in ROOT_CAUSE_TAXONOMY"
)


class ProcessQualityScorer:
    """Deterministic quality scorer for trade process quality."""

    def __init__(self, rules: list[ScoringRule] | None = None) -> None:
        self._rules = rules or SCORING_RULES

    def score(
        self,
        trade: Trade,
        entry_context: MarketContext | None,
        filter_decisions: list[FilterDecision],
        sizing_inputs: dict,
    ) -> tuple[int, list[str]]:
        """Score a trade's process quality.

        Returns (score_0_100, root_causes_list).
        """
        total_delta = 0
        root_causes: list[str] = []

        for rule in self._rules:
            if rule.condition_fn(trade, entry_context, filter_decisions, sizing_inputs):
                total_delta += rule.delta
                root_causes.append(rule.root_cause)

        score = max(0, min(100, 100 + total_delta))
        return score, root_causes
