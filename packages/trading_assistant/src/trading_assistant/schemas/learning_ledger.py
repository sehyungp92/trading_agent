# schemas/learning_ledger.py
"""Learning ledger schemas — ground truth tracking and retrospective synthesis."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class GroundTruthSnapshot(BaseModel):
    """Immutable performance snapshot — the system's val_bpb equivalent."""
    snapshot_date: str
    bot_id: str
    period_days: int = 30
    pnl_total: float = 0.0
    sharpe_ratio_30d: float = 0.0  # informational — not weighted in composite
    win_rate: float = 0.0  # informational — not weighted in composite
    calmar_ratio_30d: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_process_quality: float = 0.0  # 0-100
    expected_total_r: float = 0.0  # annualized net PnL
    expectancy: float = 0.0  # win_rate × (avg_win / avg_loss)
    composite_score: float = 0.5  # single number, deterministic
    trade_count: int = 0
    computed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class LearningLedgerEntry(BaseModel):
    """Weekly experiment log — autoresearch's results.tsv equivalent."""
    entry_id: str = ""
    week_start: str
    week_end: str
    ground_truth_start: dict[str, GroundTruthSnapshot] = {}
    ground_truth_end: dict[str, GroundTruthSnapshot] = {}
    composite_delta: dict[str, float] = {}  # bot_id → delta
    net_improvement: bool = False
    suggestions_proposed: int = 0
    suggestions_accepted: int = 0
    suggestions_implemented: int = 0
    experiments_concluded: int = 0
    discoveries_found: int = 0
    what_worked: list[str] = []
    what_failed: list[str] = []
    lessons_for_next_week: list[str] = []
    cycle_effectiveness: float = 0.0
    inner_suggestions_proposed: int = 0
    outer_suggestions_proposed: int = 0
    inner_positive_outcomes: int = 0
    outer_positive_outcomes: int = 0
    inner_total_outcomes: int = 0
    outer_total_outcomes: int = 0


class SynthesisItem(BaseModel):
    """A single worked/failed item in the retrospective synthesis."""
    suggestion_id: str = ""
    bot_id: str = ""
    category: str = ""
    title: str = ""
    outcome_verdict: str = ""
    ground_truth_delta: float = 0.0
    mechanism: str = ""  # from outcome reasoning


class DiscardItem(BaseModel):
    """A category/approach to stop trying."""
    bot_id: str = ""
    category: str = ""
    failure_count: int = 0
    reason: str = ""


class RetrospectiveSynthesis(BaseModel):
    """Weekly keep/discard verdict."""
    week_start: str
    week_end: str
    what_worked: list[SynthesisItem] = []
    what_failed: list[SynthesisItem] = []
    discard: list[DiscardItem] = []
    lessons: list[str] = []  # dynamic prompt instructions for next cycle
    ground_truth_deltas: dict[str, float] = {}
