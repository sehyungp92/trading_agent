# skills/hypothesis_library.py
"""HypothesisLibrary — adaptive catalog mapping observed symptoms to structural solutions.

Maintains backward compatibility with the original static API (get_all, get_by_category,
get_relevant) while adding a JSONL-backed store with lifecycle tracking.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.schemas.hypothesis import HypothesisRecord

logger = logging.getLogger(__name__)


# --- Legacy dataclass for backward compat ---

@dataclass
class Hypothesis:
    id: str
    title: str
    category: str
    description: str
    evidence_required: list[str]
    reversibility: str
    estimated_complexity: str


# Static catalog: the original 12 entries
_CATALOG: list[Hypothesis] = [
    Hypothesis(
        id="h-signal-recalibrate",
        title="Signal recalibration required",
        category="signal_decay",
        description="Signal-to-outcome correlation has declined over 30+ days. The predictive model may need retraining or indicator weights adjusted.",
        evidence_required=["signal_health declining stability", "win_correlation < 0.1 for 30+ days"],
        reversibility="easy",
        estimated_complexity="medium",
    ),
    Hypothesis(
        id="h-signal-replace",
        title="Signal component replacement",
        category="signal_decay",
        description="A specific signal component has become noise. Consider replacing with an alternative indicator.",
        evidence_required=["single component win_correlation < 0.0 for 60+ days", "negative trend_during_trade"],
        reversibility="moderate",
        estimated_complexity="high",
    ),
    Hypothesis(
        id="h-filter-loosen",
        title="Filter threshold loosening",
        category="filter_over_blocking",
        description="A filter is blocking high-quality signals. The threshold may be too aggressive.",
        evidence_required=["filter blocking > 30% of entries", "blocked trades hypothetical win rate > 60%"],
        reversibility="easy",
        estimated_complexity="low",
    ),
    Hypothesis(
        id="h-filter-restructure",
        title="Filter logic restructure",
        category="filter_over_blocking",
        description="Filter logic is fundamentally misaligned. Restructure from threshold-based to regime-conditional.",
        evidence_required=["filter ROI negative for 30+ days", "filter blocks differently across regimes"],
        reversibility="moderate",
        estimated_complexity="high",
    ),
    Hypothesis(
        id="h-exit-trailing",
        title="Switch to trailing stop",
        category="exit_timing",
        description="Fixed stops are causing premature exits. A trailing stop could capture more of the move.",
        evidence_required=["exit_efficiency < 50% on winners", "MFE >> realized PnL on winners"],
        reversibility="easy",
        estimated_complexity="medium",
    ),
    Hypothesis(
        id="h-exit-time-based",
        title="Add time-based exit component",
        category="exit_timing",
        description="Trades held too long in adverse conditions. A time-decay component could reduce holding time in losers.",
        evidence_required=["loser average duration > 2x winner duration", "MAE increases with hold time"],
        reversibility="easy",
        estimated_complexity="low",
    ),
    Hypothesis(
        id="h-fills-limit-orders",
        title="Switch to limit order entries",
        category="adverse_fills",
        description="Market orders are incurring excessive slippage. Limit orders with a tight timeout could reduce fill costs.",
        evidence_required=["avg slippage > 2 bps", "adverse_selection_detected = true"],
        reversibility="easy",
        estimated_complexity="low",
    ),
    Hypothesis(
        id="h-fills-timing",
        title="Delay entry by 1-2 bars",
        category="adverse_fills",
        description="Entering immediately on signal may cause adverse selection. A slight delay could improve fill quality.",
        evidence_required=["slippage concentrated in first minute of entry", "price reverts within 2 bars post-entry"],
        reversibility="easy",
        estimated_complexity="low",
    ),
    Hypothesis(
        id="h-regime-pause",
        title="Regime-conditional pause",
        category="regime_breakdown",
        description="Strategy performs poorly in a specific regime. Pausing during that regime could reduce drawdowns.",
        evidence_required=["PnL < 0 in specific regime for 20+ trades", "regime classification accuracy > 70%"],
        reversibility="easy",
        estimated_complexity="low",
    ),
    Hypothesis(
        id="h-regime-adaptation",
        title="Regime-adaptive parameters",
        category="regime_breakdown",
        description="Using different parameter sets per regime could improve performance. Requires monthly validation.",
        evidence_required=["significant PnL difference across regimes", "sufficient trade count per regime (>30)"],
        reversibility="moderate",
        estimated_complexity="high",
    ),
    Hypothesis(
        id="h-crowding-reduce",
        title="Reduce correlated exposure",
        category="correlation_crowding",
        description="Multiple trading/strategies are taking correlated positions, amplifying drawdowns.",
        evidence_required=["inter-bot correlation > 0.6", "concurrent drawdowns on same days"],
        reversibility="easy",
        estimated_complexity="low",
    ),
    Hypothesis(
        id="h-crowding-diversify",
        title="Introduce anti-correlated strategy",
        category="correlation_crowding",
        description="Portfolio lacks diversification. An anti-correlated strategy could reduce overall volatility.",
        evidence_required=["portfolio correlation > 0.5", "single-regime concentration > 60%"],
        reversibility="moderate",
        estimated_complexity="high",
    ),
]


# --- Legacy module-level functions (backward compat) ---

def get_all() -> list[Hypothesis]:
    """Return all hypotheses in the static catalog."""
    return list(_CATALOG)


def get_by_category(category: str) -> list[Hypothesis]:
    """Return hypotheses for a specific category from the static catalog."""
    return [h for h in _CATALOG if h.category == category]


def get_relevant(suggestions: list) -> list[Hypothesis]:
    """Match strategy engine suggestions to applicable hypotheses.

    Maps suggestion keywords/titles to hypothesis categories.
    """
    if not suggestions:
        return []

    keyword_map = {
        "signal": "signal_decay",
        "decay": "signal_decay",
        "alpha": "signal_decay",
        "filter": "filter_over_blocking",
        "block": "filter_over_blocking",
        "exit": "exit_timing",
        "premature": "exit_timing",
        "stop": "exit_timing",
        "slippage": "adverse_fills",
        "fill": "adverse_fills",
        "regime": "regime_breakdown",
        "correlation": "correlation_crowding",
        "crowding": "correlation_crowding",
        "diversif": "correlation_crowding",
    }

    matched_categories: set[str] = set()
    for suggestion in suggestions:
        title = (getattr(suggestion, "title", "") or "").lower()
        description = (getattr(suggestion, "description", "") or "").lower()
        text = f"{title} {description}"
        for keyword, category in keyword_map.items():
            if keyword in text:
                matched_categories.add(category)

    seen_ids: set[str] = set()
    result: list[Hypothesis] = []
    for cat in matched_categories:
        for h in get_by_category(cat):
            if h.id not in seen_ids:
                seen_ids.add(h.id)
                result.append(h)

    return result


# --- Adaptive JSONL-backed library ---

class HypothesisLibrary:
    """JSONL-backed adaptive hypothesis library with lifecycle tracking.

    On first load, seeds from the static catalog. Supports recording proposals,
    acceptances, rejections, and outcomes. Auto-retires ineffective hypotheses.
    """

    def __init__(self, findings_dir: Path) -> None:
        self._path = findings_dir / "hypotheses.jsonl"
        self._findings_dir = findings_dir

    def _load_all(self) -> list[HypothesisRecord]:
        """Load all hypothesis records from JSONL."""
        if not self._path.exists():
            return []
        records: list[HypothesisRecord] = []
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    records.append(HypothesisRecord(**json.loads(line)))
                except Exception:
                    pass
        return records

    def _save_all(self, records: list[HypothesisRecord]) -> None:
        """Overwrite the JSONL file with all records (atomic write)."""
        from trading_assistant.skills._atomic_write import atomic_rewrite_jsonl

        atomic_rewrite_jsonl(self._path, records)

    def seed_if_needed(self) -> None:
        """Seed from static catalog if JSONL doesn't exist or is empty."""
        if self._path.exists() and self._load_all():
            return
        records: list[HypothesisRecord] = []
        for h in _CATALOG:
            records.append(HypothesisRecord(
                id=h.id,
                title=h.title,
                category=h.category,
                description=h.description,
                evidence_required=", ".join(h.evidence_required),
                reversibility=h.reversibility,
                estimated_complexity=h.estimated_complexity,
            ))
        self._save_all(records)

    def get_active(self) -> list[HypothesisRecord]:
        """Return active hypotheses (not retired, effectiveness > -0.5)."""
        self.seed_if_needed()
        records = self._load_all()
        return [
            r for r in records
            if r.status != "retired" and r.effectiveness > -0.5
        ]

    def get_all_records(self) -> list[HypothesisRecord]:
        """Return all hypothesis records including retired."""
        self.seed_if_needed()
        return self._load_all()

    def record_proposal(self, hypothesis_id: str) -> None:
        """Increment times_proposed for a hypothesis."""
        self.seed_if_needed()
        records = self._load_all()
        for r in records:
            if r.id == hypothesis_id:
                r.times_proposed += 1
                r.last_proposed_at = datetime.now(timezone.utc).isoformat()
                break
        self._save_all(records)

    def record_acceptance(self, hypothesis_id: str) -> None:
        """Increment times_accepted for a hypothesis."""
        self.seed_if_needed()
        records = self._load_all()
        for r in records:
            if r.id == hypothesis_id:
                r.times_accepted += 1
                break
        self._save_all(records)

    def record_rejection(self, hypothesis_id: str) -> None:
        """Increment times_rejected. Auto-retire if effectiveness drops."""
        self.seed_if_needed()
        records = self._load_all()
        for r in records:
            if r.id == hypothesis_id:
                r.times_rejected += 1
                # Auto-retire: 3+ rejections AND non-positive effectiveness
                if r.times_rejected >= 3 and r.effectiveness <= 0:
                    r.status = "retired"
                break
        self._save_all(records)

    def record_outcome(self, hypothesis_id: str, positive: bool) -> None:
        """Record a measured outcome for a hypothesis."""
        self.seed_if_needed()
        records = self._load_all()
        for r in records:
            if r.id == hypothesis_id:
                if positive:
                    r.outcomes_positive += 1
                else:
                    r.outcomes_negative += 1
                # Re-check retirement (outcome could worsen effectiveness)
                if r.times_rejected >= 3 and r.effectiveness <= 0:
                    r.status = "retired"
                break
        self._save_all(records)

    def add_candidate(
        self,
        title: str,
        category: str,
        description: str,
        evidence: str = "",
    ) -> str:
        """Add a new candidate hypothesis. Returns the generated ID."""
        import hashlib

        hypothesis_id = "hyp_" + hashlib.sha256(
            f"{category}:{title}".encode()
        ).hexdigest()[:8]

        self.seed_if_needed()
        records = self._load_all()

        # Dedup by ID
        if any(r.id == hypothesis_id for r in records):
            return hypothesis_id

        records.append(HypothesisRecord(
            id=hypothesis_id,
            title=title,
            category=category,
            description=description,
            evidence_required=evidence,
            status="candidate",
        ))
        self._save_all(records)
        return hypothesis_id

    def promote_candidates(self) -> int:
        """Promote candidate hypotheses to active if proposed >= 2 times.

        Returns the number of candidates promoted.
        """
        self.seed_if_needed()
        records = self._load_all()
        promoted = 0
        for r in records:
            if r.status == "candidate" and r.times_proposed >= 2:
                r.status = "active"
                promoted += 1
        if promoted:
            self._save_all(records)
        return promoted

    def create_from_reliability(self, summary) -> list[str]:
        """Create candidate hypotheses from chronic bug classes in reliability summary.

        Args:
            summary: ReliabilitySummary instance.

        Returns:
            List of created hypothesis IDs.
        """
        created: list[str] = []
        for bug_class in summary.chronic_bug_classes:
            scorecard = summary.scorecards_by_class.get(bug_class)
            if not scorecard:
                continue
            hyp_id = self.add_candidate(
                title=f"Chronic {bug_class} reliability issue",
                category=f"reliability_{bug_class}",
                description=(
                    f"Bug class '{bug_class}' has {scorecard.intervention_count} interventions "
                    f"with recurrence rate {scorecard.recurrence_rate:.0%}. "
                    f"Root cause may need structural fix rather than per-incident patching."
                ),
                evidence=f"recurrence_rate={scorecard.recurrence_rate}, interventions={scorecard.intervention_count}",
            )
            created.append(hyp_id)
        return created

    def get_track_record(self) -> dict[str, dict]:
        """Return effectiveness scores for all hypotheses (for prompt injection)."""
        self.seed_if_needed()
        records = self._load_all()
        return {
            r.id: {
                "title": r.title,
                "effectiveness": round(r.effectiveness, 3),
                "times_proposed": r.times_proposed,
                "status": r.status,
            }
            for r in records
        }
