# skills/learning_ledger.py
"""Learning ledger — JSONL-backed experiment log (autoresearch's results.tsv equivalent).

Tracks weekly ground truth snapshots, composite score deltas, and lessons learned.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from trading_assistant.schemas.learning_ledger import GroundTruthSnapshot, LearningLedgerEntry
from trading_assistant.skills._outcome_utils import is_conclusive_outcome, is_positive_outcome


class LearningLedger:
    """JSONL-backed weekly experiment log."""

    def __init__(self, findings_dir: Path) -> None:
        self._path = findings_dir / "learning_ledger.jsonl"
        self._findings_dir = findings_dir

    def record_week(
        self,
        week_start: str,
        week_end: str,
        gt_start: dict[str, GroundTruthSnapshot] | None = None,
        gt_end: dict[str, GroundTruthSnapshot] | None = None,
        bots: list[str] | None = None,
        *,
        composite_delta: dict[str, float] | None = None,
        net_improvement: bool | None = None,
        suggestions_proposed: int = 0,
        suggestions_accepted: int = 0,
        suggestions_implemented: int = 0,
        experiments_concluded: int = 0,
        discoveries_found: int = 0,
        what_worked: list[str] | None = None,
        what_failed: list[str] | None = None,
        lessons_for_next_week: list[str] | None = None,
        cycle_effectiveness: float = 0.0,
        inner_suggestions_proposed: int = 0,
        outer_suggestions_proposed: int = 0,
        inner_positive_outcomes: int = 0,
        outer_positive_outcomes: int = 0,
        inner_total_outcomes: int = 0,
        outer_total_outcomes: int = 0,
    ) -> LearningLedgerEntry:
        """Record a week's learning data. Deduplicates by entry_id."""
        gt_start = gt_start or {}
        gt_end = gt_end or {}

        entry_id = hashlib.sha256(
            f"{week_start}:{week_end}".encode()
        ).hexdigest()[:12]

        # Check dedup
        existing = self._load_all()
        for e in existing:
            if e.get("entry_id") == entry_id:
                return LearningLedgerEntry(**e)

        # Compute composite deltas if not provided
        if composite_delta is None:
            composite_delta = {}
            all_bots = set(gt_start) | set(gt_end)
            for bot_id in all_bots:
                start_score = gt_start.get(bot_id)
                end_score = gt_end.get(bot_id)
                if start_score and end_score:
                    composite_delta[bot_id] = round(
                        end_score.composite_score - start_score.composite_score, 4,
                    )

        if net_improvement is None:
            net_improvement = sum(composite_delta.values()) > 0 if composite_delta else False

        entry = LearningLedgerEntry(
            entry_id=entry_id,
            week_start=week_start,
            week_end=week_end,
            ground_truth_start={
                k: v for k, v in gt_start.items()
            },
            ground_truth_end={
                k: v for k, v in gt_end.items()
            },
            composite_delta=composite_delta,
            net_improvement=net_improvement,
            suggestions_proposed=suggestions_proposed,
            suggestions_accepted=suggestions_accepted,
            suggestions_implemented=suggestions_implemented,
            experiments_concluded=experiments_concluded,
            discoveries_found=discoveries_found,
            what_worked=what_worked or [],
            what_failed=what_failed or [],
            lessons_for_next_week=lessons_for_next_week or [],
            cycle_effectiveness=cycle_effectiveness,
            inner_suggestions_proposed=inner_suggestions_proposed,
            outer_suggestions_proposed=outer_suggestions_proposed,
            inner_positive_outcomes=inner_positive_outcomes,
            outer_positive_outcomes=outer_positive_outcomes,
            inner_total_outcomes=inner_total_outcomes,
            outer_total_outcomes=outer_total_outcomes,
        )

        # Persist
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

        return entry

    def get_trend(self, weeks: int = 12) -> dict[str, list[float]]:
        """Return composite_score per bot over the last N weeks."""
        entries = self._load_all()
        # Sort by week_start ascending
        entries.sort(key=lambda e: e.get("week_start", ""))
        recent = entries[-weeks:] if len(entries) > weeks else entries

        trend: dict[str, list[float]] = {}
        for entry in recent:
            gt_end = entry.get("ground_truth_end", {})
            for bot_id, snapshot in gt_end.items():
                score = snapshot.get("composite_score", 0.5) if isinstance(snapshot, dict) else 0.5
                trend.setdefault(bot_id, []).append(score)

        return trend

    def get_lessons(self, weeks: int = 4) -> list[str]:
        """Return aggregated lessons from the last N weeks."""
        entries = self._load_all()
        entries.sort(key=lambda e: e.get("week_start", ""))
        recent = entries[-weeks:] if len(entries) > weeks else entries

        lessons: list[str] = []
        seen: set[str] = set()
        for entry in reversed(recent):  # most recent first
            for lesson in entry.get("lessons_for_next_week", []):
                normalized = lesson.strip().lower()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    lessons.append(lesson)
        return lessons

    def get_latest(self) -> LearningLedgerEntry | None:
        """Return the most recent ledger entry, or None."""
        entries = self._load_all()
        if not entries:
            return None
        entries.sort(key=lambda e: e.get("week_start", ""))
        return LearningLedgerEntry(**entries[-1])

    def get_curated_notes(self, max_notes: int = 30) -> list[str]:
        """Return curated, deduplicated lessons with relevance decay and outcome boosting.

        - Loads 52 weeks of lessons
        - Deduplicates by Jaccard similarity (threshold 0.6)
        - Applies 5%/week relevance decay
        - Boosts lessons corroborated by outcome measurements
        """
        entries = self._load_all()
        if not entries:
            return []

        entries.sort(key=lambda e: e.get("week_start", ""))

        # Load outcome verdicts for boosting
        outcome_keywords = self._load_outcome_keywords()

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        # Collect (lesson, score) pairs
        scored: list[tuple[str, float]] = []
        for entry in entries:
            week_start = entry.get("week_start", "")
            try:
                entry_date = datetime.strptime(week_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                weeks_ago = max(0, (now - entry_date).days / 7.0)
            except (ValueError, TypeError):
                weeks_ago = 52.0

            decay = 0.95 ** weeks_ago  # 5%/week decay

            for lesson in entry.get("lessons_for_next_week", []):
                if not lesson.strip():
                    continue
                # Boost if lesson keywords match outcome data
                boost = 1.0
                lesson_lower = lesson.lower()
                for kw in outcome_keywords:
                    if kw in lesson_lower:
                        boost = 1.5
                        break
                scored.append((lesson.strip(), decay * boost))

        # Deduplicate by Jaccard similarity
        curated: list[tuple[str, float]] = []
        for lesson, score in scored:
            tokens = set(lesson.lower().split())
            is_duplicate = False
            for i, (existing, existing_score) in enumerate(curated):
                existing_tokens = set(existing.lower().split())
                intersection = tokens & existing_tokens
                union = tokens | existing_tokens
                jaccard = len(intersection) / len(union) if union else 0
                if jaccard >= 0.6:
                    # Keep higher-scored version
                    if score > existing_score:
                        curated[i] = (lesson, score)
                    is_duplicate = True
                    break
            if not is_duplicate:
                curated.append((lesson, score))

        # Sort by score descending, take top N
        curated.sort(key=lambda x: x[1], reverse=True)
        return [lesson for lesson, _ in curated[:max_notes]]

    def record_outcome_lessons(
        self, week_start: str, lessons: list[str], *, source: str = "outcomes",
    ) -> None:
        """Append outcome-derived lessons as a standalone ledger entry.

        Uses a distinct entry_id prefix so it won't collide with the main
        weekly entry created by record_week().

        Args:
            week_start: The week identifier.
            lessons: Lesson strings to record.
            source: Entry source prefix for dedup (e.g. "outcomes", "corrections").
        """
        if not lessons:
            return
        entry_id = hashlib.sha256(
            f"{source}:{week_start}".encode()
        ).hexdigest()[:12]

        existing = self._load_all()
        for e in existing:
            if e.get("entry_id") == entry_id:
                return  # already recorded

        entry = LearningLedgerEntry(
            entry_id=entry_id,
            week_start=week_start,
            week_end=week_start,
            lessons_for_next_week=lessons,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    def compute_cycle_effectiveness(
        self,
        composite_delta: dict[str, float],
        suggestions_proposed: int,
        suggestions_implemented: int,
        lessons: list[str],
        week_start: str,
        week_end: str,
    ) -> float:
        """Compute a normalized 0.0-1.0 cycle effectiveness score.

        Four equal-weight components:
        - improvement_magnitude: sigmoid(mean(composite_delta values) × 10)
        - conversion_rate: implemented / max(proposed, 1)
        - outcome_quality: positive / max(total, 1) for that week
        - lesson_yield: min(1.0, len(lessons) / 5)
        """
        # 1. Improvement magnitude via sigmoid
        if composite_delta:
            mean_delta = sum(composite_delta.values()) / len(composite_delta)
        else:
            mean_delta = 0.0
        improvement_magnitude = 1.0 / (1.0 + math.exp(-mean_delta * 10))

        # 2. Conversion rate
        conversion_rate = suggestions_implemented / max(suggestions_proposed, 1)

        # 3. Outcome quality for this week
        outcome_quality = self._compute_week_outcome_quality(week_start, week_end)

        # 4. Lesson yield
        lesson_yield = min(1.0, len(lessons) / 5)

        return round(
            (improvement_magnitude + conversion_rate + outcome_quality + lesson_yield) / 4,
            4,
        )

    def _compute_week_outcome_quality(self, week_start: str, week_end: str) -> float:
        """Compute positive outcome ratio for a given week from outcomes.jsonl."""
        outcomes_path = self._findings_dir / "outcomes.jsonl"
        if not outcomes_path.exists():
            return 0.0
        positive = 0
        total = 0
        try:
            for line in outcomes_path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                ts = entry.get("measurement_date") or entry.get("measured_at") or entry.get("timestamp", "")
                if ts and len(ts) >= 10 and week_start <= ts[:10] <= week_end:
                    if not is_conclusive_outcome(entry):
                        continue
                    total += 1
                    if is_positive_outcome(entry):
                        positive += 1
        except (json.JSONDecodeError, OSError):
            pass
        return positive / max(total, 1)

    def _load_outcome_keywords(self) -> set[str]:
        """Extract bot_id and category keywords from outcomes for lesson boosting."""
        outcomes_path = self._findings_dir / "outcomes.jsonl"
        if not outcomes_path.exists():
            return set()
        keywords: set[str] = set()
        try:
            for line in outcomes_path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                bot_id = entry.get("bot_id", "")
                if bot_id:
                    keywords.add(bot_id.lower())
                verdict = entry.get("verdict", "")
                if verdict in ("positive", "negative"):
                    keywords.add(verdict)
        except (json.JSONDecodeError, OSError):
            pass
        return keywords

    def _load_all(self) -> list[dict]:
        """Load all ledger entries from JSONL."""
        if not self._path.exists():
            return []
        entries: list[dict] = []
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries
