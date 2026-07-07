"""JSONL-backed operational priors from authoritative outcomes."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.schemas.monthly_outcome import MonthlyOutcomeRecord, MonthlyOutcomeVerdict, OutcomeSource
from trading_assistant.schemas.outcome_priors import (
    GateStrictness,
    OutcomePrior,
    RollbackPriority,
    make_outcome_prior_id,
)
from trading_assistant.skills._atomic_write import atomic_rewrite_jsonl


class OutcomePriorStore:
    """Maintains bounded priors used by future monthly candidate generation."""

    def __init__(self, findings_dir: Path) -> None:
        self._path = Path(findings_dir) / "outcome_priors.jsonl"

    def record_outcome(self, outcome: MonthlyOutcomeRecord) -> OutcomePrior | None:
        if outcome.source == OutcomeSource.EARLY_WARNING:
            return None

        family = outcome.mutation_family or outcome.category or "unknown"
        prior_id = make_outcome_prior_id(
            bot_id=outcome.bot_id,
            strategy_id=outcome.strategy_id,
            mutation_family=family,
            category=outcome.category,
        )
        priors = {prior.prior_id: prior for prior in self.load_all()}
        prior = priors.get(prior_id) or OutcomePrior(
            prior_id=prior_id,
            bot_id=outcome.bot_id,
            strategy_id=outcome.strategy_id,
            mutation_family=family,
            category=outcome.category,
        )
        if outcome.outcome_id in prior.source_outcome_ids:
            return prior

        if outcome.verdict == MonthlyOutcomeVerdict.KEEP:
            if outcome.is_positive_prior_eligible:
                prior.positive_count += 1
                if outcome.persistence_confirmed:
                    prior.confirmed_positive_count += 1
            else:
                prior.inconclusive_count += 1
        elif outcome.is_negative:
            prior.negative_count += 1
        else:
            prior.inconclusive_count += 1

        prior.latest_verdict = outcome.verdict.value
        prior.latest_outcome_id = outcome.outcome_id
        prior.source_outcome_ids = _dedupe([*prior.source_outcome_ids, outcome.outcome_id])
        prior.evidence_paths = _dedupe([*prior.evidence_paths, *outcome.evidence_paths])[-20:]
        prior.updated_at = datetime.now(timezone.utc)
        self._derive_controls(prior, outcome)
        priors[prior_id] = prior
        self._write_all(list(priors.values()))
        return prior

    def get_prior(
        self,
        *,
        bot_id: str,
        strategy_id: str = "",
        mutation_family: str = "",
        category: str = "",
    ) -> OutcomePrior | None:
        wanted_family = mutation_family or category or "unknown"
        wanted = make_outcome_prior_id(
            bot_id=bot_id,
            strategy_id=strategy_id,
            mutation_family=wanted_family,
            category=category,
        )
        priors = self.load_all()
        for prior in priors:
            if prior.prior_id == wanted:
                return prior
        flexible = self._find_flexible_match(
            priors,
            bot_id=bot_id,
            strategy_id=strategy_id,
            mutation_family=mutation_family,
            category=category,
        )
        if flexible is not None:
            return flexible
        if strategy_id:
            return self.get_prior(
                bot_id=bot_id,
                mutation_family=mutation_family,
                category=category,
            )
        return None

    @staticmethod
    def _find_flexible_match(
        priors: list[OutcomePrior],
        *,
        bot_id: str,
        strategy_id: str,
        mutation_family: str,
        category: str,
    ) -> OutcomePrior | None:
        tokens = {value for value in (mutation_family, category) if value}
        if not tokens:
            return None
        matches: list[OutcomePrior] = []
        for prior in priors:
            if prior.bot_id != bot_id:
                continue
            if strategy_id and prior.strategy_id not in {strategy_id, ""}:
                continue
            prior_tokens = {value for value in (prior.mutation_family, prior.category) if value}
            if tokens & prior_tokens:
                matches.append(prior)
        if not matches:
            return None
        return max(matches, key=lambda prior: prior.updated_at)

    def snapshot(
        self,
        *,
        bot_id: str = "",
        strategy_id: str = "",
    ) -> list[dict]:
        rows: list[dict] = []
        for prior in self.load_all():
            if bot_id and prior.bot_id != bot_id:
                continue
            if strategy_id and prior.strategy_id not in {strategy_id, ""}:
                continue
            rows.append(prior.model_dump(mode="json"))
        return sorted(rows, key=lambda row: row.get("updated_at", ""), reverse=True)

    def load_all(self) -> list[OutcomePrior]:
        if not self._path.exists():
            return []
        rows: list[OutcomePrior] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(OutcomePrior.model_validate(json.loads(line)))
            except Exception:
                continue
        return rows

    def _write_all(self, priors: list[OutcomePrior]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        atomic_rewrite_jsonl(self._path, priors)

    @staticmethod
    def _derive_controls(prior: OutcomePrior, outcome: MonthlyOutcomeRecord) -> None:
        positive_pressure = min(0.30, 0.05 * prior.positive_count + 0.10 * prior.confirmed_positive_count)
        negative_pressure = min(0.60, 0.15 * prior.negative_count)
        prior.allocation_multiplier = round(max(0.40, min(1.30, 1.0 + positive_pressure - negative_pressure)), 3)
        prior.required_confirmation_count = max(1, 1 + prior.negative_count)

        if prior.negative_count >= 1:
            prior.gate_strictness = GateStrictness.STRICTER
        elif prior.confirmed_positive_count >= 2:
            prior.gate_strictness = GateStrictness.RELAXED_EXPLORATION
        else:
            prior.gate_strictness = GateStrictness.NORMAL

        if outcome.verdict == MonthlyOutcomeVerdict.QUARANTINE:
            prior.rollback_priority = RollbackPriority.CRITICAL
        elif outcome.verdict == MonthlyOutcomeVerdict.ROLLBACK:
            prior.rollback_priority = (
                RollbackPriority.CRITICAL
                if _severe_degradation(outcome)
                else RollbackPriority.HIGH
            )
        elif outcome.verdict == MonthlyOutcomeVerdict.REPAIR:
            prior.rollback_priority = RollbackPriority.WATCH
        elif prior.negative_count:
            prior.rollback_priority = RollbackPriority.HIGH
        else:
            prior.rollback_priority = RollbackPriority.NONE


def _severe_degradation(outcome: MonthlyOutcomeRecord) -> bool:
    return (
        outcome.live_vs_expected_objective_delta <= -0.15
        or outcome.drawdown_delta >= 0.15
        or outcome.execution_slippage_delta >= 0.08
    )


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
