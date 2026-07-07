"""Outcome-prior controls for monthly candidate ordering and gates."""
from __future__ import annotations

from typing import Any

from trading_assistant.schemas.outcome_priors import GateStrictness
from trading_assistant.skills.outcome_prior_store import OutcomePriorStore


class SearchAllocationPolicy:
    """Applies outcome priors without overriding hard safety gates."""

    def __init__(self, prior_store: OutcomePriorStore) -> None:
        self._prior_store = prior_store

    def allocation_multiplier(
        self,
        *,
        bot_id: str,
        strategy_id: str = "",
        mutation_family: str = "",
        category: str = "",
    ) -> float:
        prior = self._prior_store.get_prior(
            bot_id=bot_id,
            strategy_id=strategy_id,
            mutation_family=mutation_family,
            category=category,
        )
        return prior.allocation_multiplier if prior else 1.0

    def requires_stronger_evidence(
        self,
        *,
        bot_id: str,
        strategy_id: str = "",
        mutation_family: str = "",
        category: str = "",
    ) -> bool:
        prior = self._prior_store.get_prior(
            bot_id=bot_id,
            strategy_id=strategy_id,
            mutation_family=mutation_family,
            category=category,
        )
        return bool(prior and prior.gate_strictness == GateStrictness.STRICTER)

    def order_candidates(self, candidates: list[Any]) -> list[Any]:
        """Order candidates by prior allocation while preserving stable ties."""

        def _score(indexed: tuple[int, Any]) -> tuple[float, int]:
            index, candidate = indexed
            mult = self.allocation_multiplier(
                bot_id=getattr(candidate, "bot_id", ""),
                strategy_id=getattr(candidate, "strategy_id", ""),
                mutation_family=getattr(candidate, "family", ""),
                category=str(getattr(candidate, "change_kind", "") or ""),
            )
            return (-mult, index)

        return [candidate for _, candidate in sorted(enumerate(candidates), key=_score)]
