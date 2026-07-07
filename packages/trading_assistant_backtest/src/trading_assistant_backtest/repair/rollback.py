"""Rollback repair candidate builders."""

from __future__ import annotations

from collections.abc import Iterable

from trading_assistant_backtest.auto.types import Candidate


def build_rollback_candidates(prior_mutations: Iterable[dict]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for index, mutation in enumerate(prior_mutations, start=1):
        mutation_id = str(mutation.get("mutation_id") or mutation.get("candidate_id") or index)
        candidates.append(
            Candidate(
                candidate_id=f"rollback-{mutation_id}",
                family="rollback",
                payload={
                    "repair_type": "rollback",
                    "mutation_id": mutation_id,
                },
            )
        )
    return candidates
