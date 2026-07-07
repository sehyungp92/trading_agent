"""Local perturbation repair candidate builders."""

from __future__ import annotations

from trading_assistant_backtest.auto.types import Candidate


def build_local_perturbations(base: Candidate, *, count: int = 2) -> list[Candidate]:
    return [
        Candidate(
            candidate_id=f"{base.candidate_id}-perturb-{index}",
            family=f"{base.family}_perturbation",
            payload={
                **base.payload,
                "repair_type": "local_perturbation",
                "perturbation_index": index,
                "source_candidate_id": base.candidate_id,
            },
        )
        for index in range(1, max(0, count) + 1)
    ]
