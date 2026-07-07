"""Targeted repair candidate builders."""

from __future__ import annotations

from trading_assistant_backtest.auto.types import Candidate


def build_targeted_candidates(failure_analysis: dict) -> list[Candidate]:
    primary = str(failure_analysis.get("primary_failure") or "unknown")
    if primary in {"none", "data_contract"}:
        return []
    family = (
        "replay_evaluator_enablement"
        if primary == "missing_replay_evaluator"
        else "targeted_repair"
    )
    return [
        Candidate(
            candidate_id=f"repair-{family}",
            family=family,
            payload={
                "repair_type": "targeted",
                "primary_failure": primary,
            },
        )
    ]
