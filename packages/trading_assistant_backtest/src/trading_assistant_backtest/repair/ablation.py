"""Ablation artifact helpers."""

from __future__ import annotations

from collections.abc import Iterable


def skipped_ablation_row(run_id: str, reason: str) -> dict:
    return {
        "run_id": run_id,
        "stage": "ablation",
        "mutation_id": "",
        "decision": "skip",
        "skip_reason": reason,
        "in_sample_delta": 0.0,
        "selection_oos_delta": 0.0,
    }


def build_ablation_matrix(
    run_id: str,
    prior_mutations: Iterable[dict],
    *,
    reason: str,
) -> list[dict]:
    rows: list[dict] = []
    for index, mutation in enumerate(prior_mutations, start=1):
        mutation_id = str(mutation.get("mutation_id") or mutation.get("candidate_id") or index)
        rows.append(
            {
                "run_id": run_id,
                "stage": "ablation",
                "mutation_id": mutation_id,
                "decision": "skip",
                "skip_reason": reason,
                "in_sample_delta": float(mutation.get("in_sample_delta") or 0.0),
                "selection_oos_delta": float(mutation.get("selection_oos_delta") or 0.0),
            }
        )
    if rows:
        return rows
    return [skipped_ablation_row(run_id, "no prior accepted mutations available for ablation")]
