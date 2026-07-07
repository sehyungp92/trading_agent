"""Deterministic repair failure analysis."""

from __future__ import annotations


def empty_failure_analysis(run_id: str) -> dict:
    return {"run_id": run_id, "primary_failure": "none", "repair_triggered": False}


def analyze_failure(
    run_id: str,
    *,
    data_errors: list[str] | None = None,
    rejected_candidates: list[dict] | None = None,
    repair_triggered: bool = False,
) -> dict:
    errors = data_errors or []
    rejected = rejected_candidates or []
    reasons = [
        str(row.get("reason") or row.get("skip_reason") or "")
        for row in rejected
        if str(row.get("reason") or row.get("skip_reason") or "")
    ]
    if errors:
        primary = "data_contract"
    elif any("replay-backed evaluation" in reason for reason in reasons):
        primary = "shadow_replay_candidate_gate"
    elif any("replay" in reason or "plugin" in reason for reason in reasons):
        primary = "missing_replay_evaluator"
    elif rejected:
        primary = "candidate_gate_failure"
    else:
        primary = "none"
    return {
        "run_id": run_id,
        "primary_failure": primary,
        "repair_triggered": repair_triggered,
        "data_errors": errors,
        "rejected_candidate_count": len(rejected),
        "top_rejection_reasons": list(dict.fromkeys(reasons))[:10],
    }
