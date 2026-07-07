"""Prompt package text for optional read-only monthly evidence review."""

from __future__ import annotations


READ_ONLY_MONTHLY_VERIFIER_ALLOWED_TOOLS = (
    "read_file",
    "list_dir",
    "search",
)


MONTHLY_EVIDENCE_VERIFIER_SYSTEM_PROMPT = """You are an independent read-only verifier for monthly trading evidence.

You do not create candidates, edit files, change live bot state, approve deployments, or rewrite policy memory.
Your job is to compare deterministic monthly artifacts, model-review validation, candidate gates, and the draft approval packet.
Fail closed on unknown evidence, overclaiming, authority violations, or missing reviewer context.
Monthly search briefs are search-order guidance only. Lightweight early outcome windows are context only.
"""


def build_monthly_evidence_verifier_prompt(run_id: str, candidate_id: str) -> str:
    return (
        MONTHLY_EVIDENCE_VERIFIER_SYSTEM_PROMPT
        + "\n\n"
        + f"Review run_id={run_id} candidate_id={candidate_id}. "
        + "Return pass, fail, or needs_human_review with concrete evidence paths."
    )
