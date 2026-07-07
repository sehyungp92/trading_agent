"""Artifact authority classification for loop and approval evidence."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ArtifactAuthority(str, Enum):
    BINDING = "binding"
    APPROVAL_GATE = "approval_gate"
    ADVISORY = "advisory"
    DIAGNOSTICS_ONLY = "diagnostics_only"
    GENERATED = "generated"
    HUMAN_OWNED = "human_owned"


class ArtifactRegistryEntry(BaseModel):
    artifact_type: str
    authority: ArtifactAuthority
    owner_package: str = "trading_assistant"
    schema_ref: str = ""
    source_contract: str = ""
    path_patterns: list[str] = Field(default_factory=list)
    may_satisfy_approval_gate: bool = False
    readers: list[str] = Field(default_factory=list)
    writers: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _derive_gate_eligibility(self) -> "ArtifactRegistryEntry":
        if self.authority == ArtifactAuthority.ADVISORY and self.may_satisfy_approval_gate:
            raise ValueError(f"{self.artifact_type} is advisory and cannot satisfy approval gates")
        if self.authority == ArtifactAuthority.HUMAN_OWNED and self.writers:
            raise ValueError(f"{self.artifact_type} is human-owned and cannot list autonomous writers")
        return self


class ArtifactAuthorityIssue(BaseModel):
    artifact: str
    message: str
    am_row: str = "AM-06"
    remediation: str = ""


def artifact_type_from_path(path: str) -> str:
    raw = str(path or "").replace("\\", "/")
    if not raw:
        return ""
    lowered = raw.lower()
    if "/memory/policies/" in lowered or lowered.endswith("/memory/policies"):
        return "memory_policies"
    name = Path(raw).name.lower()
    stem = name.rsplit(".", 1)[0]
    if name == "coverage_manifest.json" or name.endswith(".coverage_manifest.json"):
        return "coverage_manifest"
    aliases = {
        "run_manifest": "monthly_run_manifest",
        "telemetry_manifest": "telemetry_manifest",
        "learning_sufficiency_manifest": "learning_sufficiency_manifest",
        "strategy_discovery_packet": "strategy_discovery_packet",
        "replay_parity_report": "replay_parity_report",
        "runner_observability": "runner_observability",
        "monthly_validation_result": "monthly_validation_result",
        "model_review_validation": "model_review_validation",
        "model_review_request": "model_review_request",
        "model_review_prompt": "model_review_prompt",
        "model_review": "monthly_model_review",
        "candidate_gate_report": "candidate_gate_report",
        "candidate_generation_summary": "candidate_generation_summary",
        "objective_breakdown": "objective_breakdown",
        "selected_candidates": "selected_candidates",
        "rejected_candidates": "rejected_candidates",
        "candidate_results": "candidate_results",
        "artifact_index": "artifact_index",
        "outcome_priors_snapshot": "outcome_priors_snapshot",
    }
    if stem in aliases:
        return aliases[stem]
    wildcard_aliases = {
        "approval_packet_": "approval_packet",
        "monthly_evidence_verification_": "monthly_evidence_verification",
    }
    for prefix, artifact_type in wildcard_aliases.items():
        if stem.startswith(prefix):
            return artifact_type
    return stem


def payload_paths(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item)]
    return []
