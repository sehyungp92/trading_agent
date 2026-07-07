"""Weekly-to-monthly bounded search-prior contract."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class MonthlySearchBrief(BaseModel):
    """Non-authoritative search guidance for a monthly validation run."""

    monthly_search_brief_id: str = ""
    run_month: str
    bot_id: str = ""
    strategy_id: str = ""
    report_only: bool = True
    experiment_focus_hints: list[dict[str, Any]] = Field(default_factory=list)
    phased_auto_priority_families: list[dict[str, Any]] = Field(default_factory=list)
    phase_order_hints: list[str] = Field(default_factory=list)
    seed_candidates: list[dict[str, Any]] = Field(default_factory=list)
    conditional_oos_repair_ablation_priorities: list[dict[str, Any]] = Field(default_factory=list)
    rollback_candidates: list[dict[str, Any]] = Field(default_factory=list)
    negative_priors: list[dict[str, Any]] = Field(default_factory=list)
    confidence_caps: list[dict[str, Any]] = Field(default_factory=list)
    weekly_focus: dict[str, Any] = Field(default_factory=dict)
    weekly_focus_rotation: list[dict[str, Any]] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    source_weekly_signal_ids: list[str] = Field(default_factory=list)
    attribution: dict[str, list[str]] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    brief_version: str = "monthly_search_brief_v1"

    def model_post_init(self, __context: object) -> None:
        if not self.monthly_search_brief_id:
            raw = "|".join([
                self.run_month,
                self.bot_id,
                self.strategy_id,
                str((self.weekly_focus or {}).get("focus_id") or ""),
                ",".join(self.source_weekly_signal_ids[:20]),
                ",".join(self.evidence_paths[:20]),
            ])
            self.monthly_search_brief_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_optimizer_guidance(self) -> dict[str, Any]:
        """Return the bounded subset intended for optimizer planning."""
        required_families = _dedupe([
            *_families_from(self.phased_auto_priority_families),
            *_families_from(self.seed_candidates),
        ])
        rollback_families = _dedupe(_families_from(self.rollback_candidates))
        negative_families = _dedupe(_families_from(self.negative_priors))
        focus_families = _dedupe(_string_list((self.weekly_focus or {}).get("portfolio_families")))
        focus_strategy_ids = _dedupe(_string_list((self.weekly_focus or {}).get("strategy_ids")))
        return {
            "authority": "search_order_only",
            "brief_id": self.monthly_search_brief_id,
            "weekly_focus": self.weekly_focus,
            "weekly_focus_rotation": self.weekly_focus_rotation,
            "phase_order_hints": self.phase_order_hints[:8],
            "priority_families": self.phased_auto_priority_families[:12],
            "seed_candidates": self.seed_candidates[:20],
            "negative_priors": self.negative_priors[:20],
            "rollback_candidates": self.rollback_candidates[:12],
            "conditional_oos_repair_ablation_priorities": (
                self.conditional_oos_repair_ablation_priorities[:20]
            ),
            "confidence_caps": self.confidence_caps[:20],
            "source_weekly_signal_ids": self.source_weekly_signal_ids[:50],
            "plan_requirements": {
                "candidate_families": required_families[:12],
                "rollback_families": rollback_families[:12],
                "negative_prior_families": negative_families[:12],
                "weekly_focus_id": str((self.weekly_focus or {}).get("focus_id") or ""),
                "focus_portfolio_families": focus_families[:12],
                "focus_strategy_ids": focus_strategy_ids[:20],
                "source_weekly_signal_ids": self.source_weekly_signal_ids[:50],
            },
        }

    def apply_to_experiment_plan_payload(
        self,
        plan: dict[str, Any],
        *,
        brief_path: str = "",
    ) -> dict[str, Any]:
        """Deterministically project bounded guidance into an optimizer plan payload."""
        updated = dict(plan)
        if brief_path:
            updated["evidence_paths"] = _dedupe([
                *_string_list(updated.get("evidence_paths")),
                brief_path,
            ])
        updated["source_weekly_signal_ids"] = _dedupe([
            *_string_list(updated.get("source_weekly_signal_ids")),
            *self.source_weekly_signal_ids,
        ])
        phase_order = _string_list(updated.get("phase_order"))
        for hint in self.phase_order_hints[:4]:
            token = _slug(str(hint))
            if token:
                phase_order.append(f"brief_{token[:40]}")
        updated["phase_order"] = _dedupe(phase_order)

        families = list(updated.get("candidate_families") or [])
        for item in [*self.phased_auto_priority_families, *self.seed_candidates]:
            family = _family(item)
            if not family:
                continue
            families.append({
                "family": family,
                "phase": item.get("phase", "brief_guided"),
                "priority": item.get("priority") or item.get("seed_type") or "brief_guided",
                "source_weekly_signal_id": item.get("source_weekly_signal_id", ""),
                "authority": "search_order_only",
                "monthly_search_brief_id": self.monthly_search_brief_id,
            })
        focus = self.weekly_focus or {}
        for family in _string_list(focus.get("portfolio_families")):
            families.append({
                "family": family,
                "phase": "weekly_focus_rotation",
                "priority": "focus_scope",
                "source_weekly_signal_id": f"weekly_focus:{focus.get('focus_id', '')}",
                "authority": "search_order_only",
                "monthly_search_brief_id": self.monthly_search_brief_id,
            })
        updated["candidate_families"] = _dedupe_dicts(families)

        gate_expectations = _string_list(updated.get("gate_expectations"))
        if self.weekly_focus:
            gate_expectations.append(
                "Use weekly_focus as search-order scope only; it cannot satisfy monthly gates."
            )
        if self.confidence_caps:
            gate_expectations.append("Apply monthly_search_brief confidence caps as search-order constraints only.")
        if self.rollback_candidates:
            gate_expectations.append("Inspect rollback families during search; do not bypass approval gates.")
        updated["gate_expectations"] = _dedupe(gate_expectations)

        overfit_risks = _string_list(updated.get("overfit_risks"))
        for family in _dedupe([
            *_families_from(self.negative_priors),
            *_families_from(self.rollback_candidates),
        ])[:12]:
            overfit_risks.append(f"monthly_search_brief caution for family:{family}")
        updated["overfit_risks"] = _dedupe(overfit_risks)
        return updated


def _family(item: dict[str, Any]) -> str:
    return str(item.get("family") or item.get("category") or item.get("mutation_family") or "").strip()


def _families_from(items: list[dict[str, Any]]) -> list[str]:
    return [_family(item) for item in items if _family(item)]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item).strip()]
    return []


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _dedupe_dicts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        key = str(value.get("family") or "") + "|" + str(value.get("source_weekly_signal_id") or "")
        if not key.strip("|"):
            key = repr(sorted(value.items()))
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _slug(value: str) -> str:
    chars: list[str] = []
    prev_sep = False
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
            prev_sep = False
        elif not prev_sep:
            chars.append("_")
            prev_sep = True
    return "".join(chars).strip("_")
