"""Assistant proposal lineage shared by monthly artifacts and runtime events."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class AssistantLineage(BaseModel):
    """Stable IDs that connect assistant suggestions to deployed outcomes."""

    weekly_signal_ids: list[str] = Field(default_factory=list)
    monthly_search_brief_id: str = ""
    proposal_ids: list[str] = Field(default_factory=list)
    suggestion_ids: list[str] = Field(default_factory=list)
    hypothesis_ids: list[str] = Field(default_factory=list)
    experiment_id: str = ""
    variant_id: str = ""
    parameter_set_id: str = ""
    deployment_id: str = ""
    strategy_change_record_ids: list[str] = Field(default_factory=list)
    monthly_outcome_id: str = ""

    @model_validator(mode="before")
    @classmethod
    def _from_legacy_payload(cls, data: Any) -> Any:
        if data is None:
            return {}
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        payload["weekly_signal_ids"] = _dedupe([
            *_string_list(payload.get("weekly_signal_ids")),
            *_string_list(payload.get("source_weekly_signal_ids")),
            *_string_list(payload.get("weekly_signal_id")),
        ])
        payload["proposal_ids"] = _dedupe([
            *_string_list(payload.get("proposal_ids")),
            *_string_list(payload.get("proposal_id")),
        ])
        payload["suggestion_ids"] = _dedupe([
            *_string_list(payload.get("suggestion_ids")),
            *_string_list(payload.get("suggestion_id")),
        ])
        payload["hypothesis_ids"] = _dedupe([
            *_string_list(payload.get("hypothesis_ids")),
            *_string_list(payload.get("hypothesis_id")),
        ])
        payload["strategy_change_record_ids"] = _dedupe([
            *_string_list(payload.get("strategy_change_record_ids")),
            *_string_list(payload.get("strategy_change_record_id")),
        ])
        return payload

    def has_any(self) -> bool:
        return any(value not in ("", None, [], {}) for value in self.model_dump().values())


def assistant_lineage_from_fields(
    *,
    weekly_signal_ids: list[str] | None = None,
    monthly_search_brief_id: str = "",
    proposal_ids: list[str] | None = None,
    suggestion_ids: list[str] | None = None,
    hypothesis_ids: list[str] | None = None,
    experiment_id: str = "",
    variant_id: str = "",
    parameter_set_id: str = "",
    deployment_id: str = "",
    strategy_change_record_ids: list[str] | None = None,
    monthly_outcome_id: str = "",
) -> AssistantLineage:
    return AssistantLineage(
        weekly_signal_ids=_dedupe(weekly_signal_ids or []),
        monthly_search_brief_id=monthly_search_brief_id,
        proposal_ids=_dedupe(proposal_ids or []),
        suggestion_ids=_dedupe(suggestion_ids or []),
        hypothesis_ids=_dedupe(hypothesis_ids or []),
        experiment_id=experiment_id,
        variant_id=variant_id,
        parameter_set_id=parameter_set_id,
        deployment_id=deployment_id,
        strategy_change_record_ids=_dedupe(strategy_change_record_ids or []),
        monthly_outcome_id=monthly_outcome_id,
    )


def _string_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "")]
    return [str(value)]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
