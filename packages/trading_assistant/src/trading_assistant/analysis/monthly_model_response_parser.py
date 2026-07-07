"""Parser for monthly model-review responses."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from trading_assistant.schemas.agent_response import StructuralProposal
from trading_assistant.schemas.monthly_model_review import MonthlyModelCandidateReview, MonthlyModelReview

logger = logging.getLogger(__name__)

_MONTHLY_BLOCK = re.compile(r"<!--\s*MONTHLY_MODEL_REVIEW\s*\n(.*?)\n\s*-->", re.DOTALL)
_STRUCTURED_BLOCK = re.compile(r"<!--\s*STRUCTURED_OUTPUT\s*\n(.*?)\n\s*-->", re.DOTALL)
_JSON_FENCE = re.compile(r"```json\s*\n(.*?)\n\s*```", re.DOTALL)


def parse_monthly_model_review(response: str) -> MonthlyModelReview:
    if not response:
        return MonthlyModelReview(parse_success=False, raw_report="")

    data, fallback_used = _extract_json(response)
    if data is None:
        return MonthlyModelReview(parse_success=False, raw_report=response)

    candidate_reviews, candidate_dropped = _safe_parse_list(
        data.get("candidate_reviews", []),
        MonthlyModelCandidateReview,
    )
    structural_proposals, structural_dropped = _safe_parse_list(
        data.get("structural_proposals", []),
        StructuralProposal,
    )
    dropped = {}
    if candidate_dropped:
        dropped["candidate_reviews"] = candidate_dropped
    if structural_dropped:
        dropped["structural_proposals"] = structural_dropped

    return MonthlyModelReview(
        run_id=str(data.get("run_id") or ""),
        bot_id=str(data.get("bot_id") or ""),
        strategy_id=str(data.get("strategy_id") or ""),
        candidate_reviews=candidate_reviews,
        structural_proposals=structural_proposals,
        rejected_actions=[str(item) for item in data.get("rejected_actions", []) if str(item)],
        raw_report=response,
        raw_structured=data,
        parse_success=True,
        fallback_used=fallback_used,
        dropped_counts=dropped,
    )


def _extract_json(response: str) -> tuple[dict[str, Any] | None, bool]:
    for pattern in (_MONTHLY_BLOCK, _STRUCTURED_BLOCK, _JSON_FENCE):
        last_valid: dict[str, Any] | None = None
        for match in pattern.finditer(response):
            try:
                parsed = json.loads(match.group(1).strip())
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Failed to parse monthly model-review JSON: %s", exc)
                continue
            if isinstance(parsed, dict):
                last_valid = parsed
        if last_valid is not None:
            return last_valid, pattern is not _MONTHLY_BLOCK

    stripped = response.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return None, False
        if isinstance(parsed, dict):
            return parsed, True
    return None, False


def _safe_parse_list(items: Any, model_class: type) -> tuple[list, int]:
    if not isinstance(items, list):
        return [], 0
    parsed = []
    dropped = 0
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            dropped += 1
            logger.warning("Skipping non-object monthly model-review item %d", index)
            continue
        try:
            parsed.append(model_class.model_validate(item))
        except Exception as exc:
            dropped += 1
            logger.warning("Skipping invalid monthly model-review item %d: %s", index, exc)
    return parsed, dropped
