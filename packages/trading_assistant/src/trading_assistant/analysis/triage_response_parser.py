"""Parse structured bug-triage agent output."""
from __future__ import annotations

import json
import logging
import re

from trading_assistant.schemas.bug_triage import TriageRepairProposal

logger = logging.getLogger(__name__)

_COMMENT_BLOCK = re.compile(
    r"<!--\s*(?:TRIAGE_RESULT|STRUCTURED_OUTPUT)\s*\n(.*?)\n\s*-->",
    re.DOTALL,
)
_JSON_FENCE = re.compile(r"```json\s*(.*?)```", re.DOTALL)


def parse_triage_response(response: str) -> TriageRepairProposal | None:
    """Parse a structured triage response from comment block or JSON fence."""
    if not response:
        return None

    candidates = []
    for pattern in (_COMMENT_BLOCK, _JSON_FENCE):
        match = pattern.search(response)
        if match:
            candidates.append(match.group(1).strip())

    if not candidates:
        return None

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            logger.warning("Failed to parse triage structured JSON", exc_info=True)
            continue
        try:
            return TriageRepairProposal(**data)
        except Exception:
            logger.warning("Invalid triage structured payload", exc_info=True)
    return None
