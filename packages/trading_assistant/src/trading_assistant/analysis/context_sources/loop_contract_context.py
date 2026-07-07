"""Loop-contract prompt context source."""

from __future__ import annotations

from pathlib import Path

from trading_assistant.skills.loop_contract_store import LoopContractStore


_LOOP_ALIASES = {
    "weekly_analysis": "weekly_summary",
    "monthly_model_review": "monthly_validation",
    "triage": "bug_triage",
}


class LoopContractContextSource:
    def __init__(self, memory_dir: Path) -> None:
        self._memory_dir = Path(memory_dir)

    def load(self, agent_type: str = "") -> dict:
        if not agent_type:
            return {}
        loop_id = _LOOP_ALIASES.get(agent_type, agent_type)
        return LoopContractStore(self._memory_dir).context_for_prompt(loop_id)
