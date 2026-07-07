"""EvidenceMemory-backed ContextBuilder compatibility loaders."""

from __future__ import annotations

from datetime import datetime

from trading_assistant.analysis.context_sources.common import FINDINGS_MAX_AGE_DAYS as _FINDINGS_MAX_AGE_DAYS



class SourceFacadeContextMixin:
    def load_corrections(
        self,
        bot_id: str = "",
        *,
        max_age_days: int = _FINDINGS_MAX_AGE_DAYS,
        as_of: datetime | None = None,
    ) -> list[dict]:
        """Load manual corrections from findings/corrections.jsonl.

        Applies temporal decay and caps at 50 entries.
        If bot_id is provided, only returns corrections relevant to that bot.
        """
        return self._evidence_memory().findings.load_corrections(
            bot_id,
            max_age_days=max_age_days,
            as_of=as_of,
        )

    def load_failure_log(self, bot_id: str = "") -> list[dict]:
        """Load failure log entries from findings/failure-log.jsonl.

        Applies temporal decay: sorted by recency, capped at 90 days / 50 entries.
        If bot_id is provided, only returns entries relevant to that bot.
        Drops entries pinned to retired strategies.
        """
        return self._evidence_memory().findings.load_failure_log(bot_id)

    def load_rejected_suggestions(self) -> list[dict]:
        """Load rejected suggestions from findings/suggestions.jsonl.

        Drops entries pinned to retired strategies.
        """
        return self._evidence_memory().findings.load_rejected_suggestions()

    def load_outcome_measurements(
        self, min_quality: str = "medium",
    ) -> tuple[list[dict], list[dict]]:
        """Load outcome measurements from findings/outcomes.jsonl."""
        return self._evidence_memory().findings.load_outcome_measurements(min_quality)

    def load_monthly_outcomes(
        self, bot_id: str = "", days: int = 365, max_entries: int = 30,
    ) -> list[dict]:
        """Load authoritative monthly/follow-up outcome verdicts."""
        return self._evidence_memory().findings.load_monthly_outcomes(
            bot_id=bot_id,
            days=days,
            max_entries=max_entries,
        )

    def load_outcome_priors(
        self, bot_id: str = "", max_entries: int = 30,
    ) -> list[dict]:
        """Load operational priors that steer monthly search/allocation."""
        return self._evidence_memory().findings.load_outcome_priors(
            bot_id=bot_id,
            max_entries=max_entries,
        )

    def load_allocation_history(self) -> list[dict]:
        """Load allocation history from findings/allocation_history.jsonl.

        Applies temporal decay: sorted by recency, capped at 90 days / 50 entries.
        """
        return self._evidence_memory().findings.load_allocation_history()

    def load_loop_contract_context(self, agent_type: str = "") -> dict:
        """Load the checked loop contract for a recurring workflow."""
        return self._evidence_memory().loop_contracts.load(agent_type)

    def load_recent_work_log_entries(
        self,
        *,
        agent_type: str = "",
        bot_id: str = "",
        limit: int = 10,
    ) -> list[dict]:
        """Load bounded loop activity projection entries for prompt continuity."""
        return self._evidence_memory().findings.load_recent_work_log_entries(
            agent_type=agent_type,
            bot_id=bot_id,
            limit=limit,
        )

    def load_recent_performance_learning_entries(
        self,
        *,
        bot_id: str = "",
        strategy_id: str = "",
        portfolio_id: str = "",
        limit: int = 10,
    ) -> list[dict]:
        """Load bounded performance-learning summaries for prompt continuity."""
        return self._evidence_memory().performance_learning.load(
            bot_id=bot_id,
            strategy_id=strategy_id,
            portfolio_id=portfolio_id,
            limit=limit,
        )
